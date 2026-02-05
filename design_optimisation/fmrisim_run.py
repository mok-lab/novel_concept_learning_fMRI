#!/usr/bin/env python3
"""
fmrisim + CORnet patterns simulation with LOC ROI defined from coordinates or mask fractions (no ROI mask file)

What this does
--------------
- Loads your design CSV and CORnet-derived encoding patterns (patterns_enc_200 from NPZ)
- Builds trial-wise HRF-convolved regressors for 3 events: encoding, decision, feedback
- Simulates ROI signal time series (T x 200) using:
    Y_sig = X_enc @ P_enc + X_dec @ P_dec + X_fb @ P_fb
  where P_dec/P_fb are noisy mixtures of P_enc (per-trial overlap control)
- Estimates a noise model from a real 4D NIfTI using BrainIAK fmrisim.calc_noise
- Generates matched synthetic noise with fmrisim.generate_noise
- Defines a single LOC ROI as a ~200-voxel spherical blob centered at a coordinate:
    default: estimated from brain-mask fractions for native EPI space
  The blob is built in the *voxel space* of your noise NIfTI using its affine.
- Samples exactly n_vox voxels from that ROI to produce ROI noise and combines with signal
- Fits LS-A and LS-S trial-wise encoding patterns and reports recovery (row-wise correlation)

Requirements
------------
pip install brainiak nibabel numpy pandas scipy

Key assumptions
---------------
1) Your CORnet patterns NPZ was generated from the *same* CSV and with the same ordering:
   sorted by run_id then img_onset.
2) The LOC center can be defined in MNI space or estimated in native EPI space using brain-mask
   fractions; choose the appropriate --loc_space for your data.

References
----------
- BrainIAK fmrisim docs: https://brainiak.org/docs/brainiak.utils.html
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

from brainiak.utils import fmrisim


REQUIRED_COLS = {"run_id","trial_id","img_onset","img_dur","dec_onset_est","isi2_dur","fb_dur"}

from scipy.stats import spearmanr
from scipy.stats import gamma
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, label

def get_double_gamma_hrf(temporal_resolution, duration=32.0):
    """
    Manually generate a double-gamma HRF vector (1D).
    This mimics the standard SPM/Glover HRF used by BrainIAK but 
    ensures the output is a flat 1D array to prevent dimension errors.
    """
    # Parameters (Standard SPM/Glover defaults)
    peak_delay = 6.0
    undershoot_delay = 12.0
    peak_disp = 1.0
    undershoot_disp = 1.0
    p_u_ratio = 1.0 / 6.0
    
    # Create time vector
    tr_step = 1.0 / temporal_resolution
    t = np.arange(0, duration, tr_step)
    
    # Calculate PDF
    peak = gamma.pdf(t, peak_delay / peak_disp, scale=peak_disp)
    undershoot = gamma.pdf(t, undershoot_delay / undershoot_disp, scale=undershoot_disp)
    
    hrf = peak - p_u_ratio * undershoot
    
    # Normalize max amplitude to 1
    if np.max(np.abs(hrf)) > 0:
        hrf = hrf / np.max(np.abs(hrf))
        
    return hrf


# ------------------------
# Basic GLM utilities
# ------------------------
def spm_dctmtx(N: int, K: int) -> np.ndarray:
    K = max(1, int(K))
    n = np.arange(N)[:, None]
    C = np.zeros((N, K), dtype=float)
    C[:, 0] = 1.0 / math.sqrt(N)
    if K > 1:
        k = np.arange(1, K)[None, :]
        C[:, 1:] = math.sqrt(2.0 / N) * np.cos(math.pi * (2.0 * n + 1.0) * k / (2.0 * N))
    return C

def dct_basis(n_scans: int, TR: float, cutoff: float) -> np.ndarray:
    K = int(math.floor(2.0 * (n_scans * TR) / cutoff + 1.0))
    return spm_dctmtx(n_scans, K)

def pinv_beta(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X) @ Y

def corr_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A0 = A - A.mean(axis=1, keepdims=True)
    B0 = B - B.mean(axis=1, keepdims=True)
    num = np.sum(A0 * B0, axis=1)
    den = np.sqrt(np.sum(A0**2, axis=1) * np.sum(B0**2, axis=1))
    return num / np.maximum(den, eps)


# ------------------------
# Simple AR(1) prewhitening (SPM-like idea, simplified)
# ------------------------
def _estimate_ar1_rho_from_resid(resid: np.ndarray, eps: float = 1e-8) -> float:
    """Estimate a single AR(1) rho from residuals (T x V) using median across voxels."""
    if resid.shape[0] < 3:
        return 0.0
    r0 = resid[:-1, :]
    r1 = resid[1:, :]
    num = np.sum(r1 * r0, axis=0)
    den = np.sum(r0 * r0, axis=0) + eps
    rho_v = num / den
    rho = float(np.nanmedian(rho_v))
    # keep stable
    return float(np.clip(rho, -0.99, 0.99))


def ar1_prewhiten(Y: np.ndarray, X_list: list[np.ndarray], C_hp: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Prewhiten Y and design matrices with an estimated AR(1) rho.

    We estimate rho on residuals after removing high-pass DCT (C_hp), then apply
    the same filter to Y and each X in X_list:
        y'_t = y_t - rho * y_{t-1}
    The first sample is left unchanged.
    """
    # Residualize Y by high-pass only to estimate rho robustly
    try:
        Bhp = np.linalg.pinv(C_hp) @ Y
        resid = Y - (C_hp @ Bhp)
    except Exception:
        resid = Y - Y.mean(axis=0, keepdims=True)
    rho = _estimate_ar1_rho_from_resid(resid)

    def filt(M: np.ndarray) -> np.ndarray:
        M = np.asarray(M)
        out = M.copy()
        out[1:, ...] = out[1:, ...] - rho * out[:-1, ...]
        return out

    Yw = filt(Y)
    Xw = [filt(X) for X in X_list]
    return Yw, Xw, rho



# ------------------------
# Anchored RSA recovery metrics
# ------------------------
def _zscore_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

def _rowwise_pearson_to_set(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return C where C[i, j] = corr(A[i], B[j]) (Pearson), computed efficiently.

    A: (nA, v), B: (nB, v)
    """
    A = _zscore_rows(A, eps)
    B = _zscore_rows(B, eps)
    return (A @ B.T) / A.shape[1]

def anchored_trial_rsa(G: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Per-trial anchored RSA score.

    For each trial i:
      v_gt = corr(G[i], G[others])
      v_an = corr(E[i], G[others])   # anchored against GT patterns to avoid E-E collinearity artifacts
      score_i = Spearman(v_gt, v_an)
    """
    if G.shape != E.shape:
        raise ValueError(f"G shape {G.shape} != E shape {E.shape}")
    n = G.shape[0]
    C_gt = _rowwise_pearson_to_set(G, G)   # (n, n)
    C_an = _rowwise_pearson_to_set(E, G)   # (n, n)

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        v_gt = C_gt[i, mask]
        v_an = C_an[i, mask]
        rho = spearmanr(v_gt, v_an).correlation
        if rho is None or np.isnan(rho):
            rho = 0.0
        scores[i] = float(rho)
    return scores


def category_labels_from_df(df_run: pd.DataFrame) -> np.ndarray | None:
    """Return category labels (one per trial) if present, else None."""
    for col in ("ObjectSpace","group","class_id", "category", "cat", "label", "condition"):
        if col in df_run.columns:
            return df_run[col].to_numpy()
    return None


def anchored_category_rsa(G: np.ndarray, E: np.ndarray, cats: np.ndarray) -> dict:
    """Anchored RSA computed at the *category-mean* level.

    Steps (within a run):
      1) Average patterns within each category for GT (G) and estimated (E)
      2) Compute anchored RSA per-category, analogous to anchored_trial_rsa
      3) Also compute a single summary correlation between (GT category RDM) and
         (anchored estimated-to-GT category RDM).

    Returns dict with fields:
      - scores: per-category anchored RSA (Spearman), shape (n_cat,)
      - mean: mean(scores)
      - rdm_spearman: Spearman between upper triangles of category similarity matrices
    """
    if cats is None:
        return {"scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": 0}
    if G.shape != E.shape:
        raise ValueError(f"G shape {G.shape} != E shape {E.shape}")
    if len(cats) != G.shape[0]:
        return {"scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": 0}

    # Stable category ordering
    cats = np.asarray(cats)
    uniq = pd.unique(cats)
    if len(uniq) < 2:
        return {"scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": int(len(uniq))}

    Gc = []
    Ec = []
    for c in uniq:
        idx = np.where(cats == c)[0]
        if idx.size == 0:
            continue
        Gc.append(G[idx].mean(axis=0))
        Ec.append(E[idx].mean(axis=0))
    Gc = np.asarray(Gc, dtype=np.float64)
    Ec = np.asarray(Ec, dtype=np.float64)
    n_cat = Gc.shape[0]
    if n_cat < 2:
        return {"scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": int(n_cat)}

    # Per-category anchored RSA
    C_gt = _rowwise_pearson_to_set(Gc, Gc)   # (n_cat, n_cat)
    C_an = _rowwise_pearson_to_set(Ec, Gc)   # (n_cat, n_cat)
    scores = np.zeros(n_cat, dtype=np.float64)
    for i in range(n_cat):
        mask = np.ones(n_cat, dtype=bool); mask[i] = False
        rho = spearmanr(C_gt[i, mask], C_an[i, mask]).correlation
        if rho is None or np.isnan(rho):
            rho = 0.0
        scores[i] = float(rho)

    # RDM-level summary (upper triangle Spearman)
    iu = np.triu_indices(n_cat, k=1)
    v_gt = C_gt[iu]
    v_an = C_an[iu]
    rdm_rho = spearmanr(v_gt, v_an).correlation
    if rdm_rho is None or np.isnan(rdm_rho):
        rdm_rho = 0.0

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "rdm_spearman": float(rdm_rho),
        "n_cat": int(n_cat),
    }

def anchored_trial_rsa_voxelshuffle_null(
    G: np.ndarray,
    E: np.ndarray,
    null_perms: int,
    rng: np.random.Generator,
    shuffle_mode: str = "global",
) -> dict:
    """Voxel-shuffle null for anchored RSA.

    Shuffles voxel values within the *estimated* patterns E to break voxel-to-voxel correspondence,
    then recomputes the *mean* anchored RSA score across trials.

    shuffle_mode:
      - 'global': same voxel permutation applied to all trials each perm (fast, usually sufficient)
      - 'per_row': independent voxel permutation per trial each perm (stronger null)
    """
    if not null_perms or int(null_perms) <= 0:
        return {"null_mean": float("nan"), "null_p05": float("nan"), "null_p95": float("nan")}

    if G.shape != E.shape:
        raise ValueError(f"G shape {G.shape} != E shape {E.shape}")
    n, v = G.shape

    # Pre-zscore GT once for fast Pearson correlation via dot-products
    Gz = _zscore_rows(G)
    Gz_T = Gz.T
    C_gt = (Gz @ Gz_T) / v

    def corr_E_to_G(E_in: np.ndarray) -> np.ndarray:
        Ez = _zscore_rows(E_in)
        return (Ez @ Gz_T) / v

    null_means = np.zeros(int(null_perms), dtype=np.float64)
    for p in range(int(null_perms)):
        if shuffle_mode == "global":
            perm_vox = rng.permutation(v)
            E_shuf = E[:, perm_vox]
        elif shuffle_mode == "per_row":
            E_shuf = np.empty_like(E)
            for i in range(n):
                E_shuf[i] = E[i, rng.permutation(v)]
        else:
            raise ValueError("shuffle_mode must be 'global' or 'per_row'")

        C_an = corr_E_to_G(E_shuf)
        srow = np.zeros(n, dtype=np.float64)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            v_gt = C_gt[i, mask]
            v_an = C_an[i, mask]
            rho = spearmanr(v_gt, v_an).correlation
            if rho is None or np.isnan(rho):
                rho = 0.0
            srow[i] = float(rho)
        null_means[p] = float(np.mean(srow))

    return {
        "null_mean": float(np.mean(null_means)),
        "null_p05": float(np.percentile(null_means, 5)),
        "null_p95": float(np.percentile(null_means, 95)),
    }

def category_mismatch_null_indices(df_run: pd.DataFrame) -> np.ndarray | None:
    """Return an array of category labels per trial, or None if not available."""
    for col in ("ObjectSpace","group","class_id", "category", "cat", "label", "condition"):
        if col in df_run.columns:
            return df_run[col].to_numpy()
    return None

def anchored_trial_rsa_category_null(
    G: np.ndarray,
    E: np.ndarray,
    cats: np.ndarray,
) -> dict:
    """Category-mismatch null for anchored RSA.

    For each trial i, we take an *estimated* pattern from a different category (within the run),
    use it as the 'E' for trial i, compute anchored RSA against G, and average across trials.
    This gives a null that respects category-level similarity structure.

    This is deterministic given the provided cats; it is not a permutation distribution, but a
    mismatch baseline.
    """
    if cats is None:
        return {"null_mean": float("nan")}
    n = G.shape[0]
    if len(cats) != n:
        return {"null_mean": float("nan")}

    # Precompute the GT geometry vectors
    C_gt = _rowwise_pearson_to_set(G, G)
    # For each i, build v_gt once
    v_gt_list = []
    masks = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        masks.append(mask)
        v_gt_list.append(C_gt[i, mask])

    # For each trial, pick a donor trial from a *different* category (first occurrence)
    mismatch_scores = []
    for i in range(n):
        donor = np.where(cats != cats[i])[0]
        if donor.size == 0:
            continue
        k = int(donor[0])
        # anchored similarities for donor pattern against GT others for trial i
        # v_an = corr(E_k, G_others_of_i)
        # compute correlations between E_k and all G_j
        c = _rowwise_pearson_to_set(E[[k]], G)[0]  # (n,)
        v_an = c[masks[i]]
        rho = spearmanr(v_gt_list[i], v_an).correlation
        if rho is None or np.isnan(rho):
            rho = 0.0
        mismatch_scores.append(float(rho))

    if len(mismatch_scores) == 0:
        return {"null_mean": float("nan")}
    return {"null_mean": float(np.mean(mismatch_scores))}

# ------------------------
# fmrisim helpers
# ------------------------
def build_stimfunction(onsets: np.ndarray, durs: np.ndarray, total_time_s: float, tres: float) -> np.ndarray:
    return fmrisim.generate_stimfunction(
        onsets=onsets.tolist(),
        event_durations=durs.tolist(),
        total_time=total_time_s,
        temporal_resolution=tres
    )

def convolve_to_TR(stim, TR, tres, n_scans, hrf_type='double_gamma'):
    """
    Convolve high-res stimulation vector to TR resolution.
    Returns 1D array of length n_scans.
    """
    if stim.ndim == 1:
        stim = stim.reshape(-1, 1)

    if isinstance(hrf_type, str):
        hrf_vector = get_double_gamma_hrf(temporal_resolution=tres).flatten()
    else:
        hrf_vector = np.asarray(hrf_type).flatten()

    sig = fmrisim.convolve_hrf(
        stimfunction=stim,
        tr_duration=TR,
        temporal_resolution=tres,
        scale_function=False,
        #hrf_library=hrf_vector,   # <-- use the HRF you computed
    )

    out = np.zeros((n_scans,), dtype=float)
    L = min(n_scans, sig.shape[0])
    out[:L] = sig[:L, 0]
    return out


def generate_noise_volume(noise_dict: Dict, mask3d: np.ndarray, template3d: np.ndarray, n_scans: int, TR: float) -> np.ndarray:
    """Generate 4D noise using the *documented* BrainIAK fmrisim API.

    The BrainIAK fmrisim multivariate example uses:
        generate_noise(dimensions=dim[0:3], stimfunction_tr=[0]*dim[3], tr_duration=int(tr), ...)

    For non-integer TR (e.g., 1.792s) some fmrisim versions accept float TR; we pass TR as float.
    """
    # Prefer the documented API used in the BrainIAK fmrisim multivariate example.
    # Different fmrisim/BrainIAK versions vary slightly in accepted kwargs, so:
    #   1) Try the doc-style call (dimensions=3D, stimfunction_tr, tr_duration)
    #   2) If that fails, fall back to a signature-filtered call.
    stimfunction_tr = [0] * int(n_scans)

    try:
        return fmrisim.generate_noise(
            dimensions=list(mask3d.shape),
            stimfunction_tr=stimfunction_tr,
            tr_duration=float(TR),
            mask=mask3d,
            template=template3d,
            noise_dict=noise_dict,
        )
    except TypeError:
        # Fallback: filter kwargs to what the installed version supports.
        import inspect
        kwargs = dict(
            dimensions=list(mask3d.shape),
            stimfunction_tr=stimfunction_tr,
            tr_duration=float(TR),
            mask=mask3d,
            template=template3d,
            noise_dict=noise_dict,
        )
        try:
            sig = inspect.signature(fmrisim.generate_noise)
            accepted = set(sig.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            pass
        return fmrisim.generate_noise(**kwargs)


# ------------------------
# Auto brain mask
# ------------------------
def auto_brain_mask_from_4d(vol4d: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """
    Very simple brain mask from mean image intensity.
    - Computes mean across time
    - Uses threshold = frac * 95th percentile of mean
    """
    mean_img = vol4d.mean(axis=3)
    p95 = np.percentile(mean_img[~np.isnan(mean_img)], 95)
    thr = float(frac) * float(p95)
    mask = mean_img > thr

    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=2)
    mask = binary_fill_holes(mask)
    labels, n_labels = label(mask)
    if n_labels > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        keep = counts.argmax()
        mask = labels == keep
    return mask.astype(np.uint8)


# ------------------------
# LOC ROI from coordinate
# ------------------------
def voxel_sizes_mm(aff: np.ndarray) -> np.ndarray:
    # approximate voxel sizes from affine columns
    return np.sqrt((aff[:3, :3] ** 2).sum(axis=0))

def spherical_roi_mask(shape3d: Tuple[int,int,int], aff: np.ndarray, center_mni: np.ndarray, radius_mm: float) -> np.ndarray:
    """
    Build a spherical mask in voxel space whose center is specified in MNI (world) mm.
    Uses the affine to compute world coords for each voxel and thresholds by Euclidean distance.
    """
    # world coord of each voxel center: use grid + affine
    # To avoid huge memory, compute bounding box in voxel space based on radius and voxel sizes.
    ijk_center = apply_affine(np.linalg.inv(aff), center_mni).astype(float)
    vs = voxel_sizes_mm(aff)
    rad_vox = np.ceil(radius_mm / vs).astype(int)
    i0, j0, k0 = np.round(ijk_center).astype(int)

    i_min = max(0, i0 - rad_vox[0]); i_max = min(shape3d[0]-1, i0 + rad_vox[0])
    j_min = max(0, j0 - rad_vox[1]); j_max = min(shape3d[1]-1, j0 + rad_vox[1])
    k_min = max(0, k0 - rad_vox[2]); k_max = min(shape3d[2]-1, k0 + rad_vox[2])

    ii, jj, kk = np.mgrid[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1]
    vox = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    xyz = apply_affine(aff, vox)
    dist = np.linalg.norm(xyz - center_mni, axis=1)
    inside = dist <= radius_mm
    mask = np.zeros(shape3d, dtype=np.uint8)
    mask[vox[:, 0], vox[:, 1], vox[:, 2]] = inside.astype(np.uint8)
    return mask

def roi_mask_bbox(roi_mask: np.ndarray) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(roi_mask > 0)
    if coords.size == 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return slice(mins[0], maxs[0]), slice(mins[1], maxs[1]), slice(mins[2], maxs[2])

def estimate_loc_center_from_mask(brain_mask: np.ndarray, frac_xyz: Tuple[float, float, float]) -> np.ndarray:
    coords = np.argwhere(brain_mask > 0)
    if coords.size == 0:
        raise RuntimeError("Brain mask is empty; cannot estimate LOC center.")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = mins + (maxs - mins) * np.array(frac_xyz, dtype=float)
    return center


def snap_vox_to_brain_mask(center_vox: np.ndarray, brain_mask: np.ndarray, max_radius_vox: int = 25) -> np.ndarray:
    """Snap a (possibly non-integer) voxel coordinate to the nearest in-brain voxel."""
    if brain_mask.size == 0:
        return center_vox
    c = np.array(center_vox, dtype=float)
    c_round = np.round(c).astype(int)
    c_round = np.clip(c_round, [0,0,0], np.array(brain_mask.shape) - 1)
    if brain_mask[tuple(c_round)] > 0:
        return c_round.astype(float)

    # Expanding cube search around the rounded center
    ci, cj, ck = c_round.tolist()
    best = None
    best_d2 = 1e18
    for r in range(1, max_radius_vox + 1):
        i0, i1 = max(0, ci - r), min(brain_mask.shape[0], ci + r + 1)
        j0, j1 = max(0, cj - r), min(brain_mask.shape[1], cj + r + 1)
        k0, k1 = max(0, ck - r), min(brain_mask.shape[2], ck + r + 1)
        sub = brain_mask[i0:i1, j0:j1, k0:k1]
        if not np.any(sub):
            continue
        coords = np.argwhere(sub > 0) + np.array([i0, j0, k0])
        d2 = np.sum((coords - c[None, :])**2, axis=1)
        idx = int(np.argmin(d2))
        if float(d2[idx]) < best_d2:
            best_d2 = float(d2[idx])
            best = coords[idx]
        # Early exit if we found something very close
        if best is not None and best_d2 <= 1.0:
            break
    if best is None:
        # Fall back to any in-brain voxel (centroid)
        coords = np.argwhere(brain_mask > 0)
        if coords.size == 0:
            return c_round.astype(float)
        best = np.round(np.mean(coords, axis=0)).astype(int)
    return best.astype(float)

def estimate_occipital_center_from_mask(brain_mask: np.ndarray, aff: np.ndarray, posterior_quantile: float = 0.10) -> np.ndarray:
    """Pick a conservative posterior (occipital-ish) in-brain center using world Y (posterior-most quantile)."""
    coords = np.argwhere(brain_mask > 0)
    if coords.size == 0:
        raise RuntimeError("Brain mask is empty; cannot estimate occipital center.")
    # Convert to world and use Y as posterior/anterior axis
    xyz = apply_affine(aff, coords)
    y = xyz[:, 1]
    thr = float(np.quantile(y, posterior_quantile))
    post = coords[y <= thr]
    if post.size == 0:
        post = coords
    center = np.median(post, axis=0)
    center = snap_vox_to_brain_mask(center, brain_mask, max_radius_vox=25)
    return center
def roi_mask_target_voxels(shape3d: Tuple[int,int,int],
                           aff: np.ndarray,
                           brain_mask: np.ndarray,
                           center_mni: np.ndarray,
                           target_n_vox: int,
                           r0_mm: float = 10.0,
                           max_iter: int = 10) -> Tuple[np.ndarray, float]:
    """
    Find a spherical radius that gives ~target_n_vox voxels within the brain mask.
    Returns (roi_mask, radius_mm)
    """
    radius = float(r0_mm)
    best = None
    best_diff = 10**9

    for _ in range(max_iter):
        sph = spherical_roi_mask(shape3d, aff, center_mni, radius)
        roi = (sph & (brain_mask > 0)).astype(np.uint8)
        n = int(roi.sum())
        diff = abs(n - target_n_vox)
        if diff < best_diff:
            best = roi
            best_diff = diff
        if n == 0:
            radius *= 1.5
            continue
        if diff <= max(5, int(0.03 * target_n_vox)):
            return roi, radius
        # update radius using cube-root scaling (volume ~ r^3)
        radius = radius * ((target_n_vox / n) ** (1/3))
        radius = float(np.clip(radius, 1.0, 60.0))

    return best, radius



# ------------------------
# Noise caching + doc-style pipeline (BrainIAK example)
# ------------------------
def _stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _mask_hash(mask: np.ndarray) -> str:
    """Stable short hash for a mask array used in cache keys.

    We hash the *content* (not object id) so cache invalidates if the mask changes.
    """
    if mask is None:
        return "none"
    arr = np.asarray(mask)
    # Make deterministic + compact
    arr_u8 = np.ascontiguousarray((arr > 0).astype(np.uint8))
    h = hashlib.sha1()
    h.update(str(arr_u8.shape).encode("utf-8"))
    h.update(arr_u8.tobytes())
    return h.hexdigest()[:12]



def _sanitize_name(s: str) -> str:
    """Filesystem-safe identifier."""
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "unnamed"


def infer_design_group_from_csv(csv_path: Path) -> str:
    """Infer the design 'group' folder from a CSV path.

    Expected layout:
      .../experimental_task/task_designs/<DESIGN_GROUP>/session-1/<file>.csv

    Robust fallbacks:
      - If 'task_designs' appears in the path parts, return the segment after it
      - Else if parent is 'session-*', return parent's parent
      - Else return immediate parent name
    """
    p = Path(csv_path)
    parts = list(p.parts)
    if "task_designs" in parts:
        i = parts.index("task_designs")
        if i + 1 < len(parts):
            return _sanitize_name(parts[i + 1])
    # fallback: .../<group>/session-1/<file>.csv
    if p.parent.name.lower().startswith("session-") and p.parent.parent is not None:
        return _sanitize_name(p.parent.parent.name)
    return _sanitize_name(p.parent.name)


def infer_dataset_id_from_noise_path(noise_nii: str) -> str:
    """Dataset identifier used for shared noise caching."""
    p = Path(noise_nii)
    # handle .nii.gz
    name = p.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = p.stem
    return _sanitize_name(stem)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_or_make_noise_dict(vol4d: np.ndarray, noise_nii_path: str, TR: float, cache_dir: Path, use_cache: bool) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Follow the BrainIAK fmrisim multivariate example as closely as possible:
      mask, template = fmrisim.mask_brain(volume=volume, mask_self=True)
      noise_dict = fmrisim.calc_noise(volume=volume, mask=mask, template=template, noise_dict=None)

    Returns (noise_dict, brain_mask, template).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # mask + template per docs
    brain_mask, template = fmrisim.mask_brain(volume=vol4d, mask_self=True)

    # Build a cache key that changes if the underlying NIfTI changes
    try:
        st = Path(noise_nii_path).stat()
        nii_fingerprint = {"path": str(Path(noise_nii_path).resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}
    except Exception:
        nii_fingerprint = {"path": str(noise_nii_path)}

    key = _stable_hash({
        "nii": nii_fingerprint,
        "TR": float(TR),
        "mask_self": True,
        "shape": tuple(vol4d.shape),
        "fmrisim": getattr(fmrisim, "__file__", "unknown"),
    })
    pkl_path = cache_dir / f"noise_dict_{key}.pkl"

    if use_cache and pkl_path.exists():
        with open(pkl_path, "rb") as f:
            noise_dict = pickle.load(f)
        return noise_dict, brain_mask.astype(np.uint8), template.astype(np.float32)

    noise_dict = fmrisim.calc_noise(volume=vol4d, mask=brain_mask, template=template, noise_dict=None)

    if use_cache:
        with open(pkl_path, "wb") as f:
            pickle.dump(noise_dict, f)

    return noise_dict, brain_mask.astype(np.uint8), template.astype(np.float32)

def load_or_make_run_noise_2d_doc(
    run_id,
    n_scans: int,
    TR: float,
    noise_dict: dict,
    brain_mask: np.ndarray,
    template: np.ndarray,
    roi_mask: np.ndarray,
    n_vox: int,
    cache_dir: Path,
    use_cache: bool,
    noise_cache_seed: int,
) -> np.ndarray:
    """
    Generate noise using the documented BrainIAK example call:
        noise = fmrisim.generate_noise(dimensions=dim[0:3],
                                      stimfunction_tr=[0]*dim[3],
                                      tr_duration=tr,
                                      mask=mask,
                                      template=template,
                                      noise_dict=noise_dict)

    Then sample *n_vox* voxels from roi_mask and return a (T x n_vox) matrix.

    We cache the sampled (T x n_vox) noise per run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    rid_int = int(run_id) if str(run_id).isdigit() else (hash(str(run_id)) & 0xFFFFFFFF)
    key = _stable_hash({
        "rid": str(run_id),
        "rid_int": rid_int,
        "n_scans": int(n_scans),
        "TR": float(TR),
        "n_vox": int(n_vox),
        "noise_cache_seed": int(noise_cache_seed),
        "roi_vox": int(np.sum(roi_mask > 0)),
        "brain_vox": int(np.sum(brain_mask > 0)),
    })
    npz_path = cache_dir / f"runnoise_{key}.npz"

    if use_cache and npz_path.exists():
        dat = np.load(npz_path)
        noise_2d = dat["noise_2d"].astype(np.float32)
        if noise_2d.shape != (n_scans, n_vox):
            raise ValueError(f"Cached run noise has shape {noise_2d.shape} but expected {(n_scans, n_vox)}.")
        return noise_2d


# ------------------------
# Shared noise bank (ROI-only) for cross-design reuse
# ------------------------
def _noise_bank_key(
    noise_nii_path: str,
    TR: float,
    n_vox: int,
    roi_mask: np.ndarray,
    brain_mask: np.ndarray,
    noise_dict: dict,
    max_n_scans: int,
    n_noise_reps: int,
    noise_cache_seed: int,
) -> str:
    """Key that changes when underlying noise inputs or requirements change."""
    try:
        st = Path(noise_nii_path).stat()
        nii_fingerprint = {"path": str(Path(noise_nii_path).resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}
    except Exception:
        nii_fingerprint = {"path": str(noise_nii_path)}

    return _stable_hash(
        {
            "nii": nii_fingerprint,
            "TR": float(TR),
            "n_vox": int(n_vox),
            "max_n_scans": int(max_n_scans),
            "n_noise_reps": int(n_noise_reps),
            "noise_cache_seed": int(noise_cache_seed),
            "roi_vox": int(np.sum(roi_mask > 0)),
            "brain_vox": int(np.sum(brain_mask > 0)),
            # A light fingerprint of noise_dict content (not fully serializing large items)
            "noise_dict_keys": sorted(list(noise_dict.keys())) if isinstance(noise_dict, dict) else str(type(noise_dict)),
        }
    )


def _pick_roi_voxels(roi_mask: np.ndarray, brain_mask: np.ndarray, n_vox: int, seed: int) -> np.ndarray:
    roi_bool = (roi_mask > 0) & (brain_mask > 0)
    roi_lin = np.flatnonzero(roi_bool.reshape(-1))
    if roi_lin.size < n_vox:
        raise ValueError(f"ROI has {roi_lin.size} voxels inside brain_mask but n_vox={n_vox} requested.")
    rng = np.random.default_rng(int(seed))
    return rng.choice(roi_lin, size=int(n_vox), replace=False)


def _generate_one_noise_rep_roi(
    rep_idx: int,
    max_n_scans: int,
    TR: float,
    brain_mask: np.ndarray,
    template: np.ndarray,
    noise_dict: dict,
    pick_lin: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Worker: generate one whole-brain noise 4D, then sample ROI voxels -> (T x n_vox)."""
    # Make a rep-specific RNG state by offsetting the seed inside noise_dict if possible.
    # BrainIAK uses values inside noise_dict; many variants respect 'seed' / numpy global RNG.
    # We set numpy's bitgen via default_rng and use it to perturb template slightly if needed.
    # Most importantly: we randomize fmrisim's internal draws by setting the global seed.
    np.random.seed(int(seed) + int(rep_idx) * 10007)

    stimfunction_tr = [0] * int(max_n_scans)
    noise_4d = fmrisim.generate_noise(
        dimensions=list(brain_mask.shape),
        stimfunction_tr=stimfunction_tr,
        tr_duration=float(TR),
        mask=brain_mask,
        template=template,
        noise_dict=noise_dict,
    )

    noise_2d = noise_4d.reshape(-1, max_n_scans)[pick_lin].T.astype(np.float32)  # (T, n_vox)
    return noise_2d


def load_or_make_noise_bank_roi(
    noise_nii_path: str,
    TR: float,
    n_vox: int,
    roi_mask: np.ndarray,
    brain_mask: np.ndarray,
    template: np.ndarray,
    noise_dict: dict,
    max_n_scans: int,
    n_noise_reps: int,
    cache_dir: Path,
    use_cache: bool,
    noise_cache_seed: int,
    n_parallel: int = 1,
) -> tuple[np.ndarray, dict]:
    """Create/load a bank of ROI noise time-series for cross-design reuse.

    Returns:
      noise_bank: (n_noise_reps, max_n_scans, n_vox) float32
      meta: dict with bookkeeping (key, max_n_scans, n_noise_reps, pick_lin hash)
    """
    ensure_dir(cache_dir)

    # Use a fixed voxel sample for the ROI across all reps & designs (important for comparability)
    pick_lin = _pick_roi_voxels(roi_mask, brain_mask, n_vox=n_vox, seed=int(noise_cache_seed))

    key = _noise_bank_key(
        noise_nii_path=noise_nii_path,
        TR=TR,
        n_vox=n_vox,
        roi_mask=roi_mask,
        brain_mask=brain_mask,
        noise_dict=noise_dict,
        max_n_scans=max_n_scans,
        n_noise_reps=n_noise_reps,
        noise_cache_seed=noise_cache_seed,
    )
    bank_path = cache_dir / f"noise_bank_roi_{key}.npz"
    meta_path = cache_dir / f"noise_bank_roi_{key}.json"

    if use_cache and bank_path.exists():
        dat = np.load(bank_path)
        noise_bank = dat["noise_bank"].astype(np.float32)
        if noise_bank.shape != (int(n_noise_reps), int(max_n_scans), int(n_vox)):
            raise ValueError(
                f"Cached noise bank has shape {noise_bank.shape} but expected {(int(n_noise_reps), int(max_n_scans), int(n_vox))}."
            )
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            meta = {}
        return noise_bank, meta

    # Otherwise generate from scratch (cannot 'extend' reliably without regenerating due to model dependence on T)
    noise_bank = np.zeros((int(n_noise_reps), int(max_n_scans), int(n_vox)), dtype=np.float32)

    if int(n_parallel) <= 1:
        for r in range(int(n_noise_reps)):
            noise_bank[r] = _generate_one_noise_rep_roi(
                rep_idx=r,
                max_n_scans=max_n_scans,
                TR=TR,
                brain_mask=brain_mask,
                template=template,
                noise_dict=noise_dict,
                pick_lin=pick_lin,
                seed=int(noise_cache_seed),
            )
    else:
        # ProcessPool: avoids GIL; keep args pickleable
        with ProcessPoolExecutor(max_workers=int(n_parallel)) as ex:
            futs = []
            for r in range(int(n_noise_reps)):
                futs.append(
                    ex.submit(
                        _generate_one_noise_rep_roi,
                        r,
                        int(max_n_scans),
                        float(TR),
                        brain_mask,
                        template,
                        noise_dict,
                        pick_lin,
                        int(noise_cache_seed),
                    )
                )
            for fut in as_completed(futs):
                pass
            # Collect in submission order to keep deterministic indexing
            for r, fut in enumerate(futs):
                noise_bank[r] = fut.result()

    if use_cache:
        np.savez_compressed(bank_path, noise_bank=noise_bank)
        meta = {
            "key": key,
            "noise_nii_path": str(noise_nii_path),
            "TR": float(TR),
            "n_vox": int(n_vox),
            "max_n_scans": int(max_n_scans),
            "n_noise_reps": int(n_noise_reps),
            "noise_cache_seed": int(noise_cache_seed),
            "pick_lin_sha1": hashlib.sha1(pick_lin.tobytes()).hexdigest(),
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
    else:
        meta = {"key": key}

    return noise_bank, meta


    # Whole-brain noise generation per docs
    stimfunction_tr = [0] * int(n_scans)
    noise_4d = fmrisim.generate_noise(
        dimensions=list(brain_mask.shape),
        stimfunction_tr=stimfunction_tr,
        tr_duration=float(TR),   # docs cast to int for TR=2; float is necessary for TR=1.792
        mask=brain_mask,
        template=template,
        noise_dict=noise_dict,
    )

    # Sample ROI voxels deterministically (for caching / optimisation stability)
    roi_bool = (roi_mask > 0) & (brain_mask > 0)
    roi_lin = np.flatnonzero(roi_bool.reshape(-1))
    if roi_lin.size < n_vox:
        raise ValueError(f"ROI has {roi_lin.size} voxels inside brain_mask but n_vox={n_vox} requested.")
    rng = np.random.default_rng(int(noise_cache_seed) + int(rid_int))
    pick_lin = rng.choice(roi_lin, size=n_vox, replace=False)

    noise_2d = noise_4d.reshape(-1, n_scans)[pick_lin].T.astype(np.float32)

    if use_cache:
        np.savez_compressed(npz_path, noise_2d=noise_2d)

    return noise_2d


# ------------------------
# Run-length noise cache (ROI-only): runs × reps, cached per dataset for realism
# ------------------------
def _run_noise_key(
    noise_nii_path: str,
    TR: float,
    n_vox: int,
    roi_mask: np.ndarray,
    brain_mask: np.ndarray,
    noise_dict: dict,
    noise_cache_seed: int,
) -> str:
    """Key for run-length noise files (excluding n_scans, run_id, rep_idx)."""
    try:
        nii_stat = os.stat(noise_nii_path)
        nii_sig = f"{Path(noise_nii_path).name}:{int(nii_stat.st_size)}:{int(nii_stat.st_mtime)}"
    except Exception:
        nii_sig = f"{Path(noise_nii_path).name}"
    parts = [
        f"nii={nii_sig}",
        f"TR={float(TR):.6f}",
        f"n_vox={int(n_vox)}",
        f"seed={int(noise_cache_seed)}",
        f"roi={_mask_hash(roi_mask)}",
        f"brain={_mask_hash(brain_mask)}",
        f"ndict={json.dumps(noise_dict, sort_keys=True, default=str)}",
    ]
    return _stable_hash("|".join(parts))[:16]


def _dataset_id_from_noise_nii(noise_nii_path: str) -> str:
    """Folder-friendly dataset identifier from noise file name."""
    base = Path(noise_nii_path).name
    for suf in [".nii.gz", ".nii", ".mgz", ".mgh", ".gz"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return base or "noise"


def load_or_make_run_noise_roi_set(
    noise_nii_path: str,
    TR: float,
    n_vox: int,
    roi_mask: np.ndarray,
    brain_mask: np.ndarray,
    template: np.ndarray,
    noise_dict: dict,
    run_n_scans: Dict,
    n_noise_reps: int,
    cache_dir: Path,
    use_cache: bool,
    noise_cache_seed: int,
    n_parallel: int = 1,
) -> tuple[Dict, dict]:
    """Create/load run-length ROI noise for each (run_id, rep_idx).

    Returns:
      run_noise: dict[(run_id, rep_idx)] -> (n_scans_run, n_vox) float32
      meta: dict with bookkeeping (key, dataset_id, pick_lin hash)
    """
    ensure_dir(cache_dir)

    dataset_id = _dataset_id_from_noise_nii(noise_nii_path)
    ds_dir = cache_dir / dataset_id
    ensure_dir(ds_dir)

    # Fixed voxel sample for ROI across all runs & reps (comparability)
    pick_lin = _pick_roi_voxels(roi_mask, brain_mask, n_vox=n_vox, seed=int(noise_cache_seed))
    key = _run_noise_key(
        noise_nii_path=noise_nii_path,
        TR=TR,
        n_vox=n_vox,
        roi_mask=roi_mask,
        brain_mask=brain_mask,
        noise_dict=noise_dict,
        noise_cache_seed=noise_cache_seed,
    )

    # Helper to build file paths
    def _paths_for(rid, rep_idx: int) -> tuple[Path, Path]:
        """New cache layout: exactly one file per (dataset_id, run_id, rep_idx, key)."""
        run_dir = ds_dir / f"run-{rid}"
        ensure_dir(run_dir)
        npz = run_dir / f"noise_roi_{key}_rep{int(rep_idx):03d}.npz"
        meta = run_dir / f"noise_roi_{key}_rep{int(rep_idx):03d}.json"
        return npz, meta

    def _headroom_T(T_needed: int, frac: float = 1.1) -> int:
        return int(max(int(T_needed), int(math.ceil(float(T_needed) * float(frac)))))

    def _valid_cached_array(arr: np.ndarray, T_needed: int) -> bool:
        if arr is None:
            return False
        if arr.ndim != 2:
            return False
        if int(arr.shape[1]) != int(n_vox):
            return False
        if int(arr.shape[0]) < int(T_needed):
            return False
        return True

    # Backward compatible cache loading:
    # - Prefer new filename (no _T{n_scans})
    # - Otherwise, look for old *_T*.npz files, pick the largest T, and rewrite to new name
    def _load_if_ok(rid, rep_idx: int, T_needed: int) -> Optional[np.ndarray]:
        run_dir = ds_dir / f"run-{rid}"
        if not run_dir.exists():
            return None

        # 1) New filename
        new_npz, _ = _paths_for(rid, rep_idx)
        if new_npz.exists():
            try:
                dat = np.load(new_npz)
                arr = dat["noise_2d"].astype(np.float32)
                # T_cached stored for bookkeeping; prefer it if present
                T_cached = int(dat["T_cached"]) if "T_cached" in dat.files else int(arr.shape[0])
                if int(arr.shape[0]) != int(T_cached):
                    # shape/metadata mismatch
                    return None
                if _valid_cached_array(arr, T_needed=int(T_needed)):
                    return arr[: int(T_needed)]
            except Exception:
                return None

        # 2) Old filename(s): noise_roi_{key}_rep###_T*.npz
        cand = sorted(run_dir.glob(f"noise_roi_{key}_rep{int(rep_idx):03d}_T*.npz"))
        if not cand:
            return None

        best_arr = None
        best_T = -1
        best_path = None
        for npz_path in cand:
            try:
                dat = np.load(npz_path)
                arr = dat["noise_2d"].astype(np.float32)
                T_cached = int(arr.shape[0])
                if int(arr.shape[1]) != int(n_vox):
                    continue
                if T_cached > best_T:
                    best_T = T_cached
                    best_arr = arr
                    best_path = npz_path
            except Exception:
                continue

        if best_arr is None:
            return None

        if _valid_cached_array(best_arr, T_needed=int(T_needed)):
            # Rewrite into the new filename for future loads
            try:
                np.savez_compressed(new_npz, noise_2d=best_arr.astype(np.float32), T_cached=int(best_arr.shape[0]))
            except Exception:
                pass
            return best_arr[: int(T_needed)]

        # Old cache exists but is too short
        return None

    # Worker to generate one (rid, rep_idx)
    # Helper: stable seed per (run_id, rep_idx) so caching is deterministic.
    def _seed_for(rid, rep_idx: int) -> int:
        rid_h = int(_stable_hash(str(rid))[:8], 16)
        return (int(noise_cache_seed) + rid_h + int(rep_idx) * 10007) % (2**32 - 1)

# Plan tasks
    tasks = []
    run_noise: Dict = {}
    # Keep any freshly generated headroom arrays around so we can persist them,
    # while still returning exact-length slices to callers.
    run_noise_full: Dict = {}

    for rid, n_scans in run_n_scans.items():
        n_scans = int(n_scans)
        for rep_idx in range(int(n_noise_reps)):
            if use_cache:
                cached = _load_if_ok(rid, rep_idx, n_scans)
                if cached is not None:
                    run_noise[(rid, rep_idx)] = cached
                    continue
            tasks.append((rid, rep_idx, n_scans, _seed_for(rid, rep_idx)))

    # Generate missing tasks (optionally parallel)
    if tasks:
        if int(n_parallel) <= 1:
            for rid, rep_idx, n_scans, seed in tasks:
                T_needed = int(n_scans)
                T_gen = _headroom_T(T_needed)
                arr_full = _generate_one_noise_rep_roi(
                    rep_idx=int(rep_idx),
                    max_n_scans=int(T_gen),
                    TR=float(TR),
                    brain_mask=brain_mask,
                    template=template,
                    noise_dict=noise_dict,
                    pick_lin=pick_lin,
                    seed=int(seed),
                )
                run_noise_full[(rid, rep_idx)] = arr_full
                run_noise[(rid, rep_idx)] = arr_full[:T_needed]
        else:
            with ProcessPoolExecutor(max_workers=int(n_parallel)) as ex:
                futs = []
                for rid, rep_idx, n_scans, seed in tasks:
                    T_needed = int(n_scans)
                    T_gen = _headroom_T(T_needed)
                    futs.append(ex.submit(_generate_one_noise_rep_roi, int(rep_idx), int(T_gen), float(TR), brain_mask, template, noise_dict, pick_lin, int(seed)))
                for (rid, rep_idx, n_scans, seed), fut in zip(tasks, futs):
                    T_needed = int(n_scans)
                    arr_full = fut.result()
                    run_noise_full[(rid, rep_idx)] = arr_full
                    run_noise[(rid, rep_idx)] = arr_full[:T_needed]

    # Persist any newly generated arrays
    if use_cache:
        for (rid, rep_idx), arr in run_noise.items():
            # Always persist the *cached* (potentially headroom) noise into the single canonical file.
            # If a shorter/invalid file exists, overwrite it.
            npz_path, meta_path = _paths_for(rid, rep_idx)

            # Prefer writing the full headroom array if we generated it this call.
            arr_to_write = run_noise_full.get((rid, rep_idx), arr)

            # Load existing (if any) to decide whether overwrite is necessary
            need_write = True
            if npz_path.exists():
                try:
                    dat = np.load(npz_path)
                    cur = dat["noise_2d"].astype(np.float32)
                    T_cached = int(dat["T_cached"]) if "T_cached" in dat.files else int(cur.shape[0])
                    need_write = not (
                        int(cur.shape[0]) == int(T_cached)
                        and int(cur.shape[1]) == int(n_vox)
                        and int(cur.shape[0]) >= int(arr_to_write.shape[0])
                    )
                except Exception:
                    need_write = True

            if need_write:
                np.savez_compressed(
                    npz_path,
                    noise_2d=arr_to_write.astype(np.float32),
                    T_cached=int(arr_to_write.shape[0]),
                )

            meta = {
                "key": key,
                "dataset_id": dataset_id,
                "noise_nii_path": str(noise_nii_path),
                "TR": float(TR),
                "n_vox": int(n_vox),
                "T_cached": int(arr_to_write.shape[0]),
                "run_id": str(rid),
                "rep_idx": int(rep_idx),
                "pick_lin_hash": _stable_hash(pick_lin.tobytes())[:16],
                "noise_cache_seed": int(noise_cache_seed),
                "noise_dict": noise_dict,
            }
            try:
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

    meta = {
        "key": key,
        "dataset_id": dataset_id,
        "noise_nii_path": str(noise_nii_path),
        "TR": float(TR),
        "n_vox": int(n_vox),
        "n_noise_reps": int(n_noise_reps),
        "pick_lin_hash": _stable_hash(pick_lin.tobytes())[:16],
    }
    return run_noise, meta


def compute_n_scans_for_run(df_run: pd.DataFrame, TR: float, pad_s: float, dec_dur_s: float) -> int:
    dec_on = df_run["dec_onset_est"].to_numpy(float)
    fb_on = dec_on + float(dec_dur_s) + df_run["isi2_dur"].to_numpy(float)
    fb_dur = df_run["fb_dur"].to_numpy(float)
    total_time_s = float(np.max(fb_on + fb_dur)) + float(pad_s)
    return int(math.ceil(total_time_s / float(TR)))

# ------------------------
# 3-event trial-wise regressors
# ------------------------
def build_event_mats(df_run: pd.DataFrame,
                     TR: float,
                     pad_s: float,
                     dec_dur_s: float,
                     tres: float,
                     hrf_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc_on = df_run["img_onset"].to_numpy(float)
    enc_dur = df_run["img_dur"].to_numpy(float)

    dec_on = df_run["dec_onset_est"].to_numpy(float)
    dec_dur = np.full_like(dec_on, float(dec_dur_s))

    fb_on = dec_on + float(dec_dur_s) + df_run["isi2_dur"].to_numpy(float)
    fb_dur = df_run["fb_dur"].to_numpy(float)

    total_time_s = float(np.max(fb_on + fb_dur)) + float(pad_s)
    n_scans = int(math.ceil(total_time_s / TR))

    n_trials = len(df_run)
    X_enc = np.zeros((n_scans, n_trials), float)
    X_dec = np.zeros((n_scans, n_trials), float)
    X_fb  = np.zeros((n_scans, n_trials), float)

    for j in range(n_trials):
        stim = build_stimfunction(np.array([enc_on[j]]), np.array([enc_dur[j]]), total_time_s, tres)
        X_enc[:, j] = convolve_to_TR(stim, TR, tres, n_scans, hrf_type)

        stim = build_stimfunction(np.array([dec_on[j]]), np.array([dec_dur[j]]), total_time_s, tres)
        X_dec[:, j] = convolve_to_TR(stim, TR, tres, n_scans, hrf_type)

        stim = build_stimfunction(np.array([fb_on[j]]), np.array([fb_dur[j]]), total_time_s, tres)
        X_fb[:, j] = convolve_to_TR(stim, TR, tres, n_scans, hrf_type)

    return X_enc, X_dec, X_fb


# ------------------------
# Patterns
# ------------------------
def rownorm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def make_noisy_event_patterns(P_enc: np.ndarray, rng: np.random.Generator, mix: float) -> np.ndarray:
    R = rng.normal(size=P_enc.shape)
    R = rownorm(R)
    P = (1.0 - mix) * P_enc + mix * R
    return rownorm(P)

def apply_voxel_latency_shift(
    Y_sig: np.ndarray,
    TR: float,
    rng: np.random.Generator,
    sd_s: float,
    return_shifts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if sd_s <= 0:
        shifts = np.zeros(Y_sig.shape[1], dtype=float)
        return (Y_sig, shifts) if return_shifts else Y_sig

    T, V = Y_sig.shape
    t = np.arange(T) * TR
    out = np.empty_like(Y_sig)
    shifts = rng.normal(0.0, sd_s, size=V)
    for v in range(V):
        out[:, v] = np.interp(t, t - shifts[v], Y_sig[:, v], left=Y_sig[0, v], right=Y_sig[-1, v])

    return (out, shifts) if return_shifts else out


def apply_voxel_latency_shift_to_components(
    comps: list[np.ndarray],
    TR: float,
    shifts: np.ndarray,
) -> list[np.ndarray]:
    """Apply pre-sampled voxel shifts to each component time series."""
    if shifts is None:
        return comps
    T, V = comps[0].shape
    t = np.arange(T) * TR
    out_list = []
    for comp in comps:
        out = np.empty_like(comp)
        for v in range(V):
            out[:, v] = np.interp(t, t - shifts[v], comp[:, v], left=comp[0, v], right=comp[-1, v])
        out_list.append(out)
    return out_list


def interactive_plot_true_components(
    TR: float,
    comp_enc: np.ndarray,
    comp_dec: np.ndarray,
    comp_fb: np.ndarray,
    Y_total: np.ndarray | None = None,
    title_prefix: str = "",
):
    """
    Interactive plot (matplotlib) to browse voxels.
    X-axis is TR index (0..T-1).
    """
    T, V = comp_enc.shape
    x = np.arange(T)

    # Initial voxel
    v0 = 0

    fig, ax = plt.subplots(figsize=(11, 5))
    plt.subplots_adjust(bottom=0.25, right=0.82)

    l_enc, = ax.plot(x, comp_enc[:, v0], label="Encoding")
    l_dec, = ax.plot(x, comp_dec[:, v0], label="Decision")
    l_fb,  = ax.plot(x, comp_fb[:, v0],  label="Feedback")
    l_sum, = ax.plot(x, comp_enc[:, v0] + comp_dec[:, v0] + comp_fb[:, v0], label="Sum")

    l_y = None
    if Y_total is not None:
        l_y, = ax.plot(x, Y_total[:, v0], label="Y (signal+noise)")

    ax.set_xlabel("TR index")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"{title_prefix}Voxel {v0} | TR={TR:.3f}s | Time={T} TRs")
    ax.legend(loc="upper left")

    # Slider for voxel index
    ax_vox = plt.axes([0.10, 0.10, 0.60, 0.04])
    s_vox = Slider(ax_vox, "Voxel", 0, V - 1, valinit=v0, valstep=1)

    # Checkbuttons to toggle lines
    labels = ["Encoding", "Decision", "Feedback", "Sum"] + (["Y (signal+noise)"] if l_y is not None else [])
    visibility = [True, True, True, True] + ([True] if l_y is not None else [])
    ax_chk = plt.axes([0.84, 0.35, 0.15, 0.25])
    chk = CheckButtons(ax_chk, labels, visibility)

    line_map = {
        "Encoding": l_enc,
        "Decision": l_dec,
        "Feedback": l_fb,
        "Sum": l_sum,
    }
    if l_y is not None:
        line_map["Y (signal+noise)"] = l_y

    def update(vox_idx):
        v = int(s_vox.val)
        l_enc.set_ydata(comp_enc[:, v])
        l_dec.set_ydata(comp_dec[:, v])
        l_fb.set_ydata(comp_fb[:, v])
        l_sum.set_ydata(comp_enc[:, v] + comp_dec[:, v] + comp_fb[:, v])
        if l_y is not None:
            l_y.set_ydata(Y_total[:, v])

        ax.set_title(f"{title_prefix}Voxel {v} | TR={TR:.3f}s | Time={T} TRs")
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def toggle(label):
        line = line_map[label]
        line.set_visible(not line.get_visible())
        fig.canvas.draw_idle()

    s_vox.on_changed(update)
    chk.on_clicked(toggle)

    plt.show()


# ------------------------
# Simulation / fitting
# ------------------------
def simulate_one_run(df_run: pd.DataFrame,
                     P_enc: np.ndarray,
                     TR: float,
                     hp_cutoff: float,
                     pad_s: float,
                     dec_dur_s: float,
                     tres: float,
                     hrf_type_sim: str,
                     noise_2d: np.ndarray,
                     n_vox: int,
                     rng: np.random.Generator,
                     decision_mix: float,
                     feedback_mix: float,
                     hrf_latency_mismatch_sd: float,
                     nuisance_model: str,
                     prewhiten: str = "ar1",
                     rsa_null_perms: int = 0,
                     rsa_shuffle_mode: str = "global",
                     signal_psc: float = 0.5) -> dict:

    # 1. Build Design Matrices (Uses the fixed convolve_to_TR)
    X_enc, X_dec, X_fb = build_event_mats(df_run, TR, pad_s, dec_dur_s, tres, hrf_type_sim)
    n_scans, n_trials = X_enc.shape

    if P_enc.shape[0] != n_trials:
        raise ValueError(f"P_enc rows ({P_enc.shape[0]}) != n_trials ({n_trials}). Check ordering/slicing.")

    # 2. Create Event Patterns
    P_dec = make_noisy_event_patterns(P_enc, rng, decision_mix)
    P_fb  = make_noisy_event_patterns(P_enc, rng, feedback_mix)

    comp_enc = (X_enc @ P_enc)
    comp_dec = (X_dec @ P_dec)
    comp_fb  = (X_fb  @ P_fb)
    Y_sig = comp_enc + comp_dec + comp_fb

    # 3. Generate Signal
    if signal_psc is not None and signal_psc > 0:
        # baseline intensity from the real noise timeseries (ROI sampled)
        baseline = float(np.mean(noise_2d))  # ~550 in your plot

        # target RMS amplitude in intensity units
        target_rms = (signal_psc / 100.0) * baseline

        cur_rms = float(np.std(Y_sig))
        if cur_rms > 0:
            scale = target_rms / cur_rms
            comp_enc *= scale
            comp_dec *= scale
            comp_fb  *= scale
            Y_sig    *= scale


    # Apply voxel latency mismatch consistently
    if hrf_latency_mismatch_sd and hrf_latency_mismatch_sd > 0:
        Y_sig, shifts = apply_voxel_latency_shift(Y_sig, TR, rng, hrf_latency_mismatch_sd, return_shifts=True)
        comp_enc, comp_dec, comp_fb = apply_voxel_latency_shift_to_components([comp_enc, comp_dec, comp_fb], TR, shifts)
    else:
        shifts = np.zeros(Y_sig.shape[1], dtype=float)

    # 4. Noise (doc-style pipeline)
    # Noise is generated outside this function using the BrainIAK example pipeline
    # (mask_brain -> calc_noise -> generate_noise), then we sample n_vox voxels from the ROI
    # and pass a (T x n_vox) matrix in here.
    if noise_2d.shape[0] != n_scans:
        raise ValueError(f"noise_2d has {noise_2d.shape[0]} timepoints but design has n_scans={n_scans}.")
    if noise_2d.shape[1] != n_vox:
        raise ValueError(f"noise_2d has {noise_2d.shape[1]} voxels but n_vox={n_vox}.")

    Y = Y_sig + noise_2d

    # 7. GLM Fitting (LS-A / LS-S)
    C = dct_basis(n_scans, TR, hp_cutoff)

    # Prewhitening (after simulation, as in typical fMRI pipelines)
    # NOTE: This is a simplified global AR(1) prewhitening; SPM uses a richer model.
    ar1_rho = float("nan")
    if prewhiten and prewhiten.lower() != "none":
        if prewhiten.lower() != "ar1":
            raise ValueError("prewhiten must be 'none' or 'ar1'")
        Y, (X_enc, X_dec, X_fb, C), ar1_rho = ar1_prewhiten(
            Y=Y,
            X_list=[X_enc, X_dec, X_fb, C],
            C_hp=C,
        )

    if nuisance_model == "summed":
        x_dec = X_dec.sum(axis=1, keepdims=True)
        x_fb  = X_fb.sum(axis=1, keepdims=True)
        X_nuis = np.column_stack([x_dec, x_fb, C])
    elif nuisance_model == "trialwise":
        X_nuis = np.column_stack([X_dec, X_fb, C])
    else:
        raise ValueError("nuisance_model must be 'summed' or 'trialwise'")

    # LS-A
    X_lsa = np.column_stack([X_enc, X_nuis])
    Bhat = pinv_beta(X_lsa, Y)
    Bhat_enc_lsa = Bhat[:n_trials, :]

    # LS-S
    Bhat_enc_lss = np.zeros_like(Bhat_enc_lsa) # Initialize with correct shape
    
    # Pre-calculate X_others sum to speed up loop
    X_enc_sum = X_enc.sum(axis=1, keepdims=True)
    
    for j in range(n_trials):
        x_this = X_enc[:, [j]]
        # Efficiently calculate "others" by subtracting current from sum
        x_others = X_enc_sum - x_this
        
        X_lss = np.column_stack([x_this, x_others, X_nuis])
        b = pinv_beta(X_lss, Y)
        Bhat_enc_lss[j] = b[0, :]


    # 8. Anchored RSA recovery (per trial), plus null baselines
    rsa_anchor_lsa = anchored_trial_rsa(P_enc, Bhat_enc_lsa)
    rsa_anchor_lss = anchored_trial_rsa(P_enc, Bhat_enc_lss)

    # Category-level anchored RSA (if category labels available)
    cats = category_labels_from_df(df_run)
    rsa_cat_lsa = anchored_category_rsa(P_enc, Bhat_enc_lsa, cats) if cats is not None else {
        "scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": 0
    }
    rsa_cat_lss = anchored_category_rsa(P_enc, Bhat_enc_lss, cats) if cats is not None else {
        "scores": None, "mean": float("nan"), "rdm_spearman": float("nan"), "n_cat": 0
    }

    # Voxel-shuffle null (break voxel correspondence; better than trial-identity shuffles when categories repeat)
    rsa_anchor_lsa_null = anchored_trial_rsa_voxelshuffle_null(
    P_enc, Bhat_enc_lsa,
    null_perms=rsa_null_perms,
    rng=rng,
    shuffle_mode=rsa_shuffle_mode
    )
    rsa_anchor_lss_null = anchored_trial_rsa_voxelshuffle_null(P_enc, Bhat_enc_lss, null_perms=rsa_null_perms, rng=rng, shuffle_mode="global")

    # Optional category-mismatch null if category labels exist in df_run
    cats = category_mismatch_null_indices(df_run)
    rsa_anchor_lsa_catnull = anchored_trial_rsa_category_null(P_enc, Bhat_enc_lsa, cats) if cats is not None else {"null_mean": float("nan")}
    rsa_anchor_lss_catnull = anchored_trial_rsa_category_null(P_enc, Bhat_enc_lss, cats) if cats is not None else {"null_mean": float("nan")}

    return {
        "Y": Y,
        "P_enc": P_enc,
        "Bhat_enc_lsa": Bhat_enc_lsa,
        "Bhat_enc_lss": Bhat_enc_lss,
        "prewhiten": prewhiten,
        "ar1_rho": ar1_rho,
        "rec_lsa": corr_rows(P_enc, Bhat_enc_lsa),
        "rec_lss": corr_rows(P_enc, Bhat_enc_lss),
        "rsa_anchor_lsa": rsa_anchor_lsa,
        "rsa_anchor_lss": rsa_anchor_lss,
        "rsa_cat_lsa": rsa_cat_lsa,
        "rsa_cat_lss": rsa_cat_lss,
        "rsa_anchor_lsa_null": rsa_anchor_lsa_null,
        "rsa_anchor_lss_null": rsa_anchor_lss_null,
        "rsa_anchor_lsa_catnull": rsa_anchor_lsa_catnull,
        "rsa_anchor_lss_catnull": rsa_anchor_lss_catnull,
        "comp_enc": comp_enc,
        "comp_dec": comp_dec,
        "comp_fb": comp_fb,

    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--patterns_npz", required=True)
    ap.add_argument("--noise_nii", required=True, help="Real 4D NIfTI used to estimate noise + provide affine/shape.")
    ap.add_argument("--plot_true", action="store_true",
                    help="Interactive plot of true voxel responses (encoding/decision/feedback) for rep 0.")
    ap.add_argument("--plot_run_id", default=None,
                    help="If provided, plot only this run_id (default: first run) when --plot_true is set.")

    ap.add_argument("--TR", type=float, required=True)
    ap.add_argument("--hp_cutoff", type=float, default=128.0)
    ap.add_argument("--pad_s", type=float, default=32.0)
    ap.add_argument("--dec_dur_s", type=float, default=2.0)
    ap.add_argument("--temporal_resolution", type=float, default=100.0)
    ap.add_argument("--signal_psc", type=float, default=0.5,
                help="Scale simulated signal to target percent signal change relative to noise mean (e.g., 0.5 means 0.5%% PSC).")

    ap.add_argument("--n_vox", type=int, default=200)

    ap.add_argument("--decision_mix", type=float, default=0.55)
    ap.add_argument("--feedback_mix", type=float, default=0.80)

    ap.add_argument("--n_reps", type=int, default=1,
                    help="Number of simulation repetitions to run (each rep simulates all runs once).")
    ap.add_argument("--noise_reps", type=int, default=None,
                    help="Number of independent noise repetitions to generate/cache for cross-design reuse. "
                         "Default: same as --n_reps.")
    ap.add_argument("--n_parallel", type=int, default=1,
                    help="Parallel workers for generating independent noise repetitions (ProcessPool).")
    ap.add_argument("--rsa_null_perms", type=int, default=200,
                    help="Number of permutations for voxel-shuffle null of anchored RSA (0 to disable).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--hrf_latency_mismatch_sd", type=float, default=0.0)

    ap.add_argument("--nuisance_model", choices=["summed","trialwise"], default="summed")

    ap.add_argument("--prewhiten", choices=["none","ar1"], default="ar1",
                    help="Prewhitening applied after simulation and before GLM. 'ar1' applies a simplified global AR(1) filter; 'none' disables.")

    ap.add_argument("--loc_space", choices=["mni", "voxel", "frac"], default="frac",
                    help="Coordinate space for LOC center: MNI mm, voxel indices, or brain-mask fractions.")
    ap.add_argument("--loc_mni", type=float, nargs=3, default=[48.0, -84.0, -3.0],
                    help="LOC center in MNI mm (x y z). Used when --loc_space=mni.")
    ap.add_argument("--loc_vox", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="LOC center in voxel indices (i j k). Used when --loc_space=voxel.")
    ap.add_argument("--loc_vox_frac", type=float, nargs=3, default=[0.80, 0.15, 0.35],
                    help="LOC center as fractions of brain-mask bounds (x y z). Used when --loc_space=frac.")
    ap.add_argument("--loc_from_summary_csv", default=None,
                    help="Optional: path to a previously written recovery_summary.csv. If provided, the LOC center "
                         "(world + voxel coordinates) will be fixed to the first row of that file, overriding "
                         "--loc_space/--loc_*.")
    ap.add_argument("--roi_target_vox", type=int, default=200, help="Target voxels in the LOC blob before sampling.")
    ap.add_argument("--roi_r0_mm", type=float, default=10.0, help="Initial radius guess for ROI sphere (mm).")
    ap.add_argument("--brainmask_frac", type=float, default=0.2, help="Threshold fraction for auto brain mask.")
    ap.add_argument("--write_roi_mask", action="store_true", help="Write ROI mask NIfTI alongside outputs.")
    ap.add_argument("--out_dir", default="fmrisim_cornet_loc_out")
    ap.add_argument("--cache_noise_dict", action="store_true",
                help="Cache the fmrisim noise_dict (from calc_noise) to disk so subsequent runs load it.")
    ap.add_argument("--cache_run_noise", action="store_true",
                help="Cache the sampled ROI noise (T x n_vox) per run to disk so subsequent runs load it.")
    ap.add_argument("--noise_cache_seed", type=int, default=0,
                help="Seed used for deterministic voxel sampling when caching run noise.")
    ap.add_argument("--noise_cache_dir", default=None,
                help="Optional directory for noise cache. Default: <out_dir>/noise_cache")

    # Backwards-compatible / deprecated args (ignored in this doc-faithful pipeline)
    ap.add_argument("--noise_mask", default=None,
                    help="DEPRECATED/IGNORED. This script now estimates noise from the whole-brain mask per BrainIAK docs.")
    ap.add_argument("--noise_roi_scale", type=float, default=None,
                    help="DEPRECATED/IGNORED. Noise is generated from whole-brain per BrainIAK docs.")
    args = ap.parse_args()

    if args.noise_mask is not None or args.noise_roi_scale is not None:
        print("[fmrisim_run] NOTE: --noise_mask/--noise_roi_scale are ignored. "
              "This script follows the BrainIAK fmrisim example pipeline: estimate noise on whole brain, then generate noise.")

    # Load design and enforce ordering consistent with CORnet pattern generation
    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.sort_values(["run_id", "img_onset"]).reset_index(drop=True)

    # Load patterns
    patterns = np.load(args.patterns_npz, allow_pickle=True)
    P_all = patterns["patterns_enc_200"].astype(np.float32)
    if P_all.shape[1] != args.n_vox:
        raise ValueError(f"patterns_enc_200 has {P_all.shape[1]} dims, expected n_vox={args.n_vox}")
    if len(df) != P_all.shape[0]:
        raise ValueError(f"CSV has {len(df)} rows but patterns have {P_all.shape[0]} rows. "
                         "Regenerate patterns from the same CSV/order, or align explicitly.")

    # Load noise volume (also provides space/affine)
    noise_img = nib.load(str(args.noise_nii))
    vol4d = noise_img.get_fdata().astype(np.float32)
    aff = noise_img.affine
    shape3d = vol4d.shape[:3]
    # Noise model estimation and caching (doc-style pipeline)
    base_out_dir = Path(args.out_dir)
    design_group = infer_design_group_from_csv(Path(args.csv))
    out_dir = ensure_dir(base_out_dir / design_group)

    # Shared noise cache lives under base_out_dir/noise/<dataset_id> by default (cross-design reuse)
    dataset_id = infer_dataset_id_from_noise_path(args.noise_nii)
    noise_root = ensure_dir(base_out_dir / "noise" / dataset_id)

    # Allow override for all noise-related caches
    cache_dir = Path(args.noise_cache_dir) if args.noise_cache_dir else noise_root

    noise_dict, brain_mask, template = load_or_make_noise_dict(
        vol4d=vol4d,
        noise_nii_path=str(args.noise_nii),
        TR=args.TR,
        cache_dir=cache_dir,
        use_cache=bool(args.cache_noise_dict),
    )

    # Build LOC ROI mask in this space
    # Optionally: fix the LOC center to a previously written recovery_summary.csv
    if args.loc_from_summary_csv:
        prev = pd.read_csv(args.loc_from_summary_csv)
        need = {"loc_world_x","loc_world_y","loc_world_z","loc_vox_x","loc_vox_y","loc_vox_z"}
        if not need.issubset(set(prev.columns)):
            raise ValueError(
                "--loc_from_summary_csv is missing required columns. Expected at least: " + ", ".join(sorted(need))
            )
        r0 = prev.iloc[0]
        center_world = np.array([r0["loc_world_x"], r0["loc_world_y"], r0["loc_world_z"]], dtype=float)
        center_vox = np.array([r0["loc_vox_x"], r0["loc_vox_y"], r0["loc_vox_z"]], dtype=float)
        loc_source = "from_summary"
    else:
        loc_source = args.loc_space
        if args.loc_space == "mni":
            center_world = np.array(args.loc_mni, dtype=float)
            center_vox = apply_affine(np.linalg.inv(aff), center_world)
        elif args.loc_space == "voxel":
            center_vox = np.array(args.loc_vox, dtype=float)
            center_world = apply_affine(aff, center_vox)
        else:
            # Fractional placement is fragile in cropped/native EPIs; snap to the nearest in-brain voxel.
            center_vox = estimate_loc_center_from_mask(brain_mask, tuple(args.loc_vox_frac))
            center_vox = snap_vox_to_brain_mask(center_vox, brain_mask, max_radius_vox=40)
            center_world = apply_affine(aff, center_vox)


    roi_mask, radius_mm = roi_mask_target_voxels(
        shape3d=shape3d,
        aff=aff,
        brain_mask=brain_mask,
        center_mni=center_world,
        target_n_vox=args.roi_target_vox,
        r0_mm=args.roi_r0_mm,
        max_iter=12
    )

    n_roi = int(roi_mask.sum())
    if n_roi == 0:
        raise RuntimeError("LOC ROI mask is empty. The coordinate may be outside the brain mask. "
                           "Try a different --loc_space setting or coordinate.")

    # If frac-based placement yields a tiny ROI (common when the EPI is cropped), fall back to a conservative
    # occipital (posterior) in-brain center and retry once.
    if n_roi < args.n_vox and loc_source == "frac":
        center_vox = estimate_occipital_center_from_mask(brain_mask, aff, posterior_quantile=0.10)
        center_world = apply_affine(aff, center_vox)
        roi_mask, radius_mm = roi_mask_target_voxels(
            shape3d=shape3d,
            aff=aff,
            brain_mask=brain_mask,
            center_mni=center_world,
            target_n_vox=args.roi_target_vox,
            r0_mm=args.roi_r0_mm,
            max_iter=12
        )
        n_roi = int(roi_mask.sum())
        loc_source = "occipital_fallback"

    if n_roi < args.n_vox:
        raise RuntimeError(f"LOC ROI has only {n_roi} voxels but you requested n_vox={args.n_vox}. "
                           "Increase --roi_target_vox or --roi_r0_mm.")


    if args.write_roi_mask:
        roi_img = nib.Nifti1Image(roi_mask.astype(np.uint8), aff)
        nib.save(roi_img, str(out_dir / "loc_roi_mask.nii.gz"))

    # Precompute slices per run
    run_ids = sorted(df["run_id"].unique().tolist())
    idx0 = 0
    run_slices = {}
    for rid in run_ids:
        n = int((df["run_id"] == rid).sum())
        run_slices[rid] = slice(idx0, idx0 + n)
        idx0 += n

    # Precompute required n_scans per run, then build/load a *shared* noise bank long enough for this design.
    run_n_scans: Dict = {}
    max_n_scans = 0
    for rid in run_ids:
        df_run = df[df["run_id"] == rid].copy()
        n_scans_r = compute_n_scans_for_run(df_run, TR=args.TR, pad_s=args.pad_s, dec_dur_s=args.dec_dur_s)
        run_n_scans[rid] = int(n_scans_r)
        max_n_scans = max(max_n_scans, int(n_scans_r))

    # --- Coupled noise reps ---
    # Couple simulation repetitions (n_reps) to noise repetitions: simulation rep i uses noise rep i.
    # If --noise_reps is provided, treat it as a *minimum* cache size, but always ensure we have
    # at least n_reps noise indices available.
    n_noise_reps = max(
        1,
        int(args.n_reps),
        int(args.noise_reps) if args.noise_reps is not None else 0,
    )

    run_noise, noise_meta = load_or_make_run_noise_roi_set(
        noise_nii_path=str(args.noise_nii),
        TR=float(args.TR),
        n_vox=int(args.n_vox),
        roi_mask=roi_mask,
        brain_mask=brain_mask,
        template=template,
        noise_dict=noise_dict,
        run_n_scans=run_n_scans,
        n_noise_reps=int(n_noise_reps),
        cache_dir=cache_dir,
        use_cache=bool(args.cache_run_noise),
        noise_cache_seed=int(args.noise_cache_seed),
        n_parallel=int(args.n_parallel),
    )
    rng_master = np.random.default_rng(args.seed)
    rows = []

    for rep in tqdm(range(args.n_reps), desc="Simulating Reps"):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for rid in run_ids:
            df_run = df[df["run_id"] == rid].copy()
            P_enc = P_all[run_slices[rid]].copy()

            # Ensure the coupled noise index exists; if missing after loading, regenerate on-demand.
            if (rid, int(rep)) not in run_noise:
                run_noise, noise_meta = load_or_make_run_noise_roi_set(
                    noise_nii_path=str(args.noise_nii),
                    TR=float(args.TR),
                    n_vox=int(args.n_vox),
                    roi_mask=roi_mask,
                    brain_mask=brain_mask,
                    template=template,
                    noise_dict=noise_dict,
                    run_n_scans=run_n_scans,
                    n_noise_reps=int(rep) + 1,
                    cache_dir=cache_dir,
                    use_cache=bool(args.cache_run_noise),
                    noise_cache_seed=int(args.noise_cache_seed),
                    n_parallel=int(args.n_parallel),
                )

            res = simulate_one_run(
                df_run=df_run,
                P_enc=P_enc,
                TR=args.TR,
                hp_cutoff=args.hp_cutoff,
                pad_s=args.pad_s,
                dec_dur_s=args.dec_dur_s,
                tres=args.temporal_resolution,
                hrf_type_sim="double_gamma",
                noise_2d=run_noise[(rid, int(rep))],
                n_vox=args.n_vox,
                rng=rng,
                decision_mix=args.decision_mix,
                feedback_mix=args.feedback_mix,
                hrf_latency_mismatch_sd=args.hrf_latency_mismatch_sd,
                nuisance_model=args.nuisance_model,
                prewhiten=args.prewhiten,
                rsa_null_perms=args.rsa_null_perms,
                rsa_shuffle_mode="global",
                signal_psc=args.signal_psc
            )
            if args.plot_true and rep == 0:
                # Choose which run to plot
                want = args.plot_run_id
                if want is None or str(want) == str(rid):
                    interactive_plot_true_components(
                        TR=args.TR,
                        comp_enc=res["comp_enc"],
                        comp_dec=res["comp_dec"],
                        comp_fb=res["comp_fb"],
                        Y_total=res["Y"],  # show signal+noise too; you can set to None if you only want true components
                        title_prefix=f"Run {rid} | ",
                    )
                    # If user specified a particular run_id, avoid repeated windows
                    if want is not None:
                        args.plot_true = False

            rows.append({
                "rep": rep,
                "run_id": rid,
                "loc_space": loc_source,
                "loc_from_summary_csv": bool(args.loc_from_summary_csv),
                "loc_world_x": float(center_world[0]),
                "loc_world_y": float(center_world[1]),
                "loc_world_z": float(center_world[2]),
                "loc_vox_x": float(center_vox[0]),
                "loc_vox_y": float(center_vox[1]),
                "loc_vox_z": float(center_vox[2]),
                "roi_radius_mm": radius_mm,
                "roi_voxels": n_roi,
                "n_vox_sampled": args.n_vox,
                "hrf_latency_mismatch_sd": args.hrf_latency_mismatch_sd,
                "nuisance_model": args.nuisance_model,
                "prewhiten": str(res.get("prewhiten", "")),
                "ar1_rho": float(res.get("ar1_rho", float("nan"))),
                "lsa_mean": float(res["rec_lsa"].mean()),
                "lsa_p05": float(np.percentile(res["rec_lsa"], 5)),
                "lss_mean": float(res["rec_lss"].mean()),
                "lss_p05": float(np.percentile(res["rec_lss"], 5)),
                "rsa_anchor_lsa_mean": float(np.nanmean(res["rsa_anchor_lsa"])),
                "rsa_anchor_lsa_p05": float(np.nanpercentile(res["rsa_anchor_lsa"], 5)),
                "rsa_anchor_lsa_null_mean": float(res["rsa_anchor_lsa_null"].get("null_mean", float("nan"))),
                "rsa_anchor_lsa_null_p95": float(res["rsa_anchor_lsa_null"].get("null_p95", float("nan"))),
                "rsa_anchor_lsa_catnull_mean": float(res["rsa_anchor_lsa_catnull"].get("null_mean", float("nan"))),
                "rsa_cat_lsa_mean": float(res["rsa_cat_lsa"].get("mean", float("nan"))),
                "rsa_cat_lsa_rdm_spearman": float(res["rsa_cat_lsa"].get("rdm_spearman", float("nan"))),
                "rsa_cat_lsa_n_cat": int(res["rsa_cat_lsa"].get("n_cat", 0)),
                "rsa_anchor_lss_mean": float(np.nanmean(res["rsa_anchor_lss"])),
                "rsa_anchor_lss_p05": float(np.nanpercentile(res["rsa_anchor_lss"], 5)),
                "rsa_anchor_lss_null_mean": float(res["rsa_anchor_lss_null"].get("null_mean", float("nan"))),
                "rsa_anchor_lss_null_p95": float(res["rsa_anchor_lss_null"].get("null_p95", float("nan"))),
                "rsa_anchor_lss_catnull_mean": float(res["rsa_anchor_lss_catnull"].get("null_mean", float("nan"))),
                "rsa_cat_lss_mean": float(res["rsa_cat_lss"].get("mean", float("nan"))),
                "rsa_cat_lss_rdm_spearman": float(res["rsa_cat_lss"].get("rdm_spearman", float("nan"))),
                "rsa_cat_lss_n_cat": int(res["rsa_cat_lss"].get("n_cat", 0)),
                })

            if rep == 0:
                np.savez_compressed(
                    out_dir / f"sim_run{rid}_rep0.npz",
                    Y=res["Y"].astype(np.float32),
                    P_enc=res["P_enc"].astype(np.float32),
                    Bhat_enc_lsa=res["Bhat_enc_lsa"].astype(np.float32),
                    Bhat_enc_lss=res["Bhat_enc_lss"].astype(np.float32),
                    rec_lsa=res["rec_lsa"].astype(np.float32),
                    rec_lss=res["rec_lss"].astype(np.float32),
                )

        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "recovery_summary.csv", index=False)

    print(f"LOC source: {loc_source}")
    print(f"LOC center (world mm): {tuple(center_world.tolist())}")
    print(f"LOC center (voxel): {tuple(center_vox.tolist())}")
    print(f"ROI voxels (mask): {n_roi} | radius_mm≈{radius_mm:.2f} | sampled n_vox={args.n_vox}")
    print(summary.groupby("run_id")[["lsa_mean","lsa_p05","lss_mean","lss_p05",
                                "rsa_anchor_lsa_mean","rsa_anchor_lss_mean",
                                "rsa_anchor_lsa_null_mean","rsa_anchor_lss_null_mean",
                                "rsa_anchor_lsa_catnull_mean","rsa_anchor_lss_catnull_mean"]].mean().round(4))
    print(f"\nWrote: {out_dir / 'recovery_summary.csv'}")
    if args.write_roi_mask:
        print(f"Wrote: {out_dir / 'loc_roi_mask.nii.gz'}")


if __name__ == "__main__":
    main()
