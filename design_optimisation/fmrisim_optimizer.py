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
import os
import math
import hashlib
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple
# Optional progress bars (tqdm). Falls back to plain iterators if unavailable.
try:
    from tqdm import tqdm, trange
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x

    def trange(n, *args, **kwargs):
        return range(n)

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


# -----------------------------------------------------------------------------
# Multiprocessing-safe optimiser worker (Windows requires top-level functions)
# -----------------------------------------------------------------------------
_OPT_CTX = {}
COLL_PENALTY = 1.0        # cosine similarity max (worst possible)
COLL_PENALTY_P95 = 0.8   # strong but slightly less extreme


# Ensure weights are always a simple (enc, dec, fb) float tuple in workers.
# Windows multiprocessing is sensitive to pickling and to accidental dict iteration.
def _coerce_weights_tuple(weights):
    """Return (w_enc, w_dec, w_fb) as floats.

    Accepts:
      - tuple/list/np array of length 3
      - dict with keys enc/dec/fb
      - sequence of (key, value) pairs
    """
    if isinstance(weights, dict):
        return (float(weights["enc"]), float(weights["dec"]), float(weights["fb"]))
    # sequence of pairs?
    try:
        w_try = dict(weights)
        if all(k in w_try for k in ("enc", "dec", "fb")):
            return (float(w_try["enc"]), float(w_try["dec"]), float(w_try["fb"]))
    except Exception:
        pass
    # plain length-3 sequence
    try:
        a,b,c = weights
        return (float(a), float(b), float(c))
    except Exception as e:
        raise TypeError("weights must be (enc,dec,fb), or dict-like with enc/dec/fb") from e


def _optim_pool_init(
    template_csv: str,
    TR: float,
    hp_cutoff: float,
    pad_s: float,
    opt_tres: int,
    weights,
    trial_len_range,
    mean_trial_len_range,
    opt_isi_range,
    opt_iti_range,
    dists,
    w_coll: float,
    w_cov: float,
):
    """Initializer for ProcessPoolExecutor workers (picklable args only)."""
    global _OPT_CTX
    _OPT_CTX = {
        "template": pd.read_csv(template_csv),
        "TR": float(TR),
        "hp_cutoff": (float(hp_cutoff) if hp_cutoff is not None else None),
        "pad_s": float(pad_s),
        "opt_tres": int(opt_tres),
        "weights": _coerce_weights_tuple(weights),
        "trial_len_range": trial_len_range,
        "mean_trial_len_range": mean_trial_len_range,
        "opt_isi_range": opt_isi_range,
        "opt_iti_range": opt_iti_range,
        "dists": list(dists),
        "w_coll": float(w_coll),
        "w_cov": float(w_cov),
    }

def _optim_eval_one_impl(x_list, idx_seed: int):
    """Evaluate one candidate (runs inside worker processes)."""
    ctx = _OPT_CTX
    template = ctx["template"]
    dists = ctx["dists"]

    # x_list is [isi_lo, isi_hi, iti_lo, iti_hi, isi_di, iti_di, isi_sh, iti_sh]
    isi_lo, isi_hi, iti_lo, iti_hi, isi_di, iti_di, isi_sh, iti_sh = list(x_list)
    isi_dist = dists[int(isi_di)]
    iti_dist = dists[int(iti_di)]

    df_new = generate_design_from_template(
        template=template,
        isi_dist=isi_dist,
        iti_dist=iti_dist,
        isi_lo=float(isi_lo), isi_hi=float(isi_hi),
        iti_lo=float(iti_lo), iti_hi=float(iti_hi),
        isi_shape=float(isi_sh), iti_shape=float(iti_sh),
        TR=float(ctx["TR"]),
        trial_len_range=ctx["trial_len_range"],
        mean_trial_len_range=ctx["mean_trial_len_range"],
        opt_isi_range=ctx["opt_isi_range"],
        opt_iti_range=ctx["opt_iti_range"],
        seed=int(idx_seed),
    )
    if df_new is None:
        return None

    s1 = evaluate_design_stage1(
        df_new,
        TR=float(ctx["TR"]),
        hp_cutoff=float(ctx["hp_cutoff"]),
        pad_s=float(ctx["pad_s"]),
        tres=int(ctx["opt_tres"]),
        weights=ctx["weights"],
    )

    empty_pen = 0.0 if s1["cov_minbin_worst"] > 0 else 1000.0
    obj = (float(ctx["w_coll"]) * (s1["coll_p95_worst"] + 0.5 * s1["coll_max_worst"])
           + float(ctx["w_cov"]) * (s1["cov_chi2_worst"] + empty_pen))

    return obj, s1, df_new, {
        "isi_dist": isi_dist, "iti_dist": iti_dist,
        "isi_lo": float(isi_lo), "isi_hi": float(isi_hi),
        "iti_lo": float(iti_lo), "iti_hi": float(iti_hi),
        "isi_shape": float(isi_sh), "iti_shape": float(iti_sh),
    }

def _optim_eval_one_worker(payload):
    """Thin wrapper so executor only pickles top-level function."""
    x_list, idx_seed = payload
    return _optim_eval_one_impl(x_list, idx_seed)

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


# ------------------------
# Trialwise correlation metrics (SPM-style HRF + DCT high-pass), adapted from design_trialwise_corr.py
# ------------------------
def _interp1_linear_extrap(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """1D linear interpolation with linear extrapolation (MATLAB interp1(...,'linear','extrap')-like)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)
    yq = np.interp(xq, x, y)
    if x.size >= 2:
        left = xq < x[0]
        if np.any(left):
            slope = (y[1] - y[0]) / (x[1] - x[0])
            yq[left] = y[0] + slope * (xq[left] - x[0])
        right = xq > x[-1]
        if np.any(right):
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            yq[right] = y[-1] + slope * (xq[right] - x[-1])
    return yq

def _build_trialwise_interp(t_micro: np.ndarray, t_scan: np.ndarray,
                           onsets: np.ndarray, durs: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """Build 1 regressor per event instance (trial), HRF convolved at microtime, sampled at TR with linear extrap."""
    onsets = np.asarray(onsets, float)
    durs = np.asarray(durs, float)
    dt = float(t_micro[1] - t_micro[0])
    n_trials = int(onsets.size)
    X = np.zeros((t_scan.size, n_trials), dtype=float)
    for j in range(n_trials):
        u = np.zeros(t_micro.size, dtype=float)
        onset_idx = int(round(onsets[j] / dt))
        dur_idx = max(1, int(round(durs[j] / dt)))
        last_idx = min(u.size - 1, onset_idx + dur_idx - 1)
        if onset_idx < u.size and last_idx >= 0:
            u[max(onset_idx, 0):last_idx + 1] = 1.0
        x_micro = np.convolve(u, hrf, mode="full")[:u.size]
        X[:, j] = _interp1_linear_extrap(t_micro, x_micro, t_scan)
    return X

def _build_condition_level_interp(t_micro: np.ndarray, t_scan: np.ndarray,
                                 onsets: np.ndarray, durs: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """One regressor that is the sum of all events of that type."""
    onsets = np.asarray(onsets, float)
    durs = np.asarray(durs, float)
    dt = float(t_micro[1] - t_micro[0])
    u = np.zeros(t_micro.size, dtype=float)
    for j in range(onsets.size):
        onset_idx = int(round(onsets[j] / dt))
        dur_idx = max(1, int(round(durs[j] / dt)))
        last_idx = min(u.size - 1, onset_idx + dur_idx - 1)
        if onset_idx < u.size and last_idx >= 0:
            u[max(onset_idx, 0):last_idx + 1] += 1.0
    x_micro = np.convolve(u, hrf, mode="full")[:u.size]
    return _interp1_linear_extrap(t_micro, x_micro, t_scan)

def _dct_resid(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return X - C @ (C.T @ X)

def _partial_out(X: np.ndarray, N: np.ndarray) -> np.ndarray:
    pinvN = np.linalg.pinv(N)
    return X - N @ (pinvN @ X)

def _zscore_cols(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd

def _corr_mat_cols(X: np.ndarray) -> np.ndarray:
    return np.corrcoef(X, rowvar=False)

def _summarize_corr_abs(R: np.ndarray) -> Dict[str, float]:
    off = R[~np.eye(R.shape[0], dtype=bool)]
    abs_off = np.abs(off)
    if abs_off.size == 0:
        return {"max_abs_r": float("nan"), "p95_abs_r": float("nan"), "mean_abs_r": float("nan")}
    return {
        "max_abs_r": float(np.max(abs_off)),
        "p95_abs_r": float(np.percentile(abs_off, 95)),
        "mean_abs_r": float(np.mean(abs_off)),
    }

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


def load_or_make_noise_dict(vol4d: np.ndarray, noise_nii_path: str, TR: float, cache_dir: Path, use_cache: bool, hp_cutoff: float | None = None) -> Tuple[dict, np.ndarray, np.ndarray]:
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
        "hp_cutoff": float(hp_cutoff),
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
        "hp_cutoff": float(hp_cutoff),
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
        "hp_cutoff": float(hp_cutoff),
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
        "hp_cutoff": float(hp_cutoff),
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
    hp_cutoff: float,
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

    # Helper to build file paths (canonical filename: no exact-T requirement)
    def _paths_for(rid, rep_idx: int) -> tuple[Path, Path]:
        run_dir = ds_dir / f"run-{rid}"
        ensure_dir(run_dir)
        npz = run_dir / f"noise_roi_{key}_rep{int(rep_idx):03d}.npz"
        meta = run_dir / f"noise_roi_{key}_rep{int(rep_idx):03d}.json"
        return npz, meta

    def _meta_ok(meta: dict, rid, rep_idx: int) -> bool:
        """Validate cached meta against the current parameters that define the ROI noise."""
        if not isinstance(meta, dict):
            return False
        try:
            if meta.get("key") != key:
                return False
            if str(meta.get("dataset_id")) != str(dataset_id):
                return False
            if str(meta.get("noise_nii_path")) != str(noise_nii_path):
                return False
            if float(meta.get("TR")) != float(TR):
                return False
            if int(meta.get("n_vox")) != int(n_vox):
                return False
            if str(meta.get("run_id")) != str(rid):
                return False
            if int(meta.get("rep_idx")) != int(rep_idx):
                return False
            if str(meta.get("pick_lin_hash")) != _stable_hash(pick_lin.tobytes())[:16]:
                return False
            if int(meta.get("noise_cache_seed")) != int(noise_cache_seed):
                return False
            # noise_dict can include floats; compare via JSON round-trip to stabilise ordering
            if json.dumps(meta.get("noise_dict", {}), sort_keys=True) != json.dumps(noise_dict, sort_keys=True):
                return False
        except Exception:
            return False
        return True

    # Load cached ROI noise if it is usable. Valid if T_cached >= T_needed (slice to T_needed).
    # If the cached file is too short, regenerate and overwrite the canonical filename.
    def _load_if_ok_or_rewrite(rid, rep_idx: int, T_needed: int) -> Optional[np.ndarray]:
        run_dir = ds_dir / f"run-{rid}"
        if not run_dir.exists():
            return None

        canon_npz, canon_meta = _paths_for(rid, rep_idx)

        def _try_load(npz_path: Path, meta_path: Optional[Path]) -> Optional[np.ndarray]:
            try:
                dat = np.load(npz_path)
                if "noise_2d" not in dat:
                    return None
                arr = dat["noise_2d"].astype(np.float32)
                if arr.ndim != 2 or arr.shape[1] != int(n_vox):
                    return None
                # Prefer stored T_cached if present (backward compatible: fall back to shape[0])
                T_cached = int(dat["T_cached"]) if "T_cached" in dat.files else int(arr.shape[0])

                # Validate meta if available (params mismatch => reject)
                if meta_path is not None and meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                    except Exception:
                        meta = {}
                    if not _meta_ok(meta, rid, rep_idx):
                        return None

                if T_cached >= int(T_needed):
                    return arr[:int(T_needed)]
                else:
                    return None
            except Exception:
                return None

        # 1) Try canonical file first
        out = _try_load(canon_npz, canon_meta)
        if out is not None:
            return out

        # 2) Backward compatibility: look for old *_T*.npz, prefer largest T, and rewrite into canonical filename
        old = sorted(run_dir.glob(f"noise_roi_{key}_rep{int(rep_idx):03d}_T*.npz"))
        if old:
            # parse T from filename
            best = None
            best_T = -1
            for p in old:
                m = re.search(r"_T(\d+)\.npz$", p.name)
                if not m:
                    continue
                T = int(m.group(1))
                if T > best_T:
                    best_T = T
                    best = p
            if best is not None:
                meta_guess = best.with_suffix(".json")
                out = _try_load(best, meta_guess if meta_guess.exists() else None)
                if out is not None:
                    # Rewrite canonical file with the full cached array (not sliced) if possible
                    try:
                        dat = np.load(best)
                        arr_full = dat["noise_2d"].astype(np.float32)
                        T_cached = int(dat["T_cached"]) if "T_cached" in dat.files else int(arr_full.shape[0])
                        np.savez_compressed(canon_npz, noise_2d=arr_full.astype(np.float32), T_cached=int(T_cached))
                        # Rewrite meta as canonical
                        meta = {
                            "key": key,
                            "dataset_id": dataset_id,
                            "noise_nii_path": str(noise_nii_path),
                            "TR": float(TR),
        "hp_cutoff": float(hp_cutoff),
                            "n_vox": int(n_vox),
                            "n_scans": int(T_cached),
                            "run_id": str(rid),
                            "rep_idx": int(rep_idx),
                            "pick_lin_hash": _stable_hash(pick_lin.tobytes())[:16],
                            "noise_cache_seed": int(noise_cache_seed),
                            "noise_dict": noise_dict,
                        }
                        canon_meta.write_text(json.dumps(meta, indent=2))
                    except Exception:
                        pass
                    return out

        return None
    # Helper: stable seed per (run_id, rep_idx) so caching is deterministic.
    def _seed_for(rid, rep_idx: int) -> int:
        rid_h = int(_stable_hash(str(rid))[:8], 16)
        return (int(noise_cache_seed) + rid_h + int(rep_idx) * 10007) % (2**32 - 1)

# Plan tasks
    tasks = []
    run_noise: Dict = {}
    new_cache_arrays: Dict = {}

    for rid, n_scans in run_n_scans.items():
        n_scans = int(n_scans)
        for rep_idx in range(int(n_noise_reps)):
            if use_cache:
                cached = _load_if_ok_or_rewrite(rid, rep_idx, int(n_scans))
                if cached is not None:
                    run_noise[(rid, rep_idx)] = cached
                    continue
            T_needed = int(n_scans)
            T_gen = int(math.ceil(1.1 * T_needed))
            tasks.append((rid, rep_idx, T_needed, T_gen, _seed_for(rid, rep_idx)))

    # Generate missing tasks (optionally parallel)
    if tasks:
        if int(n_parallel) <= 1:
            for rid, rep_idx, T_needed, T_gen, seed in tasks:
                arr = _generate_one_noise_rep_roi(
                    rep_idx=int(rep_idx),
                    max_n_scans=int(T_gen),
                    TR=float(TR),
                    brain_mask=brain_mask,
                    template=template,
                    noise_dict=noise_dict,
                    pick_lin=pick_lin,
                    seed=int(seed),
                )
                # arr may be longer than needed; slice for use
                run_noise[(rid, rep_idx)] = arr[:int(T_needed)]
                new_cache_arrays[(rid, rep_idx)] = arr
        else:
            with ProcessPoolExecutor(max_workers=int(n_parallel)) as ex:
                futs = []
                for rid, rep_idx, T_needed, T_gen, seed in tasks:
                    futs.append(ex.submit(_generate_one_noise_rep_roi, int(rep_idx), int(T_gen), float(TR), brain_mask, template, noise_dict, pick_lin, int(seed)))
                for (rid, rep_idx, T_needed, T_gen, seed), fut in zip(tasks, futs):
                    arr = fut.result()
                    run_noise[(rid, rep_idx)] = arr[:int(T_needed)]
                    new_cache_arrays[(rid, rep_idx)] = arr

    # Persist any newly generated arrays
    if use_cache:
        for (rid, rep_idx), arr in new_cache_arrays.items():
            n_scans = int(arr.shape[0])
            npz_path, meta_path = _paths_for(rid, rep_idx)
            # Overwrite canonical cache file (may exist but be too short)
            try:
                np.savez_compressed(npz_path, noise_2d=arr.astype(np.float32), T_cached=int(n_scans))
            except Exception:
                pass
            meta = {
                "key": key,
                "dataset_id": dataset_id,
                "noise_nii_path": str(noise_nii_path),
                "TR": float(TR),
        "hp_cutoff": float(hp_cutoff),
                "n_vox": int(n_vox),
                "n_scans": int(n_scans),
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
        "hp_cutoff": float(hp_cutoff),
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



# ------------------------
# Design optimiser
# ------------------------

def spm_hrf(dt: float, p: Optional[np.ndarray] = None, length: float = 32.0) -> np.ndarray:
    """
    SPM canonical HRF sampled at dt seconds.
    """
    if p is None:
        # [delay1, delay2, dispersion1, dispersion2, ratio, onset, length]
        p = np.array([6.0, 16.0, 1.0, 1.0, 6.0, 0.0, length], dtype=float)
    else:
        p = np.asarray(p, dtype=float)

    # Gamma pdf implemented without scipy dependency
    def gamma_pdf(x, a, scale):
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 1e-12, None)
        # pdf = x^(a-1) * exp(-x/scale) / (gamma(a) * scale^a)
        from math import lgamma
        return np.exp((a - 1) * np.log(x) - x / scale - lgamma(a) - a * np.log(scale))

    t = np.arange(0, float(p[6]), float(dt))
    hrf = gamma_pdf(t - float(p[5]), float(p[0]) / float(p[2]), float(p[2])) - gamma_pdf(t - float(p[5]), float(p[1]) / float(p[3]), float(p[3])) / float(p[4])
    hrf[t < float(p[5])] = 0.0
    # normalise to sum=1
    s = hrf.sum()
    if s > 0:
        hrf = hrf / s
    return hrf.astype(np.float32)

def _build_trialwise_X(df_run: pd.DataFrame,
                       TR: float,
                       pad_s: float,
                       tres: int,
                       weights: Tuple[float, float, float]) -> np.ndarray:
    """
    Build trialwise design matrix with one column per trial per event type (enc/dec/fb),
    HRF-convolved, then downsampled to TR.
    Returns X of shape (T, 3*n_trials).
    """
    w_enc, w_dec, w_fb = weights
    n_trials = df_run.shape[0]

    img_on = df_run["img_onset"].to_numpy(float)
    img_dur = df_run["img_dur"].to_numpy(float)
    isi1 = df_run["isi1_dur"].to_numpy(float)
    dec_on = df_run["dec_onset_est"].to_numpy(float)
    dec_dur = df_run["max_dec_dur"].to_numpy(float)
    isi2 = df_run["isi2_dur"].to_numpy(float)
    fb_on = dec_on + dec_dur + isi2
    fb_dur = df_run["fb_dur"].to_numpy(float)

    total_time_s = float(np.max(fb_on + fb_dur)) + float(pad_s)
    T = int(math.ceil(total_time_s / float(TR)))

    dt = float(TR) / float(tres)
    Th = int(math.ceil(total_time_s / dt))
    hrf = spm_hrf(dt=dt, length=32.0)

    def _neural_boxcar(onsets, durs):
        n = np.zeros((Th,), dtype=np.float32)
        for o, d in zip(onsets, durs):
            i0 = int(round(o / dt))
            i1 = int(round((o + d) / dt))
            i0 = max(i0, 0)
            i1 = min(i1, Th)
            if i1 > i0:
                n[i0:i1] = 1.0
        return n

    # Build columns efficiently by convolving per-trial impulses/boxcars
    Xh = np.zeros((Th, 3 * n_trials), dtype=np.float32)

    # For separability, trialwise regressors are per-trial boxcars.
    for j in range(n_trials):
        # encoding
        n = np.zeros((Th,), dtype=np.float32)
        i0 = int(round(img_on[j] / dt))
        i1 = int(round((img_on[j] + img_dur[j]) / dt))
        if i1 > i0 and i0 < Th:
            n[max(i0, 0):min(i1, Th)] = 1.0
        x = np.convolve(n, hrf)[:Th]
        Xh[:, j] = w_enc * x

        # decision
        n[:] = 0.0
        i0 = int(round(dec_on[j] / dt))
        i1 = int(round((dec_on[j] + dec_dur[j]) / dt))
        if i1 > i0 and i0 < Th:
            n[max(i0, 0):min(i1, Th)] = 1.0
        x = np.convolve(n, hrf)[:Th]
        Xh[:, n_trials + j] = w_dec * x

        # feedback
        n[:] = 0.0
        i0 = int(round(fb_on[j] / dt))
        i1 = int(round((fb_on[j] + fb_dur[j]) / dt))
        if i1 > i0 and i0 < Th:
            n[max(i0, 0):min(i1, Th)] = 1.0
        x = np.convolve(n, hrf)[:Th]
        Xh[:, 2 * n_trials + j] = w_fb * x

    # Downsample to TR by averaging within TR bins
    X = np.zeros((T, Xh.shape[1]), dtype=np.float32)
    for t in range(T):
        a = int(round((t * TR) / dt))
        b = int(round(((t + 1) * TR) / dt))
        a = max(a, 0); b = min(b, Th)
        if b > a:
            X[t, :] = Xh[a:b, :].mean(axis=0)

    # Column-normalize (L2) to make Gram stats meaningful
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    X = X / norms[None, :]
    return X

def collinearity_stats_from_X(X: np.ndarray) -> Dict[str, float]:
    """Summarise off-diagonal entries of G=X^T X."""
    G = X.T @ X
    np.fill_diagonal(G, 0.0)
    off = G[np.triu_indices(G.shape[0], k=1)]
    if off.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(off.mean()),
        "p95": float(np.percentile(off, 95)),
        "max": float(off.max()),
    }

def tr_coverage_score(img_on: np.ndarray, TR: float, n_bins: int = 20) -> Dict[str, float]:
    """Uniformity of onset phase modulo TR. Lower chi2 is better."""
    phase = (img_on % float(TR)) / float(TR)
    hist, _ = np.histogram(phase, bins=n_bins, range=(0.0, 1.0))
    expected = float(hist.sum()) / float(n_bins)
    if expected <= 0:
        return {"chi2": float("inf"), "min_bin": 0.0}
    chi2 = float(np.sum((hist - expected) ** 2 / expected))
    return {"chi2": chi2, "min_bin": float(hist.min())}

def _draw_jitter(rng: np.random.Generator, dist: str, n: int, lo: float, hi: float, shape: float) -> np.ndarray:
    lo = float(lo); hi = float(hi)
    if hi <= lo:
        return np.full((n,), lo, dtype=np.float32)

    if dist == "uniform":
        x = rng.uniform(lo, hi, size=n)
    elif dist == "truncnorm":
        # shape = sd as fraction of range (default ~0.25)
        sd = max(1e-6, float(shape) * (hi - lo))
        mu = (lo + hi) / 2.0
        x = rng.normal(mu, sd, size=n)
        x = np.clip(x, lo, hi)
    elif dist == "exponential":
        # shape = scale as fraction of range
        scale = max(1e-6, float(shape) * (hi - lo))
        x = lo + rng.exponential(scale=scale, size=n)
        x = np.clip(x, lo, hi)
    else:
        x = rng.uniform(lo, hi, size=n)
    return x.astype(np.float32)

def generate_design_from_template(template: pd.DataFrame,
                                 isi_dist: str,
                                 iti_dist: str,
                                 isi_lo: float, isi_hi: float,
                                 iti_lo: float, iti_hi: float,
                                 isi_shape: float, iti_shape: float,
                                 TR: float,
                                 trial_len_range: Tuple[float, float],
                                 mean_trial_len_range: Tuple[float, float],
                                 opt_isi_range: Optional[Tuple[float, float]] = None,
                                 opt_iti_range: Optional[Tuple[float, float]] = None,
                                 seed: int = 0) -> Optional[pd.DataFrame]:
    """
    Create a new timing-realisation while preserving trial order + non-timing columns.
    Enforces per-trial inter-img-onset duration bounds and mean trial length bounds.
    """
    rng = np.random.default_rng(int(seed))
    df = template.copy()

    # Optional hard constraints
    if opt_isi_range is not None:
        isi_lo = max(isi_lo, float(opt_isi_range[0]))
        isi_hi = min(isi_hi, float(opt_isi_range[1]))
    if opt_iti_range is not None:
        iti_lo = max(iti_lo, float(opt_iti_range[0]))
        iti_hi = min(iti_hi, float(opt_iti_range[1]))

    if isi_hi <= isi_lo or iti_hi <= iti_lo:
        return None

    img_dur = df["img_dur"].to_numpy(float)
    dec_dur = df["max_dec_dur"].to_numpy(float)
    fb_dur = df["fb_dur"].to_numpy(float)

    # Draw jitters
    isi1 = _draw_jitter(rng, isi_dist, df.shape[0], isi_lo, isi_hi, isi_shape)
    isi2 = _draw_jitter(rng, isi_dist, df.shape[0], isi_lo, isi_hi, isi_shape)
    iti = _draw_jitter(rng, iti_dist, df.shape[0], iti_lo, iti_hi, iti_shape)

    trial_len = img_dur + isi1 + dec_dur + isi2 + fb_dur + iti
    tmin, tmax = trial_len_range
    if np.any(trial_len < float(tmin)) or np.any(trial_len > float(tmax)):
        return None
    m = float(trial_len.mean())
    mmin, mmax = mean_trial_len_range
    if not (float(mmin) <= m <= float(mmax)):
        return None

    # Build onsets within each run
    img_on = np.zeros((df.shape[0],), dtype=np.float32)
    dec_on = np.zeros((df.shape[0],), dtype=np.float32)

    for rid in sorted(df["run_id"].unique()):
        idx = np.where(df["run_id"].to_numpy(int) == int(rid))[0]
        t = 0.0
        for k in idx:
            img_on[k] = t
            dec_on[k] = t + float(img_dur[k]) + float(isi1[k])
            t = t + float(img_dur[k]) + float(isi1[k]) + float(dec_dur[k]) + float(isi2[k]) + float(fb_dur[k]) + float(iti[k])

    # Assign
    df["iti"] = iti.astype(float)
    df["isi1_dur"] = isi1.astype(float)
    df["isi2_dur"] = isi2.astype(float)
    df["img_onset"] = img_on.astype(float)
    df["dec_onset_est"] = dec_on.astype(float)
    df["trial_duration_max"] = (img_dur + isi1 + dec_dur + isi2 + fb_dur).astype(float)
    return df


def evaluate_design_stage1(df: pd.DataFrame,
                           TR: float,
                           hp_cutoff: float,
                           pad_s: float,
                           tres: int,
                           weights: Tuple[float, float, float]) -> Dict[str, float]:
    """Compute trial-wise collinearity diagnostics (SPM-style HRF + DCT high-pass) + TR coverage.

    Collinearity metric (Option B):
      - Residualise (DCT for "raw"; partial-out other events + DCT for "part")
      - L2-normalise each trial regressor column
      - Compute Gram matrix G = X^T X
      - Summarise abs(off-diagonal): mean / p95 / max

    Also reports how many encoding columns are "nuked" by partialling (near-zero residual norm).
    """
    df = df.copy()
    run_ids = sorted(df["run_id"].unique().tolist())

    cov_stats_by_run: List[Dict[str, float]] = []
    per_run: List[Dict[str, Dict[str, float]]] = []

    # Debug: how many encoding columns become ~0 after partialling
    enc_part_nuked_n_by_run: List[int] = []
    enc_part_nuked_frac_by_run: List[float] = []

    w_enc, w_dec, w_fb = (float(weights[0]), float(weights[1]), float(weights[2]))


    HRF_TAIL_S = 32.0  # ensure enough scans to capture HRF tail
    def _gram_abs_stats_penalised(X: np.ndarray,
                                  eps_norm: float = 1e-10,
                                  nuked_frac_thresh: float = 0.10) -> Tuple[Dict[str, float], int, float]:
        """Compute Gram-matrix collinearity stats with *finite* penalties.

        We treat trialwise regressor columns as vectors and compute cosine similarity via
        G = Xn.T @ Xn after L2-normalising columns.

        If too many columns are effectively zero ("nuked"), or if anything becomes non-finite,
        we return a conservative penalty rather than NaN (to avoid the optimiser exploiting NaNs).
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape={X.shape}")

        n_cols = int(X.shape[1])
        if n_cols <= 1:
            # Not meaningful => penalise (not "perfect")
            return (
                {"mean_abs_r": float(COLL_PENALTY), "p95_abs_r": float(COLL_PENALTY_P95), "max_abs_r": float(COLL_PENALTY)},
                n_cols,
                1.0,
            )

        norms = np.linalg.norm(X, axis=0)
        nuked = norms < float(eps_norm)
        nuked_n = int(np.sum(nuked))
        nuked_frac = float(nuked_n) / float(max(n_cols, 1))

        # If partialling nukes a substantial fraction, treat as degenerate => penalise.
        if nuked_frac > float(nuked_frac_thresh):
            return (
                {"mean_abs_r": float(COLL_PENALTY), "p95_abs_r": float(COLL_PENALTY_P95), "max_abs_r": float(COLL_PENALTY)},
                nuked_n,
                nuked_frac,
            )

        safe_norms = norms.copy()
        safe_norms[safe_norms < float(eps_norm)] = 1.0
        Xn = X / safe_norms[None, :]

        G = Xn.T @ Xn
        np.fill_diagonal(G, 0.0)

        off = G[np.triu_indices(n_cols, k=1)]
        if off.size == 0 or (not np.all(np.isfinite(off))):
            return (
                {"mean_abs_r": float(COLL_PENALTY), "p95_abs_r": float(COLL_PENALTY_P95), "max_abs_r": float(COLL_PENALTY)},
                nuked_n,
                nuked_frac,
            )

        aoff = np.abs(off)
        stats = {
            "mean_abs_r": float(np.mean(aoff)),
            "p95_abs_r": float(np.percentile(aoff, 95)),
            "max_abs_r": float(np.max(aoff)),
        }

        if not all(np.isfinite(list(stats.values()))):
            stats = {"mean_abs_r": float(COLL_PENALTY), "p95_abs_r": float(COLL_PENALTY_P95), "max_abs_r": float(COLL_PENALTY)}

        return stats, nuked_n, nuked_frac

    for rid in run_ids:
        G = df[df["run_id"] == rid].copy()
        if G.empty:
            continue

        # --- Onsets/durations (trial count comes from design file) ---
        enc_on = G["img_onset"].to_numpy(float)
        enc_dur = G["img_dur"].to_numpy(float)

        dec_on = G["dec_onset_est"].to_numpy(float)
        dec_dur = G["max_dec_dur"].to_numpy(float)

        isi2 = G["isi2_dur"].to_numpy(float)
        fb_on = dec_on + dec_dur + isi2
        fb_dur = G["fb_dur"].to_numpy(float)

        # Run length / sampling
        run_end = float(np.max(fb_on + fb_dur)) + float(pad_s) + float(HRF_TAIL_S)
        n_scans = int(math.ceil(run_end / float(TR)))
        n_scans = max(n_scans, 2)  # hard safety: should not happen for valid designs, but prevents degenerate bases

        dt = float(TR) / float(tres)
        hrf = spm_hrf(dt=dt, length=32.0)

        t_scan = np.arange(n_scans, dtype=float) * float(TR)
        t_micro = np.arange(int(math.ceil(run_end / dt)), dtype=float) * dt

        # Trialwise (1 col per trial) per event
        X_enc = _build_trialwise_interp(t_micro, t_scan, enc_on, enc_dur, hrf) * w_enc
        X_dec = _build_trialwise_interp(t_micro, t_scan, dec_on, dec_dur, hrf) * w_dec
        X_fb  = _build_trialwise_interp(t_micro, t_scan, fb_on,  fb_dur,  hrf) * w_fb

        # Condition-level nuisance regressors for partialling
        x_enc = _build_condition_level_interp(t_micro, t_scan, enc_on, enc_dur, hrf) * w_enc
        x_dec = _build_condition_level_interp(t_micro, t_scan, dec_on, dec_dur, hrf) * w_dec
        x_fb  = _build_condition_level_interp(t_micro, t_scan, fb_on,  fb_dur,  hrf) * w_fb

        C_dct = dct_basis(n_scans, float(TR), float(hp_cutoff))

        # --- RAW stats (DCT residualise then Gram stats) ---
        X_enc_raw = _dct_resid(X_enc, C_dct)
        X_dec_raw = _dct_resid(X_dec, C_dct)
        X_fb_raw  = _dct_resid(X_fb,  C_dct)

        enc_raw, _ , _ = _gram_abs_stats_penalised(X_enc_raw)
        dec_raw, _ , _ = _gram_abs_stats_penalised(X_dec_raw)
        fb_raw,  _ , _ = _gram_abs_stats_penalised(X_fb_raw)

        # --- PARTIALLED stats (partial out other events' condition-level + DCT; then Gram stats) ---
        N_enc = np.column_stack([x_dec, x_fb, C_dct])
        N_dec = np.column_stack([x_enc, x_fb, C_dct])
        N_fb  = np.column_stack([x_enc, x_dec, C_dct])

        X_enc_part = _partial_out(X_enc, N_enc)
        X_dec_part = _partial_out(X_dec, N_dec)
        X_fb_part  = _partial_out(X_fb,  N_fb)

        enc_part, enc_nuked_n, enc_nuked_frac = _gram_abs_stats_penalised(X_enc_part)
        dec_part, dec_nuked_n, dec_nuked_frac = _gram_abs_stats_penalised(X_dec_part)
        fb_part, fb_nuked_n, fb_nuked_frac = _gram_abs_stats_penalised(X_fb_part)

        # Debug counters: encoding columns nuked by partialling
        n_trials = int(X_enc_part.shape[1])
        enc_part_nuked_n_by_run.append(enc_nuked_n)
        enc_part_nuked_frac_by_run.append(float(enc_nuked_frac))

        # --- ALL EVENTS (enc+dec+fb), DCT residualise then Gram stats ---
        X_all = np.column_stack([X_enc, X_dec, X_fb])
        X_all_r = _dct_resid(X_all, C_dct)
        all_stats, _ , _ = _gram_abs_stats_penalised(X_all_r)

        # TR coverage for encoding onsets (unchanged)
        phase = np.mod(enc_on, float(TR)) / float(TR)
        n_bins = 12
        counts, _ = np.histogram(phase, bins=np.linspace(0.0, 1.0, n_bins + 1))
        expected = float(len(phase)) / float(n_bins) if len(phase) else 1.0
        chi2 = float(np.sum((counts - expected) ** 2 / max(expected, 1e-12)))
        cov_stats_by_run.append({
            "cov_chi2": chi2,
            "cov_minbin": float(np.min(counts)) if counts.size else 0.0,
        })

        per_run.append({
            "enc_raw": enc_raw, "enc_part": enc_part,
            "dec_raw": dec_raw, "dec_part": dec_part,
            "fb_raw":  fb_raw,  "fb_part":  fb_part,
            "all": all_stats,
        })

    if not per_run:
        return {
            "coll_max": float("nan"), "coll_p95": float("nan"),
            "coll_max_worst": float("nan"), "coll_p95_worst": float("nan"),
            "cov_chi2": float("nan"), "cov_minbin": float("nan"),
            "cov_chi2_worst": float("nan"), "cov_minbin_worst": float("nan"),
            "enc_part_nuked_n_worst": float("nan"),
            "enc_part_nuked_frac_worst": float("nan"),
            "enc_part_nuked_frac_mean": float("nan"),
        }

    # Aggregate across runs: worst-run for max/p95, mean for mean_abs_r
    def _agg(prefix: str, key: str) -> Dict[str, float]:
        vals_max = [r[key]["max_abs_r"] for r in per_run]
        vals_p95 = [r[key]["p95_abs_r"] for r in per_run]
        vals_mean = [r[key]["mean_abs_r"] for r in per_run]
        return {
            f"{prefix}_max_worst": float(np.nanmax(vals_max)),
            f"{prefix}_p95_worst": float(np.nanmax(vals_p95)),
            f"{prefix}_mean": float(np.nanmean(vals_mean)),
        }

    out: Dict[str, float] = {}
    out.update(_agg("enc_raw", "enc_raw"))
    out.update(_agg("enc_part", "enc_part"))
    out.update(_agg("dec_raw", "dec_raw"))
    out.update(_agg("dec_part", "dec_part"))
    out.update(_agg("fb_raw",  "fb_raw"))
    out.update(_agg("fb_part",  "fb_part"))
    out.update(_agg("all", "all"))

    cov_chi2s = [c["cov_chi2"] for c in cov_stats_by_run]
    cov_mins = [c["cov_minbin"] for c in cov_stats_by_run]
    out["cov_chi2"] = float(np.nanmax(cov_chi2s))    # worst-run
    out["cov_minbin"] = float(np.nanmin(cov_mins))   # worst-run (lowest bin count)

    # Back-compat aliases for older objective code
    out["cov_chi2_worst"] = out["cov_chi2"]
    out["cov_minbin_worst"] = out["cov_minbin"]

    # Back-compat fields used by the optimiser/objective + progress bar
    out["coll_max"] = out["all_max_worst"]
    out["coll_p95"] = out["all_p95_worst"]
    out["coll_max_worst"] = out["coll_max"]
    out["coll_p95_worst"] = out["coll_p95"]

    # Back-compat: per-event
    out["coll_enc_max"] = out["enc_part_max_worst"]
    out["coll_enc_p95"] = out["enc_part_p95_worst"]
    out["coll_dec_max"] = out["dec_part_max_worst"]
    out["coll_dec_p95"] = out["dec_part_p95_worst"]
    out["coll_fb_max"]  = out["fb_part_max_worst"]
    out["coll_fb_p95"]  = out["fb_part_p95_worst"]

    # Debug outputs: encoding columns nuked by partialling
    out["enc_part_nuked_n_worst"] = float(np.max(enc_part_nuked_n_by_run)) if enc_part_nuked_n_by_run else float("nan")
    out["enc_part_nuked_frac_worst"] = float(np.max(enc_part_nuked_frac_by_run)) if enc_part_nuked_frac_by_run else float("nan")
    out["enc_part_nuked_frac_mean"] = float(np.mean(enc_part_nuked_frac_by_run)) if enc_part_nuked_frac_by_run else float("nan")
    # Extra back-compat aliases expected elsewhere (and shown in progress bar)
    out["coll_enc_max_worst"] = out["enc_part_max_worst"]
    out["coll_enc_p95_worst"] = out["enc_part_p95_worst"]
    out["coll_dec_max_worst"] = out["dec_part_max_worst"]
    out["coll_dec_p95_worst"] = out["dec_part_p95_worst"]
    out["coll_fb_max_worst"]  = out["fb_part_max_worst"]
    out["coll_fb_p95_worst"]  = out["fb_part_p95_worst"]

    # Final safety: never allow NaN/inf collinearity to look "optimal"
    for k in [
        "all_max_worst","all_p95_worst",
        "enc_part_max_worst","enc_part_p95_worst",
        "dec_part_max_worst","dec_part_p95_worst",
        "fb_part_max_worst","fb_part_p95_worst",
        "coll_max","coll_p95","coll_max_worst","coll_p95_worst",
        "coll_enc_max","coll_enc_p95","coll_dec_max","coll_dec_p95","coll_fb_max","coll_fb_p95",
        "coll_enc_max_worst","coll_enc_p95_worst","coll_dec_max_worst","coll_dec_p95_worst","coll_fb_max_worst","coll_fb_p95_worst",
    ]:
        v = out.get(k, float("nan"))
        if not np.isfinite(v):
            out[k] = float(COLL_PENALTY)

    for k in ["all_p95_worst","enc_part_p95_worst","dec_part_p95_worst","fb_part_p95_worst",
              "coll_p95","coll_p95_worst","coll_enc_p95","coll_enc_p95_worst",
              "coll_dec_p95","coll_dec_p95_worst","coll_fb_p95","coll_fb_p95_worst"]:
        v = out.get(k, float("nan"))
        if not np.isfinite(v):
            out[k] = float(COLL_PENALTY_P95)


    return out


def _run_stage2_from_stage1(out_dir: Path, stage1_df: "pd.DataFrame", args, topk: int, *, n_reps_stage2: int = 1) -> None:
    """Run Stage-2 (fmrisim recoverability) for the top-K designs from an existing stage1_results.csv DataFrame.

    This is implemented via subprocess calls back into this script (non-optimiser mode) to reuse the fmrisim
    simulation pipeline and avoid duplicating logic.
    """
    k = int(topk)
    if k <= 0:
        return
    if stage1_df is None or len(stage1_df) == 0:
        raise ValueError("stage1_results.csv is empty; nothing to run for stage-2.")
    if "csv" not in stage1_df.columns:
        raise ValueError("stage1_results.csv missing required column: 'csv'")

    # Preserve current ordering; if an explicit rank exists, respect it for determinism.
    if "rank_stage1" in stage1_df.columns:
        stage1_df = stage1_df.sort_values("rank_stage1", kind="mergesort").reset_index(drop=True)

    k = min(k, int(len(stage1_df)))
    print(f"[optimiser] Running recoverability (fmrisim) for top {k} designs...")

    import subprocess, sys as _sys

    stage2_rows = []
    for rank in range(k):
        csv_path = str(stage1_df.iloc[rank]["csv"])
        cand_out = out_dir / "stage2_fmrisim" / f"rank{rank:04d}"
        ensure_dir(cand_out)

        cmd = [
            _sys.executable, _sys.argv[0],
            "--csv", csv_path,
            "--patterns_npz", str(args.patterns_npz),
            "--noise_nii", str(args.noise_nii),
            "--TR", str(args.TR),
            "--hp_cutoff", str(args.hp_cutoff),
            "--pad_s", str(args.pad_s),
            "--dec_dur_s", str(args.dec_dur_s),
            "--temporal_resolution", str(args.temporal_resolution),
            "--loc_space", str(args.loc_space),
            "--roi_target_vox", str(args.roi_target_vox),
            "--n_vox", str(args.n_vox),
            "--out_dir", str(cand_out),
            "--n_reps", str(int(n_reps_stage2)),
        ]

        # Pass through noise caching options if provided
        if getattr(args, "cache_noise_dict", False):
            cmd.append("--cache_noise_dict")
        if getattr(args, "cache_run_noise", False):
            cmd.append("--cache_run_noise")
        if getattr(args, "noise_cache_seed", None) is not None:
            cmd.extend(["--noise_cache_seed", str(args.noise_cache_seed)])
        if getattr(args, "noise_cache_dir", None) is not None:
            cmd.extend(["--noise_cache_dir", str(args.noise_cache_dir)])

        subprocess.run(cmd, check=True)

        rec_path = Path(cand_out) / "recovery_summary.csv"
        if rec_path.exists():
            summ = pd.read_csv(rec_path)
            stage2_rows.append({
                "rank_stage1": rank,
                "lsa_mean": float(summ["lsa_mean"].mean()) if "lsa_mean" in summ.columns else float("nan"),
                "lss_mean": float(summ["lss_mean"].mean()) if "lss_mean" in summ.columns else float("nan"),
                "rsa_anchor_lsa_mean": float(summ["rsa_anchor_lsa_mean"].mean()) if "rsa_anchor_lsa_mean" in summ.columns else float("nan"),
            })
        else:
            stage2_rows.append({
                "rank_stage1": rank,
                "lsa_mean": float("nan"),
                "lss_mean": float("nan"),
                "rsa_anchor_lsa_mean": float("nan"),
            })

    pd.DataFrame(stage2_rows).to_csv(out_dir / "stage2_recoverability.csv", index=False)



def optimise_designs(args) -> None:
    """
    Evolutionary search over jitter distributions/ranges to optimise:
      - low collinearity (p95/max off-diag of X^T X)
      - good TR coverage for img_onset
    Then optionally evaluate recoverability for the top-K designs using the existing fmrisim pipeline.
    """
    template_csv = args.template_csv or args.csv
    template = pd.read_csv(template_csv)

    # Basic sanity: keep same n_trials/run, n_runs, and event lengths as template
    n_runs = int(template["run_id"].nunique())
    n_trials = int(template.shape[0])

    out_dir = Path(args.opt_out_dir)
    ensure_dir(out_dir)

    # Stage-2 only mode: reuse existing stage1_results.csv without rerunning Stage-1.
    if getattr(args, "stage2_only", False):
        if int(getattr(args, "stage2_topk", 0)) <= 0:
            raise ValueError("--stage2_only requires --stage2_topk > 0")
        stage1_csv = out_dir / "stage1_results.csv"
        if not stage1_csv.exists():
            raise FileNotFoundError(f"--stage2_only was set but stage1_results.csv was not found at: {stage1_csv}")
        stage1_df_existing = pd.read_csv(stage1_csv)
        if len(stage1_df_existing) == 0:
            raise ValueError(f"stage1_results.csv is empty: {stage1_csv}")
        _run_stage2_from_stage1(out_dir, stage1_df_existing, args, int(args.stage2_topk), n_reps_stage2=1)
        return


    # Bounds for search: default derived from template unless user overrides
    isi_min_obs = float(min(template["isi1_dur"].min(), template["isi2_dur"].min()))
    isi_max_obs = float(max(template["isi1_dur"].max(), template["isi2_dur"].max()))
    iti_min_obs = float(template["iti"].min())
    iti_max_obs = float(template["iti"].max())

    search_isi_lo = float(args.search_isi_min if args.search_isi_min is not None else max(0.0, isi_min_obs - 1.0))
    search_isi_hi = float(args.search_isi_max if args.search_isi_max is not None else (isi_max_obs + 1.0))
    search_iti_lo = float(args.search_iti_min if args.search_iti_min is not None else max(0.0, iti_min_obs - 1.0))
    search_iti_hi = float(args.search_iti_max if args.search_iti_max is not None else (iti_max_obs + 1.0))

    opt_isi_range = None
    if args.opt_isi_min is not None and args.opt_isi_max is not None:
        opt_isi_range = (float(args.opt_isi_min), float(args.opt_isi_max))
    opt_iti_range = None
    if args.opt_iti_min is not None and args.opt_iti_max is not None:
        opt_iti_range = (float(args.opt_iti_min), float(args.opt_iti_max))

    trial_len_range = (float(args.trial_len_min), float(args.trial_len_max))
    mean_trial_len_range = (float(args.mean_trial_len_min), float(args.mean_trial_len_max))

    weights = (float(args.w_enc), float(args.w_dec), float(args.w_fb))

    rng = np.random.default_rng(int(args.opt_seed))
    pop_size = int(args.pop_size)
    generations = int(args.generations)
    n_candidates = int(args.n_candidates)
    n_workers = int(args.opt_workers)

    dists = ["uniform", "truncnorm", "exponential"]

    def _rand_ind():
        isi_lo = rng.uniform(search_isi_lo, search_isi_hi)
        isi_hi = rng.uniform(isi_lo + 1e-3, search_isi_hi)
        iti_lo = rng.uniform(search_iti_lo, search_iti_hi)
        iti_hi = rng.uniform(iti_lo + 1e-3, search_iti_hi)
        isi_dist = rng.integers(0, len(dists))
        iti_dist = rng.integers(0, len(dists))
        isi_shape = rng.uniform(0.05, 0.5)  # fraction of range
        iti_shape = rng.uniform(0.05, 0.5)
        return np.array([isi_lo, isi_hi, iti_lo, iti_hi, float(isi_dist), float(iti_dist), isi_shape, iti_shape], dtype=float)

    def _clip_ind(x):
        x = x.copy()
        x[0] = float(np.clip(x[0], search_isi_lo, search_isi_hi))
        x[1] = float(np.clip(max(x[1], x[0] + 1e-3), search_isi_lo, search_isi_hi))
        x[2] = float(np.clip(x[2], search_iti_lo, search_iti_hi))
        x[3] = float(np.clip(max(x[3], x[2] + 1e-3), search_iti_lo, search_iti_hi))
        x[4] = float(int(np.clip(round(x[4]), 0, len(dists)-1)))
        x[5] = float(int(np.clip(round(x[5]), 0, len(dists)-1)))
        x[6] = float(np.clip(x[6], 0.01, 1.0))
        x[7] = float(np.clip(x[7], 0.01, 1.0))
        return x

    def _mutate(x, sigma=0.15):
        y = x.copy()
        # jitter bounds
        y[0:4] += rng.normal(0, sigma, size=4) * np.array([1.0, 1.0, 1.0, 1.0])
        # small chance to change dist
        if rng.random() < 0.1:
            y[4] = float(rng.integers(0, len(dists)))
        if rng.random() < 0.1:
            y[5] = float(rng.integers(0, len(dists)))
        # shapes
        y[6:8] += rng.normal(0, 0.05, size=2)
        return _clip_ind(y)

    def _crossover(a, b):
        t = rng.uniform(0.25, 0.75)
        return _clip_ind(t * a + (1 - t) * b)

    # Init population
    pop = [_rand_ind() for _ in range(pop_size)]
    evaluated = []

    from concurrent.futures import ProcessPoolExecutor, as_completed
    eval_seed_base = int(args.opt_seed) * 100000 + 17

    total_evals = 0
    gen_pbar = trange(generations, desc='Optimiser generations')
    for gen in gen_pbar:
        # Evaluate population
        jobs = []
        results = []

        # Windows-safe ProcessPool: worker must be top-level + picklable. We use an initializer
        # to avoid pickling large objects (template dataframe) per task.
        if n_workers <= 1:
            for i, ind in enumerate(pop):
                r = _optim_eval_one_impl(ind.tolist(), eval_seed_base + gen * 10000 + i)
                if r is not None:
                    results.append(r)
        else:
            initargs = (
                str(template_csv),
                float(args.TR),
                float(args.hp_cutoff),
                float(args.pad_s),
                int(args.opt_tres),
                weights,
                trial_len_range,
                mean_trial_len_range,
                opt_isi_range,
                opt_iti_range,
                dists,
                float(args.w_coll),
                float(args.w_cov),
            )
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_optim_pool_init, initargs=initargs) as ex:
                for i, ind in enumerate(pop):
                    jobs.append(ex.submit(_optim_eval_one_worker, (ind.tolist(), eval_seed_base + gen * 10000 + i)))
                for fut in tqdm(as_completed(jobs), total=len(jobs), desc=f'Gen {gen+1}/{generations} candidates', leave=False):
                    r = fut.result()
                    if r is not None:
                        results.append(r)
        total_evals += len(results)
        evaluated.extend(results)

        # Select elites
        evaluated.sort(key=lambda z: z[0])
        elites = evaluated[:max(2, pop_size // 4)]

        # Save best of this gen
        best_obj, best_s1, best_df, best_params = elites[0]
        # Update progress bar with best-so-far metrics
        try:
            gen_pbar.set_postfix(
                best_obj=f"{best_obj:.4g}",
                coll_p95=f"{best_s1.get('coll_p95_worst', float('nan')):.3g}",
                coll_max=f"{best_s1.get('coll_max_worst', float('nan')):.3g}",
                coll_enc_p95=f"{best_s1.get('coll_enc_p95_worst', float('nan')):.3g}",
                coll_enc_max=f"{best_s1.get('coll_enc_max_worst', float('nan')):.3g}",
                coll_dec_p95=f"{best_s1.get('coll_dec_p95_worst', float('nan')):.3g}",
                coll_dec_max=f"{best_s1.get('coll_dec_max_worst', float('nan')):.3g}",
                coll_fb_p95=f"{best_s1.get('coll_fb_p95_worst', float('nan')):.3g}",
                coll_fb_max=f"{best_s1.get('coll_fb_max_worst', float('nan')):.3g}",
                cov_chi2=f"{best_s1.get('cov_chi2_worst', float('nan')):.3g}",
                cov_minbin=int(best_s1.get('cov_minbin_worst', -1)),
            )
        except Exception:
            pass
        gen_dir = out_dir / "stage1_top"
        ensure_dir(gen_dir)
        best_path = gen_dir / f"best_gen{gen:03d}.csv"
        best_df.to_csv(best_path, index=False)

        # Produce next generation
        new_pop = [elites[i][3] for i in range(min(len(elites), 2))]  # placeholder to keep structure? (not used)
        # Actually keep elite individuals themselves
        elite_inds = [pop[0]]  # will be overridden below
        # Reconstruct elite inds from stored params not possible; keep from evaluated by storing ind too (simpler):
        # So instead keep best individuals from current pop by reevaluating order:
        # We'll keep top individuals from results mapping by obj, where we also stored params but not vector.
        # For simplicity: rebuild pop around best params + mutations.
        def _vec_from_params(p):
            return _clip_ind(np.array([p["isi_lo"], p["isi_hi"], p["iti_lo"], p["iti_hi"],
                                       float(dists.index(p["isi_dist"])), float(dists.index(p["iti_dist"])),
                                       p["isi_shape"], p["iti_shape"]], dtype=float))
        elite_vecs = [_vec_from_params(e[3]) for e in elites[:max(2, pop_size // 5)]]

        pop = elite_vecs[:]
        while len(pop) < pop_size:
            if rng.random() < 0.5 and len(elite_vecs) >= 2:
                a, b = rng.choice(elite_vecs, size=2, replace=False)
                child = _crossover(a, b)
            else:
                a = elite_vecs[int(rng.integers(0, len(elite_vecs)))]
                child = a.copy()
            child = _mutate(child, sigma=0.35)
            pop.append(child)

        # Stop if we've hit candidate budget
        if total_evals >= n_candidates:
            break

    # Final shortlist
    evaluated.sort(key=lambda z: z[0])
    top = evaluated[:int(args.keep_top)]
    rows = []
    designs_dir = out_dir / "designs"
    ensure_dir(designs_dir)
    for rank, (obj, s1, df_new, params) in enumerate(top):
        fname = f"cand_rank{rank:04d}.csv"
        fpath = designs_dir / fname
        df_new.to_csv(fpath, index=False)
        row = {"rank_stage1": rank, "stage1_obj": float(obj), "csv": str(fpath)}
        row.update(params)
        row.update(s1)
        rows.append(row)

    stage1_df = pd.DataFrame(rows)
    stage1_df.to_csv(out_dir / "stage1_results.csv", index=False)


    # Optional: run recoverability on top-K
    if int(args.stage2_topk) > 0:
        _run_stage2_from_stage1(out_dir, stage1_df, args, int(args.stage2_topk), n_reps_stage2=1)


# ------------------------
# GUI: Candidate design bank generator (Stage-1 ranking) + optional Stage-2
# ------------------------
def launch_candidate_design_gui() -> None:
    """Tk GUI to generate a large candidate bank (many variations x iterations) and rank via stage-1 metrics.

    This is intended as an *operator tool* (like create_runs.py) that:
      1) Creates a bank of candidate CSVs by sampling timing distributions/ranges around a template CSV.
      2) Runs stage-1 diagnostics (collinearity + TR coverage) and ranks designs.
      3) Writes a shortlist as cand_rankXXXX.csv to feed stage-2 fmrisim recoverability.

    Notes:
      - We preserve all non-timing columns from the template (ObjectSpace, condition, image_file, etc.).
      - We only support the 3-event CSV schema used by this script (img/dec/fb).
    """
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        raise RuntimeError("Tkinter is required for --gui_candidates. On Linux, install python3-tk.") from e

    class _ScrollableFrame(ttk.Frame):
        """Minimal scrollable container; add widgets to `self.inner`."""

        def __init__(self, parent, **kwargs):
            super().__init__(parent)

            self.canvas = tk.Canvas(self, highlightthickness=0)
            self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vbar.set)

            self.vbar.pack(side="right", fill="y")
            self.canvas.pack(side="left", fill="both", expand=True)

            self.inner = ttk.Frame(self.canvas, **kwargs)
            self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

            self.inner.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
            )
            self.canvas.bind("<Configure>", self._on_canvas_configure)

            # Mousewheel support (Win/mac)
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
            # Linux
            self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
            self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

        def _on_canvas_configure(self, event):
            self.canvas.itemconfigure(self._win_id, width=event.width)

        def _on_mousewheel(self, event):
            step = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(step, "units")

        def _on_mousewheel_linux(self, event):
            step = -1 if event.num == 4 else 1
            self.canvas.yview_scroll(step, "units")


    class CandidateGUI(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Design Optimiser – Candidate Bank + Ranking (Session 1)")
            self.geometry("1100x820")
            self._candidates = []  # list of dicts: {var_idx, iter_idx, params, df, csv_path}
            self._bank_counter = 0  # monotonically increasing design id
            self._stage2_request = False
            self._stage2_args = {}
            self._build()

        # --- parsing helpers ---
        def _get_int(self, v: tk.StringVar, default: int) -> int:
            try: return int(str(v.get()).strip())
            except Exception: return int(default)

        def _get_float(self, v: tk.StringVar, default: float) -> float:
            try: return float(str(v.get()).strip().replace(',', '.'))
            except Exception: return float(default)

        def _parse_minmax(self, s: str, name: str) -> tuple[float,float]:
            try:
                parts = [float(x.strip().replace(',', '.')) for x in str(s).split(',')]
                if len(parts) < 2:
                    raise ValueError
                lo, hi = float(parts[0]), float(parts[1])
                if hi <= lo:
                    raise ValueError
                return lo, hi
            except Exception:
                raise ValueError(f"{name} must be 'min,max' with max>min (e.g., 0.5,1.5)")

        # --- UI ---
        def _build(self):
            # Two tabs: Candidate bank + Stage 2
            root = ttk.Frame(self)
            root.pack(fill="both", expand=True)
            nb = ttk.Notebook(root)
            nb.pack(fill="both", expand=True)

            tab1 = ttk.Frame(nb)
            tab2 = ttk.Frame(nb)
            nb.add(tab1, text="Candidate bank")
            nb.add(tab2, text="Stage 2 (fmrisim)")

            # Scrollable content for tab1 (lots of options)
            sf = _ScrollableFrame(tab1, padding=10)
            sf.pack(fill="both", expand=True)
            main = sf.inner

            # Files
            fr = ttk.LabelFrame(main, text="Template + Output")
            fr.pack(fill="x", pady=6)

            self.v_template = tk.StringVar()
            ttk.Label(fr, text="Template CSV:").pack(side="left")
            ttk.Entry(fr, textvariable=self.v_template, width=60).pack(side="left", padx=6)
            ttk.Button(fr, text="Browse", command=self._browse_template).pack(side="left")

            self.v_out = tk.StringVar(value="design_optimisation/optim_out_gui")
            ttk.Label(fr, text="Output dir:").pack(side="left", padx=(16,0))
            ttk.Entry(fr, textvariable=self.v_out, width=35).pack(side="left", padx=6)

            # Candidate budget
            fr2 = ttk.LabelFrame(main, text="Candidate bank size")
            fr2.pack(fill="x", pady=6)

            self.v_nvar = tk.StringVar(value="200")
            self.v_niter = tk.StringVar(value="5")
            self.v_seed = tk.StringVar(value="0")
            ttk.Label(fr2, text="# Variations:").pack(side="left")
            ttk.Entry(fr2, textvariable=self.v_nvar, width=7).pack(side="left", padx=4)
            ttk.Label(fr2, text="# Iterations/variation:").pack(side="left", padx=(12,0))
            ttk.Entry(fr2, textvariable=self.v_niter, width=7).pack(side="left", padx=4)
            ttk.Label(fr2, text="Base seed:").pack(side="left", padx=(12,0))
            ttk.Entry(fr2, textvariable=self.v_seed, width=7).pack(side="left", padx=4)

            # Distributions + bounds
            fr3 = ttk.LabelFrame(main, text="Timing search space (applies to isi1 + isi2, and iti)")
            fr3.pack(fill="x", pady=6)

            dist_opts = ["uniform","truncnorm","exponential"]
            self.v_isi_dist = tk.StringVar(value="uniform")
            self.v_iti_dist = tk.StringVar(value="uniform")
            self.v_isi_bounds = tk.StringVar(value="0.5,1.5")
            self.v_iti_bounds = tk.StringVar(value="1.0,3.0")
            self.v_isi_span_min = tk.StringVar(value="0.4")  # min (hi-lo) per variation
            self.v_iti_span_min = tk.StringVar(value="0.8")
            self.v_isi_shape = tk.StringVar(value="0.25")
            self.v_iti_shape = tk.StringVar(value="0.25")

            row = ttk.Frame(fr3); row.pack(fill="x", pady=2)
            ttk.Label(row, text="ISI dist:").pack(side="left")
            ttk.OptionMenu(row, self.v_isi_dist, self.v_isi_dist.get(), *dist_opts).pack(side="left", padx=6)
            ttk.Label(row, text="ISI bounds (min,max):").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_isi_bounds, width=10).pack(side="left", padx=6)
            ttk.Label(row, text="min span:").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_isi_span_min, width=6).pack(side="left", padx=6)
            ttk.Label(row, text="shape:").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_isi_shape, width=6).pack(side="left", padx=6)

            row = ttk.Frame(fr3); row.pack(fill="x", pady=2)
            ttk.Label(row, text="ITI dist:").pack(side="left")
            ttk.OptionMenu(row, self.v_iti_dist, self.v_iti_dist.get(), *dist_opts).pack(side="left", padx=6)
            ttk.Label(row, text="ITI bounds (min,max):").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_iti_bounds, width=10).pack(side="left", padx=6)
            ttk.Label(row, text="min span:").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_iti_span_min, width=6).pack(side="left", padx=6)
            ttk.Label(row, text="shape:").pack(side="left", padx=(12,0))
            ttk.Entry(row, textvariable=self.v_iti_shape, width=6).pack(side="left", padx=6)

            # Stage-1 evaluation settings (kept minimal; advanced flags remain CLI)
            fr4 = ttk.LabelFrame(main, text="Stage-1 scoring settings")
            fr4.pack(fill="x", pady=6)

            self.v_TR = tk.StringVar(value="1.792")
            self.v_hp = tk.StringVar(value="128.0")
            self.v_pad = tk.StringVar(value="32.0")
            self.v_tres = tk.StringVar(value="16")
            self.v_wenc = tk.StringVar(value="0.50")
            self.v_wdec = tk.StringVar(value="0.25")
            self.v_wfb  = tk.StringVar(value="0.25")
            self.v_wcoll = tk.StringVar(value="1.0")
            self.v_wcov  = tk.StringVar(value="1.0")

            row = ttk.Frame(fr4); row.pack(fill="x", pady=2)
            for lab, var, w in [
                ("TR", self.v_TR, 6), ("HP cutoff", self.v_hp, 7), ("Pad (s)", self.v_pad, 6), ("tres", self.v_tres, 6),
                ("w_enc", self.v_wenc, 6), ("w_dec", self.v_wdec, 6), ("w_fb", self.v_wfb, 6),
                ("w_coll", self.v_wcoll, 6), ("w_cov", self.v_wcov, 6),
            ]:
                ttk.Label(row, text=f"{lab}:").pack(side="left")
                ttk.Entry(row, textvariable=var, width=w).pack(side="left", padx=6)

            fr5 = ttk.LabelFrame(main, text="Actions")
            fr5.pack(fill="x", pady=6)

            ttk.Button(fr5, text="Create candidate bank", command=self._create_bank).pack(side="left", padx=6)
            ttk.Button(fr5, text="Run stage-1 ranking", command=self._run_stage1).pack(side="left", padx=6)

            ttk.Label(fr5, text="Top-K shortlist:").pack(side="left", padx=(24,0))
            self.v_keep = tk.StringVar(value="100")
            ttk.Entry(fr5, textvariable=self.v_keep, width=6).pack(side="left", padx=6)
            
            # Log
            frlog = ttk.LabelFrame(main, text="Log")
            frlog.pack(fill="both", expand=True, pady=6)
            self.txt = tk.Text(frlog, height=18)
            self.txt.pack(fill="both", expand=True)
            
            self._log("Tip: Create -> Run. This writes <out_dir>/designs/ and stage1_results*.csv\n")
            
            # --- Stage 2 tab ---
            fr2 = ttk.LabelFrame(tab2, text="Stage 2: recoverability (fmrisim) on ranked shortlist")
            fr2.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Note: uses the same Output dir as tab 1 (must contain stage1_results.csv + designs/cand_rank*.csv)
            
            # Paths
            self.v_patterns = tk.StringVar()
            self.v_noise = tk.StringVar()
            
            row = ttk.Frame(fr2); row.pack(fill="x", pady=4)
            ttk.Label(row, text="patterns_npz:").pack(side="left")
            ttk.Entry(row, textvariable=self.v_patterns, width=70).pack(side="left", padx=6)
            ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.v_patterns, [("NPZ files", "*.npz"), ("All files", "*.*")])).pack(side="left")
            
            row = ttk.Frame(fr2); row.pack(fill="x", pady=4)
            ttk.Label(row, text="noise_nii:").pack(side="left")
            ttk.Entry(row, textvariable=self.v_noise, width=70).pack(side="left", padx=6)
            ttk.Button(row, text="Browse", command=lambda: self._browse_file(self.v_noise, [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")])).pack(side="left")
            
            # Core parameters
            self.v_stage2_topk = tk.StringVar(value="5")
            self.v_TR = tk.StringVar(value="1.792")
            self.v_hp_cutoff = tk.StringVar(value="128.0")
            self.v_pad_s = tk.StringVar(value="32.0")
            self.v_dec_dur_s = tk.StringVar(value="2.0")
            self.v_temporal_resolution = tk.StringVar(value="100.0")
            self.v_loc_space = tk.StringVar(value="frac")
            self.v_roi_target_vox = tk.StringVar(value="200")
            self.v_n_vox = tk.StringVar(value="189")
            self.v_n_reps = tk.StringVar(value="1")
            self.v_noise_cache_seed = tk.StringVar(value="0")
            
            grid = ttk.Frame(fr2); grid.pack(fill="x", pady=8)
            def _row(r, label, var, width=10):
                rr = ttk.Frame(grid); rr.pack(fill="x", pady=2)
                ttk.Label(rr, text=label, width=18).pack(side="left")
                ttk.Entry(rr, textvariable=var, width=width).pack(side="left")
                return rr
            
            _row(0, "Top-K:", self.v_stage2_topk, width=6)
            _row(1, "TR:", self.v_TR, width=8)
            _row(2, "hp_cutoff:", self.v_hp_cutoff, width=8)
            _row(3, "pad_s:", self.v_pad_s, width=8)
            _row(4, "dec_dur_s:", self.v_dec_dur_s, width=8)
            _row(5, "temporal_res:", self.v_temporal_resolution, width=8)
            
            rr = ttk.Frame(grid); rr.pack(fill="x", pady=2)
            ttk.Label(rr, text="loc_space:", width=18).pack(side="left")
            ttk.OptionMenu(rr, self.v_loc_space, self.v_loc_space.get(), "frac", "coords").pack(side="left")
            
            _row(7, "roi_target_vox:", self.v_roi_target_vox, width=8)
            _row(8, "n_vox:", self.v_n_vox, width=8)
            _row(9, "n_reps:", self.v_n_reps, width=6)
            _row(10, "noise_cache_seed:", self.v_noise_cache_seed, width=6)
            
            fr2b = ttk.Frame(fr2); fr2b.pack(fill="x", pady=(12, 0))
            ttk.Button(fr2b, text="Run Stage 2 in terminal (close GUI)", command=self._run_stage2_and_quit).pack(side="left")




        def _log(self, s: str):
            self.txt.insert("end", s + "\n")
            self.txt.see("end")
            self.update_idletasks()

        def _browse_file(self, var: tk.StringVar, filetypes):
            p = filedialog.askopenfilename(title="Select file", filetypes=filetypes)
            if p:
                var.set(p)

        def _browse_template(self):
            f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
            if f:
                self.v_template.set(f)

        def _create_bank(self):
            try:
                template_path = Path(self.v_template.get())
                if not template_path.exists():
                    messagebox.showerror("Error", "Template CSV not found")
                    return

                out_dir = Path(self.v_out.get())
                ensure_dir(out_dir)
                bank_root = ensure_dir(out_dir / "candidate_bank")
                designs_dir = ensure_dir(bank_root / "designs")
                meta_path = bank_root / "metadata.csv"
                # NOTE: we *append* to the existing candidate bank across button presses

                template = pd.read_csv(template_path)
                base_var = (max([c['var_idx'] for c in self._candidates]) + 1) if self._candidates else 0
                nvar = self._get_int(self.v_nvar, 200)
                niter = self._get_int(self.v_niter, 5)
                base_seed = self._get_int(self.v_seed, 0)

                isi_b_lo, isi_b_hi = self._parse_minmax(self.v_isi_bounds.get(), "ISI bounds")
                iti_b_lo, iti_b_hi = self._parse_minmax(self.v_iti_bounds.get(), "ITI bounds")
                isi_span_min = max(1e-6, self._get_float(self.v_isi_span_min, 0.4))
                iti_span_min = max(1e-6, self._get_float(self.v_iti_span_min, 0.8))

                isi_dist = str(self.v_isi_dist.get()).strip()
                iti_dist = str(self.v_iti_dist.get()).strip()
                isi_shape = float(self._get_float(self.v_isi_shape, 0.25))
                iti_shape = float(self._get_float(self.v_iti_shape, 0.25))

                TR = float(self._get_float(self.v_TR, 1.792))
                # preserve existing constraints from template-free defaults
                trial_len_range = (0.0, 1e9)
                mean_trial_len_range = (0.0, 1e9)

                rng = np.random.default_rng(int(base_seed))

                self._log(f"Creating bank: {nvar} variations x {niter} iterations = {nvar*niter} designs\n")
                wrote = 0
                manifest_rows = []
                for v in range(int(nvar)):
                    # Sample a parameter set (variation)
                    # Choose lo within [bound_lo, bound_hi - span_min], then hi within [lo+span_min, bound_hi]
                    if isi_b_hi - isi_b_lo <= isi_span_min or iti_b_hi - iti_b_lo <= iti_span_min:
                        raise ValueError("Bounds too tight for requested min span.")
                    isi_lo = float(rng.uniform(isi_b_lo, isi_b_hi - isi_span_min))
                    isi_hi = float(rng.uniform(isi_lo + isi_span_min, isi_b_hi))
                    iti_lo = float(rng.uniform(iti_b_lo, iti_b_hi - iti_span_min))
                    iti_hi = float(rng.uniform(iti_lo + iti_span_min, iti_b_hi))

                    params = dict(
                        isi_dist=isi_dist, iti_dist=iti_dist,
                        isi_lo=isi_lo, isi_hi=isi_hi,
                        iti_lo=iti_lo, iti_hi=iti_hi,
                        isi_shape=isi_shape, iti_shape=iti_shape,
                    )

                    for it in range(int(niter)):
                        seed = int(base_seed) + int(v)*100000 + int(it)
                        df_new = generate_design_from_template(
                            template=template,
                            isi_dist=isi_dist,
                            iti_dist=iti_dist,
                            isi_lo=isi_lo, isi_hi=isi_hi,
                            iti_lo=iti_lo, iti_hi=iti_hi,
                            isi_shape=isi_shape, iti_shape=iti_shape,
                            TR=float(TR),
                            trial_len_range=trial_len_range,
                            mean_trial_len_range=mean_trial_len_range,
                            opt_isi_range=None,
                            opt_iti_range=None,
                            seed=seed,
                        )
                        if df_new is None:
                            continue

                        var_idx = base_var + v
                        design_id = self._bank_counter
                        self._bank_counter += 1
                        fname = f"var{var_idx:04d}_iter{it:04d}_d{design_id:06d}.csv"
                        fpath = designs_dir / fname
                        df_new.to_csv(fpath, index=False)
                        wrote += 1
                        self._candidates.append({
                            "var_idx": v,
                            "iter_idx": it,
                            "seed": seed,
                            "params": params,
                            "csv": str(fpath),
                        })
                        manifest_rows.append({"var_idx": var_idx, "iter_idx": it, "seed": seed, "csv": str(fpath), **params})
                dfm = pd.DataFrame(manifest_rows)
                if meta_path.exists():
                    dfm.to_csv(meta_path, mode="a", header=False, index=False)
                else:
                    dfm.to_csv(meta_path, index=False)
                self._log(f"Wrote {wrote} candidate CSVs to: {designs_dir}\nManifest: {out_dir/'candidates_manifest.csv'}\n")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def _run_stage1(self):
            try:
                if not self._candidates:
                    messagebox.showerror("Error", "No candidates in memory. Click 'Create candidate bank' first.")
                    return

                out_dir = Path(self.v_out.get())
                ensure_dir(out_dir)
                keep_top = max(1, self._get_int(self.v_keep, 100))

                TR = float(self._get_float(self.v_TR, 1.792))
                hp = float(self._get_float(self.v_hp, 128.0))
                pad_s = float(self._get_float(self.v_pad, 32.0))
                tres = int(self._get_int(self.v_tres, 16))
                weights = (float(self._get_float(self.v_wenc, 0.5)),
                           float(self._get_float(self.v_wdec, 0.25)),
                           float(self._get_float(self.v_wfb, 0.25)))
                w_coll = float(self._get_float(self.v_wcoll, 1.0))
                w_cov = float(self._get_float(self.v_wcov, 1.0))

                self._log(f"Stage-1 evaluating {len(self._candidates)} designs...\n")

                # Evaluate each CSV
                per_iter = []
                for i, c in enumerate(self._candidates):
                    df = pd.read_csv(c["csv"])
                    s1 = evaluate_design_stage1(df, TR=TR, hp_cutoff=hp, pad_s=pad_s, tres=tres, weights=weights)
                    empty_pen = 0.0 if s1.get("cov_minbin_worst", 0.0) > 0 else 1000.0
                    obj = (w_coll * (s1["coll_p95_worst"] + 0.5 * s1["coll_max_worst"]) + w_cov * (s1["cov_chi2_worst"] + empty_pen))
                    per_iter.append({
                        **{k: c[k] for k in ["var_idx","iter_idx","seed","csv"]},
                        **c["params"],
                        **s1,
                        "stage1_obj": float(obj),
                    })
                    if (i+1) % 50 == 0:
                        self._log(f"  ...{i+1}/{len(self._candidates)} done")

                df_iter = pd.DataFrame(per_iter)
                df_iter.to_csv(out_dir / "stage1_results_all.csv", index=False)

                # Aggregate per variation: mean objective, plus worst-case sanity
                agg = df_iter.groupby("var_idx").agg(
                    n_iter=("stage1_obj","count"),
                    stage1_obj_mean=("stage1_obj","mean"),
                    stage1_obj_p95=("stage1_obj", lambda x: float(np.percentile(x, 95)) if len(x)>0 else float('nan')),
                    stage1_obj_max=("stage1_obj","max"),
                ).reset_index()

                # Choose representative CSV per variation: best iteration (min objective)
                best_rows = df_iter.sort_values(["var_idx","stage1_obj"]).groupby("var_idx").head(1)
                agg = agg.merge(best_rows[["var_idx","csv","iter_idx"]].rename(columns={"csv":"best_csv","iter_idx":"best_iter"}), on="var_idx", how="left")

                agg = agg.sort_values(["stage1_obj_mean","stage1_obj_p95","stage1_obj_max"]).reset_index(drop=True)
                agg.to_csv(out_dir / "stage1_results_variations.csv", index=False)

                # Write a cand_rankXXXX.csv shortlist to integrate with existing stage-2 flow
                designs_dir = ensure_dir(out_dir / "designs")
                shortlist = agg.head(int(keep_top))
                rows = []
                for rank, r in enumerate(shortlist.itertuples(index=False)):
                    src = Path(r.best_csv)
                    dst = designs_dir / f"cand_rank{rank:04d}.csv"
                    try:
                        # copy file contents (avoid shutil to keep minimal deps)
                        dst.write_text(Path(src).read_text(encoding='utf-8'), encoding='utf-8')
                    except Exception:
                        import shutil
                        shutil.copyfile(src, dst)

                    # keep a compact row for stage-2 selection
                    rows.append({
                        "rank_stage1": int(rank),
                        "var_idx": int(r.var_idx),
                        "best_iter": int(r.best_iter),
                        "stage1_obj_mean": float(r.stage1_obj_mean),
                        "stage1_obj_p95": float(r.stage1_obj_p95),
                        "stage1_obj_max": float(r.stage1_obj_max),
                        "csv": str(dst),
                    })

                pd.DataFrame(rows).to_csv(out_dir / "stage1_results.csv", index=False)

                self._log(f"\nWrote:\n  {out_dir/'stage1_results_all.csv'}\n  {out_dir/'stage1_results_variations.csv'}\n  {out_dir/'stage1_results.csv'} (shortlist)\n")
                self._log(f"Shortlist CSVs: {designs_dir}\n")
                self._log("Next: run stage-2 with CLI, e.g.\n  python fmrisim_optimizer.py --optimize_design --stage2_topk 10 --opt_out_dir <out_dir> ...\n")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        def _run_stage2_and_quit(self):
            """Close GUI and request stage-2 run in the terminal."""
            try:
                out_dir = str(self.v_out.get()).strip()
                if not out_dir:
                    messagebox.showerror("Error", "Output dir is empty")
                    return
                if not str(self.v_patterns.get()).strip():
                    messagebox.showerror("Error", "patterns_npz is required")
                    return
                if not str(self.v_noise.get()).strip():
                    messagebox.showerror("Error", "noise_nii is required")
                    return

                self._stage2_request = True
                self._stage2_args = {
                    "out_dir": out_dir,
                    "patterns_npz": str(self.v_patterns.get()).strip(),
                    "noise_nii": str(self.v_noise.get()).strip(),
                    "topk": int(float(self.v_stage2_topk.get())),
                    "TR": float(self.v_TR.get()),
                    "hp_cutoff": float(self.v_hp_cutoff.get()),
                    "pad_s": float(self.v_pad_s.get()),
                    "dec_dur_s": float(self.v_dec_dur_s.get()),
                    "temporal_resolution": float(self.v_temporal_resolution.get()),
                    "loc_space": str(self.v_loc_space.get()).strip(),
                    "roi_target_vox": int(float(self.v_roi_target_vox.get())),
                    "n_vox": int(float(self.v_n_vox.get())),
                    "n_reps": int(float(self.v_n_reps.get())),
                    "noise_cache_seed": int(float(self.v_noise_cache_seed.get())),
                }

                # Close GUI; caller will run stage-2 in terminal using subprocess.
                self.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e))




    gui = CandidateGUI()
    gui.mainloop()

    if getattr(gui, "_stage2_request", False):
        # Run in terminal using the same subprocess pipeline as optimiser stage-2
        cfg = getattr(gui, "_stage2_args", {})
        out_dir = Path(cfg["out_dir"])
        stage1_csv = out_dir / "stage1_results.csv"
        if not stage1_csv.exists():
            print(f"[gui-stage2] ERROR: stage1_results.csv not found at: {stage1_csv}")
            return

        df_top = pd.read_csv(stage1_csv)
        if len(df_top) == 0:
            print("[gui-stage2] ERROR: stage1_results.csv is empty; run stage-1 ranking first.")
            return

        k = min(int(cfg.get("topk", 5)), len(df_top))
        print(f"[gui-stage2] Running recoverability (fmrisim) for top {k} designs...")

        import subprocess, sys as _sys
        for rank in range(k):
            csv_path = str(df_top.iloc[rank]["csv"])
            cand_out = out_dir / "stage2_fmrisim" / f"rank{rank:04d}"
            ensure_dir(cand_out)

            cmd = [
                _sys.executable, _sys.argv[0],
                "--csv", csv_path,
                "--patterns_npz", str(cfg["patterns_npz"]),
                "--noise_nii", str(cfg["noise_nii"]),
                "--TR", str(cfg["TR"]),
                "--hp_cutoff", str(cfg["hp_cutoff"]),
                "--pad_s", str(cfg["pad_s"]),
                "--dec_dur_s", str(cfg["dec_dur_s"]),
                "--temporal_resolution", str(cfg["temporal_resolution"]),
                "--loc_space", str(cfg["loc_space"]),
                "--roi_target_vox", str(cfg["roi_target_vox"]),
                "--n_vox", str(cfg["n_vox"]),
                "--out_dir", str(cand_out),
                "--n_reps", str(cfg["n_reps"]),
                "--noise_cache_seed", str(cfg["noise_cache_seed"]),
            ]
            # Let the subprocess print its own progress to the terminal
            subprocess.run(cmd, check=True)

        print(f"[gui-stage2] Done. Outputs under: {out_dir / 'stage2_fmrisim'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui_candidates", action="store_true", help="Launch a Tk GUI to create a large candidate design bank and rank designs via stage-1 metrics.")
    ap.add_argument("--csv", required=False)
    ap.add_argument("--patterns_npz", required=False)
    ap.add_argument("--noise_nii", required=False, help="Real 4D NIfTI used to estimate noise + provide affine/shape.")
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

    # --- Design optimiser mode ---
    ap.add_argument("--optimize_design", action="store_true",
                    help="Run a design optimiser over ISI/ITI distributions/ranges to improve recoverability, collinearity, and TR coverage.")
    ap.add_argument("--template_csv", default=None,
                    help="Template 3-event CSV to preserve trial order/non-timing columns. Defaults to --csv.")
    ap.add_argument("--opt_out_dir", default="design_optimisation/optim_out",
                    help="Output directory for optimiser artefacts (stage1_results, shortlisted CSVs, etc).")
    ap.add_argument("--n_candidates", type=int, default=2000,
                    help="Total candidate evaluations budget (stage 1).")
    ap.add_argument("--keep_top", type=int, default=100,
                    help="Number of top designs to write as CSVs.")
    ap.add_argument("--pop_size", type=int, default=64)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--opt_workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    ap.add_argument("--opt_seed", type=int, default=0)
    ap.add_argument("--opt_tres", type=int, default=16,
                    help="Temporal oversampling factor for optimiser collinearity calculations.")

    ap.add_argument("--trial_len_min", type=float, required=False, default=0.0,
                    help="Min allowable trial length (img onset to next img onset), seconds.")
    ap.add_argument("--trial_len_max", type=float, required=False, default=1e9,
                    help="Max allowable trial length (img onset to next img onset), seconds.")
    ap.add_argument("--mean_trial_len_min", type=float, required=False, default=0.0,
                    help="Min allowable mean trial length across all trials, seconds.")
    ap.add_argument("--mean_trial_len_max", type=float, required=False, default=1e9,
                    help="Max allowable mean trial length across all trials, seconds.")

    ap.add_argument("--opt_isi_min", type=float, default=None, help="Optional hard constraint: minimum ISI value.")
    ap.add_argument("--opt_isi_max", type=float, default=None, help="Optional hard constraint: maximum ISI value.")
    ap.add_argument("--opt_iti_min", type=float, default=None, help="Optional hard constraint: minimum ITI value.")
    ap.add_argument("--opt_iti_max", type=float, default=None, help="Optional hard constraint: maximum ITI value.")

    ap.add_argument("--search_isi_min", type=float, default=None, help="Search-space lower bound for ISI ranges.")
    ap.add_argument("--search_isi_max", type=float, default=None, help="Search-space upper bound for ISI ranges.")
    ap.add_argument("--search_iti_min", type=float, default=None, help="Search-space lower bound for ITI ranges.")
    ap.add_argument("--search_iti_max", type=float, default=None, help="Search-space upper bound for ITI ranges.")

    ap.add_argument("--w_enc", type=float, default=0.50, help="Collinearity weight for encoding event.")
    ap.add_argument("--w_dec", type=float, default=0.25, help="Collinearity weight for decision event.")
    ap.add_argument("--w_fb", type=float, default=0.25, help="Collinearity weight for feedback event.")
    ap.add_argument("--w_coll", type=float, default=1.0, help="Overall objective weight on collinearity.")
    ap.add_argument("--w_cov", type=float, default=1.0, help="Overall objective weight on TR coverage.")
    ap.add_argument("--stage2_topk", type=int, default=0,
                    help="If >0, run fmrisim recoverability for the top-K designs via subprocess (slow).")
    ap.add_argument("--stage2_only", action="store_true",
                    help="Run ONLY stage-2 using an existing <opt_out_dir>/stage1_results.csv (skip stage-1). "
                         "Requires --stage2_topk > 0.")
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

    if getattr(args, "gui_candidates", False):
        launch_candidate_design_gui()
        return

    if getattr(args, "optimize_design", False):
        optimise_designs(args)
        return


    # In non-GUI, non-optimiser mode we need the core inputs.
    if not args.csv or not args.patterns_npz or not args.noise_nii:
        ap.error("--csv, --patterns_npz, and --noise_nii are required unless --gui_candidates or --optimize_design is used.")

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
        hp_cutoff=float(args.hp_cutoff),
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
        hp_cutoff=float(args.hp_cutoff)
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
                    hp_cutoff=float(args.hp_cutoff)
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
