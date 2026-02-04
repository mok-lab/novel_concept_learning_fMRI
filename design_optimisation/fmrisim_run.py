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
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine

from brainiak.utils import fmrisim


REQUIRED_COLS = {"run_id","trial_id","img_onset","img_dur","dec_onset_est","isi2_dur","fb_dur"}

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
    # Force stim to be 2D (time x conditions) for brainiak requirements
    if stim.ndim == 1:
        stim = stim.reshape(-1, 1)

    if isinstance(hrf_type, str):
        hrf_vector = get_double_gamma_hrf(temporal_resolution=tres)
    else:
        hrf_vector = hrf_type

    if hasattr(hrf_vector, 'flatten'):
        hrf_vector = hrf_vector.flatten()

    sig = fmrisim.convolve_hrf(
        stimfunction=stim,
        tr_duration=TR,
        temporal_resolution=tres,
        scale_function=False,
        # hrf_type=hrf_vector # Use standard canonical (default)
    )

    n_tp_run = sig.shape[0]
    t_high = np.arange(n_tp_run, dtype=float) / float(tres)
    t_tr = np.arange(n_scans, dtype=float) * float(TR)
    return np.interp(t_tr, t_high, sig[:, 0])


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

def apply_voxel_latency_shift(Y_sig: np.ndarray, TR: float, rng: np.random.Generator, sd_s: float) -> np.ndarray:
    if sd_s <= 0:
        return Y_sig
    T, V = Y_sig.shape
    t = np.arange(T) * TR
    out = np.empty_like(Y_sig)
    shifts = rng.normal(0.0, sd_s, size=V)
    for v in range(V):
        out[:, v] = np.interp(t, t - shifts[v], Y_sig[:, v], left=Y_sig[0, v], right=Y_sig[-1, v])
    return out


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
                     nuisance_model: str) -> dict:

    # 1. Build Design Matrices (Uses the fixed convolve_to_TR)
    X_enc, X_dec, X_fb = build_event_mats(df_run, TR, pad_s, dec_dur_s, tres, hrf_type_sim)
    n_scans, n_trials = X_enc.shape

    if P_enc.shape[0] != n_trials:
        raise ValueError(f"P_enc rows ({P_enc.shape[0]}) != n_trials ({n_trials}). Check ordering/slicing.")

    # 2. Create Event Patterns
    P_dec = make_noisy_event_patterns(P_enc, rng, decision_mix)
    P_fb  = make_noisy_event_patterns(P_enc, rng, feedback_mix)

    # 3. Generate Signal
    Y_sig = (X_enc @ P_enc) + (X_dec @ P_dec) + (X_fb @ P_fb)
    Y_sig = apply_voxel_latency_shift(Y_sig, TR, rng, hrf_latency_mismatch_sd)
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

    return {
        "Y": Y,
        "P_enc": P_enc,
        "Bhat_enc_lsa": Bhat_enc_lsa,
        "Bhat_enc_lss": Bhat_enc_lss,
        "rec_lsa": corr_rows(P_enc, Bhat_enc_lsa),
        "rec_lss": corr_rows(P_enc, Bhat_enc_lss),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--patterns_npz", required=True)
    ap.add_argument("--noise_nii", required=True, help="Real 4D NIfTI used to estimate noise + provide affine/shape.")

    ap.add_argument("--TR", type=float, required=True)
    ap.add_argument("--hp_cutoff", type=float, default=128.0)
    ap.add_argument("--pad_s", type=float, default=32.0)
    ap.add_argument("--dec_dur_s", type=float, default=2.0)
    ap.add_argument("--temporal_resolution", type=float, default=100.0)

    ap.add_argument("--n_vox", type=int, default=200)

    ap.add_argument("--decision_mix", type=float, default=0.55)
    ap.add_argument("--feedback_mix", type=float, default=0.80)

    ap.add_argument("--n_reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--hrf_latency_mismatch_sd", type=float, default=0.0)

    ap.add_argument("--nuisance_model", choices=["summed","trialwise"], default="summed")

    ap.add_argument("--loc_space", choices=["mni", "voxel", "frac"], default="frac",
                    help="Coordinate space for LOC center: MNI mm, voxel indices, or brain-mask fractions.")
    ap.add_argument("--loc_mni", type=float, nargs=3, default=[48.0, -84.0, -3.0],
                    help="LOC center in MNI mm (x y z). Used when --loc_space=mni.")
    ap.add_argument("--loc_vox", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="LOC center in voxel indices (i j k). Used when --loc_space=voxel.")
    ap.add_argument("--loc_vox_frac", type=float, nargs=3, default=[0.80, 0.15, 0.35],
                    help="LOC center as fractions of brain-mask bounds (x y z). Used when --loc_space=frac.")
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
    # Allow simulating fewer voxels than exist in patterns (e.g., if your sampled ROI has fewer voxels).
    # We deterministically take the first n_vox columns to keep alignment stable.
    if args.n_vox > P_all.shape[1]:
        raise ValueError(
            f"--n_vox={args.n_vox} but patterns_enc_200 has only {P_all.shape[1]} columns. "
            "Either regenerate patterns with at least n_vox columns, or reduce --n_vox."
        )
    if args.n_vox < P_all.shape[1]:
        print(f"[fmrisim_run] NOTE: patterns_enc_200 has {P_all.shape[1]} columns but --n_vox={args.n_vox}. "
              "Using the first --n_vox columns of the patterns for simulation.")
        P_all = P_all[:, :args.n_vox]
    if len(df) != P_all.shape[0]:
        raise ValueError(f"CSV has {len(df)} rows but patterns have {P_all.shape[0]} rows. "
                         "Regenerate patterns from the same CSV/order, or align explicitly.")

    # Load noise volume (also provides space/affine)
    noise_img = nib.load(str(args.noise_nii))
    vol4d = noise_img.get_fdata().astype(np.float32)
    aff = noise_img.affine
    shape3d = vol4d.shape[:3]
    # Noise model estimation and caching (doc-style pipeline)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.noise_cache_dir) if args.noise_cache_dir else (out_dir / "noise_cache")

    noise_dict, brain_mask, template = load_or_make_noise_dict(
        vol4d=vol4d,
        noise_nii_path=str(args.noise_nii),
        TR=args.TR,
        cache_dir=cache_dir,
        use_cache=bool(args.cache_noise_dict),
    )

    # Build LOC ROI mask in this space
    if args.loc_space == "mni":
        center_world = np.array(args.loc_mni, dtype=float)
        center_vox = apply_affine(np.linalg.inv(aff), center_world)
    elif args.loc_space == "voxel":
        center_vox = np.array(args.loc_vox, dtype=float)
        center_world = apply_affine(aff, center_vox)
    else:
        center_vox = estimate_loc_center_from_mask(brain_mask, tuple(args.loc_vox_frac))
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

    # Pre-generate / cache run-specific ROI noise (T x n_vox) so we don't regenerate noise every rep
    cached_run_noise: Dict = {}
    for rid in run_ids:
        df_run = df[df["run_id"] == rid].copy()
        n_scans_r = compute_n_scans_for_run(df_run, TR=args.TR, pad_s=args.pad_s, dec_dur_s=args.dec_dur_s)
        cached_run_noise[rid] = load_or_make_run_noise_2d_doc(
            run_id=rid,
            n_scans=n_scans_r,
            TR=args.TR,
            noise_dict=noise_dict,
            brain_mask=brain_mask,
            template=template,
            roi_mask=roi_mask,
            n_vox=args.n_vox,
            cache_dir=cache_dir,
            use_cache=bool(args.cache_run_noise),
            noise_cache_seed=int(args.noise_cache_seed),
        )

    rng_master = np.random.default_rng(args.seed)
    rows = []

    for rep in tqdm(range(args.n_reps), desc="Simulating Reps"):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for rid in run_ids:
            df_run = df[df["run_id"] == rid].copy()
            P_enc = P_all[run_slices[rid]].copy()

            res = simulate_one_run(
                df_run=df_run,
                P_enc=P_enc,
                TR=args.TR,
                hp_cutoff=args.hp_cutoff,
                pad_s=args.pad_s,
                dec_dur_s=args.dec_dur_s,
                tres=args.temporal_resolution,
                hrf_type_sim="double_gamma",
                noise_2d=cached_run_noise[rid],
                n_vox=args.n_vox,
                rng=rng,
                decision_mix=args.decision_mix,
                feedback_mix=args.feedback_mix,
                hrf_latency_mismatch_sd=args.hrf_latency_mismatch_sd,
                nuisance_model=args.nuisance_model
            )

            rows.append({
                "rep": rep,
                "run_id": rid,
                "loc_space": args.loc_space,
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
                "lsa_mean": float(res["rec_lsa"].mean()),
                "lsa_p05": float(np.percentile(res["rec_lsa"], 5)),
                "lss_mean": float(res["rec_lss"].mean()),
                "lss_p05": float(np.percentile(res["rec_lss"], 5)),
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

    print(f"LOC space: {args.loc_space}")
    print(f"LOC center (world mm): {tuple(center_world.tolist())}")
    print(f"LOC center (voxel): {tuple(center_vox.tolist())}")
    print(f"ROI voxels (mask): {n_roi} | radius_mm≈{radius_mm:.2f} | sampled n_vox={args.n_vox}")
    print(summary.groupby("run_id")[["lsa_mean","lsa_p05","lss_mean","lss_p05"]].mean().round(4))
    print(f"\nWrote: {out_dir / 'recovery_summary.csv'}")
    if args.write_roi_mask:
        print(f"Wrote: {out_dir / 'loc_roi_mask.nii.gz'}")


if __name__ == "__main__":
    main()