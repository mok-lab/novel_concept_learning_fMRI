#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_scrambled.py (drop-in replacement)

Paper-level controlled colour scrambles:

Default synth_mode: ycbcr_spectrum
- Convert source images to YCbCr (0..1 floats)
- Compute mean amplitude spectrum (rFFT) for Y, Cb, Cr (global or groupwise)
- Synthesize Y, Cb, Cr via mean amplitude + random phase (rFFT/irFFT) -> real fields
- Match mean/std per channel (dataset/group) (NOT per-image min-max)
- Convert back to RGB and save

Also supports synth_mode: rgb
- Same idea but on R, G, B channels.

Key behaviours retained:
- Parent dir contains category subfolders
- Excludes scrambled_name folder case-insensitively
- Uses ALL non-scrambled images for stats (always)
- Writes ONLY n_out outputs
- groupwise option: partition ALL images into n_out groups; one output per group
- group_balance: interleave across categories before grouping
- diagnostics: radial power (luminance), luminance + contrast distributions

Examples:
  # Global stats, paper-level controlled colour scrambles:
  python localiser_task/create_scrambled.py --parent_dir images/localiser_images \
    --scrambled_name scrambled --n_out 200 --size 512 --seed 0 --diag_plot

  # Groupwise: each output summarizes a subset of images:
  python localiser_task/create_scrambled.py --parent_dir images/localiser_images \
    --scrambled_name scrambled --n_out 200 --size 512 --seed 0 --diag_plot \
    --groupwise --group_balance

  # RGB mode:
  python localiser_task/create_scrambled.py --parent_dir images/localiser_images \
    --scrambled_name scrambled --n_out 200 --size 512 --seed 0 --diag_plot \
    --synth_mode rgb --rgb_phase_mode shared
"""

import os
import argparse
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------
# IO helpers
# -----------------------------

def list_category_dirs(parent_dir: str) -> List[str]:
    cats = []
    for name in sorted(os.listdir(parent_dir)):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            cats.append(name)
    return cats


def list_images(folder: str) -> List[str]:
    out = []
    for fn in sorted(os.listdir(folder)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            out.append(os.path.join(folder, fn))
    return out


def center_crop_square_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    img = img.resize((size, size), Image.BICUBIC)
    return img


def load_rgb_u8(path: str, size: int) -> np.ndarray:
    img = Image.open(path)
    img = center_crop_square_resize(img, size)
    return np.asarray(img, dtype=np.uint8)


def rgb_to_luminance01(arr_rgb_u8: np.ndarray) -> np.ndarray:
    """Return luminance in [0,1]."""
    x = arr_rgb_u8.astype(np.float32) / 255.0
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def rgb_u8_to_ycbcr_f32(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB uint8 to YCbCr float32 in 0..1."""
    im = Image.fromarray(rgb_u8, mode="RGB").convert("YCbCr")
    ycbcr = np.asarray(im, dtype=np.float32) / 255.0
    return ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]


def ycbcr_to_rgb_u8(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    ycbcr = np.stack([Y, Cb, Cr], axis=-1)
    ycbcr_u8 = np.clip(ycbcr * 255.0, 0, 255).round().astype(np.uint8)
    im = Image.fromarray(ycbcr_u8, mode="YCbCr").convert("RGB")
    return np.asarray(im, dtype=np.uint8)


# -----------------------------
# Balanced ordering + grouping
# -----------------------------

def interleave_categories(cat_to_imgs: Dict[str, List[str]], rng_py: random.Random) -> List[str]:
    """Round-robin interleave all category images (shuffled within each category)."""
    cats = sorted(cat_to_imgs.keys())
    per_cat = {c: cat_to_imgs[c][:] for c in cats}
    for c in cats:
        rng_py.shuffle(per_cat[c])

    idxs = {c: 0 for c in cats}
    out: List[str] = []
    remaining = True
    while remaining:
        remaining = False
        for c in cats:
            if idxs[c] < len(per_cat[c]):
                out.append(per_cat[c][idxs[c]])
                idxs[c] += 1
                remaining = True
    return out


def partition_into_groups(items: List[str], n_groups: int) -> List[List[str]]:
    """Split items into n_groups, spreading modulo items over early groups."""
    n = len(items)
    base = n // n_groups
    rem = n % n_groups
    groups = []
    start = 0
    for g in range(n_groups):
        k = base + (1 if g < rem else 0)
        groups.append(items[start:start + k])
        start += k
    return groups


# -----------------------------
# rFFT synthesis
# -----------------------------

def _synth_field_from_mean_amp(mean_amp: np.ndarray,
                               target_mean: float,
                               target_std: float,
                               size: int,
                               rng: np.random.Generator,
                               phase: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Synthesize one real-valued field in 0..1 from mean amplitude spectrum in rFFT space.
    """
    H = W = size
    if phase is None:
        phase = rng.uniform(-np.pi, np.pi, size=mean_amp.shape).astype(np.float32)
    else:
        phase = phase.astype(np.float32)

    phase[0, 0] = 0.0  # stabilize DC
    Fnew = mean_amp * np.exp(1j * phase)
    y = np.fft.irfft2(Fnew, s=(H, W)).astype(np.float32)

    # Match mean/std (avoid per-image min-max)
    y = (y - float(y.mean())) / (float(y.std()) + 1e-8)
    y = y * float(target_std) + float(target_mean)
    return np.clip(y, 0.0, 1.0)


def _mean_std(arr: np.ndarray, min_std: float = 1e-4) -> Tuple[float, float]:
    m = float(arr.mean())
    s = float(arr.std())
    return m, max(s, min_std)


# -----------------------------
# Stats computation
# -----------------------------

def compute_ycbcr_mean_amp_and_stats(image_paths: List[str], size: int):
    """
    Compute mean amplitude spectra + mean/std for Y, Cb, Cr using ALL (or group) images.
    Returns:
      mean_amp_Y, mean_amp_Cb, mean_amp_Cr,
      mean_Y, std_Y, mean_Cb, std_Cb, mean_Cr, std_Cr
    """
    if not image_paths:
        raise RuntimeError("No images for stats computation.")

    ampY = ampCb = ampCr = None
    Ys, Cbs, Crs = [], [], []

    for p in image_paths:
        rgb = load_rgb_u8(p, size=size)
        Y, Cb, Cr = rgb_u8_to_ycbcr_f32(rgb)

        Ys.append(Y.reshape(-1))
        Cbs.append(Cb.reshape(-1))
        Crs.append(Cr.reshape(-1))

        FY = np.fft.rfft2(Y);  AY = np.abs(FY).astype(np.float64)
        FCb = np.fft.rfft2(Cb); ACb = np.abs(FCb).astype(np.float64)
        FCr = np.fft.rfft2(Cr); ACr = np.abs(FCr).astype(np.float64)

        if ampY is None:
            ampY = np.zeros_like(AY, dtype=np.float64)
            ampCb = np.zeros_like(ACb, dtype=np.float64)
            ampCr = np.zeros_like(ACr, dtype=np.float64)

        ampY += AY
        ampCb += ACb
        ampCr += ACr

    mean_amp_Y = (ampY / len(image_paths)).astype(np.float32)
    mean_amp_Cb = (ampCb / len(image_paths)).astype(np.float32)
    mean_amp_Cr = (ampCr / len(image_paths)).astype(np.float32)

    Ys = np.concatenate(Ys); Cbs = np.concatenate(Cbs); Crs = np.concatenate(Crs)
    mean_Y, std_Y = _mean_std(Ys)
    mean_Cb, std_Cb = _mean_std(Cbs)
    mean_Cr, std_Cr = _mean_std(Crs)

    return mean_amp_Y, mean_amp_Cb, mean_amp_Cr, mean_Y, std_Y, mean_Cb, std_Cb, mean_Cr, std_Cr


def compute_rgb_mean_amp_and_stats(image_paths: List[str], size: int):
    """
    Compute mean amplitude spectra + mean/std per RGB channel.
    Returns mean_amp_R/G/B, means(3), stds(3)
    """
    if not image_paths:
        raise RuntimeError("No images for stats computation.")

    ampR = ampG = ampB = None
    px = []

    for p in image_paths:
        rgb_u8 = load_rgb_u8(p, size=size)
        x = rgb_u8.astype(np.float32) / 255.0
        px.append(x.reshape(-1, 3))

        FR = np.fft.rfft2(x[..., 0]); AR = np.abs(FR).astype(np.float64)
        FG = np.fft.rfft2(x[..., 1]); AG = np.abs(FG).astype(np.float64)
        FB = np.fft.rfft2(x[..., 2]); AB = np.abs(FB).astype(np.float64)

        if ampR is None:
            ampR = np.zeros_like(AR, dtype=np.float64)
            ampG = np.zeros_like(AG, dtype=np.float64)
            ampB = np.zeros_like(AB, dtype=np.float64)

        ampR += AR; ampG += AG; ampB += AB

    mean_amp_R = (ampR / len(image_paths)).astype(np.float32)
    mean_amp_G = (ampG / len(image_paths)).astype(np.float32)
    mean_amp_B = (ampB / len(image_paths)).astype(np.float32)

    px = np.vstack(px)
    means = px.mean(axis=0)
    stds = px.std(axis=0)
    stds = np.maximum(stds, 1e-4)

    return mean_amp_R, mean_amp_G, mean_amp_B, (float(means[0]), float(means[1]), float(means[2])), (float(stds[0]), float(stds[1]), float(stds[2]))


# -----------------------------
# Diagnostics
# -----------------------------

def radial_profile(power2d: np.ndarray) -> np.ndarray:
    H, W = power2d.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    rmax = r.max()
    out = np.zeros(rmax + 1, dtype=np.float64)
    for rad in range(rmax + 1):
        mask = (r == rad)
        out[rad] = power2d[mask].mean() if mask.any() else 0.0
    return out


def mean_radial_power(paths: List[str], size: int, rng_py: random.Random, n_samples: int, is_scramble: bool) -> np.ndarray:
    if not paths:
        return np.zeros(1, dtype=np.float64)

    sample = paths[:]
    rng_py.shuffle(sample)
    if n_samples > 0:
        sample = sample[:min(n_samples, len(sample))]

    acc = None
    for p in sample:
        if is_scramble:
            img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.uint8)
            lum = rgb_to_luminance01(arr)
        else:
            arr = load_rgb_u8(p, size=size)
            lum = rgb_to_luminance01(arr)

        F = np.fft.fft2(lum)
        P = np.abs(F) ** 2
        P = np.fft.fftshift(P)
        rp = radial_profile(P)

        if acc is None:
            acc = np.zeros_like(rp, dtype=np.float64)
        if rp.shape[0] > acc.shape[0]:
            acc = np.pad(acc, (0, rp.shape[0] - acc.shape[0]))
        elif rp.shape[0] < acc.shape[0]:
            rp = np.pad(rp, (0, acc.shape[0] - rp.shape[0]))
        acc += rp

    return acc / max(1, len(sample))


def luminance_contrast_samples(paths: List[str], size: int, rng_py: random.Random, n_samples: int, is_scramble: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not paths:
        return np.array([]), np.array([])

    sample = paths[:]
    rng_py.shuffle(sample)
    if n_samples > 0:
        sample = sample[:min(n_samples, len(sample))]

    mus, sigmas = [], []
    for p in sample:
        if is_scramble:
            img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.uint8)
            lum = rgb_to_luminance01(arr)
        else:
            arr = load_rgb_u8(p, size=size)
            lum = rgb_to_luminance01(arr)

        mus.append(float(lum.mean()))
        sigmas.append(float(lum.std()))
    return np.array(mus), np.array(sigmas)


def save_diagnostic_plots(out_dir: str, originals: List[str], scrambles: List[str], size: int, seed: int, diag_n_samples: int):
    import matplotlib.pyplot as plt

    rng_py = random.Random(seed + 999)

    rp_o = mean_radial_power(originals, size=size, rng_py=rng_py, n_samples=diag_n_samples, is_scramble=False)
    rp_s = mean_radial_power(scrambles, size=size, rng_py=rng_py, n_samples=diag_n_samples, is_scramble=True)

    rp_o_n = rp_o / (rp_o.max() + 1e-12)
    rp_s_n = rp_s / (rp_s.max() + 1e-12)

    plt.figure()
    plt.plot(rp_o_n, label="originals (mean radial power)")
    plt.plot(rp_s_n, label="outputs (mean radial power)")
    plt.xlabel("radial frequency (pixels from center)")
    plt.ylabel("normalised power")
    plt.title("Mean radial power spectrum (luminance)")
    plt.legend()
    ps_path = os.path.join(out_dir, "diagnostic_power_spectrum.png")
    plt.savefig(ps_path, dpi=150, bbox_inches="tight")
    plt.close()

    mu_o, sd_o = luminance_contrast_samples(originals, size=size, rng_py=rng_py, n_samples=diag_n_samples, is_scramble=False)
    mu_s, sd_s = luminance_contrast_samples(scrambles, size=size, rng_py=rng_py, n_samples=diag_n_samples, is_scramble=True)

    plt.figure()
    plt.hist(mu_o, bins=40, alpha=0.6, label="original mean luminance")
    plt.hist(mu_s, bins=40, alpha=0.6, label="output mean luminance")
    plt.xlabel("mean luminance (0..1)")
    plt.ylabel("count")
    plt.title("Mean luminance per image")
    plt.legend()
    lum_path = os.path.join(out_dir, "diagnostic_luminance.png")
    plt.savefig(lum_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(sd_o, bins=40, alpha=0.6, label="original RMS contrast")
    plt.hist(sd_s, bins=40, alpha=0.6, label="output RMS contrast")
    plt.xlabel("RMS contrast (std of luminance)")
    plt.ylabel("count")
    plt.title("RMS contrast per image")
    plt.legend()
    con_path = os.path.join(out_dir, "diagnostic_contrast.png")
    plt.savefig(con_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[diagnostics] wrote:\n  {ps_path}\n  {lum_path}\n  {con_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent_dir", required=True, help="Folder containing category subfolders.")
    ap.add_argument("--scrambled_name", default="scrambled", help="Name of scrambled output folder to exclude from stats.")
    ap.add_argument("--out_dir", default="", help="Output directory (default: parent_dir/scrambled_name).")

    ap.add_argument("--n_out", type=int, required=True, help="Number of scrambled images to write.")
    ap.add_argument("--size", type=int, default=512, help="Square size (default 512).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--groupwise", action="store_true",
                    help="Partition ALL images into n_out groups; make one output per group.")
    ap.add_argument("--group_balance", action="store_true",
                    help="When groupwise, interleave images across categories before grouping (recommended).")

    ap.add_argument("--synth_mode", choices=["ycbcr_spectrum", "rgb"], default="ycbcr_spectrum",
                    help="Synthesis mode: ycbcr_spectrum (paper-level chroma control) or rgb.")

    ap.add_argument("--ycbcr_phase_mode", choices=["shared", "independent"], default="independent",
                    help="YCbCr: shared phase across Y/Cb/Cr (reduces odd chroma edges) or independent (default).")

    ap.add_argument("--rgb_phase_mode", choices=["shared", "independent"], default="shared",
                    help="RGB: shared phase across channels (default) or independent.")

    ap.add_argument("--diag_plot", action="store_true", help="Write diagnostic plots.")
    ap.add_argument("--diag_n_samples", type=int, default=400, help="Max images used in diagnostics (0=all).")

    args = ap.parse_args()

    if args.n_out <= 0:
        raise ValueError("--n_out must be > 0")

    parent_dir = os.path.abspath(args.parent_dir)
    if not os.path.isdir(parent_dir):
        raise FileNotFoundError(parent_dir)

    scrambled_name_l = args.scrambled_name.strip().lower()
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(parent_dir, args.scrambled_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build sources per category excluding scrambled folder (case-insensitive)
    cats = list_category_dirs(parent_dir)
    cat_to_imgs: Dict[str, List[str]] = {}
    for c in cats:
        if c.strip().lower() == scrambled_name_l:
            continue
        imgs = list_images(os.path.join(parent_dir, c))
        if imgs:
            cat_to_imgs[c] = imgs

    if not cat_to_imgs:
        raise RuntimeError("No source images found (non-scrambled folders).")

    all_sources: List[str] = []
    for imgs in cat_to_imgs.values():
        all_sources.extend(imgs)

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    print(f"[sources] categories={len(cat_to_imgs)} n_all_sources={len(all_sources)} (used for stats always)")
    print(f"[out] n_out={args.n_out} size={args.size} out_dir={out_dir}")
    print(f"[mode] groupwise={args.groupwise} group_balance={args.group_balance} synth_mode={args.synth_mode}")

    scrambled_paths: List[str] = []

    def make_one_output(stat_paths: List[str], out_path: str):
        if args.synth_mode == "ycbcr_spectrum":
            mean_amp_Y, mean_amp_Cb, mean_amp_Cr, mean_Y, std_Y, mean_Cb, std_Cb, mean_Cr, std_Cr = \
                compute_ycbcr_mean_amp_and_stats(stat_paths, size=args.size)

            if args.ycbcr_phase_mode == "shared":
                shared_phase = rng_np.uniform(-np.pi, np.pi, size=mean_amp_Y.shape).astype(np.float32)
                Y  = _synth_field_from_mean_amp(mean_amp_Y,  mean_Y,  std_Y,  size=args.size, rng=rng_np, phase=shared_phase)
                Cb = _synth_field_from_mean_amp(mean_amp_Cb, mean_Cb, std_Cb, size=args.size, rng=rng_np, phase=shared_phase)
                Cr = _synth_field_from_mean_amp(mean_amp_Cr, mean_Cr, std_Cr, size=args.size, rng=rng_np, phase=shared_phase)
            else:
                Y  = _synth_field_from_mean_amp(mean_amp_Y,  mean_Y,  std_Y,  size=args.size, rng=rng_np, phase=None)
                Cb = _synth_field_from_mean_amp(mean_amp_Cb, mean_Cb, std_Cb, size=args.size, rng=rng_np, phase=None)
                Cr = _synth_field_from_mean_amp(mean_amp_Cr, mean_Cr, std_Cr, size=args.size, rng=rng_np, phase=None)

            out_rgb = ycbcr_to_rgb_u8(Y, Cb, Cr)
            Image.fromarray(out_rgb, mode="RGB").save(out_path, format="PNG", optimize=True)

        else:  # rgb
            mean_amp_R, mean_amp_G, mean_amp_B, means, stds = compute_rgb_mean_amp_and_stats(stat_paths, size=args.size)

            if args.rgb_phase_mode == "shared":
                shared_phase = rng_np.uniform(-np.pi, np.pi, size=mean_amp_R.shape).astype(np.float32)
                R = _synth_field_from_mean_amp(mean_amp_R, means[0], stds[0], size=args.size, rng=rng_np, phase=shared_phase)
                G = _synth_field_from_mean_amp(mean_amp_G, means[1], stds[1], size=args.size, rng=rng_np, phase=shared_phase)
                B = _synth_field_from_mean_amp(mean_amp_B, means[2], stds[2], size=args.size, rng=rng_np, phase=shared_phase)
            else:
                R = _synth_field_from_mean_amp(mean_amp_R, means[0], stds[0], size=args.size, rng=rng_np, phase=None)
                G = _synth_field_from_mean_amp(mean_amp_G, means[1], stds[1], size=args.size, rng=rng_np, phase=None)
                B = _synth_field_from_mean_amp(mean_amp_B, means[2], stds[2], size=args.size, rng=rng_np, phase=None)

            out = np.stack([R, G, B], axis=-1)
            out_u8 = np.clip(out * 255.0, 0, 255).round().astype(np.uint8)
            Image.fromarray(out_u8, mode="RGB").save(out_path, format="PNG", optimize=True)

    if not args.groupwise:
        # GLOBAL: stats from ALL images; write n_out outputs
        stat_paths_all = all_sources
        print("[stats] global: stats computed from ALL images.")
        for i in range(args.n_out):
            out_path = os.path.join(out_dir, f"scr_global_{args.synth_mode}_{i:06d}.png")
            make_one_output(stat_paths_all, out_path)
            scrambled_paths.append(out_path)
            if (i + 1) % 100 == 0 or (i + 1) == args.n_out:
                print(f"  wrote {i+1}/{args.n_out}")

    else:
        # GROUPWISE: partition ALL images into n_out groups; each output uses its group's stats
        if args.group_balance:
            ordered = interleave_categories(cat_to_imgs, rng_py=rng_py)
        else:
            ordered = all_sources[:]
            rng_py.shuffle(ordered)

        groups = partition_into_groups(ordered, n_groups=args.n_out)
        sizes = [len(g) for g in groups]
        print(f"[stats] groupwise: {len(ordered)} images -> {len(groups)} groups "
              f"(min/mean/max={min(sizes)}/{(len(ordered)/len(groups)):.2f}/{max(sizes)})")

        for gi, grp in enumerate(groups):
            stat_paths = grp if grp else all_sources
            out_path = os.path.join(out_dir, f"scr_group_{args.synth_mode}_{gi:06d}_n{len(grp)}.png")
            make_one_output(stat_paths, out_path)
            scrambled_paths.append(out_path)
            if (gi + 1) % 100 == 0 or (gi + 1) == len(groups):
                print(f"  wrote {gi+1}/{len(groups)}")

    if args.diag_plot:
        save_diagnostic_plots(
            out_dir=out_dir,
            originals=all_sources,
            scrambles=scrambled_paths,
            size=args.size,
            seed=args.seed,
            diag_n_samples=args.diag_n_samples
        )

    print("[done]")


if __name__ == "__main__":
    main()
