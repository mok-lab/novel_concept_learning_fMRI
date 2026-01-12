#!/usr/bin/env python3
"""
label_consistency.py

The Definitive Pipeline.
1. Metrics:
   - Inter-Model Agreement (Do models agree on the same image?)
   - Intra-Model Consistency (Is a single model stable across the group?)
   - Ensemble Consistency (Is the group average stable?)
   - Outlier Detection (Which model does not agree with other models?)
2. Slices: Top-20, Mid-20 (Mass), Bottom-20.
3. Visualization: Images + Transparent Gaussian + Metrics Sidebar.

This script:
 - Loads one or more pretrained torchvision models.
 - Runs them on a directory of images (optionally recursively).
 - Computes per-image model probabilities and simple ensemble statistics.
 - Produces group-level metrics that compare slice membership (top/mid/bot)
   across models, across images, and for the ensemble.
 - Optionally emits CSV rows and visualization plots per group.

Filename convention:
    {groupid}_{f1}_{f2}.ext
Where f1/f2 are normalized coordinates used for grouping/weighting.
"""

import argparse
import csv
import sys
import os
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import models
import numpy as np

# Visualisation: import when available; otherwise skip plotting features.
try:
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# ==========================================
# 1. METRICS LOGIC
# ==========================================

def get_slice_indices(probs: torch.Tensor, slice_type: str, k: int) -> Set[int]:
    """Return set of class indices belonging to the requested slice.

    slices:
      - "top": highest-k probability classes
      - "bot": lowest-k probability classes
      - "mid": k classes centered around probability mass median

    Args:
        probs: 1D tensor of class probabilities for a single image.
        slice_type: one of "top", "mid", "bot".
        k: desired slice size (will be clipped to number of classes).

    Returns:
        Set of class indices included in the slice.
    """
    numel = probs.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(probs, descending=True)

    if slice_type == "top":
        return set(sorted_indices[:k].tolist())
    elif slice_type == "bot":
        return set(sorted_indices[-k:].tolist())
    elif slice_type == "mid":
        # Mass-based median: find where cumulative sum crosses 0.5
        cumsum = torch.cumsum(sorted_vals, dim=0)
        mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
        center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
        start = max(0, center_idx - (k // 2))
        end = min(numel, start + k)
        if end - start < k: start = max(0, end - k)
        return set(sorted_indices[start:end].tolist())
    return set()

def calculate_set_similarity(sets: List[Set[int]]) -> float:
    """Compute average pairwise Jaccard similarity across a list of sets.

    Returns 1.0 for trivial case (len < 2) to indicate perfect agreement.
    """
    if len(sets) < 2: return 1.0
    scores = []
    for s1, s2 in itertools.combinations(sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        scores.append(intersection / union if union > 0 else 0.0)
    return sum(scores) / len(scores)

def detect_divergent_model(prob_list: List[torch.Tensor], model_names: List[str]) -> str:
    """Identify the model whose predictions are furthest from the group consensus.

    Uses Euclidean distance between each model's probability vector and the
    mean probability vector across models. Returns a string "ModelName (dist)".
    """
    if len(prob_list) < 2: return "None"
    stack = torch.stack(prob_list, dim=0)  # shape: [models, classes]
    consensus = stack.mean(dim=0)
    dists = torch.norm(stack - consensus, p=2, dim=1)  # distance per model
    max_dist, max_idx = torch.max(dists, dim=0)
    return f"{model_names[max_idx.item()]} ({max_dist.item():.2f})"

# ==========================================
# 2. HELPERS
# ==========================================

# Supported image extensions for discovery.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def discover_images(root: Path, recursive: bool) -> List[Path]:
    """Return a list of image file paths under `root`.

    If recursive is True, search all subdirectories; otherwise only top-level.
    """
    if recursive:
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def load_image(path: Path) -> Image.Image:
    """Open image and convert to RGB for model preprocessing."""
    return Image.open(path).convert("RGB")

def parse_filename_features(path: Path) -> Tuple[str, Optional[float], Optional[float]]:
    """Parse group id and optional normalized coordinates from filename.

    Expected format: {group}_{f1}_{f2}.ext
    Returns ("group", f1, f2) or (group, None, None) if parsing fails.
    """
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3:
        try:
            return parts[0], float(parts[1]), float(parts[2])
        except:
            return parts[0], None, None
    if len(parts) >= 1:
        return parts[0], None, None
    return "Unknown", None, None

def compute_weight(f1, f2, tf1, tf2, sigma, uniform):
    """Compute a per-item weight used when aggregating ensemble probabilities.

    - If uniform is True: items with parsed coordinates count as weight=1.
    - If coordinates are missing or sigma==0, fallback to binary matching.
    - Otherwise use Gaussian radial kernel exp(-dist^2 / 2 sigma^2).
    """
    if uniform: return 1.0 if (f1 is not None) else 0.0
    if f1 is None: return 0.0
    dist_sq = (f1 - tf1)**2 + (f2 - tf2)**2
    if sigma <= 0: return 1.0 if dist_sq == 0 else 0.0
    return math.exp(-dist_sq / (2 * sigma**2))

def format_k(values, indices, labels):
    """Format top/mid/bot results for printing and CSV output."""
    return [f"{labels[i] if 0<=i<len(labels) else str(i)} ({v:.4f})" for v, i in zip(values.tolist(), indices.tolist())]

# ==========================================
# 3. MODEL LOADING
# ==========================================

def resolve_model_and_weights(model_name: str):
    """Return (model, transform, categories) for a torchvision model name.

    Handles both newer `get_model` API and older constructors. If categories
    (human-readable labels) are unavailable, returns None for categories.
    """
    # Newer torchvision API
    if hasattr(models, "get_model"):
        try:
            w = models.get_model_weights(model_name).DEFAULT
            return models.get_model(model_name, weights=w), w.transforms(), w.meta.get("categories")
        except:
            pass
    # Fallback to classical constructors / Weights class
    ctor = getattr(models, model_name)
    if hasattr(ctor, "Weights"):
        w = ctor.Weights.DEFAULT
        return ctor(weights=w), w.transforms(), w.meta.get("categories")
    # Very old torchvision versions: return pretrained model and a default transform
    return ctor(pretrained=True), torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ]), None

# ==========================================
# 4. PLOTTING & STATS
# ==========================================

def get_stats_slices_mass_display(scores: torch.Tensor, k: int):
    """Convenience function to return (top, mid, bot) value/index tuples for display.

    This mirrors `get_slice_indices` but returns actual values and indices for
    use when printing the top-k labels for the ensemble distribution.
    """
    numel = scores.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(scores, descending=True)
    top_vals, top_idx = sorted_vals[:k], sorted_indices[:k]
    bot_vals, bot_idx = sorted_vals[-k:], sorted_indices[-k:]
    cumsum = torch.cumsum(sorted_vals, dim=0)
    mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
    center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
    start = max(0, center_idx - (k // 2))
    end = min(numel, start + k)
    if end - start < k: start = max(0, end - k)
    med_vals, med_idx = sorted_vals[start:end], sorted_indices[start:end]
    return (top_vals, top_idx), (med_vals, med_idx), (bot_vals, bot_idx)

def generate_group_plot(group_id, items, target_f1, target_f2, sigma, uniform,
                        top_res, med_res, bot_res, metrics_str, outlier_str, output_dir):
    """Create and save a per-group visualization showing image locations and heatmap.

    Visual includes:
      - Thumbnailed images placed at parsed (f1,f2) coordinates.
      - Optional Gaussian transparency map (if not uniform).
      - Sidebar with metrics and the top/mid/bot label summaries.
    """
    if not VIZ_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)

    t_str = "UNIFORM" if uniform else f"Tgt:({target_f1},{target_f2}) Sig:{sigma}"
    ax.set_title(f"Group {group_id} | {t_str}\nOutlier Model: {outlier_str}", fontsize=14)

    # Plot each image as a small thumbnail at its coordinate (if available).
    for item in items:
        if item['f1'] is None: continue
        try:
            img_obj = Image.open(item['path']).convert("RGB")
            # Thumbnail reduces memory footprint for many images
            img_obj.thumbnail((128, 128))
            im_box = OffsetImage(np.array(img_obj), zoom=0.6)
            ab = AnnotationBbox(
                im_box, (item['f1'], item['f2']),
                frameon=True, pad=0.2, bboxprops=dict(edgecolor='gray', alpha=0.5)
            )
            ax.add_artist(ab)
            ax.scatter(item['f1'], item['f2'], c='red', s=20, zorder=10)
        except:
            # Protect plotting from corrupt images or IO errors.
            pass

    # Draw Gaussian-like transparency overlay if not using uniform weights.
    if not uniform:
        x = np.linspace(-0.1, 1.1, 100); y = np.linspace(-0.1, 1.1, 100)
        X, Y = np.meshgrid(x, y)
        dist_sq = (X - target_f1)**2 + (Y - target_f2)**2
        Z = np.exp(-dist_sq / (2 * sigma**2))
        cmap = plt.get_cmap('viridis')
        rgba_img = cmap(Z); rgba_img[..., 3] = Z * 0.6  # alpha proportional to density
        ax.imshow(rgba_img, extent=[-0.1, 1.1, -0.1, 1.1], origin='lower', aspect='auto', zorder=20)

    # Prepare sidebar text box with compact metrics and top lists.
    def clean(l): return "\n".join(l[:5])
    txt = (
        f"METRICS (Jaccard 0-1):\n{metrics_str}\n\n{'='*30}\n\n"
        f"TOP 5:\n{clean(top_res)}\n\n{'-'*20}\n\n"
        f"MID 5:\n{clean(med_res)}\n\n{'-'*20}\n\n"
        f"BOT 5:\n{clean(bot_res)}"
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    # Position outside axes to the right
    ax.text(1.15, 1.0, txt, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.subplots_adjust(right=0.70)  # make room for the sidebar
    plt.savefig(os.path.join(output_dir, f"plot_group_{group_id}.png"), dpi=100)
    plt.close(fig)

# ==========================================
# 5. AGGREGATION LOGIC
# ==========================================

def aggregate_group_metrics(items, model_count, top_k):
    """Compute ensemble distribution, effective sample count, and metrics matrix.

    Returns:
      - agg_prob: aggregated (weighted) ensemble probability vector for the group
      - eff: effective sample count (sum_w^2 / sum_w^2)
      - metrics: dictionary with keys like 'top_model_agree', 'mid_model_const', ...
      - most_common_outlier: string model name most frequently flagged as outlier
    """
    # 1. Weighted Ensemble Probability: stack per-image ensemble probs and apply item weights.
    tensors = [x['ensemble_probs'] for x in items]
    weights = [x['weight'] for x in items]

    # Device-safe stacking
    stack = torch.stack(tensors, dim=0).to(tensors[0].device)
    w_tensor = torch.tensor(weights, device=stack.device, dtype=torch.float32).view(-1, 1)

    sum_w = w_tensor.sum()
    sum_sq_w = (w_tensor ** 2).sum()
    eff = (sum_w ** 2) / sum_sq_w if sum_sq_w > 0 else 0.0

    if sum_w == 0:
        # If all weights are zero (e.g., coordinates missing), fall back to uniform distribution
        agg_prob = torch.ones(stack.shape[1], device=stack.device) / stack.shape[1]
    else:
        # Weighted average across items
        agg_prob = (stack * (w_tensor / sum_w)).sum(dim=0)
    agg_prob = agg_prob / agg_prob.sum()  # ensure normalized

    # 2. Compute metrics across three slice types
    metrics = {}
    for sl in ['top', 'mid', 'bot']:
        # A. Inter-Model Agreement: for each image, compare slice membership across models,
        #    then average across images using item weights.
        img_agrees = []
        for item in items:
            sets = [get_slice_indices(p, sl, top_k) for p in item['individual_probs']]
            img_agrees.append(calculate_set_similarity(sets))
        metrics[f"{sl}_model_agree"] = np.average(img_agrees, weights=weights) if sum_w > 0 else 0.0

        # B. Intra-Model Consistency: for each model, collect its per-image slices and
        #    compute how consistent that model is across the group's images.
        mod_consts = []
        for m_idx in range(model_count):
            m_probs = [item['individual_probs'][m_idx] for item in items]
            m_sets = [get_slice_indices(p, sl, top_k) for p in m_probs]
            mod_consts.append(calculate_set_similarity(m_sets))
        metrics[f"{sl}_model_const"] = sum(mod_consts) / len(mod_consts)

        # C. Ensemble Consistency: how stable the ensemble's slice membership is across images.
        ens_probs = [item['ensemble_probs'] for item in items]
        ens_sets = [get_slice_indices(p, sl, top_k) for p in ens_probs]
        metrics[f"{sl}_ens_const"] = calculate_set_similarity(ens_sets)

    # 3. Determine most common outlier model across images in the group
    outlier_counts = {}
    for item in items:
        name = item['outlier_info'].split(' ')[0]
        outlier_counts[name] = outlier_counts.get(name, 0) + 1
    most_common_outlier = max(outlier_counts, key=outlier_counts.get) if outlier_counts else "None"

    return agg_prob, eff, metrics, most_common_outlier

# ==========================================
# 6. MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--target-f1", type=float, default=0.5)
    ap.add_argument("--target-f2", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--plot-dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--recursive", action="store_true")

    args = ap.parse_args()

    # Prepare plot output dir if necessary
    if args.plot_dir and VIZ_AVAILABLE and not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    # Discover images; exit with error if none found.
    img_paths = discover_images(Path(args.folder), args.recursive)
    if not img_paths:
        sys.exit("[ERROR] No images.")

    # Load models and preprocessing transforms
    loaded_models = []
    for m in args.models:
        print(f"[INFO] Loading {m}...")
        mod, pre, cats = resolve_model_and_weights(m)
        mod.to(args.device).eval()
        loaded_models.append({'name': m, 'model': mod, 'pre': pre, 'cats': cats})

    # Use categories from first model if available; otherwise fabricate class names.
    labels = loaded_models[0]['cats'] if loaded_models[0]['cats'] else [f"Class_{i}" for i in range(1000)]
    model_names = [m['name'] for m in loaded_models]

    # --- INFERENCE ---
    # results_db will hold per-image info keyed by stringified path
    results_db = {}
    with torch.no_grad():
        for m_idx, rec in enumerate(loaded_models):
            batch_t, batch_p = [], []

            # Flush processes current batch: run model, compute softmax, and populate DB.
            def flush():
                if not batch_t:
                    return
                t = torch.stack(batch_t).to(args.device)
                probs = F.softmax(rec['model'](t), dim=1).cpu()
                for i, p in enumerate(probs):
                    k = str(batch_p[i])
                    if k not in results_db:
                        # Parse group and coordinates once per image
                        g, f1, f2 = parse_filename_features(batch_p[i])
                        w = compute_weight(f1, f2, args.target_f1, args.target_f2, args.sigma, args.uniform)
                        results_db[k] = {
                            'group': g,
                            'f1': f1,
                            'f2': f2,
                            'weight': w,
                            'path': batch_p[i],
                            'individual_probs': [None] * len(loaded_models)
                        }
                    # Store this model's probability vector for the image
                    results_db[k]['individual_probs'][m_idx] = p
                batch_t[:], batch_p[:] = [], []

            # Build batches using the model's preprocessing transform
            for p in img_paths:
                batch_t.append(rec['pre'](load_image(p))); batch_p.append(p)
                if len(batch_t) >= args.batch_size:
                    flush()
            flush()

    # Pre-calc Ensemble probs and outlier info per image, and group items by group id.
    group_data = {}
    for key, data in results_db.items():
        # Skip images not processed by all models
        if any(x is None for x in data['individual_probs']):
            continue

        # Ensemble: simple mean across models, re-normalize
        stack = torch.stack(data['individual_probs'])
        ensemble_p = stack.mean(dim=0)
        data['ensemble_probs'] = ensemble_p / ensemble_p.sum()

        # Outlier detection per image
        data['outlier_info'] = detect_divergent_model(data['individual_probs'], model_names)

        gid = data['group']
        if gid not in group_data:
            group_data[gid] = []
        group_data[gid].append(data)

    # Sort groups numerically when possible, otherwise lexicographically
    sorted_groups = sorted(group_data.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    # Prepare CSV writer if requested
    if args.csv:
        f_csv = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        header = ["group", "count", "eff_samples", "outlier_model"]
        for s in ['top', 'mid', 'bot']:
            header.extend([f"{s}_model_agree", f"{s}_model_const", f"{s}_ens_const"])
        header.extend(["top_labels", "mid_labels", "bot_labels"])
        writer.writerow(header)

    print(f"\n{'='*30}\n FINAL ANALYSIS \n{'='*30}")

    # Iterate groups and compute final metrics, print and optionally save plots/CSV rows.
    for gid in sorted_groups:
        items = group_data[gid]
        # NOTE: aggregate_group_metrics expects (items, model_count, top_k)
        agg_prob, eff, metrics, outlier = aggregate_group_metrics(items, len(loaded_models), args.topk)

        # Get human-readable top/mid/bot label lists for the aggregated distribution
        (tv, ti), (mv, mi), (bv, bi) = get_stats_slices_mass_display(agg_prob, args.topk)
        top_str = format_k(tv, ti, labels)
        med_str = format_k(mv, mi, labels)
        bot_str = format_k(bv, bi, labels)

        # Print summary to stdout
        print(f"\nGroup: {gid} | Eff: {eff:.1f} | Outlier: {outlier}")
        print("  --- Top ---")
        print(f"  Model Agree: {metrics['top_model_agree']:.2f} | Indiv Const: {metrics['top_model_const']:.2f} | Ens Const: {metrics['top_ens_const']:.2f}")
        print("  --- Mid ---")
        print(f"  Model Agree: {metrics['mid_model_agree']:.2f} | Indiv Const: {metrics['mid_model_const']:.2f} | Ens Const: {metrics['mid_ens_const']:.2f}")

        plot_metrics_str = (
            f"TOP: Agr={metrics['top_model_agree']:.2f}, I-Const={metrics['top_model_const']:.2f}, E-Const={metrics['top_ens_const']:.2f}\n"
            f"MID: Agr={metrics['mid_model_agree']:.2f}, I-Const={metrics['mid_model_const']:.2f}, E-Const={metrics['mid_ens_const']:.2f}\n"
            f"BOT: Agr={metrics['bot_model_agree']:.2f}, I-Const={metrics['bot_model_const']:.2f}, E-Const={metrics['bot_ens_const']:.2f}"
        )

        # Save visualization if requested
        if args.plot_dir:
            generate_group_plot(
                gid, items, args.target_f1, args.target_f2, args.sigma, args.uniform,
                top_str, med_str, bot_str, plot_metrics_str, outlier, args.plot_dir
            )

        # Write CSV row if requested
        if args.csv:
            row = [gid, len(items), f"{eff:.2f}", outlier]
            for s in ['top', 'mid', 'bot']:
                row.extend([
                    f"{metrics[f'{s}_model_agree']:.4f}",
                    f"{metrics[f'{s}_model_const']:.4f}",
                    f"{metrics[f'{s}_ens_const']:.4f}"
                ])
            row.extend(["; ".join(top_str), "; ".join(med_str), "; ".join(bot_str)])
            writer.writerow(row)

    if args.csv:
        f_csv.close()

if __name__ == "__main__":
    main()