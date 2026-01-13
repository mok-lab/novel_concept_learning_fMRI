#!/usr/bin/env python3
"""
infer_final_robust_outlier.py

Final analysis pipeline producing per-model and ensemble diagnostics
for grouped image sets. The module:

 - Runs model inference (probability vectors) and computes a weighted
   ensemble probability per image.
 - Computes three complementary metric classes to summarise behaviour
   across a group of images:
     1) Inter-Model Agreement
     2) Intra-Model Consistency (per-model)
     3) Ensemble Consistency
 - Produces per-model outlier scores (L2 distance from the consensus)
   and visualisations that place images in feature-space with metric sidebars.
   
Metric descriptions and interpretation:
 - Inter-Model Agreement
     Computed as the average pairwise Jaccard similarity of label "slices"
     (Top / Mid / Bottom) between all pairs of models, measured per image
     and averaged across the group. Range [0,1]. Higher values indicate
     that models tend to nominate the same sets of labels for a given image
     (strong agreement). Low values indicate disagreement or diverging top
     label sets between models.

 - Intra-Model Consistency
     For each individual model we compute the average pairwise Jaccard
     similarity of that model's label "slices" across the images within
     the group. This measures whether a model is *stable* in the particular
     slice across images (e.g., consistently returning a similar set of
     top labels). Range [0,1]. A high intra-model consistency means the
     model tends to produce similar slice-sets across the group's images;
     a low value indicates the model's nominated labels vary a lot across
     the group (potentially reflecting sensitivity to image differences).

 - Ensemble Consistency
     Same calculation as intra-model consistency but applied to the
     ensemble (mean probability vector). This measures whether the ensemble
     focuses on the same label sets across the group's images. High
     ensemble consistency indicates the ensemble produces a stable focus;
     low values indicate the ensemble's top/mid/bottom label sets change
     substantially across images.

 - Outlier Scores (Per-model)
     For every image the Euclidean (L2) distance between each model's
     probability vector and the consensus (mean) probability is computed.
     These distances are then averaged across images (weighted by sample
     weights) to give a per-model outlier score. Larger scores indicate
     a model is systematically farther from the group consensus.

Usage summary:
 - Run on a folder of images (optionally recursive) with a list of model
   names. Produces optional PNG visualisations and CSV summarising metrics.
"""
import argparse
import csv
import sys
import re
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
from tqdm import tqdm

# Visualization
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
    """
    Select a set of label indices corresponding to the requested 'slice'.

    Parameters
    - probs: 1D tensor of class probabilities for a single prediction.
    - slice_type: 'top' | 'mid' | 'bot'
      * 'top' -> indices of the top-k classes (highest probabilities).
      * 'bot' -> indices of the bottom-k classes (lowest probabilities).
      * 'mid' -> a mass-based median window: locate the index where the
                 cumulative sorted probability mass crosses 0.5, then take
                 a symmetric window (5+5) around that center. This is a
                 fixed-width "middle" slice used for complementary analysis.
    - k: requested number for top/bot. For 'mid' this parameter is ignored
         in favour of the fixed 10-element window (5+5) implemented here.

    Returns
    - set of selected label indices (empty set if input invalid).

    Notes:
    - Returned indices are canonical label indices into the probability vector.
    - The 'mid' strategy is intended to capture labels in the region around
      the distribution's median mass (useful when the mass is spread).
    """
    numel = probs.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(probs, descending=True)
    
    if slice_type == "top":
        return set(sorted_indices[:k].tolist())
    elif slice_type == "bot":
        return set(sorted_indices[-k:].tolist())
    elif slice_type == "mid":
        # Mass-based Median: find first index where cumulative mass > 0.5
        cumsum = torch.cumsum(sorted_vals, dim=0)
        mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
        center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
        
        # Modified: 5+5 labels around the median (Radius 5).
        # This enforces a fixed window size of up to 10 centered on median.
        start = max(0, center_idx - 5)
        end = min(numel, center_idx + 5)
        
        return set(sorted_indices[start:end].tolist())
    return set()

def calculate_set_similarity(sets: List[Set[int]]) -> float:
    """
    Average pairwise Jaccard similarity across a list of sets.

    Interpretation:
      - Returns a scalar in [0,1].
      - 1.0 => all sets are identical (perfect overlap).
      - 0.0 => no overlap between any pairs.
      - Used as a compact measure of agreement between nominative label
        sets (top/mid/bot) either across models (inter-model) or across
        images for a single model (intra-model).

    Implementation details:
      - Computes pairwise Jaccard (intersection / union) for all combinations
        and returns the mean. If fewer than two sets are provided returns 1.0
        (trivially identical).
    """
    if len(sets) < 2: return 1.0
    scores = []
    for s1, s2 in itertools.combinations(sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        scores.append(intersection / union if union > 0 else 0.0)
    return sum(scores) / len(scores)

def calculate_outlier_scores(prob_list: List[torch.Tensor], model_names: List[str]) -> Dict[str, float]:
    """
    Compute per-model L2 distance from the consensus (mean) probability vector.

    Parameters:
      - prob_list: list of 1D probability tensors (one per model) for a single image.
      - model_names: list of model names corresponding to the probability tensors.

    Returns:
      - dict mapping model name -> L2 distance to consensus.

    Interpretation:
      - Larger distance => model's prediction distribution deviates more from the
        group mean for that image (possible 'outlier' behaviour on that image).
      - These per-image distances can be averaged across images to estimate
        systematic deviation of a model relative to peers.
    """
    if len(prob_list) < 2: 
        return {name: 0.0 for name in model_names}
    
    stack = torch.stack(prob_list, dim=0) # [Models, Classes]
    consensus = stack.mean(dim=0)         # [Classes]
    
    # Calculate L2 distance for each model row against the consensus row
    dists = torch.norm(stack - consensus, p=2, dim=1) # [Models]
    
    scores = {}
    for i, name in enumerate(model_names):
        scores[name] = dists[i].item()
        
    return scores

# ==========================================
# 2. HELPERS
# ==========================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def discover_images(root: Path, recursive: bool) -> List[Path]:
    if recursive: return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def parse_filename_features(path: Path) -> Tuple[str, Optional[float], Optional[float]]:
    stem = path.stem
    # Standard underscore split
    parts = stem.split('_')
    if len(parts) >= 3:
        try: return parts[0], float(parts[1]), float(parts[2])
        except: pass
    
    # Handle alternate naming convention "object1F0Level0.5F1Level0.8"
    f0_match = re.search(r'F0Level([-+]?\d*\.\d+|\d+)', stem)
    f1_match = re.search(r'F1Level([-+]?\d*\.\d+|\d+)', stem)
    
    obj_name = path.parent.name if path.parent.name else "Unknown"
    f0 = float(f0_match.group(1)) if f0_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    
    return obj_name, f0, f1

def compute_weight(f1, f2, tf1, tf2, sigma, uniform):
    """
    Compute a sample weight used to aggregate ensemble metrics.

    - If `uniform` is True, every image with valid features receives weight 1.
    - Otherwise returns a Gaussian-like weight based on distance between (f1,f2)
      and target (tf1,tf2) with bandwidth sigma.
    - If sigma <= 0 the weight is binary (1 if exact match, 0 otherwise).
    """
    if uniform: return 1.0 if (f1 is not None) else 0.0
    if f1 is None: return 0.0
    dist_sq = (f1 - tf1)**2 + (f2 - tf2)**2
    if sigma <= 0: return 1.0 if dist_sq == 0 else 0.0
    return math.exp(-dist_sq / (2 * sigma**2))

def format_k(values, indices, labels):
    return [f"{labels[i] if 0<=i<len(labels) else str(i)} ({v:.4f})" for v, i in zip(values.tolist(), indices.tolist())]

# ==========================================
# 3. MODEL LOADING
# ==========================================

def resolve_model_and_weights(model_name: str):
    """
    Resolve a torchvision model constructor and its default preprocessing transforms.
    Handles newer `get_model` API and older constructors with `Weights` attribute.
    """
    if hasattr(models, "get_model"):
        try:
            w = models.get_model_weights(model_name).DEFAULT
            return models.get_model(model_name, weights=w), w.transforms(), w.meta.get("categories")
        except: pass
    ctor = getattr(models, model_name)
    if hasattr(ctor, "Weights"):
        w = ctor.Weights.DEFAULT
        return ctor(weights=w), w.transforms(), w.meta.get("categories")
    return ctor(pretrained=True), torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), None

# ==========================================
# 4. PLOTTING & STATS
# ==========================================

def get_stats_slices_mass_display(scores: torch.Tensor, k: int):
    """
    Helper to return top/mid/bot slices from an aggregated probability vector.

    Returns:
      - (top_vals, top_idx), (med_vals, med_idx), (bot_vals, bot_idx)
    where each idx is an index tensor identifying the nominated labels.
    """
    numel = scores.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(scores, descending=True)
    top_vals, top_idx = sorted_vals[:k], sorted_indices[:k]
    bot_vals, bot_idx = sorted_vals[-k:], sorted_indices[-k:]
    
    # Mid Logic: mass-based median window
    cumsum = torch.cumsum(sorted_vals, dim=0)
    mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
    center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
    
    # Modified: 5+5 labels around the median (Radius 5)
    start = max(0, center_idx - 5)
    end = min(numel, center_idx + 5)
    
    med_vals, med_idx = sorted_vals[start:end], sorted_indices[start:end]
    return (top_vals, top_idx), (med_vals, med_idx), (bot_vals, bot_idx)

def generate_group_plot(group_id, items, target_f1, target_f2, sigma, uniform,
                        top_res, med_res, bot_res, metrics_str, outlier_str, output_dir):
    """
    Optional plotting utility that lays out item thumbnails in feature
    coordinates (f1,f2) and displays the computed metric summary in a
    sidebar for quick manual inspection.
    """
    if not VIZ_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    
    t_str = "UNIFORM" if uniform else f"Tgt:({target_f1},{target_f2}) Sig:{sigma}"
    ax.set_title(f"Group {group_id} | {t_str}\n{outlier_str}", fontsize=14)
    
    for item in items:
        if item['f1'] is None: continue
        try:
            img_obj = Image.open(item['path']).convert("RGB")
            img_obj.thumbnail((128, 128)) 
            im_box = OffsetImage(np.array(img_obj), zoom=0.6)
            ab = AnnotationBbox(im_box, (item['f1'], item['f2']), frameon=True, pad=0.2, bboxprops=dict(edgecolor='gray', alpha=0.5))
            ax.add_artist(ab)
            ax.scatter(item['f1'], item['f2'], c='red', s=20, zorder=10)
        except: pass

    if not uniform:
        x = np.linspace(-0.1, 1.1, 100); y = np.linspace(-0.1, 1.1, 100)
        X, Y = np.meshgrid(x, y)
        dist_sq = (X - target_f1)**2 + (Y - target_f2)**2
        Z = np.exp(-dist_sq / (2 * sigma**2))
        cmap = plt.get_cmap('viridis')
        rgba_img = cmap(Z); rgba_img[..., 3] = Z * 0.6 
        im = ax.imshow(rgba_img, extent=[-0.1, 1.1, -0.1, 1.1], origin='lower', aspect='auto', zorder=20)

    def clean(l): return "\n".join(l[:5])
    txt = (f"METRICS (Jaccard 0-1):\n{metrics_str}\n\n{'='*30}\n\n"
           f"TOP 5:\n{clean(top_res)}\n\n{'-'*20}\n\n"
           f"MID (5+5 Around Median):\n{clean(med_res)}\n\n{'-'*20}\n\n"
           f"BOT 5:\n{clean(bot_res)}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    ax.text(1.15, 1.0, txt, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.subplots_adjust(right=0.70) 
    plt.savefig(os.path.join(output_dir, f"plot_group_{group_id}.png"), dpi=100)
    plt.close(fig)

# ==========================================
# 5. AGGREGATION LOGIC
# ==========================================

def aggregate_group_metrics(items, model_names, top_k):
    """
    Compute group-level aggregated metrics and per-model outlier scores.

    Inputs:
      - items: list of dicts containing keys:
          'individual_probs' -> list of tensors (one per model) with class probabilities
          'ensemble_probs'   -> tensor with ensemble (mean) probability vector
          'weight'           -> scalar weight for the image
      - model_names: list of model name strings
      - top_k: integer used to select top/bottom slices

    Outputs:
      - agg_prob: aggregated (weighted) probability vector across images (torch.Tensor)
      - eff: effective sample size computed from the sample weights (float)
      - metrics: dict with keys for each slice in ['top','mid','bot'] containing:
          * {slice}_model_agree     : Inter-model agreement averaged across images (Jaccard)
          * {slice}_outlier_{model} : Average L2 outlier score for each model (averaged across items)
          * {slice}_const_{model}   : Intra-model consistency for each model (Jaccard across images)
          * {slice}_model_const_AVG : Average intra-model consistency (mean across models)
          * {slice}_ens_const       : Ensemble consistency across images (Jaccard)

    Notes on interpretation:
      - 'eff' (effective sample size) is computed from the weights via (sum_w^2 / sum_sq_w).
        It reflects how many equally-weighted samples the weighted set is equivalent to.
        If weights are uniform eff ~= N, otherwise eff is smaller.
      - Agreement/consistency metrics are Jaccard means in [0,1]: higher is more agreement/stability.
      - Outlier scores are distances in probability-space; compare magnitudes across models to
        identify models that consistently deviate from the consensus.
    """
    model_count = len(model_names)
    
    # 1. Weighted Ensemble Probability
    tensors = [x['ensemble_probs'] for x in items]
    weights = [x['weight'] for x in items]
    
    stack = torch.stack(tensors, dim=0).to(tensors[0].device)
    w_tensor = torch.tensor(weights, device=stack.device, dtype=torch.float32).view(-1, 1)

    sum_w = w_tensor.sum()
    sum_sq_w = (w_tensor ** 2).sum()
    eff = (sum_w ** 2) / sum_sq_w if sum_sq_w > 0 else 0.0
    
    if sum_w == 0: agg_prob = torch.ones(stack.shape[1], device=stack.device)/stack.shape[1]
    else: agg_prob = (stack * (w_tensor / sum_w)).sum(dim=0)
    agg_prob = agg_prob / agg_prob.sum()

    # 2. Metrics & Outliers
    metrics = {}
    
    for sl in ['top', 'mid', 'bot']:
        # A. Inter-Model Agreement (Avg across images)
        #    For each image we compute label-sets for all models and take the
        #    average pairwise Jaccard similarity; these image-level agreements
        #    are then averaged across the group (weighted).
        img_agrees = []
        # Store per-image per-model outlier scores so we can aggregate them
        img_outliers = []
        
        for item in items:
            # Jaccard: agreement between models on this image's slice
            sets = [get_slice_indices(p, sl, top_k) for p in item['individual_probs']]
            img_agrees.append(calculate_set_similarity(sets))
            
            # Outlier Score: L2 distance from consensus for each model on this image
            scores = calculate_outlier_scores(item['individual_probs'], model_names)
            img_outliers.append(scores)
        
        # Weighted Aggregations: average image-level agreement using sample weights
        metrics[f"{sl}_model_agree"] = np.average(img_agrees, weights=weights) if sum_w > 0 else 0.0

        # Aggregate Outlier Scores (Average distance per model across the group)
        for m_name in model_names:
            m_scores = [x[m_name] for x in img_outliers]
            avg_score = np.average(m_scores, weights=weights) if sum_w > 0 else 0.0
            metrics[f"{sl}_outlier_{m_name}"] = avg_score

        # B. Intra-Model Consistency (Per Model)
        #    For each model we look across images and measure how similar that
        #    model's nominated slice-sets are across the group's images.
        mod_const_scores = []
        for m_idx, m_name in enumerate(model_names):
            m_probs = [item['individual_probs'][m_idx] for item in items]
            m_sets = [get_slice_indices(p, sl, top_k) for p in m_probs]
            val = calculate_set_similarity(m_sets)
            metrics[f"{sl}_const_{m_name}"] = val
            mod_const_scores.append(val)
        
        # Average intra-model consistency across models
        metrics[f"{sl}_model_const_AVG"] = sum(mod_const_scores) / len(mod_const_scores)

        # C. Ensemble Consistency: how stable is the ensemble's slice across images?
        ens_probs = [item['ensemble_probs'] for item in items]
        ens_sets = [get_slice_indices(p, sl, top_k) for p in ens_probs]
        metrics[f"{sl}_ens_const"] = calculate_set_similarity(ens_sets)

    return agg_prob, eff, metrics

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

    if args.plot_dir and VIZ_AVAILABLE and not os.path.exists(args.plot_dir): os.makedirs(args.plot_dir)
    img_paths = discover_images(Path(args.folder), args.recursive)
    if not img_paths: sys.exit("[ERROR] No images.")

    loaded_models = []
    for m in args.models:
        print(f"[INFO] Loading {m}...")
        mod, pre, cats = resolve_model_and_weights(m)
        mod.to(args.device).eval()
        loaded_models.append({'name': m, 'model': mod, 'pre': pre, 'cats': cats})

    labels = loaded_models[0]['cats'] if loaded_models[0]['cats'] else [f"Class_{i}" for i in range(1000)]
    model_names = [m['name'] for m in loaded_models]

    # --- INFERENCE ---
    results_db = {}
    with torch.no_grad():
        for m_idx, rec in enumerate(tqdm(loaded_models, desc="Models")):
            batch_t, batch_p = [], []
            def flush():
                if not batch_t: return
                t = torch.stack(batch_t).to(args.device)
                probs = F.softmax(rec['model'](t), dim=1).cpu()
                for i, p in enumerate(probs):
                    k = str(batch_p[i])
                    if k not in results_db:
                        g, f1, f2 = parse_filename_features(batch_p[i])
                        w = compute_weight(f1, f2, args.target_f1, args.target_f2, args.sigma, args.uniform)
                        results_db[k] = {'group': g, 'f1': f1, 'f2': f2, 'weight': w, 'path': batch_p[i], 
                                         'individual_probs': [None]*len(loaded_models)}
                    results_db[k]['individual_probs'][m_idx] = p
                batch_t[:], batch_p[:] = [], []
            for p in tqdm(img_paths, desc=f"Inference {rec['name']}", leave=False):
                batch_t.append(rec['pre'](load_image(p))); batch_p.append(p)
                if len(batch_t) >= args.batch_size: flush()
            flush()

    # Pre-calc Ensemble
    group_data = {}
    for key, data in results_db.items():
        if any(x is None for x in data['individual_probs']): continue
        stack = torch.stack(data['individual_probs'])
        ensemble_p = stack.mean(dim=0)
        data['ensemble_probs'] = ensemble_p / ensemble_p.sum()
        
        gid = data['group']
        if gid not in group_data: group_data[gid] = []
        group_data[gid].append(data)

    sorted_groups = sorted(group_data.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    if args.csv:
        f_csv = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        
        # HEADER
        header = ["group", "count", "eff_samples"]
        for s in ['top', 'mid', 'bot']:
            header.append(f"{s}_model_agree")
            header.append(f"{s}_model_const_AVG")
            
            # Per-Model Consistency
            for m_name in model_names:
                header.append(f"{s}_const_{m_name}")
            
            # Per-Model Outlier Score
            for m_name in model_names:
                header.append(f"{s}_outlier_{m_name}")

            header.append(f"{s}_ens_const")

        header.extend(["top_labels", "mid_labels", "bot_labels"])
        writer.writerow(header)

    print(f"\n{'='*30}\n FINAL ANALYSIS (With Outlier Scores) \n{'='*30}")

    for gid in sorted_groups:
        items = group_data[gid]
        agg_prob, eff, metrics = aggregate_group_metrics(items, model_names, args.topk)
        
        (tv, ti), (mv, mi), (bv, bi) = get_stats_slices_mass_display(agg_prob, args.topk)
        top_str, med_str, bot_str = format_k(tv, ti, labels), format_k(mv, mi, labels), format_k(bv, bi, labels)
        
        # Identify Max Outlier for Display
        top_outliers = {m: metrics[f"top_outlier_{m}"] for m in model_names}
        worst_model = max(top_outliers, key=top_outliers.get)
        worst_score = top_outliers[worst_model]

        print(f"\nGroup: {gid} | Eff: {eff:.1f} | Top Outlier: {worst_model} ({worst_score:.2f})")
        print("  --- Top ---")
        print(f"  Model Agree: {metrics['top_model_agree']:.2f}")
        print(f"  Avg Consistency: {metrics['top_model_const_AVG']:.2f} | Ensemble Const: {metrics['top_ens_const']:.2f}")

        plot_metrics_str = (
            f"TOP: Agr={metrics['top_model_agree']:.2f}, I-Const={metrics['top_model_const_AVG']:.2f}, E-Const={metrics['top_ens_const']:.2f}\n"
            f"MID: Agr={metrics['mid_model_agree']:.2f}, I-Const={metrics['mid_model_const_AVG']:.2f}, E-Const={metrics['mid_ens_const']:.2f}\n"
            f"BOT: Agr={metrics['bot_model_agree']:.2f}, I-Const={metrics['bot_model_const_AVG']:.2f}, E-Const={metrics['bot_ens_const']:.2f}"
        )
        
        outlier_display = f"Top Outlier: {worst_model} ({worst_score:.2f})"

        if args.plot_dir:
            generate_group_plot(gid, items, args.target_f1, args.target_f2, args.sigma, args.uniform,
                                top_str, med_str, bot_str, plot_metrics_str, outlier_display, args.plot_dir)
        
        if args.csv:
            row = [gid, len(items), f"{eff:.2f}"]
            for s in ['top', 'mid', 'bot']:
                row.append(f"{metrics[f'{s}_model_agree']:.4f}")
                row.append(f"{metrics[f'{s}_model_const_AVG']:.4f}")
                
                # Per-Model Consistency
                for m_name in model_names:
                    row.append(f"{metrics[f'{s}_const_{m_name}']:.4f}")
                
                # Per-Model Outlier Score
                for m_name in model_names:
                    row.append(f"{metrics[f'{s}_outlier_{m_name}']:.4f}")
                    
                row.append(f"{metrics[f'{s}_ens_const']:.4f}")
            
            row.extend(["; ".join(top_str), "; ".join(med_str), "; ".join(bot_str)])
            writer.writerow(row)

    if args.csv: f_csv.close()

if __name__ == "__main__":
    main()