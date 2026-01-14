#!/usr/bin/env python3
"""
infer_final_robust_outlier.py

The Definitive Pipeline (With Per-Model Outlier Scores).
1. Metrics:
   - Inter-Model Agreement & Intra-Model Consistency.
   - Ensemble Consistency.
   - Outlier Scoring (Distance from consensus for EACH model).
2. Slices: 
   - Top: Top-K (default 20).
   - Mid: Either 5+5 around Median (default) OR Next-K ranked (21-40).
   - Bot: Bottom-K.
3. Visualization: Images + Transparent Gaussian + Metrics Sidebar.
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

def get_slice_indices(probs: torch.Tensor, slice_type: str, k: int, mid_type: str = "median") -> Set[int]:
    """Returns set of indices for Top/Mid/Bot slices."""
    numel = probs.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(probs, descending=True)
    
    if slice_type == "top":
        return set(sorted_indices[:k].tolist())
    elif slice_type == "bot":
        return set(sorted_indices[-k:].tolist())
    elif slice_type == "mid":
        if mid_type == "rank":
            # Next k ranked labels (e.g., 21-40 if k=20)
            # Ensure we don't go out of bounds
            start = k
            end = min(numel, k * 2)
            return set(sorted_indices[start:end].tolist())
        else:
            # Mass-based Median (Original Logic)
            cumsum = torch.cumsum(sorted_vals, dim=0)
            mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
            center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
            
            # Modified: 5+5 labels around the median (Radius 5)
            # Note: This ignores 'k' and enforces a fixed window size of 10 centered on the median.
            start = max(0, center_idx - 5)
            end = min(numel, center_idx + 5)
            
            return set(sorted_indices[start:end].tolist())
    return set()

def calculate_set_similarity(sets: List[Set[int]]) -> float:
    """Average Pairwise Jaccard Similarity."""
    if len(sets) < 2: return 1.0
    scores = []
    for s1, s2 in itertools.combinations(sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        scores.append(intersection / union if union > 0 else 0.0)
    return sum(scores) / len(scores)

def calculate_outlier_scores(prob_list: List[torch.Tensor], model_names: List[str]) -> Dict[str, float]:
    """
    Calculates the Euclidean distance of EACH model from the group mean.
    Returns a dict: {'resnet50': 0.12, 'inception': 0.45, ...}
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
    
    # Handle "object1F0Level0.5F1Level0.8" format
    # Matches digits/decimals following 'F0Level' and 'F1Level'
    f0_match = re.search(r'F0Level([-+]?\d*\.\d+|\d+)', stem)
    f1_match = re.search(r'F1Level([-+]?\d*\.\d+|\d+)', stem)
    
    obj_name = path.parent.name if path.parent.name else "Unknown"
    f0 = float(f0_match.group(1)) if f0_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    
    return obj_name, f0, f1

def compute_weight(f1, f2, tf1, tf2, sigma, uniform):
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

def get_stats_slices_mass_display(scores: torch.Tensor, k: int, mid_type: str = "median"):
    numel = scores.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(scores, descending=True)
    top_vals, top_idx = sorted_vals[:k], sorted_indices[:k]
    bot_vals, bot_idx = sorted_vals[-k:], sorted_indices[-k:]
    
    # Mid Logic
    if mid_type == "rank":
        # Next k ranked labels
        start = k
        end = min(numel, k * 2)
        med_vals, med_idx = sorted_vals[start:end], sorted_indices[start:end]
    else:
        # Mass-based Median
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
           f"MID ({'Next-K' if 'Next-K' in metrics_str else '5+5 Around Median'}):\n{clean(med_res)}\n\n{'-'*20}\n\n"
           f"BOT 5:\n{clean(bot_res)}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    ax.text(1.15, 1.0, txt, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.subplots_adjust(right=0.70) 
    plt.savefig(os.path.join(output_dir, f"plot_group_{group_id}.png"), dpi=100)
    plt.close(fig)

# ==========================================
# 5. AGGREGATION LOGIC
# ==========================================

def aggregate_group_metrics(items, model_names, top_k, mid_type):
    """Calculates metrics including per-model breakdowns and outlier scores."""
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
        img_agrees = []
        # Store outlier scores per image, then we average them
        # Structure: list of dicts [{'resnet': 0.1, 'inc': 0.2}, ...]
        img_outliers = []
        
        for item in items:
            # Jaccard
            sets = [get_slice_indices(p, sl, top_k, mid_type) for p in item['individual_probs']]
            img_agrees.append(calculate_set_similarity(sets))
            
            # Outlier Score (Distance) for this image
            scores = calculate_outlier_scores(item['individual_probs'], model_names)
            img_outliers.append(scores)
        
        # Weighted Aggregations
        metrics[f"{sl}_model_agree"] = np.average(img_agrees, weights=weights) if sum_w > 0 else 0.0

        # Aggregate Outlier Scores (Average distance per model across the group)
        for m_name in model_names:
            m_scores = [x[m_name] for x in img_outliers]
            avg_score = np.average(m_scores, weights=weights) if sum_w > 0 else 0.0
            metrics[f"{sl}_outlier_{m_name}"] = avg_score

        # B. Intra-Model Consistency (Per Model)
        mod_const_scores = []
        for m_idx, m_name in enumerate(model_names):
            m_probs = [item['individual_probs'][m_idx] for item in items]
            m_sets = [get_slice_indices(p, sl, top_k, mid_type) for p in m_probs]
            val = calculate_set_similarity(m_sets)
            metrics[f"{sl}_const_{m_name}"] = val
            mod_const_scores.append(val)
        
        metrics[f"{sl}_model_const_AVG"] = sum(mod_const_scores) / len(mod_const_scores)

        # C. Ensemble Consistency
        ens_probs = [item['ensemble_probs'] for item in items]
        ens_sets = [get_slice_indices(p, sl, top_k, mid_type) for p in ens_probs]
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
    ap.add_argument("--mid-type", type=str, choices=["median", "rank"], default="median", 
                    help="Method for defining mid slice: 'median' (mass-based) or 'rank' (next-K ranked labels).")
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
        agg_prob, eff, metrics = aggregate_group_metrics(items, model_names, args.topk, args.mid_type)
        
        (tv, ti), (mv, mi), (bv, bi) = get_stats_slices_mass_display(agg_prob, args.topk, args.mid_type)
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