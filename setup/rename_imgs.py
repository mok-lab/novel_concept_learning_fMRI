"""
Normalize per-directory feature values and rename image files.

Treats each subdirectory as an "object space", parses embedded `F#Level`
tokens from filenames, normalizes each feature dimension to [0,1] within
that directory, and copies files to a mirrored output tree using names of
the form: "{gid}_{F0_norm}_{F1_norm}_...{ext}".

Usage:
    No args on this script. It uses hardcoded input/output paths at the bottom... (I know, I know.)
    python setup/rename_imgs.py
"""
import os
import re
import shutil
from pathlib import Path

def normalize_value(value, min_val, max_val):
    """Standardizes values to [0, 1]."""
    if max_val - min_val == 0:
        return 0.5  # Midpoint for single-value features
    return (value - min_val) / (max_val - min_val)

def process_directory(source_path, dest_path):
    """Processes images in a single directory as a discrete object space."""
    # Regex: Extracts group ID and the string containing F#Level# values
    base_pattern = re.compile(r"object(.*?)(F\d+Level.*)(\.[^.]+)$", re.IGNORECASE)
    feature_pattern = re.compile(r"F\d+Level([\d\.]+)", re.IGNORECASE)

    files_to_process = []
    max_features = 0

    # 1. Parse all files in this directory
    for entry in source_path.iterdir():
        if entry.is_file() and not entry.name.startswith('.'):
            match = base_pattern.match(entry.name)
            if match:
                gid = match.group(1)
                raw_features = [float(x) for x in feature_pattern.findall(match.group(2))]
                max_features = max(max_features, len(raw_features))
                
                files_to_process.append({
                    'path': entry,
                    'gid': gid,
                    'features': raw_features,
                    'ext': match.group(3)
                })
    
    if not files_to_process:
        return

    # 2. Find Min/Max per feature dimension within THIS directory
    min_vals = [float('inf')] * max_features
    max_vals = [float('-inf')] * max_features
    
    for f in files_to_process:
        for i, val in enumerate(f['features']):
            if val < min_vals[i]: min_vals[i] = val
            if val > max_vals[i]: max_vals[i] = val

    # 3. Normalize and save to mirrored subdirectory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for f in files_to_process:
        norm_features = []
        for i in range(max_features):
            # Handle files with inconsistent feature counts by padding
            val = f['features'][i] if i < len(f['features']) else min_vals[i]
            n = normalize_value(val, min_vals[i], max_vals[i])
            norm_features.append(n)

        # New Name: {gid}_{F0_norm}_{F1_norm}...{ext}
        # Example: 3_0.000_1.000.png
        feature_str = "_".join([f"{v:.3f}" for v in norm_features])
        new_name = f"{f['gid']}_{feature_str}{f['ext']}"
        shutil.copy2(f['path'], dest_path / new_name)

def run_renamer(source_folder, output_folder):
    """Walks directory tree and processes each subdirectory."""
    src_root = Path(source_folder)
    dst_root = Path(output_folder)
    
    print(f"--- Renaming Images in '{src_root}' ---")
    for root, dirs, files in os.walk(src_root):
        curr_src = Path(root)
        rel_path = curr_src.relative_to(src_root)
        process_directory(curr_src, dst_root / rel_path)
    print("--- Done. ---")

if __name__ == "__main__":
    # Use
    run_renamer("images/raw_images", "images/task_images")