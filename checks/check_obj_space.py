"""Streamlit app to inspect object-space stimulus layouts and distributions.

Displays a 10x10 grid of stimuli sampled from a selected object-space folder,
overlays a 2D Gaussian density (configurable mean and SD per axis), and draws
an approximate 95% confidence ellipse. Labels for the selected space are read
from a CSV file (expected columns: 'group', 'top_labels', 'mid_labels',
'bot_labels'). Image filenames are expected to follow the pattern
"{gid}_{F0}_{F1}.png" or ".jpg" where F0 and F1 are normalized coordinates.

Usage:
    streamlit run checks/check_obj_space.py

Dependencies:
    streamlit, numpy, pandas, matplotlib, scipy

Notes:
    - The parent stimuli directory and label CSV path are selected via the
      sidebar inputs in the running Streamlit app.
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
from scipy.stats import multivariate_normal

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Object Space Checker")

def get_closest_image(folder, target_coords):
    """Finds the image in the folder closest to the target normalized coordinates."""
    images = list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpg"))
    best_match, min_dist = None, float('inf')
    
    for img_path in images:
        try:
            # Format: {gid}_{F0}_{F1}.ext
            parts = img_path.stem.split('_')
            img_coords = np.array([float(p) for p in parts[1:]])
            dist = np.linalg.norm(img_coords[:2] - target_coords)
            if dist < min_dist:
                min_dist, best_match = dist, img_path
        except (ValueError, IndexError):
            continue
    return best_match

def main():
    st.title(" Object Space Distribution Checker")

    # --- SIDEBAR GUI ---
    st.sidebar.header("Path Settings")
    parent_dir = st.sidebar.text_input("Parent Stimuli Directory", "renamed_images")
    label_csv = st.sidebar.text_input("Label CSV Path", "setup/output/full_results_v1.csv")

    if not os.path.exists(label_csv):
        st.error(f"Label file not found: {label_csv}")
        return

    # Load Labels
    df_labels = pd.read_csv(label_csv)
    
    if os.path.exists(parent_dir):
        spaces = sorted([d.name for d in Path(parent_dir).iterdir() if d.is_dir()])
    else:
        st.sidebar.warning("Stimuli directory not found.")
        spaces = []

    st.sidebar.header("Parameters (SD Units)")
    selected_space = st.sidebar.selectbox("Select Object Space", spaces)
    
    mu_f0 = st.sidebar.slider("Mean F0", 0.0, 1.0, 0.5, 0.01)
    mu_f1 = st.sidebar.slider("Mean F1", 0.0, 1.0, 0.5, 0.01)
    
    # SD
    sd_f0 = st.sidebar.slider("SD F0", 0.01, 0.5, 0.15, 0.01)
    sd_f1 = st.sidebar.slider("SD F1", 0.01, 0.5, 0.15, 0.01)

    # --- MAIN DISPLAY ---
    col1, col2 = st.columns([2, 1])

    with col1:
        if selected_space:
            space_path = Path(parent_dir) / selected_space
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 1. Draw 10x10 Grid
            grid_points = np.linspace(0, 1, 10)
            for y_val in reversed(grid_points):
                for x_val in grid_points:
                    img_path = get_closest_image(space_path, np.array([x_val, y_val]))
                    if img_path:
                        img = plt.imread(img_path)
                        ax.imshow(img, extent=[x_val-0.045, x_val+0.045, y_val-0.045, y_val+0.045])

            # 2. Draw Gaussian Cloud (Density Heatmap)
            x_range = np.linspace(-0.1, 1.1, 150)
            y_range = np.linspace(-0.1, 1.1, 150)
            X, Y = np.meshgrid(x_range, y_range)
            pos = np.dstack((X, Y))
            
            # Use SD squared (variance) for the covariance matrix
            cov_matrix = [[sd_f0**2, 0], [0, sd_f1**2]]
            rv = multivariate_normal([mu_f0, mu_f1], cov_matrix)
            Z = rv.pdf(pos)
            
            # Cloud effect
            ax.contourf(X, Y, Z, levels=20, cmap='Reds', alpha=0.3)

            # 3. Draw 95% CI Band (approx 2.45 * SD for 2D Gaussian)
            ci_95 = Ellipse(xy=(mu_f0, mu_f1), width=sd_f0 * 4.89, height=sd_f1 * 4.89, 
                            edgecolor='red', fc='None', lw=2, linestyle='--', label='95% CI')
            ax.add_patch(ci_95)
            
            ax.scatter(mu_f0, mu_f1, color='black', marker='+', s=100, label='Mean')
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f"Space: {selected_space} | SD0: {sd_f0}, SD1: {sd_f1}", fontsize=15)
            ax.legend()
            st.pyplot(fig)

    with col2:
        st.subheader("Object Space Labels")
        space_data = df_labels[df_labels['group'].astype(str) == str(selected_space)]
        if not space_data.empty:
            st.info("**TOP LABELS (Congruent)**")
            st.write(space_data.iloc[0]['top_labels'])
            st.warning("**MID LABELS (Medium)**")
            st.write(space_data.iloc[0]['mid_labels'])
            st.error("**BOT LABELS (Incongruent)**")
            st.write(space_data.iloc[0]['bot_labels'])

if __name__ == "__main__":
    main()