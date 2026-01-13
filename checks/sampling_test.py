"""Interactive sampling visualization for design CSVs.

Opens a file-selection dialog to pick a design CSV produced by the pipeline,
loads coordinates sampled_f0 / sampled_f1 grouped by ObjectSpace, and produces
a saved scatterplot (and displays it).

Expectations:
 - CSV must contain columns: 'ObjectSpace', 'sampled_f0', 'sampled_f1'
 - Filenames/paths are selected interactively via a small Tk file dialog.

Usage:
    python checks/sampling_test.py

Notes:
 - This script uses tkinter for a file dialog and matplotlib/seaborn for plotting.
   On headless systems (no DISPLAY) the dialog or plt.show() may fail; however,
   the PNG preview is always written to the current working directory.
 - The saved output name is `sampling_test_<original_csv_basename>.png`.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog, Tk
import os

def run_sampling_test():
    # 1. File Selection GUI
    # Create a minimal, hidden Tk root so we can call the native file dialog.
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring dialog to front on some platforms

    print("Please select a design CSV file to visualize...")
    file_path = filedialog.askopenfilename(
        title="Select Design CSV",
        filetypes=[("CSV Files", "*.csv")]
    )
    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(file_path)

        # Validate required columns early to give a clear error message
        required_cols = ['ObjectSpace', 'sampled_f0', 'sampled_f1']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: The selected file is missing columns: {missing}")
            print("Ensure you are using a design file generated with the latest 'create_runs.py'.")
            return

    except Exception as e:
        # Surface IO / parsing errors
        print(f"Error reading CSV: {e}")
        return

    # 3. Plotting
    print(f"Plotting sampling data for: {os.path.basename(file_path)}")

    plt.figure(figsize=(10, 8))

    # Create the scatter plot:
    # - hue/style keyed by 'ObjectSpace' so each group gets a distinct marker/color.
    # - s controls marker size; alpha provides slight transparency for overlap.
    sns.scatterplot(
        data=df,
        x='sampled_f0',
        y='sampled_f1',
        hue='ObjectSpace',
        style='ObjectSpace',
        palette='viridis',
        s=100,
        alpha=0.7
    )

    # 4. Formatting the Stimulus Space
    # Keep axes square and within slightly padded [0,1] range for clarity.
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f"Gaussian Sampling Test\nFile: {os.path.basename(file_path)}", fontsize=14)
    plt.xlabel("Feature 0 (Standardized)", fontsize=12)
    plt.ylabel("Feature 1 (Standardized)", fontsize=12)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Object Space (Group)', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save a preview and show the plot.
    # The saved PNG is useful when running in environments where interactive
    # display is not available.
    output_name = f"checks/plots/sampling_test_{os.path.basename(file_path)}.png"
    plt.savefig(output_name)
    print(f"Plot saved as: {output_name}")
    plt.show()

if __name__ == "__main__":
    run_sampling_test()