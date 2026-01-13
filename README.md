# Novel Concept Learning fMRI Task

This repository contains the PsychoPy implementation for a "Novel Concept Learning" fMRI study. The experiment is designed to investigate how participants learn and categorize novel objects across different learning conditions.

## Overview

The core application runs a visual categorization task with trials where participants:

1. View a novel stimulus (image).
2. Select a category label from 4 options.
3. Receive feedback on their accuracy.

The task supports dynamic loading of image sets and label files, allowing for flexible experimental designs (e.g., 6-category & 9-category versions).

## Installation & Requirements

This project relies on **Conda** for environment management.

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/AkillesU/novel_concept_learning_fMRI
cd novel_concept_learning_fMRI

```


2. **Create the Conda environment:**
Use the requirements file included in this repository to install the necessary dependencies.
```bash
conda env create -f environment.yml
conda activate novel_concepts

```


3. **Download Stimulus Images:**
The stimulus images are not included in this repository due to size constraints.
* Download the entire `task_images/` folder from this [**OneDrive link**](https://rhul-my.sharepoint.com/:f:/g/personal/pnjt246_live_rhul_ac_uk/IgDOSCvDv20xQZYHt9jXc22WAQ0IoR5t-s1I_HV90KqWiio?e=DKrjzX).
* Place the downloaded `task_images/` folder containing all the numbered subfolders directly inside the `images/` parent folder within this repository.
* *Final structure should look like e.g.,:* `novel_concept_learning_fMRI/images/task_images/1/`



## Usage

To run the main experimental task:

```bash
python experimental_task/experimental_task.py

```

### Configuration GUI

Upon launching, a dialog box will ask for:

* **Sub:** Subject ID (e.g., `001`).
* **Design CSV:** Path to the schedule design file (inside `experimental_task/task_designs/`).
* **Label CSV:** Path to the category mapping file (e.g., inside `experimental_task/label_files/`). Remember to match the label file to the design (6-cat vs 9-cat, wide).
* **Image Dir:** Path to the folder containing stimulus images.
* **Feedback Delay:** (Boolean) Enable ISI2 delay logic.
* **Demo Mode:** (Boolean) Enable on-screen debug text and trial skipping.
* **Check below file structure graph if unsure about where things are located**

## Repository Structure

Below is an overview of the files and directories in this repository:

```text
.
├── .gitignore                   # Git configuration
├── README.md                    # Project documentation
├── checks/                      # Quality assurance and validation scripts
│   ├── check_obj_space.py       # Validates object space consistency
│   ├── label_consistency.py     # Ensures labels match across files
│   ├── sampling_test.py         # Tests the random sampling logic for distractors
│   └── task_design_vis.py       # Visualizes trial timing/distribution
├── experimental_task/           # Core experiment runtime files
│   ├── experimental_task.py     # MAIN RUNNER: The primary PsychoPy script
│   ├── practice_task.py         # Simplified version for participant training
│   ├── label_files/             # CSVs defining category labels (Targets/Distractors)
│   │   ├── practice_labels.csv  # Labels for training
│   │   ├── v1-6cat.csv          # 6-category version labels
│   │   └── v1-9cat.csv          # 9-category version labels
│   ├── resources/               # UI assets (e.g., instruction images)
│   └── task_designs/            # CSVs defining trial order, timing, and conditions
│       ├── v1_6cat/             # Designs for 6-category condition
│       ├── v1_6cat_wider/       # Designs with wider variability
│       ├── v1_9cat/             # Designs for 9-category condition
│       └── v1_9cat_wider/       # Designs for 9-category wider variability
├── images/                      # Stimulus image repository
│   └── task_images/             # (Download from OneDrive) Contains main experimental stimuli
│   └── practice_images/         # Images specific to the practice phase
└── setup/                       # Scripts for generating experimental designs
    ├── create_runs.py           # Generates the randomized trial CSVs
    ├── model_class.py           # Python class definitions for concept models
    ├── rename_imgs.py           # Utility to standardize image filenames
    ├── output/                  # Generated output from setup scripts
    └── labelling/               # External labelling files from human validation

```

## Outputs

Data is saved to the `participant_data/` directory (created automatically).

* **Filenames:** `sub-{ID}_{mode}_run-{X}.csv`
* **Content:** Includes reaction times (`rt`), accuracy, selected keys, and all original design columns.
