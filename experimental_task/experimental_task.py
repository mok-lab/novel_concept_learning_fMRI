#!/usr/bin/env python3
"""
Novel Concept Learning: Experiment Runner
Supports:
 1. Standard (2-Event: Dec -> Fb)
 2. Self-Paced (3-Event: Img -> Jitter -> Dec -> Fix -> Fb)
Features:
 - Dynamic Image Loading (Group_*.ext)
 - Dynamic Labels (Target + 3 Distractors)
 - Interactive Start Screen
 - Button Mapping: 1, 2, 9, 0
 - Feedback: Green for Correct (Always), Red for Incorrect Selection
 - Demo Mode: Displays condition text on right side
 - Centered Images: Images now appear at (0,0)

ADDED:
 - Multi-run support: if the design CSV contains a run/block column, the task will
   pause *between runs* by closing the fullscreen window, showing a native GUI dialog
   to start the next run, then re-opening fullscreen and showing the trigger screen.
   (Run 1 remains unchanged: it starts immediately after the initial trigger screen.)
 - Demo Mode trial skipping: when Demo Mode is enabled, press RIGHT ARROW to skip the
   current trial and immediately advance to the next trial.
"""

from psychopy import visual, core, event, gui
import pandas as pd
import numpy as np
import os
import random
import re
import sys
from datetime import datetime

LANGUAGE = "Japanese"  # "english" or "japanese"

# =========================== CONFIGURATION =========================== #
# Keys used by the task. Numeric keys map to response options; TRIGGER_KEY used
# to start a run (scanner trigger simulation). EXIT_KEY exits the experiment.
KEYS_RESP = ['1', '2', '9', '0']
TRIGGER_KEY = '5'
EXIT_KEY = 'escape'
SKIP_KEY = 'right'  # Demo Mode only
LANG_BTN_FONT_SCALE = { # Enlargen font size slightly for japanese buttons
    "english": 1.0,
    "japanese": 1.25
    }


# Colors used for buttons / feedback. Keep names descriptive to make intent clear.
COL_NEUTRAL   = 'silver'
COL_HOVER     = 'lightgrey'
COL_SELECT    = 'green'      # Color during selection (not used for fill)
COL_CORRECT   = 'green'      # Feedback Correct (border)
COL_INCORRECT = 'red'        # Feedback Incorrect (border)
TEXT_COLOR    = 'black'
DEMO_TEXT_COL = 'blue'

# Button line widths: normal and highlighted for feedback
LINE_W_NORMAL = 2
LINE_W_HIGHLIGHT = 9

# Selection fill color used to indicate currently chosen option during decision
COL_SELECT_FILL = 'white'

# Supported image file extensions for scanning image directories
IMG_EXTS = ['.png', '.jpg', '.jpeg', '.bmp']

# Candidate column names to detect run/block/session grouping in design CSVs.
# The first matching name in RUN_COL_CANDIDATES will be used.
RUN_COL_CANDIDATES = [
    'run', 'run_id', 'run_num', 'run_number',
    'block', 'block_id', 'block_num', 'block_number',
    'session', 'session_id',
]

# =========================== LOADERS =========================== #

def load_label_map(csv_path, key_col='ObjectSpace'):
    """Load label mapping from the stimulus info CSV (robust to NaNs + key normalization).

    Returns:
      (label_map, all_labels)
        label_map: dict space_id -> {'Congruent': str|None, 'Medium': str|None, 'Incongruent': str|None}
        all_labels: sorted list of valid label strings (no None/"nan"/"MISSING")
    """
    df = pd.read_csv(csv_path, sep=",", engine="python")

    required = {key_col, 'Congruent', 'Medium', 'Incongruent'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label CSV missing required columns: {sorted(missing)}")

    def _norm_space_id(v):
        # Normalize keys like 20, 20.0, "20.0" -> "20"
        if pd.isna(v):
            return None
        s = str(v).strip()
        # handle "20.0" form
        if re.fullmatch(r"\d+\.0+", s):
            return s.split(".")[0]
        return s

    def _clean_label(v):
        # Turn NaN/empty/"nan"/"MISSING" into None; otherwise return stripped string.
        if pd.isna(v):
            return None
        s = str(v).strip()
        if not s:
            return None
        if s.lower() == "nan":
            return None
        if s.upper() == "MISSING":
            return None
        return s

    label_map = {}
    all_labels_set = set()

    for _, row in df.iterrows():
        key = _norm_space_id(row[key_col])
        if key is None:
            continue

        entry = {
            'Congruent': _clean_label(row['Congruent']),
            'Medium': _clean_label(row['Medium']),
            'Incongruent': _clean_label(row['Incongruent']),
        }
        label_map[key] = entry

        for lbl in entry.values():
            if lbl is not None:
                all_labels_set.add(lbl)

    all_labels = sorted(all_labels_set)
    return label_map, all_labels


def build_participant_label_pool(design_df, label_map, space_col_candidates=('ObjectSpace', 'group')):
    """Build a restricted distractor pool for this participant, robustly.

    Guarantees returned list contains only non-empty strings (no None/"nan"/"MISSING"),
    so sorting cannot crash.
    """
    def _norm_space_id(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        if re.fullmatch(r"\d+\.0+", s):
            return s.split(".")[0]
        return s

    def _is_valid_label(lbl):
        if lbl is None:
            return False
        if isinstance(lbl, float) and np.isnan(lbl):
            return False
        s = str(lbl).strip()
        if not s:
            return False
        if s.lower() == "nan":
            return False
        if s.upper() == "MISSING":
            return False
        return True

    space_col = next((c for c in space_col_candidates if c in design_df.columns), None)
    cond_col = 'condition' if 'condition' in design_df.columns else None

    # Preferred: restrict to labels actually used by (space, condition) pairs
    if space_col is not None and cond_col is not None:
        pool = set()
        pairs = design_df[[space_col, cond_col]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            s = _norm_space_id(row[space_col])
            if s is None:
                continue
            cond = str(row[cond_col]).strip()
            lbl = label_map.get(s, {}).get(cond, None)
            if _is_valid_label(lbl):
                pool.add(str(lbl).strip())
        if pool:
            return sorted(pool)

    # Fallback: restrict by spaces only
    if space_col is not None:
        pool = set()
        for raw in design_df[space_col].dropna().tolist():
            s = _norm_space_id(raw)
            if s is None or s not in label_map:
                continue
            for lbl in label_map[s].values():
                if _is_valid_label(lbl):
                    pool.add(str(lbl).strip())
        if pool:
            return sorted(pool)

    # Final fallback: all labels from label_map
    pool = set()
    for d in label_map.values():
        for lbl in d.values():
            if _is_valid_label(lbl):
                pool.add(str(lbl).strip())
    return sorted(pool)


def calculate_lengths(df):
    """Return a human-readable total duration string for the design.

    Uses 'trial_duration_max' when present for the delayed/maximum timing, and
    approximates the immediate (no-isi2) total by subtracting isi2_dur contributions.
    """
    total_delayed = df['trial_duration_max'].sum() if 'trial_duration_max' in df.columns else 0
    # Immediate mode removes the isi2_dur jitter
    total_immediate = total_delayed - df['isi2_dur'].sum() if 'isi2_dur' in df.columns else total_delayed
    return f"{total_delayed/60:.1f}m (Delayed) / {total_immediate/60:.1f}m (Immediate)"


def scan_images(img_dir):
    """Scan a directory for images and group them by leading "space" token.

    The function expects filenames like: <space>_...ext and groups by the prefix
    before the first underscore. It returns a mapping space_id -> [filepaths].
    """
    files = os.listdir(img_dir)
    img_map = {}
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXTS:
            continue
        parts = name.split('_')
        if not parts:
            continue
        group = parts[0]
        if group not in img_map:
            img_map[group] = []
        img_map[group].append(os.path.join(img_dir, f))
    return img_map


def get_trial_space_id(trial):
    """Return normalized stimulus space identifier for the current trial.

    Handles pandas float coercion (e.g., 20.0) and string forms ("20.0").
    Returns None for missing/NaN IDs (e.g., fixation rows).
    """
    def _norm(v):
        if v is None:
            return None

        # pandas/np NaN
        try:
            import numpy as _np
            if isinstance(v, float) and _np.isnan(v):
                return None
        except Exception:
            pass

        # float like 20.0 -> "20"
        if isinstance(v, float):
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return str(v)

        s = str(v).strip()
        # string like "20.0" -> "20"
        if re.fullmatch(r"\d+\.0+", s):
            return s.split(".")[0]
        return s

    if 'ObjectSpace' in trial:
        return _norm(trial['ObjectSpace'])
    if 'group' in trial:
        return _norm(trial['group'])
    if 'Group' in trial:
        return _norm(trial['Group'])
    raise KeyError("Design CSV must contain either 'ObjectSpace' or 'group' column.")


def resolve_image_path(img_dir, space_id, image_file):
    """Resolve image path robustly across common directory layouts.

    The function attempts three strategies in order:
      1) Image stored in subfolder named after space_id: <img_dir>/<space_id>/<image_file>
      2) Flat layout: <img_dir>/<image_file>
      3) Recursive search under <img_dir> for the basename (returns first match)

    Returning None indicates the image could not be located.
    """
    # 1) Subfolder layout
    p1 = os.path.join(img_dir, str(space_id), str(image_file))
    if os.path.exists(p1):
        return p1
    # 2) Flat layout
    p2 = os.path.join(img_dir, str(image_file))
    if os.path.exists(p2):
        return p2
    # 3) Recursive fallback
    try:
        for root, _, files in os.walk(img_dir):
            if str(image_file) in files:
                return os.path.join(root, str(image_file))
    except Exception:
        # Ignore IO errors during recursive search; return None below
        pass
    return None

# =========================== RUN HELPERS =========================== #

def detect_run_column(design_df):
    """Return the name of the run/block column if present, else None.

    Uses RUN_COL_CANDIDATES order to allow flexibility in input CSV formats.
    """
    cols = set(design_df.columns)
    for c in RUN_COL_CANDIDATES:
        if c in cols:
            return c
    return None


def split_into_runs(design_df, run_col):
    """Split design_df into ordered runs (stable order of first appearance).

    When run_col is None the whole design is considered a single run and a
    list with (None, design_df) is returned to keep the calling code uniform.
    """
    if run_col is None:
        return [(None, design_df)]
    # Preserve order of appearance rather than sorting values lexicographically
    run_values = []
    seen = set()
    for v in design_df[run_col].tolist():
        if v not in seen:
            seen.add(v)
            run_values.append(v)
    return [(rv, design_df[design_df[run_col] == rv].copy()) for rv in run_values]




def generate_balanced_button_sequence(n_trials, n_buttons=4, max_repeat=2, rng=None):
    """Generate a per-run sequence of correct-button indices.

    Goals:
      - Counterbalance correct option positions within a run (counts differ by at most 1).
      - Avoid placing the correct option on the same button more than `max_repeat` times in a row.
      - Keep behaviour deterministic if the caller passes a seeded `rng`.

    Returns:
      list[int]: length n_trials, values in [0, n_buttons-1]
    """
    if rng is None:
        rng = random

    if n_trials <= 0:
        return []

    # Distribute counts as evenly as possible across buttons
    base = n_trials // n_buttons
    rem = n_trials % n_buttons
    counts = [base] * n_buttons
    # Randomize which buttons get the extra 1 to avoid systematic bias
    extra_buttons = list(range(n_buttons))
    rng.shuffle(extra_buttons)
    for i in range(rem):
        counts[extra_buttons[i]] += 1

    # Greedy with light backtracking: pick among eligible buttons with remaining counts.
    seq = []
    last = []
    for _ in range(n_trials):
        # Eligible buttons: remaining > 0 and not violating max_repeat constraint
        eligible = [i for i in range(n_buttons) if counts[i] > 0]
        if max_repeat is not None and max_repeat > 0 and len(last) >= max_repeat:
            if all(x == last[-1] for x in last[-max_repeat:]):
                eligible = [i for i in eligible if i != last[-1]]

        if not eligible:
            # Rare corner: restart with a fresh random distribution until feasible
            # (n_buttons=4 and max_repeat=2 makes this extremely unlikely to loop).
            return generate_balanced_button_sequence(n_trials, n_buttons=n_buttons, max_repeat=max_repeat, rng=rng)

        # Choose the button with the highest remaining count; random tie-break for balance
        max_c = max(counts[i] for i in eligible)
        top = [i for i in eligible if counts[i] == max_c]
        choice = rng.choice(top)

        seq.append(choice)
        counts[choice] -= 1
        last.append(choice)

    return seq
def between_run_dialog(next_run_idx, run_label=None):
    """Native GUI dialog shown between runs (outside fullscreen).

    This dialog helps with multi-run experiments where the display must be
    closed between runs (e.g., to allow scanner breaks or repositioning). It
    also reminds the operator which run is about to begin and which trigger key
    will start it.
    """
    if LANGUAGE == "japanese":
        title = f"施行開始 {next_run_idx}"
    else:
        title = f"Start run {next_run_idx}"
    dlg = gui.Dlg(title=title)
    if run_label is not None:
        if LANGUAGE == "japanese":
            dlg.addText(f"次回の施行値: {run_label}")
        else:
            dlg.addText(f"Next run value: {run_label}")
    if LANGUAGE == "japanese":
        dlg.addText("「OK」をクリックすると全画面で開きます。")
    else:
        dlg.addText("Click OK to open fullscreen.")
    if LANGUAGE == "japanese":
        dlg.addText(f"次にトリガー画面で '{TRIGGER_KEY}'を押して、施行 {next_run_idx} を開始します。")
    else:
        dlg.addText(f"Then press '{TRIGGER_KEY}' on the trigger screen to begin run {next_run_idx}.")
    dlg.show()
    if not dlg.OK:
        core.quit()


# =========================== VISUAL HELPERS =========================== #

def show_instruction_screen(win, text_content, image_path=None, use_scanner_buttons=False):
    """
    Standardized instruction screen with optional image.
    Anchored to relative height units for cross-screen compatibility.

    The function blocks until the participant presses the SPACE bar, and it
    will attempt to display a large illustrative image if a valid path is given.
    """
    # Standardized vertical positions (relative to height 1.0)
    Y_INSTR_TEXT_HIGH = 0.22
    Y_INSTR_TEXT_MID  = 0.10
    Y_INSTR_IMAGE     = -0.15
    Y_SPACE_PROMPT    = -0.40

    text_y = Y_INSTR_TEXT_HIGH if image_path else Y_INSTR_TEXT_MID
    msg = visual.TextStim(win, text=text_content, color='black', height=0.035, 
                          wrapWidth=0.8, pos=(0, text_y))
    if LANGUAGE == "japanese":
        cont_text = "続行するにはスペースバーを押してください"
    else:
        cont_text = "Press SPACE bar to continue"
    if use_scanner_buttons:
        if LANGUAGE == "japanese":
            cont_text = "続行するには右端のボタンを押してください"
        else:
            cont_text = "Press the rightmost button to continue"
    cont_msg = visual.TextStim(win, text=cont_text,
                               pos=(0, Y_SPACE_PROMPT), color='grey', height=0.025)
    
    img_stim = None
    if image_path and os.path.exists(image_path):
        # size (0.7, 0.4) provides a large, clear instructional icon
        img_stim = visual.ImageStim(win, image=image_path, pos=(0, Y_INSTR_IMAGE), size=(0.3, 0.3))
    
    event.clearEvents()
    while True:
        msg.draw()
        if img_stim:
            img_stim.draw()
        cont_msg.draw()
        win.flip()
        # Block until the continue key is pressed to ensure participant reads instructions
        cont_key = 'space'
        if use_scanner_buttons:
            # In scanner mode we use the rightmost response button to continue
            cont_key = '4'
        if cont_key in event.getKeys():
            break


def draw_buttons(buttons, selected_key):
    """Draw buttons during the Decision phase.

    Visual behaviour:
      - The currently selected option (if any) is indicated by a white fill.
      - Borders remain at normal thickness during the decision window.
      - Button text is drawn twice to ensure readability across systems.
    """
    for btn in buttons:
        # Fill selected option with white
        if selected_key is not None and selected_key == btn['key']:
            btn['box'].fillColor = COL_SELECT_FILL
        else:
            btn['box'].fillColor = COL_NEUTRAL

        btn['box'].lineColor = 'black'
        btn['box'].lineWidth = LINE_W_NORMAL

        btn['box'].draw()
        btn['text'].draw()
        btn['text'].draw()


def draw_buttons_feedback(buttons, response_made, correct_key):
    """Draw buttons during the Feedback phase and apply highlighting.

    Behaviour:
      - If *no response* was made (response_made is None), draw buttons in a neutral
        state (no green/red feedback borders).
      - If a response was made:
          * highlight the correct option border in green
          * if the response was incorrect, also highlight the chosen option in red
      - Make highlight thicker than standard border for visibility.
      - Keep fill neutral to preserve label readability.
    """
    no_response = (response_made is None)

    for btn in buttons:
        btn['box'].fillColor = COL_NEUTRAL
        btn['box'].lineColor = 'black'
        btn['box'].lineWidth = LINE_W_NORMAL

        if not no_response:
            # Correct option: green border
            if correct_key is not None and btn['key'] == correct_key:
                btn['box'].lineColor = COL_CORRECT
                btn['box'].lineWidth = LINE_W_HIGHLIGHT

            # Incorrect selection: red border (in addition to correct border elsewhere)
            if response_made != correct_key and btn['key'] == response_made:
                btn['box'].lineColor = COL_INCORRECT
                btn['box'].lineWidth = LINE_W_HIGHLIGHT

        btn['box'].draw()
        btn['text'].draw()
def setup_trial_visuals(trial, components, label_data, img_dir, demo_mode, target_button_idx=None):
    """Common setup logic for both timing modes.

    Responsibilities:
      - Resolve and preload the trial image if present (will set main_image).
      - Determine the target label and sample 3 distractors from the participant's pool.
      - Shuffle choices and write them into the on-screen button text objects.
      - Optionally set demo text showing the trial condition for experimenter debugging.

    Returns:
      (has_img, img_path, target_label, choices_list)
    """
    space_id = get_trial_space_id(trial)
    if space_id is None:
        # This is likely a fixation-only row or malformed trial.
        # You can return a safe placeholder or raise an error depending on your preference.
        space_id = "NA"

    cond = str(trial['condition'])
    # Treat explicit fixation rows as non-response events
    is_fixation = (cond.strip().lower() == "fixation") or (str(trial.get("event_type", "")).strip().lower() == "fixation")

    if is_fixation:
        # Fixation-only row: no image, no labels, no responses.
        # Clear button labels so nothing is shown/clickable.
        for btn in components['buttons']:
            btn['text'].text = ""
        return False, "", "", ["", "", "", ""], True

    # 1) Image
    img_path = resolve_image_path(img_dir, space_id, trial.get('image_file', ''))
    if img_path and os.path.exists(img_path):
        components['main_image'].setImage(img_path)
        has_img = True
    else:
        # Keep a best-guess path (may be used for logging) but mark as missing visually
        has_img = False
        img_path = img_path or os.path.join(img_dir, str(trial.get('image_file', '')))

    # 2) Labels
    label_map, all_labels = label_data
    # target label is looked up by space and condition; missing mapping is explicit 'MISSING'
    target_lbl = label_map.get(space_id, {}).get(cond, "MISSING")
    if (not is_fixation) and (target_lbl is None or str(target_lbl).strip() == "" or str(target_lbl).strip().lower() in ("missing", "nan")):
        raise ValueError(
            f"Missing target label for space_id={space_id!r}, condition={cond!r}. "
            f"Check ObjectSpace normalization and label CSV completeness."
        )


    # Build distractor pool restricted to the participant's label set (precomputed)
    pool = [l for l in all_labels if l != target_lbl]


    if len(pool) >= 3:
        distractors = random.sample(pool, 3)
    else:
        # If there are fewer than 3 alternatives, pad with 'NA' placeholders
        distractors = pool[:]
        while len(distractors) < 3:
            distractors.append("NA")

    choices = [target_lbl] + distractors
    # By default, keep the original behaviour: shuffle choice order freely.
    # If a target button index is provided, place the target label on that button
    # and shuffle only the distractors across the remaining buttons.
    if target_button_idx is None:
        random.shuffle(choices)
    else:
        try:
            tb = int(target_button_idx)
        except Exception:
            tb = None
        if tb is None or tb < 0 or tb >= len(KEYS_RESP):
            random.shuffle(choices)
        else:
            choices_fixed = [None] * len(KEYS_RESP)
            choices_fixed[tb] = target_lbl
            other_idx = [i for i in range(len(KEYS_RESP)) if i != tb]
            distractors_shuf = distractors[:]
            random.shuffle(distractors_shuf)
            for i, bi in enumerate(other_idx):
                choices_fixed[bi] = distractors_shuf[i] if i < len(distractors_shuf) else ""
            choices = choices_fixed

    # Apply choices to button text
    for i, btn in enumerate(components['buttons']):
        btn['text'].text = choices[i] if i < len(choices) else ""

    # 3) Demo Text for live debugging / experimenter view
    if demo_mode:
        components['demo_text'].text = f"Condition:\n{cond}"

    return has_img, img_path, target_lbl, choices, False


def check_response(components, clock, start_time):
    """Poll for keyboard responses (primary) and mouse clicks (fallback).

    Returns tuple (key, rt) where rt is relative to start_time. If no response
    is detected returns (None, None). If EXIT_KEY is pressed, terminates experiment.
    """
    keys = event.getKeys(keyList=KEYS_RESP + [EXIT_KEY], timeStamped=clock)
    if keys:
        k, t = keys[0]
        if k == EXIT_KEY:
            core.quit()
        return k, t - start_time

    # Mouse fallback: return the key associated with the clicked button if any
    if components['mouse'].getPressed()[0]:
        for btn in components['buttons']:
            if btn['box'].contains(components['mouse']):
                return btn['key'], clock.getTime() - start_time
    return None, None


def draw_extras(components, has_img, demo_mode):
    """Helper to draw trial image or 'missing' text and demo annotations.

    Keeps calling code concise while centralising display behaviour for image
    presence and demo annotations.
    """
    if has_img:
        components['main_image'].draw()
    else:
        components['missing_text'].draw()

    if demo_mode:
        components['demo_text'].draw()


def _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials, clock, trial_t0,
                      phase_name, phase_end=None, breakdown_lines=None):
    """Update and draw demo HUD.

    Left: task timer + trial counter.
    Right: trial timer + current phase + per-event countdowns (approx).

    This HUD is only active when demo_mode is True and is intended to help the
    experimenter or developer observe scheduling progress. It should not affect
    experiment timing since PsychoPy clocks control the actual flow.
    """
    if not demo_mode:
        return

    now = clock.getTime()
    task_t = task_clock.getTime() if task_clock is not None else 0.0
    trial_elapsed = max(0.0, now - trial_t0)

    left = f"Task: {task_t:6.1f}s\nTrial: {trial_idx}/{n_trials}"
    components['hud_left'].text = left

    phase_rem = ""
    if phase_end is not None:
        phase_rem = f" ({max(0.0, phase_end - now):4.1f}s left)"

    right_lines = [f"Trial: {trial_elapsed:6.1f}s", f"{phase_name}{phase_rem}"]
    if breakdown_lines:
        right_lines.append("")
        right_lines.extend(breakdown_lines)

    components['hud_right'].text = "\n".join(right_lines)

    components['hud_left'].draw()
    components['hud_right'].draw()


def _demo_skip_pressed(demo_mode):
    """Return True if the demo skip key was pressed (Demo Mode only)."""
    if not demo_mode:
        return False
    return bool(event.getKeys(keyList=[SKIP_KEY]))


def _return_skipped(img_path):
    """Standard result payload for a skipped trial.

    Used to produce a consistent row structure when demo skipping is used; the
    'skipped' flag can be used downstream to filter demo trials from analysis.
    """
    return {
        'response': None, 'rt': None,
        'target': None, 'correct_key': None,
        'accuracy': 0,
        'img': img_path,
        'skipped': 1
    }

# =========================== TRIALS =========================== #

def run_trial_standard(win, clock, trial, components, label_data, img_dir, demo_mode,
                       task_clock=None, trial_idx=1, n_trials=1, target_button_idx=None):
    """Restored 2-Event Logic with new visuals.

    Standard mode timeline is absolute: onsets in the design CSV are respected
    relative to a run clock. This function loops through phases using the
    provided PsychoPy clock to maintain precise timing.
    """
    has_img, img_path, target, choices, is_fixation = setup_trial_visuals(trial, components, label_data, img_dir, demo_mode, target_button_idx=target_button_idx)

    # Identify correct key
    try:
        target_idx = choices.index(target)
        correct_key = KEYS_RESP[target_idx]
    except ValueError:
        correct_key = None

    # Parse timing fields from the trial row (expected to exist in standard mode)
    t_dec_on  = float(trial['dec_onset'])
    t_dec_dur = float(trial['dec_dur'])
    t_fb_on   = float(trial['dec_fb_onset'])
    t_fb_dur  = float(trial['dec_fb_dur'])
    t_end     = float(trial['trial_onset']) + float(trial['trial_duration'])

    trial_t0 = float(trial.get('trial_onset', clock.getTime()))

    # --- FIXATION-ONLY TRIAL (condition == 'fixation') ---
    # Respect absolute timing from the design file in standard mode by drawing
    # a fixation cross until the trial end time.

    if is_fixation:
        # Fixation-only trial: draw an explicit '+' for the duration specified in the design.
        # Priority for duration: fix_dur (if present) -> (trial_onset+trial_duration) fallback.
        now = clock.getTime()
        # Preferred: use trial_onset + fix_dur if available (absolute schedule)
        end_fix = None
        try:
            if 'fix_dur' in trial and not pd.isna(trial.get('fix_dur')):
                end_fix = float(trial.get('trial_onset', now)) + float(trial.get('fix_dur'))
        except Exception:
            end_fix = None
        # Fallback: use standard end-of-trial timing from the design
        if end_fix is None:
            end_fix = t_end

        # Guarantee at least one visible frame even if duration is 0 or timing already passed
        if now >= end_fix:
            end_fix = now + (1.0 / 60.0)

        while clock.getTime() < end_fix:
            if event.getKeys(keyList=[EXIT_KEY]):
                core.quit()
            components.get('fixation_trial', components['fixation']).draw()
            _hud_set_and_draw(
                components, demo_mode, task_clock, trial_idx, n_trials,
                clock, trial_t0, 'Fixation', end_fix, breakdown_lines=["Fixation-only trial"]
            )
            win.flip()

        return {
            'response': None, 'rt': None,
            'target': None, 'correct_key': None,
            'accuracy': 0,
            'img': img_path,
            'skipped': 0
        }
    # Pre-Dec Fixation: render fixation until decision onset
    while clock.getTime() < t_dec_on:
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            # Fast-forward schedule to end of trial (keeps standard timing aligned)
            dt = max(0.0, t_end - clock.getTime())
            if dt > 0:
                clock.addTime(dt)
            return _return_skipped(img_path)

        components['fixation'].draw()
        breakdown = [
            f"Dec: {max(0.0, (t_dec_on + t_dec_dur) - clock.getTime()):4.1f}s",
            f"Jit: {max(0.0, t_fb_on - max(clock.getTime(), (t_dec_on + t_dec_dur))):4.1f}s",
            f"Fb : {max(0.0, (t_fb_on + t_fb_dur) - max(clock.getTime(), t_fb_on)):4.1f}s",
            f"ITI: {max(0.0, t_end - max(clock.getTime(), (t_fb_on + t_fb_dur))):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Fixation', t_dec_on, breakdown)
        win.flip()

    # Decision phase: collect keyboard/mouse responses while drawing choices
    components['mouse'].clickReset()
    event.clearEvents()
    response_made = None
    rt = None

    while clock.getTime() < (t_dec_on + t_dec_dur):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            dt = max(0.0, t_end - clock.getTime())
            if dt > 0:
                clock.addTime(dt)
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)

        if response_made is None:
            response_made, rt = check_response(components, clock, t_dec_on)

        draw_buttons(components['buttons'], response_made)
        breakdown = [
            f"Dec: {max(0.0, (t_dec_on + t_dec_dur) - clock.getTime()):4.1f}s",
            f"Jit: {max(0.0, t_fb_on - max(clock.getTime(), (t_dec_on + t_dec_dur))):4.1f}s",
            f"Fb : {max(0.0, (t_fb_on + t_fb_dur) - max(clock.getTime(), t_fb_on)):4.1f}s",
            f"ITI: {max(0.0, t_end - max(clock.getTime(), (t_fb_on + t_fb_dur))):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Decision', (t_dec_on + t_dec_dur), breakdown)
        win.flip()

    # Pre-Feedback Fixation: wait until feedback onset
    while clock.getTime() < t_fb_on:
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            dt = max(0.0, t_end - clock.getTime())
            if dt > 0:
                clock.addTime(dt)
            return _return_skipped(img_path)

        components['fixation'].draw()
        breakdown = [
            "Dec:  0.0s",
            f"Jit: {max(0.0, t_fb_on - clock.getTime()):4.1f}s",
            f"Fb : {max(0.0, (t_fb_on + t_fb_dur) - max(clock.getTime(), t_fb_on)):4.1f}s",
            f"ITI: {max(0.0, t_end - max(clock.getTime(), (t_fb_on + t_fb_dur))):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Jitter', t_fb_on, breakdown)
        win.flip()

    # Feedback: show correct/incorrect highlighting for the configured duration
    while clock.getTime() < (t_fb_on + t_fb_dur):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            dt = max(0.0, t_end - clock.getTime())
            if dt > 0:
                clock.addTime(dt)
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)
        draw_buttons_feedback(components['buttons'], response_made, correct_key)
        if response_made is None:
            components['respond_text'].draw()
        breakdown = [
            "Dec:  0.0s",
            "Jit:  0.0s",
            f"Fb : {max(0.0, (t_fb_on + t_fb_dur) - clock.getTime()):4.1f}s",
            f"ITI: {max(0.0, t_end - max(clock.getTime(), (t_fb_on + t_fb_dur))):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Feedback', (t_fb_on + t_fb_dur), breakdown)
        win.flip()

    # ITI: wait until trial end before returning results
    while clock.getTime() < t_end:
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            dt = max(0.0, t_end - clock.getTime())
            if dt > 0:
                clock.addTime(dt)
            return _return_skipped(img_path)

        components['fixation'].draw()
        breakdown = [
            "Dec:  0.0s",
            "Jit:  0.0s",
            "Fb :  0.0s",
            f"ITI: {max(0.0, t_end - clock.getTime()):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'ITI', t_end, breakdown)
        win.flip()

    return {
        'response': response_made, 'rt': rt,
        'target': target, 'correct_key': correct_key,
        'accuracy': 1 if response_made == correct_key else 0,
        'img': img_path,
        'skipped': 0
    }


def run_trial_3event(win, clock, trial, components, label_data, img_dir, demo_mode, feedback_delay,
                     fixed_decision_time,
                     task_clock=None, trial_idx=1, n_trials=1, target_button_idx=None):
    """3-event (self-paced) trial.

    Phases:
      - Image + ISI1 (image remains visible)
      - Decision (self-paced up to max_dec_dur)
      - Optional hidden ISI2 delay (only if feedback_delay and responded)
      - Feedback
      - ITI (fixation)

    The function implements a slightly more flexible flow than standard mode and
    supports a self-paced decision phase where the trial advances immediately
    after a response (useful for reaction-time based tasks).
    """
    has_img, img_path, target, choices, is_fixation = setup_trial_visuals(trial, components, label_data, img_dir, demo_mode, target_button_idx=target_button_idx)

    # Identify correct key based on shuffled choices
    try:
        target_idx = choices.index(target)
        correct_key = KEYS_RESP[target_idx]
    except ValueError:
        correct_key = None

    # Durations (seconds) -- use fallbacks for compatibility with older design files
    img_dur = float(trial.get('img_dur', 0))
    isi1_dur = float(trial.get('isi1_dur', 0))
    max_dec_dur = float(trial.get('max_dec_dur', trial.get('dec_dur', 0)))
    isi2_dur = float(trial.get('isi2_dur', 0))
    fb_dur = float(trial.get('fb_dur', trial.get('dec_fb_dur', 0)))
    iti = float(trial.get('iti', 0))

    # --- FIXATION-ONLY TRIAL (condition == 'fixation') ---
    # In 3-event mode, fixation rows should present only a fixation cross for the
    # duration specified in the design (prefer trial_duration_max/trial_duration; fall back to iti).

    if is_fixation:
        # Mark trial start for HUD/debug and ensure at least one visible frame
        trial_t0 = clock.getTime()

        # Prefer explicit per-trial duration fields from the design file.
        # (trial_duration_max is commonly present in generated 3-event designs.)
        # Use fix_dur when provided for fixation-only trials (authoritative)
        dur = trial.get('fix_dur', trial.get('trial_duration_max', trial.get('trial_duration', iti)))
        try:
            dur = float(dur)
            # Treat NaN as 0 (common when CSV has empty cells)
            if isinstance(dur, float) and np.isnan(dur):
                dur = 0.0

        except Exception:
            dur = float(iti) if iti is not None else 0.0

        # Ensure fixation is visible for at least one refresh, even if duration is 0.
        min_frame = 1.0 / 60.0
        if dur <= 0:
            dur = min_frame

        t_fix = clock.getTime()
        end_fix = t_fix + dur
        while clock.getTime() < end_fix:
            if event.getKeys(keyList=[EXIT_KEY]):
                core.quit()
            components.get('fixation_trial', components['fixation']).draw()
            _hud_set_and_draw(
                components, demo_mode, task_clock, trial_idx, n_trials,
                clock, trial_t0, 'Fixation', end_fix, breakdown_lines=["Fixation-only trial"]
            )
            win.flip()

        return {
            'response': None, 'rt': None,
            'target': None, 'correct_key': None,
            'accuracy': 0,
            'img': img_path,
            'skipped': 0
        }

    # Ensure clean input state for upcoming trial (clear any residual clicks/keys)
    components['mouse'].clickReset()
    event.clearEvents()

    # Demo HUD trial start
    trial_t0 = clock.getTime()

    # 1) Encoding + ISI1 (image visible throughout)
    t0 = trial_t0
    while clock.getTime() < (t0 + img_dur + isi1_dur):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)
        breakdown = [
            f"Img+ISI1: {max(0.0, (t0 + img_dur + isi1_dur) - clock.getTime()):4.1f}s",
            f"Dec: {max_dec_dur:4.1f}s (max)",
            f"ISI2: {isi2_dur:4.1f}s" if feedback_delay else "ISI2: 0.0s",
            f"Fb : {fb_dur:4.1f}s",
            f"ITI: {iti:4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Encoding', (t0 + img_dur + isi1_dur), breakdown)
        win.flip()

    # --- IMPORTANT: reset input state right at decision onset so earlier presses
    # do not carry over from the encoding phase. This ensures responses only have
    # an effect while the buttons are visible.
    components['mouse'].clickReset()
    event.clearEvents()

    # 2) Decision (self-paced)
    t_dec = clock.getTime()
    dec_end = t_dec + max_dec_dur
    response_made, rt = None, None

    while clock.getTime() < dec_end:
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)

        if response_made is None:
            response_made, rt = check_response(components, clock, t_dec)
            if response_made and (not fixed_decision_time):
                # Self-paced advance: show selection for one frame then move on immediately
                draw_buttons(components['buttons'], response_made)

                breakdown = [
                    f"Dec: {max(0.0, dec_end - clock.getTime()):4.1f}s",
                    f"ISI2: {isi2_dur:4.1f}s" if feedback_delay else "ISI2: 0.0s",
                    f"Fb : {fb_dur:4.1f}s",
                    f"ITI: {iti:4.1f}s",
                ]
                _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                                  clock, trial_t0, 'Decision', dec_end, breakdown)
                win.flip()
                break

        # In fixed decision-time mode, keep the decision screen up until dec_end,
        # even if a response was already made.
        draw_buttons(components['buttons'], response_made)

        breakdown = [
            f"Dec: {max(0.0, dec_end - clock.getTime()):4.1f}s",
            f"ISI2: {isi2_dur:4.1f}s" if feedback_delay else "ISI2: 0.0s",
            f"Fb : {fb_dur:4.1f}s",
            f"ITI: {iti:4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Decision', dec_end, breakdown)
        win.flip()
# 3) Optional hidden ISI2 delay (keep image + selected button on screen)
    if feedback_delay and response_made:
        t_isi2 = clock.getTime()
        while clock.getTime() < (t_isi2 + isi2_dur):
            if event.getKeys(keyList=[EXIT_KEY]):
                core.quit()
            if _demo_skip_pressed(demo_mode):
                return _return_skipped(img_path)

            draw_extras(components, has_img, demo_mode)
            draw_buttons(components['buttons'], response_made)
            breakdown = [
                f"ISI2: {max(0.0, (t_isi2 + isi2_dur) - clock.getTime()):4.1f}s",
                f"Fb : {fb_dur:4.1f}s",
                f"ITI: {iti:4.1f}s",
            ]
            _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                              clock, trial_t0, 'ISI2', (t_isi2 + isi2_dur), breakdown)
            win.flip()

    # 4) Feedback
    t_fb = clock.getTime()
    while clock.getTime() < (t_fb + fb_dur):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)
        draw_buttons_feedback(components['buttons'], response_made, correct_key)
        if response_made is None:
            components['respond_text'].draw()
        breakdown = [
            f"Fb : {max(0.0, (t_fb + fb_dur) - clock.getTime()):4.1f}s",
            f"ITI: {iti:4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Feedback', (t_fb + fb_dur), breakdown)
        win.flip()

    # 5) ITI (fixation)
    t_iti = clock.getTime()
    while clock.getTime() < (t_iti + iti):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            return _return_skipped(img_path)

        components['fixation'].draw()
        breakdown = [
            f"ITI: {max(0.0, (t_iti + iti) - clock.getTime()):4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'ITI', (t_iti + iti), breakdown)
        win.flip()

    return {
        'response': response_made, 'rt': rt,
        'target': target, 'correct_key': correct_key,
        'accuracy': 1 if (correct_key is not None and response_made == correct_key) else 0,
        'img': img_path,
        'skipped': 0
    }

# =========================== WINDOW/COMPONENT FACTORY =========================== #

def create_window_and_components(demo_mode):
    """Create a fullscreen window and all visual components tied to it.

    The window uses 'height' units which makes stimulus sizes expressed as
    fractions of the screen height. This provides reasonable portability across
    different screen resolutions. Components (TextStim/ImageStim/Rect) are
    created here and returned in a dict for convenience.
    """
    win = visual.Window([1280, 720], fullscr=True, color='lightgrey', units='height', allowGUI=False, useRetina=True)

    components = {
        'fixation': visual.TextStim(win, text='+', height=0.1, color='black'),
        # Dedicated fixation stimulus for explicit fixation trials (condition == 'fixation')
        'fixation_trial': visual.TextStim(win, text='+', height=0.12, color='black'),
        # Image centered at (0,0) as requested
        'main_image': visual.ImageStim(win, pos=(0, 0), size=(0.5, 0.5), interpolate=True, texRes=2048),
        # Messages (language-specific)
        'missing_text': visual.TextStim(
            win,
            text=('Img Missing' if LANGUAGE == 'japanese' else 'Img Missing'),
            pos=(0, 0),
            height=0.05,
            color='red'
        ),
        'respond_text': visual.TextStim(
            win,
            text=('回答してください' if LANGUAGE == 'japanese' else 'Respond'),
            pos=(0, 0.4),
            height=0.05,
            color='red'
        ),
        # Demo text on center-right
        'demo_text': visual.TextStim(win, text='', pos=(0.4, 0), height=0.04, color=DEMO_TEXT_COL),
        # Demo HUD (only drawn when Demo Mode is enabled)
        'hud_left': visual.TextStim(win, text='', pos=(-0.48, 0.48), height=0.03, color='black',
                                    alignText='left', anchorHoriz='left', anchorVert='top'),
        'hud_right': visual.TextStim(win, text='', pos=(0.48, 0.48), height=0.03, color='black',
                                     alignText='right', anchorHoriz='right', anchorVert='top'),
        'mouse': event.Mouse(win=win),
        'buttons': []
    }

    # Setup Buttons: create four rectangular buttons and centered labels
    font_scale = LANG_BTN_FONT_SCALE.get(LANGUAGE, 1.0)
    spacing = 0.22
    start_x = -((3 * spacing) / 2)
    for i, key in enumerate(KEYS_RESP):
        x = start_x + (i * spacing)
        box = visual.Rect(win, width=0.2, height=0.1, pos=(x, -0.3),
                          fillColor=COL_NEUTRAL, lineColor='black', lineWidth=LINE_W_NORMAL)
        txt = visual.TextStim(win, text=f"{key}", pos=(x, -0.3), height=0.03*font_scale, color='black')
        components['buttons'].append({'box': box, 'text': txt, 'key': key})

    return win, components


def trigger_screen(win, components, mode, demo_mode, run_idx=1, n_runs=1, run_label=None):
    """Interactive trigger screen: waits for TRIGGER_KEY.

    Displays run/mode/demo information and allows the operator to press a
    response key to visually check button mapping before the run begins. The
    function blocks until TRIGGER_KEY is pressed or EXIT_KEY is used to abort.
    """
    if LANGUAGE == "japanese":
        run_line = f"施行: {run_idx}/{n_runs}"
    else:
        run_line = f"Run: {run_idx}/{n_runs}"

    if run_label is not None:
        if LANGUAGE == "japanese":
            run_line += f" (値: {run_label})"
        else:
            run_line += f" (value: {run_label})"

    if LANGUAGE == "japanese":
        msg_text = (
            f"{run_line}\n"
            f"ボタンを押してテストします。\n\n\n"
            f"準備してください！\n\n"
            f"MRI装置の起動を待っています\n"
            + (f"Demo skip: RIGHT ARROW\n" if demo_mode else "")
        )
    else:
        msg_text = (
            f"{run_line}\n"
            f"Press the buttons to test.\n\n\n"
            f"Be ready!\n\n"
            f"Waiting for scanner to start\n"
            + (f"Demo skip: RIGHT ARROW\n" if demo_mode else "")
        )

    msg = visual.TextStim(
        win,
        text=msg_text,
        pos=(0, 0.3),
        height=0.04,
        color='black'
    )


    triggered = False
    event.clearEvents()

    while not triggered:
        msg.draw()
        pressed = event.getKeys()
        if TRIGGER_KEY in pressed:
            triggered = True
        elif EXIT_KEY in pressed:
            core.quit()

        # Visualise any pressed response key by temporarily filling the matching button
        hl_key = None
        for k in pressed:
            if k in KEYS_RESP:
                hl_key = k

        for btn in components['buttons']:
            if hl_key == btn['key']:
                btn['box'].fillColor = COL_SELECT_FILL
            else:
                btn['box'].fillColor = COL_NEUTRAL
            btn['box'].draw()

        win.flip()

# =========================== MAIN =========================== #


def _sanitize_run_label(label):
    """Make a safe filename token from a run/block label.

    Removes characters that may be problematic in filenames and truncates the
    result to a reasonable length to avoid path issues on different OSes.
    """
    if label is None:
        return "NA"
    s = str(label).strip()
    if not s:
        return "NA"
    # Keep only safe chars
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    return s[:64]  # keep it short


def _safe_write_csv(df, path):
    """Write CSV atomically to reduce risk of corruption on crash.

    Writes to a temporary file then renames it into place which is an atomic
    operation on most platforms and reduces the chance of partial files being
    left behind if the process is terminated while writing.
    """
    path = str(path)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)



class _TeeStream:
    """Lightweight stdout/stderr tee to also write console output to a file.

    Uses line-buffered writes and flushes to keep logs readable without heavy overhead.
    """
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        # Flush on newline to keep file readable in crashes
        if isinstance(s, str) and ("\n" in s):
            self.flush()

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

def _freeze_clock_for_io(clock, fn, *args, **kwargs):
    """Run IO-heavy code without shifting the task's *design clock*.

    In standard mode, trial timing is absolute relative to run_clock. Any delay caused
    by file IO between trials would otherwise shift subsequent trial onsets. This helper
    measures the elapsed time spent in `fn` and subtracts it back from the provided clock.

    For non-standard modes this is still harmless and preserves the intended pacing.
    """
    if clock is None:
        return fn(*args, **kwargs)

    try:
        t0 = clock.getTime()
    except Exception:
        return fn(*args, **kwargs)

    out = fn(*args, **kwargs)

    try:
        t1 = clock.getTime()
        dt = float(t1 - t0)
        if dt > 0:
            clock.addTime(-dt)
    except Exception:
        pass

    return out



def run_experiment():
    # 1. Start Dialog: basic participant/run configuration
    info = {
        'Sub': '001',
        'Language': 'English',  # English or Japanese
        'Design CSV': '',        # Leave blank to browse
        'Label CSV': '',         # Leave blank to browse
        'Image Dir': 'images/task_images/',   # Default directory
        'Feedback Delay': True, # Selection from previous requirements
        'Fixed Decision Time': True,  # When True, always wait max_dec_dur before ISI2/Feedback
        'Demo Mode': False,
        # Tickbox: when True use scanner button mapping (1-4). When False use PC mapping (1,2,9,0)
        'Scanner Buttons': True,
    }

    # Define a helper for browsing (uses PsychoPy's GUI helpers)
    def browse_for_path(prompt, folder=False):
        if folder:
            path = gui.dirSelectDialog(prompt=prompt)
        else:
            path = gui.fileOpenDlg(prompt=prompt, allowed="CSV files (*.csv);;All files (*.*)")
        return path[0] if path else None

    # Initial Setup GUI: allow operator to edit defaults and choose files
    dlg = gui.DlgFromDict(
        info,
        title='Study 3 Launcher',
        order=['Sub', 'Language', 'Design CSV', 'Label CSV', 'Image Dir', 'Feedback Delay', 'Fixed Decision Time', 'Demo Mode', 'Scanner Buttons'],
        tip={
            'Design CSV': 'Leave blank to open file browser',
            'Label CSV': 'Leave blank to open file browser',
            'Image Dir': 'Directory containing stimulus images'
        }
    )

    if not dlg.OK:
        core.quit()
    # Language selection (affects on-screen texts)
    global LANGUAGE
    LANGUAGE = 'japanese' if str(info.get('Language', 'English')).strip().lower().startswith('jap') else 'english'



    # 2. Trigger Browser for Empty Fields: ensure all required paths are set
    if not info['Design CSV']:
        info['Design CSV'] = browse_for_path("Select Design CSV")
    if not info['Label CSV']:
        info['Label CSV'] = browse_for_path("Select Label CSV")

    # Optional: Browse for image directory if default doesn't exist
    if not os.path.exists(info['Image Dir']):
        print(f"Warning: {info['Image Dir']} not found. Please select image directory.")
        info['Image Dir'] = browse_for_path("Select Image Directory", folder=True)

    # 3. Final Validation: ensure required inputs were provided
    if not info['Design CSV'] or not info['Label CSV'] or not info['Image Dir']:
        print("Startup Error: Missing required file paths or directories.")
        core.quit()

    # Convert GUI values to booleans robustly (handles both booleans and strings)
    def _to_bool(val):
        if isinstance(val, bool):
            return val
        if val is None:
            return False
        s = str(val).strip().lower()
        return s in ('1', 'true', 't', 'yes', 'y', 'on')

    demo_mode = _to_bool(info.get('Demo Mode'))
    feedback_delay = _to_bool(info.get('Feedback Delay'))
    fixed_decision_time = _to_bool(info.get('Fixed Decision Time'))
    # Determine button mapping based on scanner toggle (default: scanner mapping)
    use_scanner_buttons = _to_bool(info.get('Scanner Buttons'))
    # Update global keys mapping so other functions use the selected mapping
    global KEYS_RESP
    if use_scanner_buttons:
        KEYS_RESP = ['1', '2', '3', '4']
    else:
        KEYS_RESP = ['1', '2', '9', '0']

    
    # --------------------------- OUTPUT / LOGGING --------------------------- #
    # Create a clean, per-participant output folder with a per-session timestamp.
    session_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sub_token = str(info['Sub']).strip()
    base_out_dir = os.path.join('participant_data', f"sub-{sub_token}", f"session-{session_ts}")
    dir_logs = os.path.join(base_out_dir, "logs")
    dir_ckpt9 = os.path.join(base_out_dir, "checkpoints_9trials")
    dir_runs = os.path.join(base_out_dir, "per_run")
    dir_final = os.path.join(base_out_dir, "final")

    for d in (dir_logs, dir_ckpt9, dir_runs, dir_final):
        os.makedirs(d, exist_ok=True)

    # Tee all console output to a participant/session log for crash forensics.
    # Keep it lightweight; writing is line-buffered and shouldn't affect timing.
    console_log_path = os.path.join(dir_logs, f"console_sub-{sub_token}_{session_ts}.txt")
    try:
        _log_fh = open(console_log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, _log_fh)
        sys.stderr = _TeeStream(sys.__stderr__, _log_fh)
        print(f"[info] Console logging to: {console_log_path}")
    except Exception as _e:
        _log_fh = None
        print(f"[warn] Could not set up console logging: {_e}")

    try:
        # Load design and label files and build participant-specific pools
        design_df = pd.read_csv(info['Design CSV'])
        print("design csv successfully loaded")
        label_map, _all_labels = load_label_map(info['Label CSV'], key_col='ObjectSpace')
        participant_labels = build_participant_label_pool(design_df, label_map)
        label_data = (label_map, participant_labels)
        print("label csv successfully loaded")
        _ = scan_images(info['Image Dir'])

        # Determine mode based on design file columns: presence of 'img_dur' indicates 3-event
        mode = '3event' if 'img_dur' in design_df.columns else 'standard'
        run_col = detect_run_column(design_df)
        runs = split_into_runs(design_df, run_col)
        print(f"Mode detected: {mode} | Demo: {demo_mode} | Feedback Delay: {feedback_delay}")
        if run_col is None:
            print("No run/block column detected; treating entire design as a single run.")
        else:
            print(f"Run grouping detected via column: '{run_col}' | Runs: {len(runs)}")
    except Exception as e:
        print(f"Startup Error during file loading: {e}")
        core.quit()

    # Output rows (one per trial) will be accumulated and written per-run for crash safety
    out_rows = []

    n_runs = len(runs)


    # Create first fullscreen window and start as-is (run 1 unchanged)
    win, components = create_window_and_components(demo_mode)
    # Show initial experimental instructions
    # In scanner mode, do NOT mention specific key numbers on instruction screens.
    if LANGUAGE == "japanese":
        if use_scanner_buttons:
            hand_text = "指を応答ボタンの上に置いたままにします。"
        else:
            keys_str = ", ".join(f"'{k}'" for k in KEYS_RESP)
            hand_text = f"指を {keys_str} キーの上に置いたままにします。"
    else:
        if use_scanner_buttons:
            hand_text = "Keep your fingers placed on the response buttons."
        else:
            keys_str = ", ".join(f"'{k}'" for k in KEYS_RESP)
            hand_text = f"Keep your fingers placed on the {keys_str} keys."

    if LANGUAGE == "japanese":
        instr1 = (
            "実験セッション\n\n"
            "メインオブジェクトの学習タスクを開始しようとしています。\n"
            "できるだけ正確かつ迅速にオブジェクトを分類することを忘れないでください。\n\n"
            f"{hand_text}"
        )
    else:
        instr1 = (
            "EXPERIMENTAL SESSION\n\n"
            "You are about to start the main object learning task.\n"
            "Remember to categorize the objects as accurately and fast as possible.\n\n"
            f"{hand_text}"
        )

    show_instruction_screen(
        win,
        instr1,
        image_path="experimental_task/resources/instruction_image_scanner.png",
        use_scanner_buttons=use_scanner_buttons
    )

    if LANGUAGE == "japanese":
        instr2 = (
            "これから学ぶ物体は、異星から来たものです。\nこれまでに見たことのあるものと似ているものもあれば、\n似ていないものもあります。\n\n "
            "実験中、オブジェクトの名前は変更されません。"
        )
    else:
        instr2 = (
            "The objects you are going to learn are from an alien planet. Some of them can look similar to one's you've seen before "
            "while others will not.\n\n"
            "The names of the objects won't change during the experiment."
        )

    show_instruction_screen(
        win,
        instr2,
        image_path="experimental_task/resources/alien_image.png",
        use_scanner_buttons=use_scanner_buttons
    )
    trigger_screen(win, components, mode, demo_mode, run_idx=1, n_runs=n_runs, run_label=runs[0][0])

    for run_idx, (run_label, run_df) in enumerate(runs, start=1):
        # Between-run flow (run 2+): close fullscreen -> GUI -> reopen fullscreen -> trigger screen
        if run_idx > 1:
            try:
                win.close()
            except Exception:
                pass
            between_run_dialog(run_idx, run_label=run_label)

            win, components = create_window_and_components(demo_mode)
            trigger_screen(win, components, mode, demo_mode, run_idx=run_idx, n_runs=n_runs, run_label=run_label)

        # Reset clocks per run (important for standard-mode absolute onsets)
        run_clock = core.Clock()
        run_clock.reset()
        task_clock = core.Clock()
        task_clock.reset()

        trials = run_df.to_dict('records')
        n_trials = len(trials)
        # Counterbalance correct-option button position within this run
        target_btn_seq = generate_balanced_button_sequence(n_trials, n_buttons=len(KEYS_RESP), max_repeat=2, rng=random)
        run_rows = []  # per-run buffer for crash-safe saving

        for trial_idx, trial in enumerate(trials, start=1):
            if mode == 'standard':
                res = run_trial_standard(
                    win, run_clock, trial, components, label_data, info['Image Dir'],
                    demo_mode, task_clock=task_clock, trial_idx=trial_idx, n_trials=n_trials,
                    target_button_idx=target_btn_seq[trial_idx-1]
                )
            else:
                res = run_trial_3event(
                    win, run_clock, trial, components, label_data, info['Image Dir'],
                    demo_mode, feedback_delay, fixed_decision_time,
                    task_clock=task_clock, trial_idx=trial_idx, n_trials=n_trials,
                    target_button_idx=target_btn_seq[trial_idx-1]
                )

            row = dict(trial)
            row.update(res)
            if 'skipped' not in row:
                row['skipped'] = 0
            if run_col is not None and run_col not in row:
                row[run_col] = run_label
            run_rows.append(row)
            out_rows.append(row)

            # Checkpoint save every 9 trials (global, across runs) for crash robustness.
            # Use _freeze_clock_for_io to avoid shifting the design clock (especially in standard mode).
            if len(out_rows) % 9 == 0:
                ckpt_path = os.path.join(dir_ckpt9, f"sub-{sub_token}_{mode}_ckpt-trials-{len(out_rows):05d}_{session_ts}.csv")
                def _write_ckpt():
                    ckpt_df = pd.DataFrame(out_rows)
                    _safe_write_csv(ckpt_df, ckpt_path)
                _freeze_clock_for_io(run_clock, _write_ckpt)

        # Save results for this run immediately (crash-safe).
        # This ensures we still have earlier runs if the task crashes later.
        if len(run_rows) > 0:
            run_token = _sanitize_run_label(run_label)
            run_path = os.path.join(dir_runs, f"sub-{sub_token}_{mode}_run-{run_idx:02d}_{run_token}_{session_ts}.csv")
            def _write_run():
                run_out_df = pd.DataFrame(run_rows)
                _safe_write_csv(run_out_df, run_path)
            _freeze_clock_for_io(run_clock, _write_run)

    out_df = pd.DataFrame(out_rows)
    # Final joined file across all runs (crash-safe)
    joined_path = os.path.join(dir_final, f"sub-{sub_token}_{mode}_joined_{session_ts}.csv")
    _freeze_clock_for_io(None, _safe_write_csv, out_df, joined_path)
    # Also write/keep the legacy filename for compatibility
    legacy_path = os.path.join(dir_final, f"sub-{sub_token}_{mode}_{session_ts}.csv")
    _freeze_clock_for_io(None, _safe_write_csv, out_df, legacy_path)

    try:
        win.close()
    except Exception:
        pass

    # Close console log file handle if we created one
    try:
        if '_log_fh' in locals() and _log_fh is not None:
            _log_fh.close()
    except Exception:
        pass

    core.quit()

if __name__ == "__main__":
    run_experiment()