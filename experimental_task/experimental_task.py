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

# =========================== CONFIGURATION =========================== #
# Keys used by the task. Numeric keys map to response options; TRIGGER_KEY used
# to start a run (scanner trigger simulation). EXIT_KEY exits the experiment.
KEYS_RESP = ['1', '2', '9', '0']
TRIGGER_KEY = '5'
EXIT_KEY = 'escape'
SKIP_KEY = 'right'  # Demo Mode only

# Colors used for buttons / feedback. Keep names descriptive to make intent clear.
COL_NEUTRAL   = 'grey'
COL_HOVER     = 'lightgrey'
COL_SELECT    = 'green'      # Color during selection (not used for fill)
COL_CORRECT   = 'green'      # Feedback Correct (border)
COL_INCORRECT = 'red'        # Feedback Incorrect (border)
TEXT_COLOR    = 'black'
DEMO_TEXT_COL = 'blue'

# Button line widths: normal and highlighted for feedback
LINE_W_NORMAL = 2
LINE_W_HIGHLIGHT = 6

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
    """Load label mapping from the stimulus info CSV.

    The label CSV is expected to contain at least:
      - a key column identifying an object space (default: 'ObjectSpace')
      - 'Congruent', 'Medium', 'Incongruent' columns providing label strings

    Returns:
      tuple: (label_map, all_labels)
        - label_map: dict mapping space_id -> {'Congruent':..., 'Medium':..., 'Incongruent':...}
        - all_labels: sorted list of unique label strings across all spaces (used as distractor pool)

    Notes:
      - Uses pandas with engine='python' to allow flexible CSV separators when needed.
      - Raises ValueError if expected columns are missing so the caller can fail fast.
    """
    df = pd.read_csv(csv_path, sep=None, dtype={key_col: str}, engine="python")
    required = {key_col, 'Congruent', 'Medium', 'Incongruent'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label CSV missing required columns: {sorted(missing)}")

    label_map = {}
    all_labels = set()
    for _, row in df.iterrows():
        key = str(row[key_col])
        label_map[key] = {
            'Congruent': str(row['Congruent']),
            'Medium': str(row['Medium']),
            'Incongruent': str(row['Incongruent'])
        }
        all_labels.update(label_map[key].values())

    # Keep deterministic order for reproducibility (sampling is still random)
    all_labels = sorted(all_labels)
    return label_map, all_labels


def build_participant_label_pool(design_df, label_map, space_col_candidates=('ObjectSpace','group')):
    """Build a restricted distractor pool for this participant.

    The pool is limited to labels belonging to ObjectSpaces that actually appear in the
    participant's design CSV. This reduces the chance of presenting distractors from
    object spaces the participant never sees, which could be confusing.

    If no recognized space column exists in the design file the function falls back to
    returning all labels from the label_map (safe default).
    """
    space_col = None
    for c in space_col_candidates:
        if c in design_df.columns:
            space_col = c
            break
    if space_col is None:
        # Fallback: use all labels when the design has no recognizable space column
        pool = set()
        for d in label_map.values():
            pool.update(d.values())
        return sorted(pool)

    # Gather unique space identifiers present in the design file
    spaces = set(str(x) for x in design_df[space_col].dropna().astype(str).tolist())
    pool = set()
    for s in spaces:
        if s in label_map:
            pool.update(label_map[s].values())
    # If nothing matched (e.g., formatting mismatch), fallback to all labels
    if not pool:
        for d in label_map.values():
            pool.update(d.values())
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
    """Return the stimulus 'space' identifier for the current trial.

    Supports both legacy 'group' and newer 'ObjectSpace' design files. Raises
    KeyError if neither column is present so callers can signal malformatted designs.
    """
    if 'ObjectSpace' in trial:
        return str(trial['ObjectSpace'])
    if 'group' in trial:
        return str(trial['group'])
    if 'Group' in trial:
        return str(trial['Group'])
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


def between_run_dialog(next_run_idx, run_label=None):
    """Native GUI dialog shown between runs (outside fullscreen).

    This dialog helps with multi-run experiments where the display must be
    closed between runs (e.g., to allow scanner breaks or repositioning). It
    also reminds the operator which run is about to begin and which trigger key
    will start it.
    """
    title = f"Start run {next_run_idx}"
    dlg = gui.Dlg(title=title)
    if run_label is not None:
        dlg.addText(f"Next run value: {run_label}")
    dlg.addText("Click OK to open fullscreen.")
    dlg.addText(f"Then press '{TRIGGER_KEY}' on the trigger screen to begin run {next_run_idx}.")
    dlg.show()
    if not dlg.OK:
        core.quit()

# =========================== VISUAL HELPERS =========================== #

def show_instruction_screen(win, text_content, image_path=None):
    """
    Standardized instruction screen with optional image.
    Anchored to relative height units for cross-screen compatibility.

    The function blocks until the participant presses the SPACE bar, and it
    will attempt to display a large illustrative image if a valid path is given.
    """
    # Standardized vertical positions (relative to height 1.0)
    Y_INSTR_TEXT_HIGH = 0.22
    Y_INSTR_TEXT_MID  = 0.10
    Y_INSTR_IMAGE     = -0.10
    Y_SPACE_PROMPT    = -0.40

    text_y = Y_INSTR_TEXT_HIGH if image_path else Y_INSTR_TEXT_MID
    msg = visual.TextStim(win, text=text_content, color='black', height=0.035, 
                          wrapWidth=0.8, pos=(0, text_y))
    cont_msg = visual.TextStim(win, text="Press SPACE bar to continue", 
                               pos=(0, Y_SPACE_PROMPT), color='grey', height=0.025)
    
    img_stim = None
    if image_path and os.path.exists(image_path):
        # size (0.7, 0.4) provides a large, clear instructional icon
        img_stim = visual.ImageStim(win, image=image_path, pos=(0, Y_INSTR_IMAGE), size=(0.7, 0.4))
    
    event.clearEvents()
    while True:
        msg.draw()
        if img_stim:
            img_stim.draw()
        cont_msg.draw()
        win.flip()
        # Block until space pressed to ensure participant reads instructions
        if 'space' in event.getKeys():
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
    """Draws buttons during Feedback phase and apply highlighting.

    Behavioural requirements implemented here:
      - Always highlight the correct option border in green to reveal the target.
      - If the participant selected an incorrect option, highlight the selected
        option border in red to indicate the error.
      - Make the highlight thicker than the standard border for visibility.
      - Fill remains neutral to keep label text readable.
    """
    for btn in buttons:
        btn['box'].fillColor = COL_NEUTRAL
        btn['box'].lineColor = 'black'
        btn['box'].lineWidth = LINE_W_NORMAL

        # Correct option: green border (always)
        if correct_key is not None and btn['key'] == correct_key:
            btn['box'].lineColor = COL_CORRECT
            btn['box'].lineWidth = LINE_W_HIGHLIGHT

        # Incorrect selection: red border (in addition to correct border elsewhere)
        if response_made is not None and response_made != correct_key and btn['key'] == response_made:
            btn['box'].lineColor = COL_INCORRECT
            btn['box'].lineWidth = LINE_W_HIGHLIGHT

        btn['box'].draw()
        btn['text'].draw()


def setup_trial_visuals(trial, components, label_data, img_dir, demo_mode):
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
    cond = str(trial['condition'])

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
    random.shuffle(choices)

    # Apply choices to button text
    for i, btn in enumerate(components['buttons']):
        btn['text'].text = choices[i] if i < len(choices) else ""

    # 3) Demo Text for live debugging / experimenter view
    if demo_mode:
        components['demo_text'].text = f"Condition:\n{cond}"

    return has_img, img_path, target_lbl, choices


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
                       task_clock=None, trial_idx=1, n_trials=1):
    """Restored 2-Event Logic with new visuals.

    Standard mode timeline is absolute: onsets in the design CSV are respected
    relative to a run clock. This function loops through phases using the
    provided PsychoPy clock to maintain precise timing.
    """
    has_img, img_path, target, choices = setup_trial_visuals(trial, components, label_data, img_dir, demo_mode)

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
                     task_clock=None, trial_idx=1, n_trials=1):
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
    has_img, img_path, target, choices = setup_trial_visuals(trial, components, label_data, img_dir, demo_mode)

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

    # Ensure clean input state
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

    # 2) Decision (self-paced)
    t_dec = clock.getTime()
    response_made, rt = None, None
    while clock.getTime() < (t_dec + max_dec_dur):
        if event.getKeys(keyList=[EXIT_KEY]):
            core.quit()
        if _demo_skip_pressed(demo_mode):
            return _return_skipped(img_path)

        draw_extras(components, has_img, demo_mode)

        if response_made is None:
            response_made, rt = check_response(components, clock, t_dec)
            if response_made:
                # Draw one last frame with the selected option, then move on immediately
                draw_buttons(components['buttons'], response_made)

                breakdown = [
                    f"Dec: {max(0.0, (t_dec + max_dec_dur) - clock.getTime()):4.1f}s",
                    f"ISI2: {isi2_dur:4.1f}s" if feedback_delay else "ISI2: 0.0s",
                    f"Fb : {fb_dur:4.1f}s",
                    f"ITI: {iti:4.1f}s",
                ]
                _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                                  clock, trial_t0, 'Decision', (t_dec + max_dec_dur), breakdown)
                win.flip()
                break

        draw_buttons(components['buttons'], response_made)

        breakdown = [
            f"Dec: {max(0.0, (t_dec + max_dec_dur) - clock.getTime()):4.1f}s",
            f"ISI2: {isi2_dur:4.1f}s" if feedback_delay else "ISI2: 0.0s",
            f"Fb : {fb_dur:4.1f}s",
            f"ITI: {iti:4.1f}s",
        ]
        _hud_set_and_draw(components, demo_mode, task_clock, trial_idx, n_trials,
                          clock, trial_t0, 'Decision', (t_dec + max_dec_dur), breakdown)
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
    win = visual.Window([1280, 720], fullscr=True, color='white', units='height', allowGUI=False, useRetina=True)

    components = {
        'fixation': visual.TextStim(win, text='+', height=0.1, color='black'),
        # Image centered at (0,0) as requested
        'main_image': visual.ImageStim(win, pos=(0, 0), size=(0.5, 0.5), interpolate=True, texRes=2048),
        'missing_text': visual.TextStim(win, text='Img Missing', pos=(0, 0), height=0.05, color='red'),
        'respond_text': visual.TextStim(win, text='Respond!', pos=(0, 0.4), height=0.05, color='red'),
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
    spacing = 0.22
    start_x = -((3 * spacing) / 2)
    for i, key in enumerate(KEYS_RESP):
        x = start_x + (i * spacing)
        box = visual.Rect(win, width=0.2, height=0.1, pos=(x, -0.3),
                          fillColor=COL_NEUTRAL, lineColor='black', lineWidth=LINE_W_NORMAL)
        txt = visual.TextStim(win, text=f"{key}", pos=(x, -0.3), height=0.03, color='black')
        components['buttons'].append({'box': box, 'text': txt, 'key': key})

    return win, components


def trigger_screen(win, components, mode, demo_mode, run_idx=1, n_runs=1, run_label=None):
    """Interactive trigger screen: waits for TRIGGER_KEY.

    Displays run/mode/demo information and allows the operator to press a
    response key to visually check button mapping before the run begins. The
    function blocks until TRIGGER_KEY is pressed or EXIT_KEY is used to abort.
    """
    run_line = f"Run: {run_idx}/{n_runs}"
    if run_label is not None:
        run_line += f" (value: {run_label})"

    msg = visual.TextStim(
        win,
        text=(
            f"{run_line}\n"
            f"Mode: {mode}\n"
            f"Demo: {demo_mode}\n"
            f"Press {KEYS_RESP} to test.\n"
            f"Scanner Trigger: '{TRIGGER_KEY}'\n"
            + (f"Demo skip: RIGHT ARROW\n" if demo_mode else "")
        ),
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
            btn['text'].draw()

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


def run_experiment():
    # 1. Start Dialog: basic participant/run configuration
    info = {
        'Sub': '001',
        'Design CSV': '',        # Leave blank to browse
        'Label CSV': '',         # Leave blank to browse
        'Image Dir': 'blender_output_renamed',   # Default directory
        'Feedback Delay': False, # Selection from previous requirements
        'Demo Mode': False
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
        order=['Sub', 'Design CSV', 'Label CSV', 'Image Dir', 'Feedback Delay', 'Demo Mode'],
        tip={
            'Design CSV': 'Leave blank to open file browser',
            'Label CSV': 'Leave blank to open file browser',
            'Image Dir': 'Directory containing stimulus images'
        }
    )

    if not dlg.OK:
        core.quit()

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

    # Convert boolean GUI results into native Python booleans
    demo_mode = info['Demo Mode'] == 'True'
    feedback_delay = info['Feedback Delay'] == 'True'

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

    # Ensure output directory exists early so per-run files can be written safely
    os.makedirs('data', exist_ok=True)

    # Create first fullscreen window and start as-is (run 1 unchanged)
    win, components = create_window_and_components(demo_mode)
    # Show initial experimental instructions
    show_instruction_screen(win, (
        "EXPERIMENTAL SESSION\n\n"
        "You are about to start the main task.\n"
        "Remember to categorize the objects as accurately as possible.\n\n"
        "Keep your fingers placed on the '1', '2', '9', and '0' keys."
    ), image_path="instruction_image_1.png")
    show_instruction_screen(win, (
        "The objects you are going to learn are from an alien planet. Some of them can look similar to one's you've seen before\
while others will not. \n\nThe names of the objects won't change during the experiment."
    ))
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
        run_rows = []  # per-run buffer for crash-safe saving

        for trial_idx, trial in enumerate(trials, start=1):
            if mode == 'standard':
                res = run_trial_standard(
                    win, run_clock, trial, components, label_data, info['Image Dir'],
                    demo_mode, task_clock=task_clock, trial_idx=trial_idx, n_trials=n_trials
                )
            else:
                res = run_trial_3event(
                    win, run_clock, trial, components, label_data, info['Image Dir'],
                    demo_mode, feedback_delay, task_clock=task_clock, trial_idx=trial_idx, n_trials=n_trials
                )

            row = dict(trial)
            row.update(res)
            if 'skipped' not in row:
                row['skipped'] = 0
            if run_col is not None and run_col not in row:
                row[run_col] = run_label
            run_rows.append(row)
            out_rows.append(row)

        # Save results for this run immediately (crash-safe).
        # This ensures we still have earlier runs if the task crashes later.
        if len(run_rows) > 0:
            run_out_df = pd.DataFrame(run_rows)
            run_token = _sanitize_run_label(run_label)
            run_path = f"data/sub-{info['Sub']}_{mode}_run-{run_idx:02d}_{run_token}.csv"
            _safe_write_csv(run_out_df, run_path)

    out_df = pd.DataFrame(out_rows)
    # Final joined file across all runs (crash-safe)
    joined_path = f"data/sub-{info['Sub']}_{mode}_joined.csv"
    _safe_write_csv(out_df, joined_path)
    # Also write/keep the legacy filename for compatibility
    legacy_path = f"data/sub-{info['Sub']}_{mode}.csv"
    _safe_write_csv(out_df, legacy_path)

    try:
        win.close()
    except Exception:
        pass
    core.quit()

if __name__ == "__main__":
    run_experiment()