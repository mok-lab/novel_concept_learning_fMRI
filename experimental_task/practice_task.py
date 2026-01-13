#!/usr/bin/env python3
"""
Practice task runner (standardised layout).

Provides a small standalone practice session that mirrors the main experiment's
visual layout but uses fixed, simplified timing and a short trial list.

Behaviour:
 - Uses the same visual components and button layout as the main task (via demo_task).
 - Presents a short sequence of practice trials (slow -> fast blocks).
 - Shows on-screen guidance during slow trials (prompts + key labels).
 - For fast repeat trials, incorrect responses may be repeated until correct (configurable).
 - Feedback text is positioned in a 'safe zone' above the centered image to keep
   important UI elements away from the image area for cross-screen compatibility.

Expectations:
 - A label CSV (PRACTICE_LABEL_CSV) with an 'ObjectSpace' key and Congruent/Medium/Incongruent labels.
 - Practice images stored under PRACTICE_IMG_DIR named like '<ObjectSpace>.png' (or other supported extensions).
 - This file imports core display/loader helpers from `demo_task.py`.

Usage:
    python experimental_task/practice_task.py

Notes:
 - Layout constants use PsychoPy 'height' units (fractions of screen height).
 - No changes are made to the main experiment code; this module is intended for operator/participant familiarisation.
"""
from psychopy import visual, core, event
import pandas as pd
import os

# Import core loaders and logic from demo_task.py so visual behaviour stays consistent.
# demo_task provides the window factory, stimulus setup and common drawing/response helpers.
from demo_task import (
    create_window_and_components, 
    setup_trial_visuals, 
    load_label_map, 
    build_participant_label_pool,
    draw_extras,
    draw_buttons,
    draw_buttons_feedback,
    check_response,
    show_instruction_screen
)

# =========================== LAYOUT CONSTANTS (Standardised) =========================== #
# Units relative to screen height (PsychoPy 'height' units). Using standardized
# positions keeps the practice display consistent with the main experiment.
IMG_HEIGHT = 0.5
IMG_TOP_EDGE = 0.25  # Top edge when image is centered at (0,0) and height=0.5

# Feedback and prompt positions placed in a 'safe zone' above the image so
# feedback text does not overlap the stimulus content.
Y_FEEDBACK_HINT = IMG_TOP_EDGE + 0.03  # 0.28
Y_FEEDBACK_MAIN = Y_FEEDBACK_HINT + 0.07  # 0.35
Y_PROMPT        = Y_FEEDBACK_MAIN + 0.07  # 0.42

# Instructional layout positions (re-used from main task conventions)
Y_INSTR_TEXT_HIGH = 0.22
Y_INSTR_TEXT_MID  = 0.10
Y_INSTR_IMAGE     = -0.10
Y_SPACE_PROMPT    = -0.40
Y_BUTTON_LABELS   = -0.38

# =========================== CONFIGURATION =========================== #
# Paths and small practice design used by the script.
PRACTICE_IMG_DIR = "practice_images"
PRACTICE_LABEL_CSV = "practice_labels.csv"

# Short practice design demonstrating slow/fast and repeat behaviours.
# Each dictionary represents one trial template (ObjectSpace used to resolve image and label).
PRACTICE_DESIGN = [
    {'ObjectSpace': '1', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': False},
    {'ObjectSpace': '2', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': False},
    {'ObjectSpace': '1', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': True},
    {'ObjectSpace': '2', 'condition': 'Congruent', 'speed': 'slow', 'is_repeat': True},
    
    {'ObjectSpace': '3', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': False},
    {'ObjectSpace': '4', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': False},
    {'ObjectSpace': '3', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': True}, 
    {'ObjectSpace': '4', 'condition': 'Congruent', 'speed': 'fast', 'is_repeat': True}
]

def run_custom_practice_trial(win, clock, trial, components, label_data, img_dir):
    """Execute a single practice trial using the standardised layout.

    The flow:
      1) Encoding (image visible + short ISI)
      2) Decision (response window, slow trials show on-screen prompts)
      3) Feedback (text feedback + colored border highlights)
      4) ITI (fixation)

    Parameters:
      - win: PsychoPy Window (not closed by this function)
      - clock: PsychoPy Clock used for trial timing
      - trial: dict with trial parameters (expects keys like ObjectSpace, condition, speed, is_repeat)
      - components: dict of pre-created visual components (created by create_window_and_components)
      - label_data: tuple (label_map, label_pool) used by setup_trial_visuals
      - img_dir: directory containing practice images

    Returns:
      dict with 'accuracy' (1 or 0). No file I/O is performed here.
    """
    # Prepare stimuli and choices for this trial; returns whether image was found
    has_img, img_path, target, choices = setup_trial_visuals(trial, components, label_data, img_dir, False)
    
    # Map the shuffled choice index to the physical key labels used in the experiment
    try:
        correct_key = ['1', '2', '9', '0'][choices.index(target)]
    except ValueError:
        # If target label not present in choices for any reason, treat as missing target
        correct_key = None

    # Speed and repeat flags affect timing and feedback policies
    is_slow = (trial.get('speed') == 'slow')
    is_repeat = trial.get('is_repeat', False)
    
    # Pre-defined durations for slow vs fast practice trials (kept intentionally simple)
    durations = {
        'img': 2.0 if is_slow else 1.0,
        'isi1': 1.0 if is_slow else 0.5,
        'dec': 10.0 if is_slow else 3.0,
        'fb': 5.0 if is_slow else 1.5,
        'iti': 1.5 if is_slow else 2.0
    }

    t0 = clock.getTime()
    
    # 1. Encoding: show image (or missing placeholder) during image+isi1 window
    while clock.getTime() < (t0 + durations['img'] + durations['isi1']):
        draw_extras(components, has_img, False)
        win.flip()

    # 2. Decision: draw choices and handle keyboard/mouse responses.
    # For slow trials we overlay a textual prompt and per-button labels to guide participants.
    t_dec = clock.getTime()
    response, rt = None, None
    while clock.getTime() < (t_dec + durations['dec']):
        draw_extras(components, has_img, False)
        if is_slow:
            # Show explicit prompt and key labels for novice participants
            components['prompt'].draw()
            for lbl in components['key_labels']:
                lbl.draw()
        
        if response is None:
            # check_response returns (key, rt) or (None, None)
            response, rt = check_response(components, clock, t_dec)
            if response:
                # On first response we break out (decision completed)
                break
        
        # Draw the option buttons (visual feedback of selection occurs inside draw_buttons)
        draw_buttons(components['buttons'], response)
        win.flip()

    # 3. Feedback Phase: set appropriate explanatory text and draw colored borders
    t_fb = clock.getTime()
    is_correct = (response == correct_key)
    components['fb_main'].text = ""
    components['fb_hint'].text = ""
    
    # Provide more descriptive feedback for slow trials; for fast repeat trials only
    # show a brief hint and allow repetition if the trial is configured that way.
    if is_slow:
        if is_correct:
            components['fb_main'].text = "The green highlight indicates that your choice was correct."
            components['fb_main'].color = 'green'
        else:
            components['fb_main'].text = "The red highlight indicates that your choice was incorrect."
            components['fb_main'].color = 'red'
            components['fb_hint'].text = "The correct option is highlighted in green."
    elif not is_correct and not is_slow and is_repeat:
        # For fast repeat trials, show concise corrective prompt
        components['fb_main'].text = "Check the feedback and try again"
        components['fb_main'].color = 'red'
        components['fb_hint'].text = "The correct option is highlighted in green."

    # Render feedback for the configured duration
    while clock.getTime() < (t_fb + durations['fb']):
        draw_extras(components, has_img, False)
        draw_buttons_feedback(components['buttons'], response, correct_key)
        if response is None:
            components['respond_text'].draw()
        components['fb_main'].draw()
        components['fb_hint'].draw()
        win.flip()

    # 4. ITI (simple fixation)
    t_iti = clock.getTime()
    while clock.getTime() < (t_iti + durations['iti']):
        components['fixation'].draw()
        win.flip()

    return {'accuracy': 1 if is_correct else 0}

def run_practice():
    """Create window, prepare components and run the short practice loop.

    Responsibilities:
      - Instantiate the window and shared visual components.
      - Initialize small practice-specific text elements positioned using the
        standardized layout constants above.
      - Load the practice label CSV and build the participant-specific label pool.
      - Iterate through PRACTICE_DESIGN, presenting trials and handling optional repeats.
      - Close the window at the end of practice.
    """
    # Build window and shared visual components using the same factory as the main task
    win, components = create_window_and_components(demo_mode=False)
    
    # Initialize practice-only text components positioned relative to the image.
    components['prompt'] = visual.TextStim(
        win, text="Choose an option by pressing a key", pos=(0, Y_PROMPT),
        height=0.035, color='blue'
    )
    components['fb_main'] = visual.TextStim(
        win, text="", pos=(0, Y_FEEDBACK_MAIN), height=0.03, wrapWidth=0.8
    )
    components['fb_hint'] = visual.TextStim(
        win, text="", pos=(0, Y_FEEDBACK_HINT), height=0.03, wrapWidth=0.8, color='green'
    )
    
    # Per-button key labels shown for slow trials so novices can map buttons to keys
    components['key_labels'] = []
    for i, key in enumerate(['1', '2', '9', '0']):
        x = components['buttons'][i]['box'].pos[0]
        components['key_labels'].append(
            visual.TextStim(win, text=key, pos=(x, Y_BUTTON_LABELS), height=0.025, color='black')
        )

    # Load practice labels and make a label pool restricted to the practice design's spaces.
    label_map, _ = load_label_map(PRACTICE_LABEL_CSV, 'ObjectSpace')
    label_pool = build_participant_label_pool(pd.DataFrame(PRACTICE_DESIGN), label_map)
    label_data = (label_map, label_pool)

    # 1. Instructions: show short practice instructions then a reminder with visual.
    show_instruction_screen(
        win,
        "PRACTICE TASK\n\nThis is a practice version of the task you will complete in the MRI scanner.\n\n"
        "The aim is to learn the correct names for different objects. The names for each object won't change during the experiment.\n\n"
        "You will first see an object on the screen. After a delay, options will appear under the image.\n"
        "Your task is to choose the option you think is correct. This will be followed by feedback and the correct option will be shown if you were incorrect in your choice."
    )

    show_instruction_screen(
        win,
        "SLOW START\n\nThe first few trials will be slow.\n\nPrompts on screen will guide you.\n\nPlace your fingers like this onto the '1', '2', '9', and '0' keys",
        image_path="instruction_image_1.png"
    )

    # 2. Trial Loop: iterate practice trials and enforce repeat rules when configured
    clock = core.Clock()
    for i, trial in enumerate(PRACTICE_DESIGN):
        # When transitioning from slow -> fast, show a short transition message
        if trial['speed'] == 'fast' and PRACTICE_DESIGN[i-1]['speed'] == 'slow':
            show_instruction_screen(
                win,
                "Well done!\n\nTrials will now move faster at the actual experimental speed.\n"
                "Guidance prompts will also be removed.\n\nTry your best to make correct choices."
            )

        repeat_enabled = (trial['speed'] == 'fast' and trial['is_repeat'])
        success = False
        # Run each trial, optionally repeating until success if repeat_enabled
        while not success:
            # Compose a trial dict with image_file resolved from ObjectSpace (simple convention)
            trial_row = dict(trial, image_file=f"{trial['ObjectSpace']}.png")
            result = run_custom_practice_trial(win, clock, trial_row, components, label_data, PRACTICE_IMG_DIR)
            if result['accuracy'] == 1 or not repeat_enabled:
                success = True

    # Final message and clean shutdown
    show_instruction_screen(win, "PRACTICE COMPLETE.\n\nPlease inform the experimenter that you are done.")
    win.close()

if __name__ == "__main__":
    run_practice()