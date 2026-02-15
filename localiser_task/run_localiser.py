#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Category localiser with 1-back task (PsychoPy)

VISUALS (matched to experimental_task.py):
- White background
- Black fixation cross (+)
- Window units='height' with useRetina=True
- Images displayed at a constant footprint: ~60% of screen height (size=(0.6, 0.6))
- Image *texture* is center-cropped and resized to a target pixel resolution (img_tex_size)
  so you can standardize the stimulus resolution without changing on-screen footprint.

TASK:
- 10 categories (your 9 + "scrambled") -> 10 blocks per run (single pass each)
- Block: 20 images, 300 ms image + 500 ms blank (= 16 s)
- Inter-block blank: 4 s (between blocks)
- Baseline: 16 s at start and 16 s at end
- 2 or 3 1-back targets per block (immediate repeat of previous image)
- Counterbalancing: rotation across participants + run index
- Response keys:
    * scanner mode (default): 1,2,3,4
    * pc mode: 1,2,9,0
- GUI: participant/session, n_runs, image parent dir, button mode, targets/block,
       texture resolution, fullscreen, screen index
- Saves per-trial CSV and a simple run summary TXT.

Folder structure expected:
parent_dir/
  Faces/
    img1.jpg ...
  Scenes/
  Bodies/
  Buildings/
  Objects/
  ScrambledObjects/   (or your label)
  ... (your 9 categories total + scrambled = 10 folders)

Dependencies:
pip install psychopy pillow numpy
"""

import os
import sys
import csv
import time
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from psychopy import visual, core, event, gui
from psychopy.hardware import keyboard

from PIL import Image, ImageOps
import numpy as np


# Language for on-screen text (manual toggle)
LANGUAGE = "english"  # "english" or "japanese"
# -----------------------------
# Config / helpers
# -----------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_BASELINE_S = 16.0
DEFAULT_INTERBLOCK_S = 4.0
DEFAULT_IMG_S = 0.300
DEFAULT_ISI_S = 0.500
DEFAULT_TRIALS_PER_BLOCK = 20

# Pixel resolution used for the image texture (NOT on-screen size).
# On-screen size is controlled by DISPLAY_IMG_HEIGHT_FRAC below.
DEFAULT_IMG_TEX_SIZE = 512

# On-screen footprint (units='height'): 1.0 == full screen height.
DISPLAY_IMG_HEIGHT_FRAC = 0.60

# Match experimental_task.py look
BG_COLOR = "lightgrey"
FG_COLOR = "black"

SCANNER_KEYS = ["1", "2", "3", "4"]
PC_KEYS = ["1", "2", "9", "0"]
TRIGGER_KEY = "5"  # scanner trigger
QUIT_KEYS = ["escape"]

# PsychoPy ImageStim texture resolution (power-of-two). Larger helps retina displays.
DEFAULT_TEXRES = 2048


@dataclass
class Params:
    participant: str
    session: str
    n_runs: int
    parent_dir: str
    button_mode: str  # "scanner" or "pc"
    n_targets_per_run: int  # total one-back targets across the entire run
    img_tex_size: int
    fullscreen: bool
    screen_index: int
    baseline_s: float = DEFAULT_BASELINE_S
    interblock_s: float = DEFAULT_INTERBLOCK_S
    img_s: float = DEFAULT_IMG_S
    isi_s: float = DEFAULT_ISI_S
    trials_per_block: int = DEFAULT_TRIALS_PER_BLOCK


def list_category_folders(parent_dir: str) -> List[str]:
    cats = []
    for name in sorted(os.listdir(parent_dir)):
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            cats.append(name)
    return cats


def list_images_in_folder(folder: str) -> List[str]:
    files = []
    for fn in sorted(os.listdir(folder)):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, fn))
    return files


def ensure_data_dir(participant: str, session: str) -> str:
    out = os.path.join(os.getcwd(), "data", f"sub-{participant}", f"ses-{session}")
    os.makedirs(out, exist_ok=True)
    return out


def parse_participant_number(participant: str) -> int:
    digits = "".join([c for c in participant if c.isdigit()])
    if digits:
        return int(digits)
    return sum(ord(c) for c in participant) % 10_000


def counterbalanced_order(categories: List[str], participant_num: int, run_idx: int) -> List[str]:
    """
    Simple counterbalancing:
    - Rotate base order by (participant_num + run_idx) mod n
    - Flip direction for even/odd participant to diversify sequences
    """
    n = len(categories)
    base = categories[:]  # already sorted
    offset = (participant_num + run_idx) % n
    order = base[offset:] + base[:offset]
    if participant_num % 2 == 0:
        if run_idx % 2 == 1:
            order = list(reversed(order))
    return order


def choose_target_positions_across_run(
    n_blocks: int,
    trials_per_block: int,
    n_targets_total: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    """Assign *exactly* n_targets_total one-back targets across the whole run.

    We place targets *within* blocks only (no cross-block repeats), at trial indices
    t where the stimulus on trial t repeats trial t-1.

    Constraints:
      - t must be >= 1
      - avoid adjacent targets within the same block (t and t+1)

    Returns a dict: block_index (1-based) -> sorted list of target trial indices.
    """
    if n_targets_total < 0:
        raise ValueError("n_targets_per_run must be >= 0")

    # Candidate slots are (block, t) where t in [1..trials_per_block-1]
    candidates: List[Tuple[int, int]] = []
    for b in range(1, n_blocks + 1):
        for t in range(1, trials_per_block):
            candidates.append((b, t))
    rng.shuffle(candidates)

    chosen_by_block: Dict[int, List[int]] = {b: [] for b in range(1, n_blocks + 1)}
    chosen_total = 0

    # First pass: enforce non-adjacent constraint strictly
    for b, t in candidates:
        if chosen_total >= n_targets_total:
            break
        if any(abs(t - c) <= 1 for c in chosen_by_block[b]):
            continue
        chosen_by_block[b].append(t)
        chosen_total += 1

    # If we couldn't place enough (rare unless n_targets_total is huge), relax adjacency slightly.
    if chosen_total < n_targets_total:
        for b, t in candidates:
            if chosen_total >= n_targets_total:
                break
            if t in chosen_by_block[b]:
                continue
            chosen_by_block[b].append(t)
            chosen_total += 1

    for b in chosen_by_block:
        chosen_by_block[b].sort()

    # Hard clamp (in case user requests more than possible)
    if chosen_total < n_targets_total:
        # We don't crash mid-run; we just deliver the maximum possible.
        # (Still, this situation should be visible in the summary.)
        pass

    return chosen_by_block


def preprocess_image_to_cache(path: str, target_size: int, cache_dir: str) -> str:
    """
    Preprocess an image for presentation:
    - Apply EXIF orientation (prevents rotated/flipped JPEGs)
    - Convert to standard RGB (handles grayscale/CMYK/RGBA safely)
    - Center-crop to square
    - Resize to (target_size, target_size) pixels
    - Save as an 8-bit PNG in a cache directory

    Returns the cached PNG filepath.

    Why this approach:
    PsychoPy can interpret numpy-array textures as needing values in [-1, 1]
    (depending on version / backend). Passing a filename is the most robust
    and color-faithful approach across PsychoPy installs.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Cache key: path + mtime + size + target_size
    try:
        st = os.stat(path)
        key_str = f"{path}|{st.st_mtime_ns}|{st.st_size}|{int(target_size)}"
    except OSError:
        # Fallback if stat fails (e.g., network hiccup)
        key_str = f"{path}|{int(target_size)}"

    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(cache_dir, f"{h}_{int(target_size)}.png")
    if os.path.exists(out_path):
        return out_path

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    w, h0 = img.size
    min_dim = min(w, h0)
    left = (w - min_dim) // 2
    top = (h0 - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    # Use a high-quality downsampler; LANCZOS is best for shrink, BICUBIC fine for mixed.
    resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC
    img = img.resize((int(target_size), int(target_size)), resample=resample)

    # Save without optimization tricks that can be slow; PNG preserves exact 8-bit RGB.
    img.save(out_path, format="PNG")
    return out_path


def safe_wait(duration: float, kb: keyboard.Keyboard, allowed_keys: List[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    Wait for 'duration' seconds while collecting the FIRST keypress in allowed_keys.
    Returns (key, rt) relative to the start of this wait, or (None, None).
    """
    t0 = core.getTime()
    got_key = None
    got_rt = None
    while (core.getTime() - t0) < duration:
        if event.getKeys(QUIT_KEYS):
            core.quit()
        keys = kb.getKeys(keyList=allowed_keys + QUIT_KEYS, waitRelease=False, clear=False)
        if keys and got_key is None:
            k = keys[0]
            if k.name in QUIT_KEYS:
                core.quit()
            got_key = k.name
            got_rt = k.rt
        core.wait(0.001)
    return got_key, got_rt


def show_text_screen(win: visual.Window, text: str, kb: keyboard.Keyboard, advance_keys: List[str]):
    """Instruction / pause screen styled like experimental_task.py (white bg, black text)."""
    stim = visual.TextStim(
        win,
        text=text,
        height=0.04,
        wrapWidth=0.90,
        color=FG_COLOR,
        pos=(0, 0.10),
    )
    if LANGUAGE == "japanese":
        cont_text = "続行するにはいずれかのボタンを押してください"
    else:
        cont_text = "Press any button to continue"

    cont = visual.TextStim(
        win,
        text=cont_text,
        height=0.03,
        color="grey",
        pos=(0, -0.40),
    )

    while True:
        if event.getKeys(QUIT_KEYS):
            core.quit()
        stim.draw()
        cont.draw()
        win.flip()

        keys = kb.getKeys(keyList=advance_keys + ["space", "return"] + QUIT_KEYS, waitRelease=False, clear=True)
        if keys:
            if keys[0].name in QUIT_KEYS:
                core.quit()
            return
        core.wait(0.01)



def wait_for_trigger(
    win: visual.Window,
    kb: keyboard.Keyboard,
    trigger_key: str = TRIGGER_KEY,
    allow_skip_keys: Optional[List[str]] = None,
    text: str = None,
) -> str:
    """
    Block until we receive the scanner trigger key (default '5').

    - In scanner mode, you typically want the run clock to start from this trigger.
    - Optionally allow 'allow_skip_keys' (e.g., response keys) for manual testing.
    Returns the key name that started the run (trigger or skip key).
    """
    allow_skip_keys = allow_skip_keys or []

    if text is None:
        if LANGUAGE == "japanese":
            text = "MRI装置の起動を待っています"
        else:
            text = "Waiting for scanner to start"

    stim = visual.TextStim(
        win,
        text=text,
        height=0.04,
        wrapWidth=0.90,
        color=FG_COLOR,
        pos=(0, 0.05),
    )


    kb.clearEvents()
    while True:
        if event.getKeys(QUIT_KEYS):
            core.quit()
        stim.draw()
        win.flip()

        keys = kb.getKeys(
            keyList=[trigger_key] + QUIT_KEYS,
            waitRelease=False,
            clear=True,
        )
        if keys:
            k = keys[0].name
            if k in QUIT_KEYS:
                core.quit()
            return k

def create_window(params: Params) -> visual.Window:
    """Create a PsychoPy window that matches experimental_task.py styling."""
    if params.fullscreen:
        win = visual.Window(
            fullscr=True,
            screen=params.screen_index,
            color=BG_COLOR,
            units="height",
            allowGUI=False,
            useRetina=True,
        )
    else:
        win = visual.Window(
            size=[1280, 720],
            fullscr=False,
            screen=params.screen_index,
            color=BG_COLOR,
            units="height",
            allowGUI=True,
            useRetina=True,
        )
    return win


# -----------------------------
# Main task logic
# -----------------------------

def run_localiser(params: Params):
    resp_keys = SCANNER_KEYS if params.button_mode.lower() == "scanner" else PC_KEYS
    use_scanner_trigger = (params.button_mode.lower() == "scanner")
    start_keys = resp_keys + ([TRIGGER_KEY] if use_scanner_trigger else [])

    if not params.parent_dir or not os.path.isdir(params.parent_dir):
        raise FileNotFoundError(f"Parent dir not found: {params.parent_dir}")

    categories = list_category_folders(params.parent_dir)
    if len(categories) != 10:
        raise RuntimeError(
            f"Expected exactly 10 category folders (9 + scrambled), found {len(categories)}:\n{categories}\n"
            f"Please ensure parent_dir contains exactly 10 subfolders."
        )

    # Load image lists
    cat_to_imgs: Dict[str, List[str]] = {}
    for c in categories:
        imgs = list_images_in_folder(os.path.join(params.parent_dir, c))
        if len(imgs) < 2:
            raise RuntimeError(f"Category '{c}' has too few images ({len(imgs)}). Need at least 2.")
        cat_to_imgs[c] = imgs

    out_dir = ensure_data_dir(params.participant, params.session)

    cache_dir = os.path.join(out_dir, '_stim_cache_png')

    win = create_window(params)
    kb = keyboard.Keyboard()

    # Cache preprocessed (cropped+resized) image arrays to avoid repeated PIL work
    img_cache: Dict[Tuple[str, int], str] = {}  # cached PNG paths

    # Image stim is created ONCE and re-used, so display settings stay identical across trials
    img_stim = visual.ImageStim(
        win,
        image=None,
        pos=(0, 0),
        size=(DISPLAY_IMG_HEIGHT_FRAC, DISPLAY_IMG_HEIGHT_FRAC),
        interpolate=True,
        texRes=DEFAULT_TEXRES,
    )

    # Fixation / text styling to match experimental_task.py
    fixation = visual.TextStim(win, text="+", height=0.10, color=FG_COLOR)

    participant_num = parse_participant_number(params.participant)
    all_run_summaries = []

    # Instructions (bilingual placeholders)
    if LANGUAGE == "japanese":
        instr1 = (
            "画像閲覧\n\n"
            "さまざまな画像のシーケンスが表示されます"
        )
        instr2 = (
            "同じ画像が2回続けて表示されることもあります。\n"
            "繰り返し画像（連続）が表示されたら、\n任意の応答ボタンを押します。"
        )
        instr3 = (
            "迅速かつ正確に応答するようにしてください。\n応答が遅いと思われても心配しないでください。\n"
            "シーケンス間の固定クロスに注目してください\n"
            "開始するには任意のボタンを押してください。"
        )
    else:
        instr1 = (
            "Image viewing\n\n"
            "You will see different sequences of images"
        )
        instr2 = (
            "Sometimes the SAME picture will appear twice in a row.\n"
            "When you see a repeat image (back-to-back), press ANY response button."
        )
        instr3 = (
            "Try to respond quickly and accurately\nbut don't worry if your response seems slow.\n\n"
            "Keep your eyes on the fixation cross between sequences.\n\n"
            "Press any button to start."
        )


    show_text_screen(win, instr1, kb, advance_keys=resp_keys)
    show_text_screen(win, instr2, kb, advance_keys=resp_keys)
    show_text_screen(win, instr3, kb, advance_keys=start_keys)

    try:
        for run_idx in range(1, params.n_runs + 1):
            rng = random.Random((participant_num * 10_000) + run_idx)
            order = counterbalanced_order(categories, participant_num, run_idx)

            # Decide *once* which trials are 1-back targets for this run.
            # This ensures we get exactly params.n_targets_per_run targets across the whole run,
            # rather than per-block.
            targets_by_block = choose_target_positions_across_run(
                n_blocks=len(order),
                trials_per_block=params.trials_per_block,
                n_targets_total=params.n_targets_per_run,
                rng=rng,
            )

            ts = time.strftime("%Y%m%d-%H%M%S")
            csv_path = os.path.join(
                out_dir,
                f"localiser_sub-{params.participant}_ses-{params.session}_run-{run_idx:02d}_{ts}.csv",
            )

            fieldnames = [
                "participant", "session", "run",
                "block_index", "category",
                "trial_in_block", "image_path",
                "is_target",
                "resp_key", "resp_rt_s", "correct",
                "trial_onset_s", "img_onset_s", "isi_onset_s",
                "img_tex_size", "img_display_height_frac",
            ]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                if LANGUAGE == "japanese":
                    trigger_text = (
                        f"施行: {run_idx}/{params.n_runs}"
                        "MRI装置の起動を待っています"
                    )
                else:
                    trigger_text = (
                        f"Run: {run_idx}/{params.n_runs}"
                        "Waiting for scanner to start…"
                    )

                wait_for_trigger(
                    win,
                    kb,
                    trigger_key=TRIGGER_KEY,
                    allow_skip_keys=resp_keys,  # handy for keyboard testing
                    text=trigger_text,
                )

                kb.clearEvents()  # ensure the trigger key doesn't count as a response
                run_start = core.getTime()
                kb.clock.reset()

                # Baseline start
                fixation.draw()
                win.flip()
                safe_wait(params.baseline_s, kb, allowed_keys=resp_keys)

                hits = misses = fas = crs = 0

                # Decide EXACTLY which trials will be one-back targets across the whole run
                # (not per block). Targets are still constrained to occur within blocks.
                targets_by_block = choose_target_positions_across_run(
                    n_blocks=len(order),
                    trials_per_block=params.trials_per_block,
                    n_targets_total=params.n_targets_per_run,
                    rng=rng,
                )

                for b_i, cat in enumerate(order, start=1):
                    imgs = cat_to_imgs[cat]

                    # Sample a sequence without accidental immediate repeats (we'll add targets explicitly)
                    seq = [rng.choice(imgs)]
                    while len(seq) < params.trials_per_block:
                        nxt = rng.choice(imgs)
                        if nxt == seq[-1]:
                            continue
                        seq.append(nxt)

                    target_positions = targets_by_block.get(b_i, [])
                    is_target = [False] * params.trials_per_block
                    for t in target_positions:
                        seq[t] = seq[t - 1]
                        is_target[t] = True

                    for t_i in range(params.trials_per_block):
                        trial_onset = core.getTime() - run_start
                        img_path = seq[t_i]

                        cache_key = (img_path, params.img_tex_size)
                        if cache_key not in img_cache:
                            img_cache[cache_key] = preprocess_image_to_cache(img_path, params.img_tex_size, cache_dir)

                        # Set image *data* (RGB uint8) but keep on-screen *size* constant (~60% height)
                        img_stim.image = img_cache[cache_key]

                        # IMAGE
                        img_onset = core.getTime() - run_start
                        img_stim.draw()
                        win.flip()

                        kb.clearEvents()
                        key_img, rt_img = safe_wait(params.img_s, kb, allowed_keys=resp_keys)

                        # ISI (blank white screen)
                        isi_onset = core.getTime() - run_start
                        win.flip()
                        key_isi, rt_isi = safe_wait(params.isi_s, kb, allowed_keys=resp_keys)

                        # First response within full trial window
                        resp_key = resp_rt = None
                        if key_img is not None:
                            resp_key, resp_rt = key_img, rt_img
                        elif key_isi is not None:
                            resp_key, resp_rt = key_isi, rt_isi

                        target = is_target[t_i]
                        if target and resp_key is not None:
                            correct = 1; hits += 1
                        elif target and resp_key is None:
                            correct = 0; misses += 1
                        elif (not target) and resp_key is not None:
                            correct = 0; fas += 1
                        else:
                            correct = 1; crs += 1

                        writer.writerow({
                            "participant": params.participant,
                            "session": params.session,
                            "run": run_idx,
                            "block_index": b_i,
                            "category": cat,
                            "trial_in_block": t_i + 1,
                            "image_path": img_path,
                            "is_target": int(target),
                            "resp_key": resp_key if resp_key is not None else "",
                            "resp_rt_s": f"{resp_rt:.4f}" if resp_rt is not None else "",
                            "correct": int(correct),
                            "trial_onset_s": f"{trial_onset:.4f}",
                            "img_onset_s": f"{img_onset:.4f}",
                            "isi_onset_s": f"{isi_onset:.4f}",
                            "img_tex_size": int(params.img_tex_size),
                            "img_display_height_frac": float(DISPLAY_IMG_HEIGHT_FRAC),
                        })

                    # Inter-block blank (white) except after last block
                    if b_i < len(order):
                        win.flip()
                        safe_wait(params.interblock_s, kb, allowed_keys=resp_keys)

                # Baseline end
                fixation.draw()
                win.flip()
                safe_wait(params.baseline_s, kb, allowed_keys=resp_keys)

                total_targets = hits + misses
                total_nontargets = fas + crs
                hit_rate = hits / total_targets if total_targets else 0.0
                fa_rate = fas / total_nontargets if total_nontargets else 0.0

                summary = {
                    "run": run_idx,
                    "hits": hits, "misses": misses, "false_alarms": fas, "correct_rejects": crs,
                    "hit_rate": hit_rate, "fa_rate": fa_rate,
                    "order": order,
                    "csv": os.path.basename(csv_path),
                    "targets_requested": int(params.n_targets_per_run),
                    "targets_placed": int(total_targets),
                }
                all_run_summaries.append(summary)

                if LANGUAGE == "japanese":
                    end_text = (
                        f"施行 {run_idx}の終わり"
                        f"正確な: {hits}/{total_targets}  誤報: {fas}"
                        "続行するにはいずれかのボタンを押してください"
                    )
                else:
                    end_text = (
                        f"End of run {run_idx}."
                        f"Hits: {hits} / {total_targets}   False alarms: {fas}"
                        "Press a button to continue."
                    )

                show_text_screen(
                    win,
                    end_text,

                    kb,
                    advance_keys=resp_keys,
                )

        summ_path = os.path.join(
            out_dir,
            f"SUMMARY_localiser_sub-{params.participant}_ses-{params.session}_{time.strftime('%Y%m%d-%H%M%S')}.txt",
        )
        with open(summ_path, "w", encoding="utf-8") as f:
            f.write("Category localiser summary\n")
            f.write(f"participant={params.participant} session={params.session}\n")
            f.write(f"parent_dir={params.parent_dir}\n")
            f.write(f"n_runs={params.n_runs} img_tex_size={params.img_tex_size} button_mode={params.button_mode}\n")
            f.write(f"display_img_height_frac={DISPLAY_IMG_HEIGHT_FRAC}\n")
            f.write(f"timing: baseline={params.baseline_s}s interblock={params.interblock_s}s img={params.img_s}s isi={params.isi_s}s\n")
            f.write(f"trials_per_block={params.trials_per_block} targets_per_run={params.n_targets_per_run}\n\n")
            for s in all_run_summaries:
                f.write(f"RUN {s['run']:02d}\n")
                f.write(f"  hits={s['hits']} misses={s['misses']} false_alarms={s['false_alarms']} correct_rejects={s['correct_rejects']}\n")
                f.write(f"  hit_rate={s['hit_rate']:.3f} fa_rate={s['fa_rate']:.3f}\n")
                f.write(f"  targets_requested={s.get('targets_requested','')} targets_placed={s.get('targets_placed','')}\n")
                f.write(f"  order={s['order']}\n")
                f.write(f"  csv={s['csv']}\n\n")

        if LANGUAGE == "japanese":
            final_text = "終了した"
        else:
            final_text = "All runs complete."
        show_text_screen(win, final_text, kb, advance_keys=resp_keys)

    finally:
        try:
            win.close()
        except Exception:
            pass
        core.quit()


# -----------------------------
# GUI
# -----------------------------

def get_params_from_gui() -> Params:
    info = {
        "language": "english",  # "english" or "japanese"
        "participant": "001",
        "session": "01",
        "n_runs": 6,
        "parent_dir": "images/localiser_images/",
        "button_mode": "scanner",  # default
        "n_targets_per_run": 5,  # total one-back repeats across the whole run
        "img_tex_size": DEFAULT_IMG_TEX_SIZE,
        "fullscreen": True,
        "screen_index": 0,
    }

    dlg = gui.DlgFromDict(
        dictionary=info,
        title="fMRI Localiser Setup",
        order=[
            "language",
            "participant", "session",
            "n_runs",
            "parent_dir",
            "button_mode",
            "n_targets_per_run",
            "img_tex_size",
            "fullscreen",
            "screen_index",
        ],
        tip={
            "language": "On-screen language: english or japanese (japanese currently prefixes strings with [japanese]).",
            "parent_dir": "Folder that contains 10 subfolders (one per category) with images inside.",
            "button_mode": "scanner: keys 1,2,3,4 | pc: keys 1,2,9,0",
            "n_targets_per_run": "Total number of one-back repeats across the entire run.",
            "img_tex_size": "Pixel resolution of loaded images after crop/resize (display size is fixed at ~60% screen height).",
            "screen_index": "Which monitor to use (0 = primary).",
        }
    )
    if not dlg.OK:
        sys.exit(0)
    # Set global language
    global LANGUAGE
    lang = str(info.get("language", "english")).strip().lower()
    LANGUAGE = "japanese" if lang.startswith("jap") else "english"

    parent_dir = info["parent_dir"]
    if not parent_dir or not os.path.isdir(parent_dir):
        picked = gui.fileOpenDlg(tryFilePath=os.getcwd(), prompt="Select the image parent directory", allowed=None)
        if picked and len(picked) > 0:
            p = picked[0]
            parent_dir = p if os.path.isdir(p) else os.path.dirname(p)
        else:
            raise FileNotFoundError("No valid parent_dir selected.")

    return Params(
        participant=str(info["participant"]),
        session=str(info["session"]),
        n_runs=int(info["n_runs"]),
        parent_dir=parent_dir,
        button_mode=str(info["button_mode"]).strip().lower(),
        n_targets_per_run=int(info["n_targets_per_run"]),
        img_tex_size=int(info["img_tex_size"]),
        fullscreen=bool(info["fullscreen"]),
        screen_index=int(info["screen_index"]),
    )


if __name__ == "__main__":
    params = get_params_from_gui()
    run_localiser(params)
