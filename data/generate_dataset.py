#!/usr/bin/env python3
"""
Offline dataset generator for NESS FDTD guitar synthesis.

Generates random chord+pattern samples, runs the NESS C++ simulator,
and saves (audio, parameter_tensor) pairs for training.

Usage:
    python generate_dataset.py --output_dir ./data/ness_guitar \
                               --num_samples 50000 \
                               --ness_binary ./ness/ness-framework \
                               --num_workers 8
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────
# 1. Physics constants (NESS-safe values)
# ─────────────────────────────────────────────────────────────────────
NUM_STRINGS = 6
DURATION = 4.0
SAMPLE_RATE_NESS = 48000
SAMPLE_RATE_OUT = 44100
L_FIXED = 0.68
E_FIXED = 2e11
RHO_FIXED = 7850.0

RADII = [0.0002, 0.00015, 0.00015, 0.00015, 0.0001, 0.0001]
TENSIONS = [12.1, 12.3, 21.9, 39.2, 27.6, 49.2]
T60_1000_VALS = [5, 5, 5, 7, 5, 8]
TARGET_FREQS = [81.43, 109.47, 146.13, 195.49, 246.03, 328.43]

# ─────────────────────────────────────────────────────────────────────
# 2. Musical vocabulary
# ─────────────────────────────────────────────────────────────────────
CHORDS = {
    "A": [0, 0, 2, 2, 2, 0], "Am": [0, 0, 2, 2, 1, 0],
    "B": [2, 2, 4, 4, 4, 2], "Bm": [2, 2, 4, 4, 3, 2],
    "C": [0, 3, 2, 0, 1, 0], "Cm": [3, 3, 5, 5, 4, 3],
    "D": [2, 0, 0, 2, 3, 2], "Dm": [1, 0, 0, 2, 3, 1],
    "E": [0, 2, 2, 1, 0, 0], "Em": [0, 2, 2, 0, 0, 0],
    "F": [1, 3, 3, 2, 1, 1], "Fm": [1, 3, 3, 1, 1, 1],
    "G": [3, 2, 0, 0, 0, 3], "Gm": [3, 5, 5, 3, 3, 3],
}

CHORD_ROOTS = {
    "A": 2, "Am": 2, "B": 2, "Bm": 2, "C": 2, "Cm": 2,
    "D": 3, "Dm": 3,
    "E": 1, "Em": 1, "F": 1, "Fm": 1, "G": 1, "Gm": 1,
}

PATTERNS = {
    "Forward_Roll":     [["B"], ["I"], ["M"], ["R"], ["B"], ["I"], ["M"], ["R"]],
    "Backward_Roll":    [["B"], ["R"], ["M"], ["I"], ["B"], ["R"], ["M"], ["I"]],
    "Forward_Backward": [["B"], ["I"], ["M"], ["R"], ["M"], ["I"], ["B"], ["I"]],
    "Travis_Style":     [["B"], ["M"], ["I"], ["B"], ["R"], ["M"], ["B"], ["I"]],
    "Pinch_Pattern":    [["B", "R"], ["I"], ["M"], ["I"], ["B", "R"], ["I"], ["M"], ["I"]],
    "Outside_In":       [["B"], ["R"], ["M"], ["I"], ["M"], ["R"], ["B"], ["R"]],
    "Waltz_6_8":        [["B"], ["I"], ["M"], ["R"], ["M"], ["I"]],
    "Block_Chords":     [["B", "I", "M", "R"], ["I"], ["B"], ["I"],
                         ["B", "I", "M", "R"], ["I"], ["B"], ["I"]],
}

MAX_PLUCKS = 40

# Reparameterisation bounds (must match dereparameterize in inference)
AMP_MIN, AMP_MAX = 0.3, 1.4
DUR_MIN, DUR_MAX = 0.0012, 0.0018
TIME_MIN, TIME_MAX = 0.0, DURATION


def get_fret_position(fret_number, string_idx=None):
    if fret_number == 0:
        return 0.01
    math_fret = 1 - (2 ** (-fret_number / 12.0))
    if string_idx is None:
        return math_fret - (0.0010 * (fret_number / 12.0))
    comp_factors = [0.0015, 0.0012, 0.0010, 0.0008, 0.0006, 0.0005]
    return math_fret - (comp_factors[string_idx - 1] * (fret_number / 12.0))


def resolve_pattern(pattern_name, root_string):
    pattern = PATTERNS[pattern_name]
    resolved = []
    for beat in pattern:
        strings_to_pluck = []
        for char in beat:
            if char == "B":
                strings_to_pluck.append(root_string)
            elif char == "I":
                strings_to_pluck.append(4)
            elif char == "M":
                strings_to_pluck.append(5)
            elif char == "R":
                strings_to_pluck.append(6)
        resolved.append(strings_to_pluck)
    return resolved


# ─────────────────────────────────────────────────────────────────────
# 3. Reparameterisation: guitar_tab → flat tensor
# ─────────────────────────────────────────────────────────────────────
def scale_to_unit(value, vmin, vmax):
    """Scale [vmin, vmax] → [-1, 1]."""
    return ((value - vmin) / (vmax - vmin)) * 2.0 - 1.0


def reparameterize_to_tensor(guitar_tab, max_plucks=MAX_PLUCKS):
    """
    Convert a variable-length list of pluck events into a fixed-size
    (max_plucks, 31)-shaped tensor, then flatten to (max_plucks * 31,).

    Per-pluck layout (31 dims):
        [0]      exists flag (1.0 if real event, 0.0 if padding)
        [1]      start_time scaled to [-1, 1]
        [2:8]    string index one-hot (6 dims, strings 1-6)
        [8:29]   fret one-hot (21 dims, frets 0-20)
        [29]     amplitude scaled to [-1, 1]
        [30]     pluck_duration scaled to [-1, 1]
    """
    params = torch.zeros((max_plucks, 31), dtype=torch.float32)

    for i, event in enumerate(guitar_tab):
        if i >= max_plucks:
            break

        t_start, s_idx, fret, _step_dur, amp, _exp_f0, pluck_dur = event

        params[i, 0] = 1.0  # exists
        params[i, 1] = scale_to_unit(t_start, TIME_MIN, TIME_MAX)
        params[i, 2 + (s_idx - 1)] = 1.0  # string one-hot
        params[i, 8 + fret] = 1.0  # fret one-hot
        params[i, 29] = scale_to_unit(amp, AMP_MIN, AMP_MAX)
        params[i, 30] = scale_to_unit(pluck_dur, DUR_MIN, DUR_MAX)

    return params.view(-1)  # (1240,)


def dereparameterize_tensor(flat_params):
    """Inverse of reparameterize_to_tensor — recover guitar_tab list."""
    params = flat_params.view(MAX_PLUCKS, 31)
    events = []
    for i in range(MAX_PLUCKS):
        if params[i, 0] < 0.5:
            continue
        t_start = (params[i, 1].item() + 1.0) / 2.0 * (TIME_MAX - TIME_MIN) + TIME_MIN
        s_idx = params[i, 2:8].argmax().item() + 1
        fret = params[i, 8:29].argmax().item()
        amp = (params[i, 29].item() + 1.0) / 2.0 * (AMP_MAX - AMP_MIN) + AMP_MIN
        pluck_dur = (params[i, 30].item() + 1.0) / 2.0 * (DUR_MAX - DUR_MIN) + DUR_MIN
        exp_f0 = TARGET_FREQS[s_idx - 1] * (2 ** (fret / 12.0))
        step_dur = 0.0  # not used for resynthesis
        events.append([t_start, s_idx, fret, step_dur, amp, exp_f0, pluck_dur])
    return events


# ─────────────────────────────────────────────────────────────────────
# 4. NESS XML generation
# ─────────────────────────────────────────────────────────────────────
def write_instrument_xml(path):
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<instrument>\n')
        f.write(f"  <samplerate>{SAMPLE_RATE_NESS}</samplerate>\n  <guitar>\n")
        for i in range(NUM_STRINGS):
            f.write(f'    <guitarString name="string{i+1}">\n')
            f.write(f"      <length>{L_FIXED}</length>")
            f.write(f"<youngsModulus>{E_FIXED:.1e}</youngsModulus>\n")
            f.write(f"      <tension>{TENSIONS[i]:.4f}</tension>")
            f.write(f"<radius>{RADII[i]}</radius>\n")
            f.write(f"      <density>{RHO_FIXED}</density>")
            f.write(f"<T60_0>15.0</T60_0>")
            f.write(f"<T60_1000>{T60_1000_VALS[i]}</T60_1000>\n")
            f.write(f"    </guitarString>\n")

        f.write("    <backboard><b0>-0.002</b0><b1>-0.001</b1>")
        f.write("<b2>-0.0002</b2></backboard>\n")
        for fret_num in range(1, 21):
            pos = get_fret_position(fret_num)
            f.write(f"    <fret><pos>{pos:.6f}</pos><height>-0.001</height></fret>\n")
        f.write("    <Kb>1e10</Kb><alphab>1.3</alphab><betab>10.0</betab>\n")
        f.write("    <finger><Mf>0.005</Mf><Kf>1e7</Kf>")
        f.write("<alphaf>3.3</alphaf><betaf>100.0</betaf></finger>\n")
        f.write("  </guitar>\n")

        for i in range(NUM_STRINGS):
            pan = np.linspace(-0.8, 0.8, NUM_STRINGS)[i]
            f.write(f'  <output interpolated="true">')
            f.write(f"<component>string{i+1}</component>")
            f.write(f"<x>0.9</x><pan>{pan:.2f}</pan></output>\n")
        f.write("</instrument>\n")


def write_score_xml(path, guitar_tab, chord_frets):
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<score>\n')
        f.write(f"  <duration>{DURATION}</duration>\n")

        # Plucks
        for e in guitar_tab:
            t, s_idx, fret, dur, amp, _, pluck_dur = e
            f.write(f'  <pluck interpolated="true">\n')
            f.write(f"    <component>string{s_idx}</component>\n")
            f.write(f"    <x>0.8</x><startTime>{t:.4f}</startTime>\n")
            f.write(f"    <duration>{pluck_dur:.6f}</duration>")
            f.write(f"<amplitude>{amp:.4f}</amplitude>\n")
            f.write(f"  </pluck>\n")

        # Left hand fingers (staggered to avoid mesh shockwave)
        for s_idx in range(1, NUM_STRINGS + 1):
            fret = chord_frets[s_idx - 1]
            if fret > 0:
                pos = get_fret_position(fret, s_idx) - 0.01
                settle_time = 0.02 + (s_idx * 0.02)
                f.write(f"  <finger>\n")
                f.write(f"    <component>string{s_idx}</component>\n")
                f.write(f"    <initialPosition>{pos:.6f}</initialPosition>")
                f.write(f"<initialVelocity>0.0</initialVelocity>\n")
                f.write(f"    <gesture>\n")
                f.write(f"      <time>0.0</time><position>{pos:.6f}</position>")
                f.write(f"<force>0.0</force>\n")
                f.write(f"      <time>{settle_time:.4f}</time>")
                f.write(f"<position>{pos:.6f}</position><force>1.0</force>\n")
                f.write(f"      <time>{DURATION:.4f}</time>")
                f.write(f"<position>{pos:.6f}</position><force>1.0</force>\n")
                f.write(f"    </gesture>\n  </finger>\n")

        f.write("</score>\n")


# ─────────────────────────────────────────────────────────────────────
# 5. Single sample generation
# ─────────────────────────────────────────────────────────────────────
def generate_one_sample(sample_idx, output_dir, ness_binary, seed=None):
    """Generate one (audio, params) pair. Returns metadata dict or None on failure."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**31))

    chord_name = random.choice(list(CHORDS.keys()))
    chord_frets = CHORDS[chord_name]
    root_string = CHORD_ROOTS[chord_name]
    pattern_name = random.choice(list(PATTERNS.keys()))
    resolved_pattern = resolve_pattern(pattern_name, root_string)

    bpm = random.uniform(70.0, 280.0)
    step_sec = (60.0 / bpm) / 2.0
    base_amplitude = random.uniform(0.5, 0.8)

    guitar_tab = []
    current_time = 0.25  # let fingers settle
    step_idx = 0

    while current_time < DURATION - 0.5 and len(guitar_tab) < MAX_PLUCKS:
        strings_to_pluck = resolved_pattern[step_idx % len(resolved_pattern)]
        for s_idx in strings_to_pluck:
            if len(guitar_tab) >= MAX_PLUCKS:
                break
            fret = chord_frets[s_idx - 1]
            exp_freq = TARGET_FREQS[s_idx - 1] * (2 ** (fret / 12.0))
            amp = base_amplitude + random.uniform(-0.05, 0.05)
            if s_idx == root_string:
                amp += 0.10
            amp = max(AMP_MIN, min(AMP_MAX, amp))
            pluck_dur = random.uniform(DUR_MIN, DUR_MAX)
            guitar_tab.append(
                [current_time, s_idx, fret, step_sec, amp, exp_freq, pluck_dur]
            )
        current_time += step_sec
        step_idx += 1

    # ── Run NESS in a temp directory ──
    with tempfile.TemporaryDirectory() as tmpdir:
        inst_path = os.path.join(tmpdir, "instrument.xml")
        score_path = os.path.join(tmpdir, "score.xml")
        basename = os.path.join(tmpdir, "out")

        write_instrument_xml(inst_path)
        write_score_xml(score_path, guitar_tab, chord_frets)

        try:
            subprocess.run(
                [ness_binary, "-i", inst_path, "-s", score_path,
                 "-o", basename, "-c", "stereo"],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            return None

        # Find and load the stereo mix
        mix_path = os.path.join(tmpdir, "out-mix.wav")
        if not os.path.exists(mix_path):
            # Sometimes NESS names it without -mix for single output
            candidates = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
            if not candidates:
                return None
            mix_path = os.path.join(tmpdir, candidates[0])

        try:
            y, sr = sf.read(mix_path)
        except Exception:
            return None

    # ── Resample 48kHz → 44.1kHz ──
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)  # mono → stereo
    if sr != SAMPLE_RATE_OUT:
        # Resample each channel
        y_left = librosa.resample(y[:, 0], orig_sr=sr, target_sr=SAMPLE_RATE_OUT)
        y_right = librosa.resample(y[:, 1], orig_sr=sr, target_sr=SAMPLE_RATE_OUT)
        y = np.stack([y_left, y_right], axis=-1)

    # Trim/pad to exactly 4s at 44.1kHz
    target_len = int(DURATION * SAMPLE_RATE_OUT)
    if y.shape[0] > target_len:
        y = y[:target_len]
    elif y.shape[0] < target_len:
        pad = np.zeros((target_len - y.shape[0], y.shape[1]))
        y = np.concatenate([y, pad], axis=0)

    # ── Save outputs ──
    sample_name = f"sample_{sample_idx:06d}"
    audio_path = os.path.join(output_dir, f"{sample_name}.wav")
    params_path = os.path.join(output_dir, f"{sample_name}.pt")
    meta_path = os.path.join(output_dir, f"{sample_name}.json")

    sf.write(audio_path, y, SAMPLE_RATE_OUT)
    param_tensor = reparameterize_to_tensor(guitar_tab)
    torch.save(param_tensor, params_path)

    meta = {
        "chord": chord_name,
        "pattern": pattern_name,
        "bpm": bpm,
        "num_plucks": len(guitar_tab),
        "base_amplitude": base_amplitude,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return meta


def _worker(args):
    """Wrapper for ProcessPoolExecutor."""
    idx, output_dir, ness_binary, base_seed = args
    seed = base_seed + idx if base_seed is not None else None
    return generate_one_sample(idx, output_dir, ness_binary, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Generate NESS guitar dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--ness_binary", type=str, default="./ness/ness-framework")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = [
        (i, args.output_dir, args.ness_binary, args.seed)
        for i in range(args.num_samples)
    ]

    successes = 0
    failures = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        pbar = tqdm(total=len(tasks), desc="Generating samples")
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                successes += 1
            else:
                failures += 1
            pbar.update(1)
            pbar.set_postfix(ok=successes, fail=failures)
        pbar.close()

    # Save dataset-level metadata
    dataset_meta = {
        "num_samples": successes,
        "num_failures": failures,
        "sample_rate": SAMPLE_RATE_OUT,
        "duration": DURATION,
        "max_plucks": MAX_PLUCKS,
        "params_per_pluck": 31,
        "total_param_dim": MAX_PLUCKS * 31,
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\nDataset generation complete: {successes} samples, {failures} failures")


if __name__ == "__main__":
    main()