#!/usr/bin/env python3
"""
Compute dataset-level mel spectrogram statistics for standardization.

Run after generate_dataset.py:
    python compute_stats.py --data_dir ./data/ness_guitar --num_samples 5000
"""

import argparse
import glob
import os
import json

import torch
import torchaudio
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of samples to use for statistics (use -1 for all)")
    args = parser.parse_args()

    audio_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if args.num_samples > 0:
        audio_files = audio_files[: args.num_samples]

    sample_rate = 44100
    n_fft = int(0.025 * sample_rate)   # 25ms = 1102 samples
    hop_length = int(0.010 * sample_rate)  # 10ms = 441 samples

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        f_min=60.0,
        f_max=16000.0,
    )

    # Online mean/variance computation (Welford's algorithm)
    count = 0
    mean = 0.0
    m2 = 0.0

    for path in tqdm(audio_files, desc="Computing statistics"):
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        mel = mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        values = log_mel.flatten()
        for v in [values.mean().item()]:  # batch-level contribution
            pass

        # Per-element statistics
        batch_mean = log_mel.mean().item()
        batch_var = log_mel.var().item()
        batch_count = log_mel.numel()

        # Welford update
        delta = batch_mean - mean
        count += batch_count
        mean += delta * batch_count / count
        m2 += batch_var * batch_count + delta**2 * batch_count * (count - batch_count) / count

    variance = m2 / count
    std = variance**0.5

    stats = {
        "mel_mean": mean,
        "mel_std": std,
        "num_files_used": len(audio_files),
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": 128,
        "f_min": 60.0,
        "f_max": 16000.0,
    }

    out_path = os.path.join(args.data_dir, "mel_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Mel statistics saved to {out_path}")
    print(f"  mean = {mean:.4f}")
    print(f"  std  = {std:.4f}")


if __name__ == "__main__":
    main()