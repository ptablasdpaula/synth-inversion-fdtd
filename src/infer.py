#!/usr/bin/env python3
"""
Inference script: given audio file(s), predict NESS guitar parameters
using a trained flow matching model.

Usage:
    python infer.py --checkpoint outputs/2026-04-24_12-00-00/checkpoints/last.ckpt \
                    --audio_dir ./test_audio \
                    --output_dir ./predictions \
                    --num_samples 5 \
                    --steps 100 \
                    --cfg_strength 2.0
"""

import argparse
import glob
import json
import os

import torch
import torchaudio
from tqdm import tqdm

from generate_dataset import (
    dereparameterize_tensor,
    write_instrument_xml,
    write_score_xml,
    CHORDS,
    MAX_PLUCKS,
    SAMPLE_RATE_OUT,
    DURATION,
)
from src.models.ness_flow_module import NESSFlowMatchingModule


def load_and_preprocess_audio(
    audio_path: str,
    sample_rate: int = 44100,
    mel_mean: float = -5.0,
    mel_std: float = 2.5,
    target_frames: int = 401,
):
    """Load audio file and compute standardised stereo log-mel spectrogram."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    n_fft = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=128,
        f_min=60.0,
        f_max=16000.0,
    )

    mel = mel_transform(waveform)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    log_mel = (log_mel - mel_mean) / mel_std

    T = log_mel.shape[-1]
    if T < target_frames:
        pad = torch.zeros(log_mel.shape[0], log_mel.shape[1], target_frames - T)
        log_mel = torch.cat([log_mel, pad], dim=-1)
    elif T > target_frames:
        log_mel = log_mel[:, :, :target_frames]

    return log_mel  # (2, 128, 401)


def main():
    parser = argparse.ArgumentParser(description="NESS guitar parameter inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of flow samples per audio (for diversity)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--mel_mean", type=float, default=-5.0)
    parser.add_argument("--mel_std", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = NESSFlowMatchingModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.eval()
    model.to(device)

    # Find audio files
    audio_files = sorted(
        glob.glob(os.path.join(args.audio_dir, "*.wav"))
    )
    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Inferring"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]

        mel = load_and_preprocess_audio(
            audio_path,
            mel_mean=args.mel_mean,
            mel_std=args.mel_std,
        )
        mel = mel.unsqueeze(0).to(device)  # (1, 2, 128, 401)

        # Sample multiple predictions
        mel_batch = mel.repeat(args.num_samples, 1, 1, 1)
        noise = torch.randn(
            args.num_samples, model.hparams.num_params, device=device
        )

        with torch.no_grad():
            pred_params = model._sample(
                mel_batch, noise, args.steps, args.cfg_strength
            )

        # Dereparameterise and save
        for s in range(args.num_samples):
            flat = pred_params[s].cpu()
            events = dereparameterize_tensor(flat)

            out_name = f"{basename}_sample{s:02d}"

            # Save raw tensor
            torch.save(flat, os.path.join(args.output_dir, f"{out_name}.pt"))

            # Save human-readable JSON
            readable = []
            for ev in events:
                readable.append({
                    "time": round(ev[0], 4),
                    "string": int(ev[1]),
                    "fret": int(ev[2]),
                    "amplitude": round(ev[4], 4),
                    "pluck_duration": round(ev[6], 6),
                    "expected_f0": round(ev[5], 2),
                })
            with open(os.path.join(args.output_dir, f"{out_name}.json"), "w") as f:
                json.dump(readable, f, indent=2)

            # Optionally write NESS XML for resynthesis
            xml_dir = os.path.join(args.output_dir, "ness_xml", out_name)
            os.makedirs(xml_dir, exist_ok=True)
            write_instrument_xml(os.path.join(xml_dir, "instrument.xml"))

            # We need chord_frets for the finger gestures — infer from predicted events
            chord_frets = [0] * 6
            for ev in events:
                s_idx = int(ev[1])
                fret = int(ev[2])
                chord_frets[s_idx - 1] = fret

            write_score_xml(
                os.path.join(xml_dir, "score.xml"),
                events,
                chord_frets,
            )

    print(f"Predictions saved to {args.output_dir}")


if __name__ == "__main__":
    main()