"""
PyTorch Dataset & Lightning DataModule for the NESS FDTD guitar dataset.

Loads pre-generated (wav, pt) pairs, computes log-mel spectrograms on-the-fly,
and formats batches for the SurgeFlowMatchingModule.
"""

import glob
import json
import os
from typing import Optional

import torch
import torchaudio
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


class NESSGuitarDataset(Dataset):
    """
    Loads audio → log-mel spectrogram and pre-saved parameter tensors.

    Each __getitem__ returns a dict:
        {
            "mel_spec": (2, 128, T_frames),   # stereo log-mel
            "params":   (1240,),               # flat reparameterised params
            "noise":    (1240,),               # standard Gaussian noise
        }
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 60.0,
        f_max: float = 16000.0,
        mel_mean: float = -5.0,
        mel_std: float = 2.5,
        target_frames: int = 401,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.target_frames = target_frames

        # Discover samples by looking for .pt files
        self.param_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        assert len(self.param_files) > 0, f"No .pt files found in {data_dir}"

        # Mel spectrogram settings (matching Hayes et al.)
        if n_fft is None:
            n_fft = int(0.025 * sample_rate)  # 25ms window
        if hop_length is None:
            hop_length = int(0.010 * sample_rate)  # 10ms hop

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

        self.mel_mean = mel_mean
        self.mel_std = mel_std

    def __len__(self):
        return len(self.param_files)

    def _get_audio_path(self, param_path: str) -> str:
        return param_path.replace(".pt", ".wav")

    def __getitem__(self, idx):
        param_path = self.param_files[idx]
        audio_path = self._get_audio_path(param_path)

        # ── Load & process audio ──
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # Log-mel spectrogram: (2, 128, T)
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # Standardize
        log_mel = (log_mel - self.mel_mean) / self.mel_std

        # Pad/trim time axis to target_frames
        T = log_mel.shape[-1]
        if T < self.target_frames:
            pad = torch.zeros(
                log_mel.shape[0], log_mel.shape[1], self.target_frames - T
            )
            # Pad with the mean value (0 after standardization)
            log_mel = torch.cat([log_mel, pad], dim=-1)
        elif T > self.target_frames:
            log_mel = log_mel[:, :, : self.target_frames]

        # ── Load parameters ──
        params = torch.load(param_path, weights_only=True)  # (1240,)

        # ── Flow matching noise ──
        noise = torch.randn_like(params)

        return {
            "mel_spec": log_mel,      # (2, 128, 401)
            "params": params,          # (1240,)
            "noise": noise,            # (1240,)
        }


class NESSDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapping NESSGuitarDataset.

    Supports either:
      - A single data_dir with automatic 80/10/10 split
      - Separate train_dir / val_dir / test_dir
    """

    def __init__(
        self,
        data_dir: str = "./data/ness_guitar",
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        sample_rate: int = 44100,
        n_mels: int = 128,
        f_min: float = 60.0,
        f_max: float = 16000.0,
        mel_mean: float = -5.0,
        mel_std: float = 2.5,
        target_frames: int = 401,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        hp = self.hparams

        # Try loading computed stats
        stats_path = os.path.join(hp.data_dir, "mel_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            mel_mean = stats["mel_mean"]
            mel_std = stats["mel_std"]
        else:
            mel_mean = hp.mel_mean
            mel_std = hp.mel_std

        dataset_kwargs = dict(
            sample_rate=hp.sample_rate,
            n_mels=hp.n_mels,
            f_min=hp.f_min,
            f_max=hp.f_max,
            mel_mean=mel_mean,
            mel_std=mel_std,
            target_frames=hp.target_frames,
        )

        if hp.train_dir is not None:
            self.train_ds = NESSGuitarDataset(hp.train_dir, **dataset_kwargs)
            self.val_ds = NESSGuitarDataset(hp.val_dir, **dataset_kwargs)
            self.test_ds = NESSGuitarDataset(hp.test_dir, **dataset_kwargs)
        else:
            full = NESSGuitarDataset(hp.data_dir, **dataset_kwargs)
            n = len(full)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            n_test = n - n_train - n_val
            self.train_ds, self.val_ds, self.test_ds = (
                torch.utils.data.random_split(
                    full,
                    [n_train, n_val, n_test],
                    generator=torch.Generator().manual_seed(hp.seed),
                )
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )