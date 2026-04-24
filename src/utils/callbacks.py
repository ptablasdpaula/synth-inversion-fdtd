import os
import tempfile
import subprocess
import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from data.generate_dataset import (
    dereparameterize_tensor,
    write_instrument_xml,
    write_score_xml,
)

class LogPerParamMSE(Callback):
    """
    Logs per-parameter-field MSE breakdown on validation.
    Fields: exists, time, string, fret, amplitude, duration.
    """

    def __init__(self, max_plucks: int = 40, params_per_pluck: int = 31):
        super().__init__()
        self.max_plucks = max_plucks
        self.params_per_pluck = params_per_pluck

    def on_validation_epoch_end(self, trainer, pl_module):
        # The per-field metrics are already logged inside validation_step
        pass


class LogValidationAudioCallback(Callback):
    """
    Runs full inference on one validation sample per epoch, synthesizes
    the audio using the NESS FDTD binary, and logs the audio to WandB.
    """
    def __init__(
        self, 
        ness_binary: str = "./ness/ness-framework", 
        num_steps: int = 50, 
        cfg_strength: float = 2.0
    ):
        super().__init__()
        self.ness_binary = ness_binary
        self.num_steps = num_steps
        self.cfg_strength = cfg_strength
        self.validation_batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Cache the very first validation batch so we can sample from it at the end of the epoch
        if batch_idx == 0:
            self.validation_batch = batch

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Skip during the initial sanity check or if no batch was cached
        if trainer.sanity_checking or self.validation_batch is None:
            return

        # Check if WandB is active
        wandb_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                wandb_logger = logger
                break
        
        if wandb_logger is None:
            return

        device = pl_module.device
        
        # 1. Grab ONE mel spectrogram from the cached validation batch
        mel = self.validation_batch["mel_spec"][0:1].to(device)
        noise = torch.randn(1, pl_module.hparams.num_params, device=device)

        # 2. Run Flow Matching Inference
        with torch.no_grad():
            pred_params = pl_module._sample(mel, noise, self.num_steps, self.cfg_strength)

        # 3. Dereparameterize the flat tensor into readable events
        flat = pred_params[0].cpu()
        events = dereparameterize_tensor(flat)

        chord_frets = [0] * 6
        for ev in events:
            s_idx = int(ev[1])
            fret = int(ev[2])
            chord_frets[s_idx - 1] = fret

        # 4. Synthesize Audio in a Temporary Directory
        with tempfile.TemporaryDirectory() as tmpdir:
            inst_xml = os.path.join(tmpdir, "instrument.xml")
            score_xml = os.path.join(tmpdir, "score.xml")
            out_wav = os.path.join(tmpdir, "output.wav")

            write_instrument_xml(inst_xml)
            write_score_xml(score_xml, events, chord_frets)

            # Run the NESS C++ binary
            try:
                subprocess.run(
                    [self.ness_binary, inst_xml, score_xml, out_wav],
                    check=True,
                    stdout=subprocess.DEVNULL,  # Suppress terminal spam
                    stderr=subprocess.DEVNULL
                )
                
                # 5. Log the resulting WAV file to WandB
                if os.path.exists(out_wav):
                    wandb_logger.experiment.log({
                        "val/predicted_audio": wandb.Audio(
                            out_wav, 
                            sample_rate=44100, 
                            caption=f"Epoch {trainer.current_epoch}"
                        )
                    })
            except Exception as e:
                # If NESS fails (e.g. invalid parameters), we don't want to crash the whole training loop
                print(f"\n[Warning] Audio synthesis failed at epoch {trainer.current_epoch}: {e}")