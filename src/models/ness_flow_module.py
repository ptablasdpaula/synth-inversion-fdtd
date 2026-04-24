"""
NESSFlowMatchingModule: Lightning module for conditional flow matching
over NESS guitar synthesizer parameters.

Adapted from Ben Hayes' SurgeFlowMatchingModule with:
  - NESS-specific parameter dimensions (40 plucks × 31 dims = 1240)
  - Guitar-aware validation metrics
  - Dereparameterisation utilities for inference
"""

import math
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm


# ─────────────────────────────────────────────────────────────────────
# CFG + RK4 helpers (from Ben's code)
# ─────────────────────────────────────────────────────────────────────
def call_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    y_c = f(x, t, conditioning)
    y_u = f(x, t, None)
    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    f = partial(call_with_cfg, f, conditioning=conditioning, cfg_strength=cfg_strength)
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class NESSFlowMatchingModule(LightningModule):
    """
    Conditional flow matching for NESS guitar parameter estimation.

    Architecture:
        encoder:      AudioSpectrogramTransformer (mel → conditioning embeddings)
        vector_field: ApproxEquivTransformer with LearntProjection (Param2Tok)
                      or KSinParamToTokenProjection

    The flow transforms Gaussian noise → parameter vectors, conditioned on
    mel spectrograms of guitar audio.
    """

    def __init__(
        self,
        encoder: nn.Module,
        vector_field: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        conditioning: Literal["mel"] = "mel",
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 2.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 2.0,
        compile: bool = False,
        num_params: int = 1240,
        max_plucks: int = 40,
        params_per_pluck: int = 31,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.vector_field = vector_field

    # ── Flow matching core ──

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, 1, device=device)

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _rectified_probability_path(self, x0, x1, t):
        return x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

    def _sample_probability_path(self, x0, x1, t):
        return self._rectified_probability_path(x0, x1, t)

    def _rectified_vector_field(self, x0, x1):
        return x1 - x0

    def _evaluate_target_field(self, x0, x1, x_t, t):
        return self._rectified_vector_field(x0, x1)

    def _get_conditioning_from_batch(self, batch):
        return batch["mel_spec"]

    # ── Training ──

    def _train_step(self, batch):
        conditioning = self._get_conditioning_from_batch(batch)
        params = batch["params"]
        noise = batch["noise"]

        # Encode audio → conditioning vector(s)
        conditioning = self.encoder(conditioning)
        z = self.vector_field.apply_dropout(
            conditioning, self.hparams.cfg_dropout_rate
        )

        with torch.no_grad():
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)
            x0 = noise
            x1 = params
            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        prediction = self.vector_field(x_t, t, z)

        loss = (prediction - target).square().mean(dim=-1)
        loss = loss * w
        loss = loss.mean()

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return loss, penalty

    def training_step(self, batch, batch_idx):
        loss, penalty = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if penalty is not None and penalty != 0.0:
            self.log("train/penalty", penalty, on_step=True, on_epoch=True)
            return loss + penalty
        return loss

    # ── Sampling / inference ──

    def _warp_time(self, t):
        return t

    def _sample(self, conditioning, noise, steps, cfg_strength):
        if conditioning is not None:
            conditioning = self.encoder(conditioning)

        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps
        sample = noise

        for _ in range(steps):
            warped_t = self._warp_time(t)
            warped_t_plus_dt = self._warp_time(t + dt)
            warped_dt = warped_t_plus_dt - warped_t
            sample = rk4_with_cfg(
                self.vector_field, sample, warped_t, warped_dt,
                conditioning, cfg_strength,
            )
            t = t + dt
        return sample

    # ── Validation ──

    def validation_step(self, batch, batch_idx):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
        )

        per_param_mse = (pred_params - batch["params"]).square().mean(dim=0)
        param_mse = per_param_mse.mean()
        self.log("val/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        # Per-field breakdown (averaged over pluck slots)
        params_2d = pred_params.view(-1, self.hparams.max_plucks, self.hparams.params_per_pluck)
        target_2d = batch["params"].view(-1, self.hparams.max_plucks, self.hparams.params_per_pluck)
        diff = (params_2d - target_2d).square().mean(dim=(0, 1))

        self.log("val/exists_mse", diff[0], on_step=False, on_epoch=True)
        self.log("val/time_mse", diff[1], on_step=False, on_epoch=True)
        self.log("val/string_mse", diff[2:8].mean(), on_step=False, on_epoch=True)
        self.log("val/fret_mse", diff[8:29].mean(), on_step=False, on_epoch=True)
        self.log("val/amp_mse", diff[29], on_step=False, on_epoch=True)
        self.log("val/dur_mse", diff[30], on_step=False, on_epoch=True)

        return {"param_mse": param_mse, "per_param_mse": per_param_mse}

    # ── Test ──

    def test_step(self, batch, batch_idx):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.test_sample_steps,
            self.hparams.test_cfg_strength,
        )
        param_mse = (pred_params - batch["params"]).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)
        return param_mse

    # ── Predict (for inference scripts) ──

    def predict_step(self, batch, batch_idx):
        conditioning = self._get_conditioning_from_batch(batch)
        return (
            self._sample(
                conditioning,
                torch.randn(
                    conditioning.shape[0] if conditioning.ndim == 2 else conditioning.shape[0],
                    self.hparams.num_params,
                    device=conditioning.device if hasattr(conditioning, 'device') else self.device,
                ),
                self.hparams.test_sample_steps,
                self.hparams.test_cfg_strength,
            ),
            batch,
        )

    # ── Setup & optimizers ──

    def setup(self, stage: str) -> None:
        if not self.hparams.compile:
            return
        self.vector_field = torch.compile(self.vector_field)
        self.encoder = torch.compile(self.encoder)

    def on_before_optimizer_step(self, optimizer) -> None:
        vf_norms = grad_norm(self.vector_field, 2.0)
        enc_norms = grad_norm(self.encoder, 2.0)
        self.log_dict(
            {f"grad/vf_{k}": v for k, v in vf_norms.items()},
            on_step=True, on_epoch=False,
        )
        self.log_dict(
            {f"grad/enc_{k}": v for k, v in enc_norms.items()},
            on_step=True, on_epoch=False,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-10, 1.0, self.hparams.warmup_steps
            )
        else:
            warmup = None

        if self.hparams.scheduler is not None:
            sched = self.hparams.scheduler(optimizer=optimizer)
        else:
            sched = None

        if warmup is not None and sched is None:
            sched = warmup
        elif warmup is not None and sched is not None:
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, sched],
                milestones=[self.hparams.warmup_steps],
            )

        if sched is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}