"""
Utility functions: logging, Hydra helpers, callbacks.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import lightning as L
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import hydra


class RankedLogger(logging.LoggerAdapter):
    """Logger that only logs on rank 0 in distributed settings."""

    def __init__(self, name: str = __name__, rank_zero_only: bool = True):
        logger = logging.getLogger(name)
        super().__init__(logger, {})
        self.rank_zero_only = rank_zero_only

    def log(self, level, msg, *args, **kwargs):
        if self.rank_zero_only:
            kwargs.setdefault("stacklevel", 2)
            # Only log on rank 0
            if not L.pytorch.utilities.rank_zero_only.rank == 0:
                return
        super().log(level, msg, *args, **kwargs)


def extras(cfg: DictConfig):
    """Optional utilities before training (print config, etc.)."""
    if cfg.get("print_config"):
        print(OmegaConf.to_yaml(cfg))


def instantiate_callbacks(callbacks_cfg: Optional[DictConfig]) -> List[Callback]:
    """Instantiate callbacks from Hydra config."""
    callbacks = []
    if not callbacks_cfg:
        return callbacks
    for _, cb_conf in callbacks_cfg.items():
        if cb_conf is not None and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: Optional[DictConfig]) -> List[Logger]:
    """Instantiate loggers from Hydra config."""
    loggers = []
    if not logger_cfg:
        return loggers
    for _, lg_conf in logger_cfg.items():
        if lg_conf is not None and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]):
    """Log hyperparameters to all loggers."""
    hparams = {}
    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    hparams["cfg"] = cfg

    trainer = object_dict["trainer"]
    model = object_dict["model"]

    if trainer.logger:
        trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely get a metric value from dict."""
    if not metric_name:
        return None
    if metric_name in metric_dict:
        return metric_dict[metric_name].item() if hasattr(metric_dict[metric_name], 'item') else metric_dict[metric_name]
    return None


def register_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def task_wrapper(func):
    """Wrapper for training task to handle exceptions gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Training failed: {e}")
            raise
    return wrapper