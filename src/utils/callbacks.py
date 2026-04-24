"""Custom callbacks for NESS guitar flow matching training."""

from lightning import Callback


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