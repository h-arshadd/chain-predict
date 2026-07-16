# crypto_pipeline/ml/deep_learning/callbacks.py

"""
callbacks.py
------------
Training callbacks for deep learning models (PDF heading 7).

Currently just early stopping, since that's the only callback the PDF
asks for ("Epochs, Early stopping" under configurable options). Lives
as its own small class so trainer.py's epoch loop stays readable --
it just calls callback.step(val_loss) and checks callback.should_stop
instead of tracking best-loss/patience-counter state inline.
"""

from typing import Dict, Optional

import torch
from torch import nn


class EarlyStopping:
    """
    Tracks validation loss across epochs and signals when to stop.

    Args:
        patience: number of epochs with no improvement before stopping.
            0 or None disables early stopping entirely (trains for the
            full configured number of epochs regardless of val_loss).
        min_delta: minimum decrease in val_loss to count as an improvement.
    """

    def __init__(self, patience: Optional[int] = 10, min_delta: float = 0.0):
        self.patience = patience or 0
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.epochs_without_improvement = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """Call once per epoch with the current validation loss."""
        if not self.patience:
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-seen weights back into `model`, if any were captured."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)