# crypto_pipeline/ml/deep_learning/trainer.py

"""
trainer.py
----------
Shared training loop for deep learning models (PDF heading 7).

One epoch loop used by every architecture (mlp.py, lstm.py, gru.py) via
base_network.py -- optimizer/scheduler step, forward/backward pass,
early stopping (via callbacks.py's EarlyStopping) all live here once,
instead of being copy-pasted into each architecture file. Architecture
files only ever define the network shape; this file is what actually
runs it.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from crypto_pipeline.ml.deep_learning.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def train_network(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    val_loader: Optional[DataLoader] = None,
    scheduler=None,
    early_stopping_patience: Optional[int] = 10,
) -> nn.Module:
    """
    Run the full training loop for `epochs`, with optional validation,
    LR scheduling, and early stopping.

    Args:
        model: an already-constructed, uninitialized-weights nn.Module
        train_loader / val_loader: DataLoaders yielding (X_batch, y_batch)
        optimizer: already built (see base_network.py's _build_optimizer)
        loss_fn: from losses.py's get_loss_fn()
        device: cpu or cuda, model must already be moved to it
        epochs: max number of epochs to run
        scheduler: optional torch.optim.lr_scheduler instance, or None
        early_stopping_patience: epochs with no val improvement before
            stopping; 0/None disables it (only takes effect if val_loader given)

    Returns:
        The same model, with weights set to the best-validation-loss
        checkpoint if early stopping was active, otherwise the final
        epoch's weights.
    """
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = _run_epoch(model, train_loader, device, loss_fn, optimizer=optimizer)
        log_line = f"Epoch {epoch}/{epochs} - train_loss={train_loss:.6f}"

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = _run_epoch(model, val_loader, device, loss_fn, optimizer=None)
            log_line += f" - val_loss={val_loss:.6f}"

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            early_stopping.step(val_loss, model)
            if early_stopping.should_stop:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no val improvement for {early_stopping_patience} epochs)"
                )
                break
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        logger.info(log_line)

    early_stopping.restore_best(model)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    """One pass over `loader`. Backprop happens only if optimizer is given (training vs eval)."""
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        output = model(X_batch)
        loss = loss_fn(output, y_batch)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)