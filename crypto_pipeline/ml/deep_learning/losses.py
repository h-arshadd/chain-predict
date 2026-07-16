# crypto_pipeline/ml/deep_learning/losses.py

"""
losses.py
---------
Loss functions for deep learning models (PDF heading 7).

One place both regression and classification networks pull their loss
from, by name -- base_regressor_network.py / base_classifier_network.py
call get_loss_fn("mse") / get_loss_fn("cross_entropy") instead of each
hardcoding `nn.MSELoss()` / `nn.CrossEntropyLoss()` inline. Keeps the
loss choice itself config-driven and swappable without touching either
base network file.
"""

from typing import Dict, Type

from torch import nn

LOSSES: Dict[str, Type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "cross_entropy": nn.CrossEntropyLoss,
}


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    """
    Build a loss function by name.

    Args:
        name: key into LOSSES, e.g. "mse" or "cross_entropy"
        **kwargs: forwarded to the loss class constructor (e.g.
            huber's `delta`), empty for most losses.

    Returns:
        An instantiated torch loss module.
    """
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(LOSSES.keys())}")
    return LOSSES[name](**kwargs)