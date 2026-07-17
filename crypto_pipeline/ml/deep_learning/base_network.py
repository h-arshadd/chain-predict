# crypto_pipeline/ml/deep_learning/base_network.py

"""
base_network.py
----------------
Common base class for every deep learning model (PDF heading 7),
matching the PDF's recommended tree: one base_network.py, not split
into separate regression/classification base files.

Same interface as ml/regressors/base_regressor.py and
ml/classifiers/base_classifier.py:
    train()
    predict()
    save()
    load()

Kept deliberately independent from BaseRegressor/BaseClassifier (per
the PDF: "Deep learning models should be implemented independently
from traditional machine learning models") since save()/load() need
torch.save/torch.load instead of joblib, and train() needs an actual
epoch loop instead of a single sklearn .fit() call.

Regression vs classification: this base class defaults to regression
behavior (loss="mse", scalar output, 1D predictions). Classifier
subclasses (see mlp.py's MLPClassifierModel, etc.) override four small
hooks -- _default_loss(), _output_dim(), _make_target_tensor(),
_postprocess_predictions() -- plus add predict_proba()/classes_.
Nothing here branches on "if classification: ..."; each concrete class
just is one or the other via which hooks it overrides. The epoch loop
itself lives in trainer.py, early stopping in callbacks.py, and loss
functions in losses.py -- this file wires hyperparams into those
three, builds the DataLoaders, and owns save()/load().
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crypto_pipeline.ml.deep_learning.losses import get_loss_fn
from crypto_pipeline.ml.deep_learning.trainer import train_network

logger = logging.getLogger(__name__)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

SCHEDULERS = {
    "none": None,
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def _require(hyperparams: dict, key: str):
    """
    Fetch `key` from hyperparams (i.e. ml/config.yaml's model.params),
    raising instead of silently falling back to a hardcoded default.
    Every deep learning hyperparameter must be set explicitly in
    config.yaml, same as every traditional regressor/classifier's params.
    """
    if key not in hyperparams:
        raise ValueError(
            f"Missing required deep learning param '{key}' in ml/config.yaml's "
            f"model.params -- deep learning hyperparameters are not defaulted, "
            f"they must be set explicitly."
        )
    return hyperparams[key]


class BaseNetwork(ABC):
    """
    Args:
        **hyperparams: all configurable per the PDF:
            hidden_layers (int), hidden_units (int or list[int]),
            activation (str, key into ACTIVATIONS),
            dropout (float), batch_norm (bool),
            optimizer (str, key into OPTIMIZERS), learning_rate (float),
            scheduler (str, key into SCHEDULERS) + scheduler_params (dict),
            batch_size (int), epochs (int),
            early_stopping_patience (int, 0/None disables it),
            loss (str, key into losses.py's LOSSES -- defaults to
                _default_loss() if not set),
            random_seed (int)
        Never inspected here beyond wiring -- subclasses read what they
        need from self.hyperparams in _build_network().
    """

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model: Optional[nn.Module] = None  # set by train()/load()
        self.input_dim: Optional[int] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # random_seed (like every other deep learning hyperparam) is
        # required, but only once training actually happens -- see
        # train() below. It's NOT required here in __init__, because
        # ml/persistence/model_loader.py constructs this class with zero
        # hyperparams (model_cls()) before calling .load(), which then
        # restores self.hyperparams from the saved checkpoint. Requiring
        # random_seed in __init__ would make every saved deep learning
        # model impossible to reload.

    # ------------------------------------------------------------------
    # Subclasses implement this (architecture)
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_network(self, input_dim: int) -> nn.Module:
        """Construct and return the torch.nn.Module for this architecture."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Regression/classification hooks -- default to regression;
    # classifier subclasses override all four (see mlp.py, lstm.py, gru.py)
    # ------------------------------------------------------------------
    def _default_loss(self) -> str:
        """Key into losses.py's LOSSES to use when hyperparams doesn't set 'loss' explicitly."""
        return "mse"

    def _output_dim(self) -> int:
        """Number of output units. Regression: always 1. Classifiers override to len(classes)."""
        return 1

    def _make_target_tensor(self, y: pd.Series) -> torch.Tensor:
        """Convert y into the tensor shape/dtype the loss function expects. Default: regression target."""
        return torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def _postprocess_predictions(self, raw_output: torch.Tensor) -> np.ndarray:
        """Convert raw network output into the standardized prediction format (PDF heading 8)."""
        # Regression default: predicted value -- flatten (n, 1) to a 1D array.
        return raw_output.cpu().numpy().reshape(-1)

    # ------------------------------------------------------------------
    # Training -- delegates the actual loop to trainer.train_network()
    # ------------------------------------------------------------------
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "BaseNetwork":
        """
        Fit the model on training data. X_val/y_val are optional -- if
        given, they drive early stopping and ReduceLROnPlateau; if not,
        the model just trains for the configured number of epochs.
        """
        self.input_dim = X_train.shape[1]

        seed = _require(self.hyperparams, "random_seed")
        torch.manual_seed(seed)

        self.model = self._build_network(self.input_dim).to(self.device)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        loss_fn = get_loss_fn(self.hyperparams["loss"] if "loss" in self.hyperparams else self._default_loss())

        self.model = train_network(
            model=self.model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=self.device,
            epochs=_require(self.hyperparams, "epochs"),
            val_loader=val_loader,
            scheduler=scheduler,
            early_stopping_patience=_require(self.hyperparams, "early_stopping_patience"),
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._require_trained()
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return self._postprocess_predictions(output)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _extra_save_state(self) -> dict:
        """
        Hook for subclasses to add extra keys to the saved checkpoint.
        Default: nothing extra (plain BaseNetwork regressors don't need
        this). BaseClassifierNetwork overrides this to persist
        _classes_ -- see that class for why.
        """
        return {}

    def _restore_extra_state(self, checkpoint: dict) -> None:
        """
        Hook for subclasses to restore whatever _extra_save_state()
        added, called by load() before _build_network() runs (some
        subclasses' _build_network(), e.g. BaseClassifierNetwork's,
        needs that restored state to size the output layer).
        """
        pass

    def save(self, path: str) -> None:
        """Persist the fitted model as a PyTorch checkpoint (PDF heading 11)."""
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}: cannot save before train() has been called")
        checkpoint = {
            "class_name": type(self).__name__,
            "state_dict": self.model.state_dict(),
            "hyperparams": self.hyperparams,
            "input_dim": self.input_dim,
        }
        checkpoint.update(self._extra_save_state())
        torch.save(checkpoint, path)

    def load(self, path: str) -> "BaseNetwork":
        """Load a previously-saved checkpoint from `path` into this instance."""
        checkpoint = torch.load(path, map_location=self.device)
        saved_class = checkpoint.get("class_name")
        if saved_class != type(self).__name__:
            raise ValueError(
                f"Model file at {path} was saved as '{saved_class}', "
                f"but you're loading it into a '{type(self).__name__}'. "
                f"Load it with the matching class instead."
            )
        self.hyperparams = checkpoint["hyperparams"]
        self.input_dim = checkpoint["input_dim"]
        self._restore_extra_state(checkpoint)
        self.model = self._build_network(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Shared building blocks
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = _require(self.hyperparams, "optimizer")
        if name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZERS.keys())}")
        lr = _require(self.hyperparams, "learning_rate")
        return OPTIMIZERS[name](self.model.parameters(), lr=lr)

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        name = _require(self.hyperparams, "scheduler")
        if name not in SCHEDULERS:
            raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULERS.keys())}")
        scheduler_cls = SCHEDULERS[name]
        if scheduler_cls is None:
            return None
        scheduler_params = _require(self.hyperparams, "scheduler_params") or {}
        return scheduler_cls(optimizer, **scheduler_params)

    def _make_loader(self, X: pd.DataFrame, y: pd.Series, shuffle: bool) -> DataLoader:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = self._make_target_tensor(y)
        batch_size = _require(self.hyperparams, "batch_size")
        return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    def _mlp_block(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Shared MLP head-builder used by mlp.py directly, and by lstm.py /
        gru.py to turn their final hidden state into an output. Reads
        hidden_layers / hidden_units / activation / dropout / batch_norm
        from self.hyperparams -- this is the one place those five configs
        are actually wired into torch layers.
        """
        hidden_layers = _require(self.hyperparams, "hidden_layers")
        hidden_units = _require(self.hyperparams, "hidden_units")
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units] * hidden_layers

        activation_name = _require(self.hyperparams, "activation")
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation_name}'. Available: {list(ACTIVATIONS.keys())}")
        activation_cls = ACTIVATIONS[activation_name]

        dropout = _require(self.hyperparams, "dropout")
        batch_norm = _require(self.hyperparams, "batch_norm")

        layers = []
        prev_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _require_trained(self) -> None:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__}: predict() called before train() (or load())")


class BaseClassifierNetwork(BaseNetwork):
    """
    Classification variant of BaseNetwork, living in this same file since
    the PDF's tree has one base_network.py, not a separate file per task
    type. Overrides the hooks BaseNetwork leaves as regression defaults,
    and adds predict_proba()/classes_ so it matches
    ml/classifiers/base_classifier.py's interface (PDF heading 6) exactly.

    mlp.py / lstm.py / gru.py subclass BaseNetwork directly for their
    *Regressor* classes, and BaseClassifierNetwork for their *Classifier*
    classes -- _build_network() is the only method each concrete class
    still has to write itself.

    Labels: fit on whatever labels are present in y_train (e.g. the
    target_pipeline's -1/0/1 triple-barrier labels) and remember the
    mapping in self._classes_ so predict()/predict_proba() can translate
    back from internal class indices to the original label values.
    """

    def __init__(self, **hyperparams):
        super().__init__(**hyperparams)
        self._classes_: Optional[np.ndarray] = None

    def _default_loss(self) -> str:
        return "cross_entropy"

    def _output_dim(self) -> int:
        return len(self._classes_)

    def _extra_save_state(self) -> dict:
        """
        Persist _classes_ alongside the checkpoint. Needed at reload
        time before _build_network() runs, since _output_dim() (used to
        size the network's output layer) reads len(self._classes_).
        """
        return {"classes_": self._classes_}

    def _restore_extra_state(self, checkpoint: dict) -> None:
        self._classes_ = checkpoint["classes_"]

    def _make_target_tensor(self, y: pd.Series) -> torch.Tensor:
        if self._classes_ is None:
            self._classes_ = np.sort(y.unique())
        label_to_index = {label: i for i, label in enumerate(self._classes_)}
        indices = y.map(label_to_index).values
        return torch.tensor(indices, dtype=torch.long)

    def _postprocess_predictions(self, raw_output: torch.Tensor) -> np.ndarray:
        # PDF heading 8, classification predict(): predicted class (original label values).
        probs = torch.softmax(raw_output, dim=1)
        pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        return self._classes_[pred_indices]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None) -> "BaseClassifierNetwork":
        # Resolve class labels (and therefore _output_dim()) before
        # BaseNetwork.train() builds the network.
        self._classes_ = np.sort(y_train.unique())
        return super().train(X_train, y_train, X_val, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, columns ordered per self.classes_ (PDF heading 6)."""
        self._require_trained()
        self.model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            raw_output = self.model(X_tensor)
        return torch.softmax(raw_output, dim=1).cpu().numpy()

    @property
    def classes_(self) -> np.ndarray:
        self._require_trained()
        return self._classes_