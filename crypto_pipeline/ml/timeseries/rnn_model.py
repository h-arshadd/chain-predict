# crypto_pipeline/ml/timeseries/rnn_model.py

"""
rnn_model.py
------------
LSTM/GRU (darts.models.BlockRNNModel) wrappers -- same family as
nbeats_model.py / tcn_model.py, living here rather than under
ml/deep_learning/ because this is the only place in the codebase that
actually feeds a model a real sequence of past steps
(input_chunk_length) instead of one row at a time.

The LSTM/GRU that used to live under ml/deep_learning/ fed the network
exactly one timestep per row (batch, 1, n_features). With a sequence
length of 1 an LSTM/GRU cell reduces to a linear projection over the
current row -- no past steps are ever seen, so those implementations
captured no more temporal information than the MLP sitting right next
to them; they were LSTM/GRU in name only. Real sequence modeling needs
a lookback window of multiple past rows per prediction, which is
exactly what input_chunk_length/output_chunk_length already give every
model in this family (see nbeats_model.py, tcn_model.py's docstrings).
Wrapping LSTM/GRU here means they share that same windowing machinery
instead of a second, separate (and non-functional) implementation.

Practical effect: LSTM/GRU are now selected via
model_type: timeseries (ml/data_prep/config.yaml), same as nbeats/tcn
-- not via model_type: regression/classification any more. They now
also need a target *series* (the darts.TimeSeries machinery
base_timeseries_model.py already sets up), not a flat (X, y) table, so
regression vs classification isn't a choice for them either way: like
nbeats/tcn, they always forecast the target series n steps ahead, and
the resulting forecast is turned into a Buy/Sell/Hold signal by
ml/signals/timeseries_signals.py exactly like nbeats/tcn already are.

BlockRNNModel's `model` constructor argument picks the cell type
("LSTM" or "GRU") -- everything else about the two classes below is
identical, so both just set that one class attribute and forward the
rest of self.hyperparams straight through to
darts.models.BlockRNNModel, same as nbeats_model.py/tcn_model.py do
for their own underlying Darts class.
"""

from typing import Optional

import numpy as np

from crypto_pipeline.ml.timeseries.base_timeseries_model import BaseTimeseriesModel, TimeSeries

try:
    from darts.models import BlockRNNModel
except ImportError:
    BlockRNNModel = None


class _RNNTimeseriesModel(BaseTimeseriesModel):
    """Shared base for LSTM/GRU -- subclasses only set _CELL_TYPE."""

    _CELL_TYPE: str = None  # "LSTM" or "GRU", set by each subclass below

    def train(self, target_series: "TimeSeries", past_covariates: Optional["TimeSeries"] = None) -> "_RNNTimeseriesModel":
        if BlockRNNModel is None:
            raise ImportError(f"darts is required for {type(self).__name__} but is not installed")
        self.model = BlockRNNModel(model=self._CELL_TYPE, **self.hyperparams)
        self.model.fit(target_series, past_covariates=past_covariates)
        return self

    def predict(self, n: int, past_covariates: Optional["TimeSeries"] = None) -> np.ndarray:
        self._require_trained()
        forecast = self.model.predict(n=n, past_covariates=past_covariates)
        return forecast.values().flatten()

    def _darts_model_class(self):
        return BlockRNNModel


class LSTMTimeseriesModel(_RNNTimeseriesModel):
    """
    Args (forwarded to darts.models.BlockRNNModel(model="LSTM", ...)):
        input_chunk_length: int, required -- how many past steps the
            model looks at to produce a forecast.
        output_chunk_length: int, required -- how many future steps it
            predicts per forward pass.
        hidden_dim, n_rnn_layers, dropout: architecture hyperparams
            (Darts defaults used if not given).
        n_epochs, batch_size, random_state: training hyperparams.
        Any other kwarg BlockRNNModel's constructor accepts.
    """
    _CELL_TYPE = "LSTM"


class GRUTimeseriesModel(_RNNTimeseriesModel):
    """Same as LSTMTimeseriesModel, just model="GRU" -- see that class's docstring."""
    _CELL_TYPE = "GRU"