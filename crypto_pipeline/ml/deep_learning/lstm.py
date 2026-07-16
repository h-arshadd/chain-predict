# crypto_pipeline/ml/deep_learning/lstm.py

"""
LSTM, for both regression and classification (PDF heading 7).

The feature pipeline hands this module a flat (n_rows, n_features)
table -- same as every other model -- so each row is treated as a
single timestep of length 1 (shape (batch, 1, n_features) going into
the LSTM). hidden_layers/hidden_units here control the LSTM's own
num_layers/hidden_size (per the PDF's "configurable hidden layers /
hidden units" requirement), and the final hidden state is then passed
through the shared _mlp_block() head to produce the output -- this
reuses the same dropout/batch_norm/activation wiring as mlp.py instead
of duplicating it.
"""

import torch
from torch import nn


class _LSTMCore(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, head: nn.Module):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, 1, n_features), one timestep per row.
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size), final layer's hidden state
        return self.head(last_hidden)


def _build_lstm(base_network, input_dim: int, output_dim: int) -> nn.Module:
    hidden_layers = base_network.hyperparams.get("hidden_layers", 2)
    hidden_units = base_network.hyperparams.get("hidden_units", 64)
    if isinstance(hidden_units, list):
        hidden_units = hidden_units[0]  # LSTM hidden_size is a single int, not per-layer
    dropout = base_network.hyperparams.get("dropout", 0.0)

    head = base_network._mlp_block(hidden_units, output_dim)
    return _LSTMCore(input_dim, hidden_units, hidden_layers, dropout, head)


from crypto_pipeline.ml.deep_learning.base_network import BaseClassifierNetwork, BaseNetwork


class LSTMRegressorModel(BaseNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return _build_lstm(self, input_dim, output_dim=self._output_dim())


class LSTMClassifierModel(BaseClassifierNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return _build_lstm(self, input_dim, output_dim=self._output_dim())