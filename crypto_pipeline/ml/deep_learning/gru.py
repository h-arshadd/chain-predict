# crypto_pipeline/ml/deep_learning/gru.py

"""
GRU, for both regression and classification (PDF heading 7).

Same shape convention as lstm.py: a flat (n_rows, n_features) table,
each row treated as one timestep (batch, 1, n_features). See lstm.py's
docstring for the reasoning -- this file mirrors it exactly, swapping
nn.LSTM for nn.GRU (no cell state to track).
"""

import torch
from torch import nn


class _GRUCore(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, head: nn.Module):
        super().__init__()
        self.gru = nn.GRU(
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
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (batch, hidden_size), final layer's hidden state
        return self.head(last_hidden)


def _build_gru(base_network, input_dim: int, output_dim: int) -> nn.Module:
    hidden_layers = _require(base_network.hyperparams, "hidden_layers")
    hidden_units = _require(base_network.hyperparams, "hidden_units")
    if isinstance(hidden_units, list):
        hidden_units = hidden_units[0]  # GRU hidden_size is a single int, not per-layer
    dropout = _require(base_network.hyperparams, "dropout")

    head = base_network._mlp_block(hidden_units, output_dim)
    return _GRUCore(input_dim, hidden_units, hidden_layers, dropout, head)


from crypto_pipeline.ml.deep_learning.base_network import BaseClassifierNetwork, BaseNetwork, _require


class GRURegressorModel(BaseNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return _build_gru(self, input_dim, output_dim=self._output_dim())


class GRUClassifierModel(BaseClassifierNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return _build_gru(self, input_dim, output_dim=self._output_dim())