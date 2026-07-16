# crypto_pipeline/ml/deep_learning/mlp.py

"""Multi-Layer Perceptron, for both regression and classification (PDF heading 7)."""

from torch import nn

from crypto_pipeline.ml.deep_learning.base_network import BaseClassifierNetwork, BaseNetwork


class MLPRegressorModel(BaseNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return self._mlp_block(input_dim, output_dim=self._output_dim())


class MLPClassifierModel(BaseClassifierNetwork):
    def _build_network(self, input_dim: int) -> nn.Module:
        return self._mlp_block(input_dim, output_dim=self._output_dim())