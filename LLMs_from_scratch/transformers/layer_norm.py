import torch
from torch import nn

from LLMs_from_scratch.configs.models import LayerNormConfig


class LayerNorm(torch.nn.Module):
    def __init__(self, config: LayerNormConfig):
        super().__init__()
        self._config = config
        self._eps = self._config.eps
        self.scale = nn.Parameter(torch.ones(self._config.in_features))
        self.shift = nn.Parameter(torch.zeros(self._config.in_features))
        self._unbiased = self._config.unbiased

    def forward(self, x: torch.Tensor):
        _mean = x.mean(dim=-1, keepdim=True)
        _var = x.var(dim=-1, keepdim=True, unbiased=self._unbiased)
        norm = (x - _mean) / torch.sqrt(_var + self._eps)
        return self.scale * norm + self.shift
