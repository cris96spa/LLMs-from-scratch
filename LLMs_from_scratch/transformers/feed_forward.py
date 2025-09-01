import torch
from torch import nn

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import (
    FeedForwardConfig,
    SimpleSkipConnectionNetworkConfig,
)
from LLMs_from_scratch.transformers.activations import GELU
from utils.utilities import print_gradients


class FeedForward(nn.Module):
    def __init__(self, config: FeedForwardConfig):
        super().__init__()
        self._config = config
        self.layers = nn.Sequential(
            nn.Linear(self._config.in_features, 4 * self._config.in_features),
            nn.GELU(),
            nn.Linear(4 * self._config.in_features, self._config.in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleSkipConnectionNetwork(nn.Module):
    def __init__(self, config: SimpleSkipConnectionNetworkConfig):
        super().__init__()
        self._config = config
        self._use_shortcut = self._config.use_shortcut
        in_out_features = zip(
            self._config.in_features[:-1], self._config.in_features[1:]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_features, out_features), GELU())
                for in_features, out_features in in_out_features
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_output = layer(x)
            if self._use_shortcut and x.shape == layer_output.shape:
                x = layer_output + x
            else:
                x = layer_output
        return x


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    skip_network = SimpleSkipConnectionNetwork(
        ConfigsProvider().simple_skip_connection_network_config
    )

    print_gradients(skip_network, batch_example)

    output = skip_network(batch_example)

    print(batch_example)
    print("-" * 60)
    print(output)
