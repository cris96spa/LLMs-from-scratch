import torch
from torch import nn

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import TransformerBlockConfig
from LLMs_from_scratch.transformers.feed_forward import FeedForward
from LLMs_from_scratch.transformers.layer_norm import LayerNorm
from LLMs_from_scratch.transformers.multi_headed_attention import MultiHeadedAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self._config = config
        self.multi_headed_attention = MultiHeadedAttention(
            self._config.attention_config
        )
        self.feedforward = FeedForward(self._config.feed_forward_config)
        self.layer_norm_pre_attention = LayerNorm(self._config.layer_norm_config)
        self.layer_norm_pre_feedforward = LayerNorm(self._config.layer_norm_config)
        self.drop_shortcut = nn.Dropout(self._config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.layer_norm_pre_attention(x)
        x = self.multi_headed_attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_pre_feedforward(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


if __name__ == "__main__":
    torch.manual_seed(123)

    x = torch.rand(2, 4, 768)
    transformer_block_config = ConfigsProvider().transformer_block_config
    transformer_block = TransformerBlock(transformer_block_config)
    output = transformer_block(x)
    pass
