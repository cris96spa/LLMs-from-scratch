import torch
from torch import nn

from LLMs_from_scratch.configs.models import SelfAttentionConfig


class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self._config: SelfAttentionConfig = config
        self.W_query = nn.Linear(
            config.in_features,
            config.out_features,
            config.qkv_bias,
        )
        self.W_key = nn.Linear(
            config.in_features,
            config.out_features,
            config.qkv_bias,
        )

        self.W_value = nn.Linear(
            config.in_features,
            config.out_features,
            config.qkv_bias,
        )
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            name="mask",
            tensor=torch.triu(
                torch.ones(self._config.context_length, self._config.context_length),
                diagonal=1,
            ),
        )
        self.mask: torch.Tensor = self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, in_features = x.shape

        if in_features != self._config.in_features:
            raise ValueError(
                f"Input features {in_features} do not match config in_features {self._config.in_features}"
            )

        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values
        return context_vectors
