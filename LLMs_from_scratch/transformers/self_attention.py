import torch
from torch import nn

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
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


inputs = torch.tensor(
    [
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ],
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ],
    ]
)


def main():
    torch.manual_seed(789)
    self_attention = SelfAttention(ConfigsProvider().self_attention_config)
    print(self_attention(inputs))
    self_attention.state_dict()
    pass


if __name__ == "__main__":
    main()
