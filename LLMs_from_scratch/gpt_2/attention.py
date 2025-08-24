import torch
from torch import nn

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import SelfAttentionConfig

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
    in_features = inputs.shape[-1]
    out_features = 2

    print("-" * 50)
    print(f"Inputs (shape: {inputs.shape}):")
    print(inputs)

    W_query = nn.Parameter(torch.rand(in_features, out_features), requires_grad=False)
    W_key = nn.Parameter(torch.rand(in_features, out_features), requires_grad=False)
    W_value = nn.Parameter(torch.rand(in_features, out_features), requires_grad=False)

    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value

    attention_scores = queries @ keys.T
    print("-" * 50)
    print(f"Attention Scores (Queries @ Keys^T) (shape: {attention_scores.shape}):")
    print(attention_scores)

    d_k = keys.shape[1]
    attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
    print("-" * 50)
    print(
        f"Attention Weights (Softmax of Attention Scores) (shape {attention_weights.shape}):"
    )
    print(attention_weights)

    contextualized_embeddings = attention_weights @ values
    print("-" * 50)
    print(
        f"Contextualized Embeddings (Attention Weights @ Inputs) (shape: {contextualized_embeddings.shape}):"
    )
    print(contextualized_embeddings)
    pass


def main_2():
    torch.manual_seed(789)
    self_attention = MultiHeadedAttention(ConfigsProvider().self_attention_config)
    print(self_attention(inputs))
    pass


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


class MultiHeadedAttention(nn.Module):
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        if config.out_features % config.num_heads != 0:
            raise ValueError(
                f"The output dimension must be a multiple of the number of heads. Got out_features:{config.out_features} num_heads: {config.num_heads}"
            )
        self._config: SelfAttentionConfig = config
        self._out_features: int = self._config.out_features
        self._num_heads: int = config.num_heads
        self._head_dim: int = config.out_features // self._num_heads

        self.W_query: nn.Linear = nn.Linear(
            in_features=self._config.in_features,
            out_features=self._config.out_features,
            bias=self._config.qkv_bias,
        )
        self.W_key: nn.Linear = nn.Linear(
            in_features=self._config.in_features,
            out_features=self._config.out_features,
            bias=self._config.qkv_bias,
        )
        self.W_value: nn.Linear = nn.Linear(
            in_features=self._config.in_features,
            out_features=self._config.out_features,
            bias=self._config.qkv_bias,
        )

        self.out_projection: nn.Linear = nn.Linear(
            self._config.out_features, self._config.out_features
        )

        self.register_buffer(
            name="mask",
            tensor=torch.triu(
                torch.ones(self._config.context_length, self._config.context_length),
                diagonal=1,
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, in_features = x.shape

        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        # Split the Q, K and V matrices to get individual heads on a new dimension
        queries = queries.view(batch, num_tokens, self._num_heads, self._head_dim)
        keys = keys.view(batch, num_tokens, self._num_heads, self._head_dim)
        values = values.view(batch, num_tokens, self._num_heads, self._head_dim)

        # Transpose from (batch, num_tokens, num_head, head_dim) -> (batch, num_head, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # Compute the context vector and transpose back from
        # (batch, num_head, num_tokens, head_dim) -> (batch, num_tokens, num_head, head_dim)
        context_vector: torch.Tensor = (attention_weights @ values).transpose(1, 2)

        # Merge heads
        context_vector = context_vector.contiguous().view(
            batch, num_tokens, self._out_features
        )
        context_vector = self.out_projection(context_vector)

        return context_vector


if __name__ == "__main__":
    main_2()
