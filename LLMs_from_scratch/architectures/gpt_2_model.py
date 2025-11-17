import torch
import torch.nn as nn

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import Gpt2ModelConfig, TransformerBlockConfig
from LLMs_from_scratch.transformers.layer_norm import LayerNorm
from LLMs_from_scratch.transformers.transformer import TransformerBlock


class GPT2Model(nn.Module):
    def __init__(
        self, config: Gpt2ModelConfig, transformer_block_config: TransformerBlockConfig
    ):
        super().__init__()

        self._config: Gpt2ModelConfig = config
        self._transformer_block_config: TransformerBlockConfig = (
            transformer_block_config
        )

        # First we create the token embedding layer
        # that has an element for each token in the vocabulary, each represented
        # by a vector of size embedding_dim.
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
        )

        # Next, we create the position embedding layer
        # that has an element for each position in the context window
        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length, embedding_dim=config.embedding_dim
        )

        self.drop_embedding = nn.Dropout(p=config.dropout)

        # Use a placeholder for the transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(self._transformer_block_config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = LayerNorm(self._config.layer_norm_config)
        self.output_projection = nn.Linear(
            in_features=config.embedding_dim, out_features=config.vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, sequence_length = x.shape

        # Get the embeddings for tokens
        token_embeddings = self.token_embedding(x)

        # Get the position indices and their embeddings
        position_indices = torch.arange(sequence_length, device=x.device)
        position_embeddings = self.position_embedding(position_indices)

        # Combine token and position embeddings
        # and apply dropout
        embeddings = token_embeddings + position_embeddings
        embeddings = self.drop_embedding(embeddings)

        # Pass through the transformer blocks
        transformer_output = self.transformer_blocks(embeddings)

        # Final layer normalization
        normalized_output = self.final_norm(transformer_output)

        # Project to vocabulary size to get logits
        logits = self.output_projection(normalized_output)

        return logits


if __name__ == "__main__":
    torch.manual_seed(123)
    config = ConfigsProvider().gpt_2_model_config
    transformer_block_config = ConfigsProvider().transformer_block_config
    model = GPT2Model(config, transformer_block_config)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt_strings = ["Every effort moves you", "Every day holds a"]

    for txt in txt_strings:
        token_ids = tokenizer.encode(txt)
        batch.append(torch.tensor(token_ids))

    tensor_batch = torch.stack(batch, dim=0)
    print(tensor_batch)

    logits = model(tensor_batch)
    print(logits.shape)
    print(logits)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.token_embedding.weight.shape)
    print("Output layer shape:", model.output_projection.weight.shape)

    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.output_projection.parameters()
    )
    print(
        f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}"
    )
