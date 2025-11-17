from pathlib import Path
from typing import AbstractSet, Literal

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from tiktoken import Encoding

from utils.configs import YamlBaseConfig


class TokenizerConfig(YamlBaseConfig):
    encoding: Encoding | None = Field(
        description="The tokenizer encoding", default=None
    )
    model_name: str | None = Field(
        description="The name of the tokenizer model", default=None
    )
    allowed_special: Literal["all"] | AbstractSet[str] = Field(
        description="The allowed special tokens", default="all"
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/dataloader/tokenizer.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_tokenizer_config(self) -> "TokenizerConfig":
        if not self.encoding and not self.model_name:
            raise ValueError("Either 'encoding' or 'model_name' must be set.")
        return self


class GPTDatasetConfig(YamlBaseConfig):
    file_path: Path = Field(description="The file path to the dataset")
    max_length: int = Field(description="The maximum length of the input sequences")
    stride: int = Field(description="The stride length for overlapping chunks")

    model_config = SettingsConfigDict(
        yaml_file="configs/dataloader/gpt_dataset.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class DataLoaderConfig(YamlBaseConfig):
    batch_size: int = Field(description="The batch size of the data loader")
    shuffle: bool = Field(description="Whether to shuffle the data loader")
    num_workers: int = Field(
        description="The number of worker processes for data loading"
    )
    drop_last: bool = Field(description="Whether to drop the last incomplete batch")

    model_config = SettingsConfigDict(
        yaml_file="configs/dataloader/dataloader.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class SelfAttentionConfig(YamlBaseConfig):
    in_features: int = Field(
        description="Number of input features for the self-attention layer", ge=1
    )
    out_features: int = Field(
        description="Number of output features for the self-attention layer", ge=1
    )
    qkv_bias: bool = Field(
        description="Whether to use bias in the query, key, and value linear layers"
    )
    dropout: float = Field(
        description="Dropout rate for attention weights", ge=0.0, lt=1.0
    )
    context_length: int = Field(
        description="Length of the context window for attention", ge=1
    )
    num_heads: int = Field(description="Number of attention heads", ge=1)

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/self_attention.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_num_heads(self):
        if self.in_features % self.num_heads != 0:
            raise ValueError(
                "in_features must be divisible by num_heads. "
                f"Got in_features: {self.in_features}, num_heads: {self.num_heads}"
            )
        return self


class LayerNormConfig(YamlBaseConfig):
    in_features: int = Field(
        description="Number of input features for the layer normalization", ge=1
    )
    eps: float = Field(
        description="Small constant for numerical stability.", default=1e-5
    )
    unbiased: bool = Field(description="Boolean flag for variance bias.", default=False)

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/layer_norm.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class FeedForwardConfig(YamlBaseConfig):
    in_features: int = Field(
        description="Number of input features for the feed-forward layer", ge=1
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/feed_forward.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class SimpleSkipConnectionNetworkConfig(YamlBaseConfig):
    in_features: list[int] = Field(
        description="Number of input features for each layer of the feed-forward layer",
    )

    use_shortcut: bool = Field(
        description="Whether to use skip connections", default=True
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/simple_skip_connection_network.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @property
    def num_layers(self) -> int:
        return len(self.in_features) - 1


class TransformerBlockConfig(YamlBaseConfig):
    attention_config: SelfAttentionConfig = Field(
        description="Configuration for self attention blocks"
    )
    layer_norm_config: LayerNormConfig = Field(
        description="Configuration for layer normalization blocks"
    )
    feed_forward_config: FeedForwardConfig = Field(
        description="Configuration for feed forward blocks"
    )
    dropout: float = Field(
        description="Dropout rate for attention weights", ge=0.0, lt=1.0
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/transformer_block.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_in_features(self):
        if (
            self.attention_config.in_features
            != self.feed_forward_config.in_features
            != self.layer_norm_config.in_features
        ):
            raise ValueError(
                "All sub-configs must have the same in_features. Got: "
                f"attention_config.in_features: {self.attention_config.in_features}, "
                f"feed_forward_config.in_features: {self.feed_forward_config.in_features}, "
                f"layer_norm_config.in_features: {self.layer_norm_config.in_features}"
            )
        return self


class Gpt2ModelConfig(YamlBaseConfig):
    vocab_size: int = Field(description="Size of the vocabulary", ge=1)
    context_length: int = Field(description="Length of the context window", ge=1)
    embedding_dim: int = Field(description="Dimension of the embeddings", ge=1)
    num_layers: int = Field(description="Number of transformer layers", ge=1)
    dropout: float = Field(
        description="Dropout rate for attention weights", ge=0.0, lt=1.0
    )
    layer_norm_config: LayerNormConfig = Field(
        description="Configuration for final layer normalization"
    )

    model_config = SettingsConfigDict(
        yaml_file="configs/transformers/gpt_2_model.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_embedding_dim(self):
        if self.embedding_dim != self.layer_norm_config.in_features:
            raise ValueError(
                "embedding_dim must be equal to transformer_block_config.attention_config.in_features. "
                f"Got embedding_dim: {self.embedding_dim}, "
                f"transformer_block_config.attention_config.in_features: {self.transformer_block_config.attention_config.in_features}, "
                f"layer_norm_config.in_features: {self.layer_norm_config.in_features}"
            )
        return self
