from pathlib import Path
from typing import AbstractSet, Literal

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict
from tiktoken import Encoding

from utils.configs import YamlBaseConfig


class TokenizerConfig(YamlBaseConfig):
    encoding: Encoding | None = None
    model_name: str | None = None
    allowed_special: Literal["all"] | AbstractSet[str]

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
    file_path: Path
    max_length: int
    stride: int

    model_config = SettingsConfigDict(
        yaml_file="configs/dataloader/gpt_dataset.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class DataLoaderConfig(YamlBaseConfig):
    batch_size: int
    shuffle: bool
    num_workers: int
    drop_last: bool

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
        yaml_file="configs/trasformers/self_attention.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )
