from typing import AbstractSet, Literal

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from tiktoken import Encoding

from utils.configs import YamlBaseConfig


class TokenizerConfig(YamlBaseConfig):
    encoding: Encoding | None = None
    model_name: str | None = None
    allowed_special: Literal["all"] | AbstractSet[str]

    model_config = SettingsConfigDict(
        yaml_file="configs/tokenizer/tokenizer.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )

    @model_validator(mode="after")
    def validate_tokenizer_config(self) -> "TokenizerConfig":
        if not self.encoding and not self.model_name:
            raise ValueError("Either 'encoding' or 'model_name' must be set.")
        return self
