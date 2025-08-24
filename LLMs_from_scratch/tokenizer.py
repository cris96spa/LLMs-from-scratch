from abc import ABC, abstractmethod
from typing import override

import tiktoken

from LLMs_from_scratch.configs.models import TokenizerConfig


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into text."""
        pass


class Tokenizer(BaseTokenizer):
    def __init__(self, config: TokenizerConfig):
        self._config: TokenizerConfig = config
        self.tokenizer: tiktoken.Encoding = (
            config.encoding or tiktoken.encoding_for_model(config.model_name)  # type: ignore[arg-type]
        )
        self._allowed_special = config.allowed_special

    @override
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: The encoded token IDs.
        """
        return self.tokenizer.encode(text, allowed_special=self._allowed_special)

    @override
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into text.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(ids)
