import tiktoken

from LLMs_from_scratch.configs.models import TokenizerConfig


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self._config: TokenizerConfig = config
        self._tokenizer: tiktoken.Encoding = (
            config.encoding or tiktoken.encoding_for_model(config.model_name)  # type: ignore
        )
        self._allowed_special = config.allowed_special

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, allowed_special=self._allowed_special)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)
