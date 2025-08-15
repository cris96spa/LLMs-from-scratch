from LLMs_from_scratch.configs.models import TokenizerConfig
from utils.configs_provider import BaseConfigProvider


class ConfigsProvider(BaseConfigProvider):
    """Configs provider that extends the base configs provider."""

    def __init__(self):
        super().__init__()
        self._tokenizer_config = TokenizerConfig()

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        """Get the tokenizer config."""
        return self._tokenizer_config


if __name__ == "__main__":
    provider = ConfigsProvider()
    print(provider.tokenizer_config)
