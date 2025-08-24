from LLMs_from_scratch.configs.models import (
    DataLoaderConfig,
    GPTDatasetConfig,
    SelfAttentionConfig,
    TokenizerConfig,
)
from utils.configs_provider import BaseConfigProvider


class ConfigsProvider(BaseConfigProvider):
    """Configs provider that extends the base configs provider."""

    def __init__(self):
        super().__init__()
        self._tokenizer_config = TokenizerConfig()
        self._gpt_dataset_config = GPTDatasetConfig()
        self._dataloader_config = DataLoaderConfig()

        self._self_attention_config = SelfAttentionConfig()

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        """Get the tokenizer config."""
        return self._tokenizer_config

    @property
    def gpt_dataset_config(self) -> GPTDatasetConfig:
        """Get the GPT dataset config."""
        return self._gpt_dataset_config

    @property
    def dataloader_config(self) -> DataLoaderConfig:
        """Get the DataLoader config."""
        return self._dataloader_config

    @property
    def self_attention_config(self) -> SelfAttentionConfig:
        """Get the self-attention config."""
        return self._self_attention_config


if __name__ == "__main__":
    provider = ConfigsProvider()
    print(provider.tokenizer_config)
    print(provider.gpt_dataset_config)
    print(provider.dataloader_config)
    print(provider.self_attention_config)
