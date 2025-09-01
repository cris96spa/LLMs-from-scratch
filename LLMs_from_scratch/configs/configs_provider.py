from LLMs_from_scratch.configs.models import (
    DataLoaderConfig,
    FeedForwardConfig,
    GPTDatasetConfig,
    LayerNormConfig,
    SelfAttentionConfig,
    SimpleSkipConnectionNetworkConfig,
    TokenizerConfig,
    TransformerBlockConfig,
)
from utils.configs_provider import BaseConfigProvider


class ConfigsProvider(BaseConfigProvider):
    """Configs provider that extends the base configs provider."""

    def __init__(self):
        super().__init__()
        self._tokenizer_config: TokenizerConfig | None = None

        self._gpt_dataset_config: GPTDatasetConfig | None = None
        self._dataloader_config: DataLoaderConfig | None = None

        self._self_attention_config: SelfAttentionConfig = None
        self._layer_norm_config: LayerNormConfig | None = None
        self._feed_forward_config: FeedForwardConfig | None = None
        self._transformer_block_config: TransformerBlockConfig | None = None

        self._simple_skip_connection_network_config: (
            SimpleSkipConnectionNetworkConfig | None
        ) = None

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        """Get the tokenizer config."""
        if self._tokenizer_config is None:
            self._tokenizer_config = TokenizerConfig()
        return self._tokenizer_config

    @property
    def gpt_dataset_config(self) -> GPTDatasetConfig:
        """Get the GPT dataset config."""
        if self._gpt_dataset_config is None:
            self._gpt_dataset_config = GPTDatasetConfig()  # type: ignore[call-arg]
        return self._gpt_dataset_config

    @property
    def dataloader_config(self) -> DataLoaderConfig:
        """Get the DataLoader config."""
        if self._dataloader_config is None:
            self._dataloader_config = DataLoaderConfig()  # type: ignore[call-arg]
        return self._dataloader_config

    @property
    def self_attention_config(self) -> SelfAttentionConfig:
        """Get the self-attention config."""
        if self._self_attention_config is None:
            self._self_attention_config = SelfAttentionConfig()
        return self._self_attention_config

    @property
    def layer_norm_config(self) -> LayerNormConfig:
        """Get the layer norm config"""
        if self._layer_norm_config is None:
            self._layer_norm_config = LayerNormConfig()  # type: ignore[call-arg]
        return self._layer_norm_config

    @property
    def feed_forward_config(self):
        """Get the feed-forward config."""
        if self._feed_forward_config is None:
            self._feed_forward_config = FeedForwardConfig()
        return self._feed_forward_config

    @property
    def simple_skip_connection_network_config(self):
        """Get the simple skip connection network config."""
        if self._simple_skip_connection_network_config is None:
            self._simple_skip_connection_network_config = (
                SimpleSkipConnectionNetworkConfig()
            )
        return self._simple_skip_connection_network_config

    @property
    def transformer_block_config(self):
        """Get the transformer block config."""
        if self._transformer_block_config is None:
            self._transformer_block_config = TransformerBlockConfig()
        return self._transformer_block_config
