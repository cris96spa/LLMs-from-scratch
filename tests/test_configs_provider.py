import pytest

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
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


@pytest.fixture
def configs_provider() -> ConfigsProvider:
    return ConfigsProvider()


def test_tokenizer_config(configs_provider: ConfigsProvider):
    tokenizer_config = configs_provider.tokenizer_config
    assert isinstance(tokenizer_config, TokenizerConfig)


def test_gpt_dataset_config(configs_provider: ConfigsProvider):
    gpt_dataset_config = configs_provider.gpt_dataset_config
    assert isinstance(gpt_dataset_config, GPTDatasetConfig)


def test_dataloader_config(configs_provider: ConfigsProvider):
    dataloader_config = configs_provider.dataloader_config
    assert isinstance(dataloader_config, DataLoaderConfig)


def test_self_attention_config(configs_provider: ConfigsProvider):
    self_attention_config = configs_provider.self_attention_config
    assert isinstance(self_attention_config, SelfAttentionConfig)


def test_layer_norm_config(configs_provider: ConfigsProvider):
    layer_norm_config = configs_provider.layer_norm_config
    assert isinstance(layer_norm_config, LayerNormConfig)


def test_feed_forward_config(configs_provider: ConfigsProvider):
    feed_forward_config = configs_provider.feed_forward_config
    assert isinstance(feed_forward_config, FeedForwardConfig)


def test_simple_skip_connection_network_config(configs_provider: ConfigsProvider):
    simple_skip_connection_network_config = (
        configs_provider.simple_skip_connection_network_config
    )
    assert isinstance(
        simple_skip_connection_network_config, SimpleSkipConnectionNetworkConfig
    )


def test_transformer_block_config(configs_provider: ConfigsProvider):
    transformer_block_config = configs_provider.transformer_block_config
    assert isinstance(transformer_block_config, TransformerBlockConfig)
