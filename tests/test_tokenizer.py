import pytest
import tiktoken

from LLMs_from_scratch.configs.models import TokenizerConfig
from LLMs_from_scratch.tokenizer import Tokenizer


@pytest.fixture
def sample_text() -> str:
    return "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough."


@pytest.fixture
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(model_name="gpt2", allowed_special={"<|endoftext|>"})


@pytest.fixture
def tokenizer(tokenizer_config: TokenizerConfig) -> Tokenizer:
    return Tokenizer(tokenizer_config)


def test_encode_returns_token_ids(tokenizer: Tokenizer, sample_text: str):
    token_ids = tokenizer.encode(sample_text)
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    assert len(token_ids) > 0


def test_decode_returns_string(tokenizer: Tokenizer, sample_text: str):
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0


def test_encode_decode_round_trip(tokenizer: Tokenizer, sample_text: str):
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert sample_text.strip() == decoded.strip()


def test_special_token_handling():
    config = TokenizerConfig(model_name="gpt2", allowed_special={"<|endoftext|>"})
    tokenizer = Tokenizer(config)

    special_token = "<|endoftext|>"
    text_with_special = f"Hello {special_token} world"

    encoded = tokenizer.encode(text_with_special)
    decoded = tokenizer.decode(encoded)

    assert special_token in decoded


def test_disallowed_special_token_raises():
    config = TokenizerConfig(
        model_name="gpt2",
        allowed_special=set(),
    )
    tokenizer = Tokenizer(config)

    text_with_special = "This text includes <|endoftext|> token"
    with pytest.raises(ValueError):
        tokenizer.encode(text_with_special)


def test_config_encoding():
    encoding = tiktoken.get_encoding("gpt2")
    config = TokenizerConfig(encoding=encoding, allowed_special=set())
    tokenizer = Tokenizer(config)
    assert tokenizer.tokenizer is encoding


def test_invalid_config():
    with pytest.raises(ValueError):
        TokenizerConfig(encoding=None, model_name=None, allowed_special=set())
