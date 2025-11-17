import pytest
import torch
from pydantic import ValidationError

from LLMs_from_scratch.configs.configs_provider import ConfigsProvider
from LLMs_from_scratch.configs.models import SelfAttentionConfig
from LLMs_from_scratch.transformers.multi_headed_attention import MultiHeadedAttention


@pytest.fixture
def device() -> str:
    return ConfigsProvider().device


@pytest.fixture
def base_config() -> SelfAttentionConfig:
    return SelfAttentionConfig(
        in_features=8,
        out_features=8,
        num_heads=2,
        qkv_bias=False,
        context_length=64,
        dropout=0.0,
    )


@pytest.fixture
def model(base_config, device: str) -> MultiHeadedAttention:
    mha = MultiHeadedAttention(base_config).to(device)
    mha.eval()
    return mha


@pytest.fixture
def inputs(device: str) -> torch.Tensor:
    # (batch=2, seq=6, in_features=8)
    torch.manual_seed(123)
    x = torch.randn(2, 6, 8, device=device)
    return x


def test_output_shape(
    model: MultiHeadedAttention, inputs: torch.Tensor, base_config: SelfAttentionConfig
):
    y: torch.Tensor = model(inputs)
    assert y.shape == (inputs.shape[0], inputs.shape[1], base_config.out_features)


def test_invalid_heads_raises(device: str):
    with pytest.raises(ValidationError):
        bad_cfg = SelfAttentionConfig(
            in_features=8,
            out_features=10,
            num_heads=3,
            qkv_bias=False,
            context_length=64,
            dropout=0.0,
        )  # 10 % 3 != 0
        _ = MultiHeadedAttention(bad_cfg).to(device)


def test_state_dict_keys(model: MultiHeadedAttention):
    keys = set(model.state_dict().keys())
    expected = {
        "W_query.weight",
        "W_key.weight",
        "W_value.weight",
        "out_projection.weight",
        "out_projection.bias",
        "mask",
    }
    assert expected.issubset(keys)
    biases = {
        "W_query.bias",
        "W_key.bias",
        "W_value.bias",
    }
    if model._qkv_bias:
        assert biases.issubset(keys)
    else:
        assert not biases.intersection(keys)


def test_train_mode_dropout_changes_outputs(base_config, device):
    """With dropout>0 and train() mode, two forward passes (without reseeding) should differ."""
    cfg = SelfAttentionConfig(
        in_features=8,
        out_features=8,
        num_heads=2,
        qkv_bias=False,
        context_length=32,
        dropout=0.5,  # enable dropout
    )
    model = MultiHeadedAttention(cfg).to(device)
    model.train()

    torch.manual_seed(42)
    x = torch.randn(2, 6, 8, device=device)

    y1: torch.Tensor = model(x)
    y2: torch.Tensor = model(x)

    # There should be some difference introduced by dropout
    diff = (y1 - y2).abs().max().item()
    assert diff > 0, "Dropout in train() mode did not change outputs as expected"


def test_eval_mode_is_deterministic(device: str):
    """In eval mode with dropout=0.5, outputs should be deterministic (dropout disabled in eval)."""
    cfg = SelfAttentionConfig(
        in_features=8,
        out_features=8,
        num_heads=2,
        qkv_bias=False,
        context_length=32,
        dropout=0.5,  # still deterministic in eval()
    )
    model = MultiHeadedAttention(cfg).to(device)
    model.eval()

    torch.manual_seed(999)
    x = torch.randn(2, 6, 8, device=device)

    y1: torch.Tensor = model(x)
    y2: torch.Tensor = model(x)
    assert torch.allclose(y1, y2, atol=1e-7)


def test_gradients_flow(model: MultiHeadedAttention, inputs: torch.Tensor):
    """Make sure we can backprop through the module and its parameters receive grads."""
    x: torch.Tensor = inputs.clone().requires_grad_(True)
    y: torch.Tensor = model(x)
    loss = y.sum()
    loss.backward()

    # input grad present
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # some parameter grads present
    params_with_grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in params_with_grads)
    assert all(torch.isfinite(g).all() for g in params_with_grads if g is not None)


def test_head_split_merge_dimensions(device: str):
    """
    Sanity check on head dimension math: out_features = num_heads * head_dim,
    and output preserves (batch, seq, out_features).
    """
    cfg = SelfAttentionConfig(
        in_features=12,
        out_features=24,
        num_heads=6,
        qkv_bias=True,
        context_length=32,
        dropout=0.0,
    )
    m = MultiHeadedAttention(cfg).to(device).eval()
    x = torch.randn(3, 5, 12, device=device)
    y: torch.Tensor = m(x)
    assert y.shape == (3, 5, 24)
