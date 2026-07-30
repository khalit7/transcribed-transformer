"""Context extension.

Arm A needs 32k and every candidate trunk ships at 8k, so this runs on the real
deliverable. The mistake it most guards against is rescaling the local layers:
that still runs, still trains, and quietly degrades attention inside a window
that never grew.
"""

import pytest

from tt.models.context import (
    GLOBAL_KEY,
    LOCAL_KEY,
    adjusted_base,
    extend_context,
    plan_extension,
)


class _Config:
    """Minimal stand-in shaped like a ModernBertConfig."""

    def __init__(self, length: int = 8192, glob: float = 160_000.0, loc: float = 10_000.0):
        self.max_position_embeddings = length
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.rope_parameters: dict[str, dict[str, float]] = {
            LOCAL_KEY: {"rope_theta": loc},
            GLOBAL_KEY: {"rope_theta": glob},
        }


class _FlatConfig:
    """Older transformers exposed the thetas flat rather than nested."""

    def __init__(self) -> None:
        self.max_position_embeddings = 8192
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.global_rope_theta = 160_000.0
        self.local_rope_theta = 10_000.0


def test_base_grows_with_the_length_ratio() -> None:
    assert adjusted_base(160_000.0, 4.0, 64) > 160_000.0 * 4
    assert adjusted_base(160_000.0, 1.0, 64) == pytest.approx(160_000.0)


def test_exponent_correction_shrinks_as_heads_widen() -> None:
    """d/(d-2) tends to 1, so it is a small correction rather than a large one."""
    narrow = adjusted_base(10_000.0, 4.0, 16)
    wide = adjusted_base(10_000.0, 4.0, 256)
    assert narrow > wide > 10_000.0 * 4


def test_tiny_head_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        adjusted_base(10_000.0, 2.0, 2)


def _theta(cfg: _Config, key: str) -> float:
    return cfg.rope_parameters[key]["rope_theta"]


def test_only_the_global_theta_changes() -> None:
    """The load-bearing assertion.

    A 128-token sliding window spans 128 tokens whether the input is 8k or 32k,
    so rescaling local rope stretches resolution across a window that never grew.
    """
    cfg = _Config()
    extend_context(cfg, 32768)
    assert _theta(cfg, GLOBAL_KEY) > 160_000.0
    assert _theta(cfg, LOCAL_KEY) == 10_000.0


def test_declared_length_is_raised() -> None:
    cfg = _Config()
    extend_context(cfg, 32768)
    assert cfg.max_position_embeddings == 32768


def test_plan_does_not_mutate() -> None:
    """Plan first, so the change can be logged as config rather than inferred later."""
    cfg = _Config()
    plan = plan_extension(cfg, 32768)
    assert cfg.max_position_embeddings == 8192
    assert _theta(cfg, GLOBAL_KEY) == 160_000.0
    assert plan.scale == pytest.approx(4.0)
    assert "669" in plan.describe() or "->" in plan.describe()


def test_shortening_is_refused() -> None:
    with pytest.raises(ValueError, match="not longer"):
        extend_context(_Config(), 4096)
    with pytest.raises(ValueError, match="not longer"):
        extend_context(_Config(), 8192)


def test_flat_config_layout_is_supported() -> None:
    cfg = _FlatConfig()
    extend_context(cfg, 32768)
    assert cfg.global_rope_theta > 160_000.0
    assert cfg.local_rope_theta == 10_000.0
    assert cfg.max_position_embeddings == 32768


def test_model_without_rope_is_refused() -> None:
    class NoRope:
        max_position_embeddings = 512
        hidden_size = 768
        num_attention_heads = 12

    with pytest.raises(ValueError, match="no global RoPE theta"):
        plan_extension(NoRope(), 1024)
