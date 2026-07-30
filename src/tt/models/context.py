"""Extending an encoder's context window.

Arm A needs 32k. Every candidate trunk ships at 8k: ModernBERT-large declares
``max_position_embeddings=8192`` and Ettin 7999. So context extension is a
required phase, not an optional one.

## Only the global layers need touching

ModernBERT-style encoders alternate attention: every ``global_attn_every_n_layers``
th layer attends to the whole sequence, and the rest use a sliding window of
``local_attention`` tokens. The two use **separate RoPE frequencies**, 160,000 for
global and 10,000 for local in ModernBERT-large.

A sliding window of 128 tokens spans 128 tokens whether the input is 8k or 32k,
so the local layers' rope is already correct at any length and rescaling it would
actively damage them: it would stretch positional resolution across a window that
never grew. Only the global layers see distances beyond the window, and only they
are rescaled. Two thirds of the layers are left alone.

Getting this backwards is the easy mistake, and a quiet one. Scaling both still
runs, still trains, and simply degrades local attention for no reason.

## Why adjusted base frequency rather than naive interpolation

Scaling the RoPE base ("ABF", also called NTK-aware scaling) preserves
high-frequency components that carry local word order while stretching the
low-frequency ones that carry long-range position. Plain position interpolation
squeezes all frequencies equally and costs short-range accuracy, which for
transcripts is where most of the signal is.

The exponent ``d/(d-2)`` corrects for the finite head dimension; it tends to 1 as
``d`` grows, so it is a small correction rather than a large one, but it is what
makes the scaled model match the unscaled one on short inputs.

Extension changes the model's positional behaviour, so a model whose context has
been extended **must be trained on long inputs afterwards**. Setting the theta and
declaring victory produces a model that accepts 32k tokens and understands them
worse than the 8k original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GLOBAL_KEY = "full_attention"
LOCAL_KEY = "sliding_attention"


@dataclass(frozen=True)
class ExtensionPlan:
    """What extension will do, computed before anything is mutated."""

    original_length: int
    target_length: int
    scale: float
    head_dim: int
    original_theta: float
    new_theta: float
    local_theta_unchanged: float | None

    def describe(self) -> str:
        local = (
            f", local rope left at {self.local_theta_unchanged:,.0f}"
            if self.local_theta_unchanged is not None
            else ""
        )
        return (
            f"{self.original_length:,} -> {self.target_length:,} tokens "
            f"({self.scale:.1f}x): global rope theta "
            f"{self.original_theta:,.0f} -> {self.new_theta:,.0f}{local}"
        )


def adjusted_base(theta: float, scale: float, head_dim: int) -> float:
    """RoPE base for a longer context, NTK-aware.

    ``theta * scale ** (d / (d - 2))``. The exponent corrects for finite head
    dimension and tends to 1 as ``d`` grows.
    """
    if head_dim <= 2:
        raise ValueError(f"head_dim must exceed 2, got {head_dim}")
    scaled: float = float(theta) * float(scale) ** (head_dim / (head_dim - 2))
    return scaled


def _head_dim(config: Any) -> int:
    explicit = getattr(config, "head_dim", None)
    if explicit:
        return int(explicit)
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    return hidden // heads


def _read_theta(config: Any, key: str) -> float | None:
    """Read a rope theta across the two layouts transformers has used.

    Newer versions nest per attention type under ``rope_parameters``; older ones
    expose flat ``global_rope_theta`` / ``local_rope_theta``.
    """
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and key in params:
        value = params[key].get("rope_theta")
        return None if value is None else float(value)
    flat = "global_rope_theta" if key == GLOBAL_KEY else "local_rope_theta"
    value = getattr(config, flat, None)
    return None if value is None else float(value)


def _write_theta(config: Any, key: str, value: float) -> None:
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and key in params:
        params[key]["rope_theta"] = value
        return
    flat = "global_rope_theta" if key == GLOBAL_KEY else "local_rope_theta"
    if hasattr(config, flat):
        setattr(config, flat, value)
        return
    raise ValueError(f"config exposes no rope theta for {key!r}")


def plan_extension(config: Any, target_length: int) -> ExtensionPlan:
    """Work out the new rope base without touching the config."""
    original = int(config.max_position_embeddings)
    if target_length <= original:
        raise ValueError(
            f"target {target_length:,} is not longer than the current "
            f"{original:,}; extension would shorten or no-op."
        )

    theta = _read_theta(config, GLOBAL_KEY)
    if theta is None:
        raise ValueError(
            "no global RoPE theta on this config. Context extension here assumes "
            "rotary position embeddings; a learned-embedding model needs a different "
            "method entirely."
        )

    scale = target_length / original
    head_dim = _head_dim(config)
    return ExtensionPlan(
        original_length=original,
        target_length=target_length,
        scale=scale,
        head_dim=head_dim,
        original_theta=theta,
        new_theta=adjusted_base(theta, scale, head_dim),
        local_theta_unchanged=_read_theta(config, LOCAL_KEY),
    )


def extend_context(config: Any, target_length: int) -> ExtensionPlan:
    """Rescale global RoPE and raise the declared context length, in place.

    Returns the plan so the change can be logged as a run config rather than
    inferred later from a checkpoint.
    """
    plan = plan_extension(config, target_length)
    _write_theta(config, GLOBAL_KEY, plan.new_theta)
    config.max_position_embeddings = target_length
    return plan
