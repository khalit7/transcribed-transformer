"""Run configuration.

Config over code: adding a model, dataset or head means adding a class and
pointing a YAML at it. If launching a run requires editing the training loop,
the abstraction is wrong and should be fixed instead.

The mandatory tag fields exist because results tables are generated from the
wandb API filtered on them. An untagged run is invisible to the results
pipeline, so the tags are required at construction rather than left optional.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tt.data.schema import Track


class Arm(StrEnum):
    """Which research arm a run belongs to. See the README design table."""

    A = "arm-a"
    """Adapted native encoder."""

    B = "arm-b"
    """Encoder-decoder."""

    C = "arm-c"
    """Bidirectional-prefix decoder."""

    D = "arm-d"
    """Baselines."""

    E = "arm-e"
    """From-scratch controlled ablation."""

    P0 = "arm-p0"
    """Environment validation and throughput measurement. Not a research arm."""


class WandbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project: str = Field(
        description="One per arm family: tt-trunk, tt-heads, tt-scratch, tt-encdec."
    )
    entity: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = Field(
        default="online", description="'offline' for long unattended runs."
    )
    notes: str | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name_or_path: str
    attn_implementation: str = "sdpa"

    # Attention pattern, needed for honest MFU accounting. ModernBERT-style
    # models make every ``global_every``-th layer global and the rest local.
    global_every: int | None = None
    local_window: int | None = None


class OptimConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    grad_clip: float = 1.0


class RunConfig(BaseModel):
    """A complete run. Logged to wandb in full, so a run is reproducible from wandb alone."""

    model_config = ConfigDict(frozen=True)

    name: str
    arm: Arm
    track: Track
    seed: int = 0

    model: ModelConfig
    optim: OptimConfig = OptimConfig()
    wandb: WandbConfig

    seq_len: int = 8192
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    steps: int = 100
    warmup_steps: int = 0
    mlm_probability: float = 0.3
    activation_checkpointing: bool = False
    log_every: int = 1

    @property
    def tags(self) -> list[str]:
        """Mandatory wandb tags. Results tables filter on these."""
        return [self.track.value, self.arm.value, f"base:{self.model.name_or_path}"]

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.model_validate(raw)

    def resolved(self) -> dict[str, Any]:
        """The fully resolved config, for logging as the wandb run config.

        Logged in full rather than as a path to a YAML, so the run does not
        depend on this repository's working tree to be interpretable later.
        """
        return self.model_dump(mode="json")
