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

    # Tokenizer facts the MLM objective needs. Populated from the tokenizer by
    # the launcher rather than hand-written, and required for MLM: without a
    # mask token the corruption step silently does nothing and the model trains
    # on an identity task that looks like it is converging beautifully.
    mask_token_id: int | None = None
    vocab_size: int | None = None

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


class Objective(StrEnum):
    """What the model is trained to do. Selects a loss, not a training loop.

    Adding one means registering a new objective, never editing the loop. If a
    new arm needs the loop changed, the abstraction is wrong.
    """

    MLM = "mlm"
    """Masked language modelling. Arms A and E phase 2."""

    CLM = "clm"
    """Causal language modelling. Arm E phase 1 of the biphasic schedule."""


class ScheduleConfig(BaseModel):
    """Learning-rate schedule."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 0
    min_lr_ratio: float = Field(
        default=0.1, description="Floor as a fraction of peak lr, so decay does not reach zero."
    )


class CheckpointConfig(BaseModel):
    """Checkpointing policy.

    Runs here are measured in days on a machine that is also someone's desktop,
    so resuming cleanly is a requirement rather than a nicety.
    """

    model_config = ConfigDict(frozen=True)

    dir: Path = Path("checkpoints")
    every_steps: int = Field(default=1000, description="0 disables periodic checkpoints.")
    keep_last: int = Field(default=2, description="Older checkpoints are deleted. 0 keeps all.")
    resume: bool = Field(
        default=True,
        description="Resume from the latest checkpoint in dir if one exists. A long run that "
        "silently restarts from scratch after a crash wastes days before anyone notices.",
    )


class DataConfig(BaseModel):
    """Where training tokens come from.

    ``sources`` names loaders by key; the mixing weights decide sampling
    proportions. Track is enforced at load time, since a mixed-track batch makes
    a model research-only and the contamination is invisible afterwards.
    """

    model_config = ConfigDict(frozen=True)

    sources: dict[str, float] = Field(
        default_factory=dict, description="Source name to mixing weight. Weights are normalised."
    )
    cache_dir: Path = Path("data/raw")
    variant: str | None = Field(
        default=None, description="Transcript layer where a loader offers one, e.g. 'asr'."
    )
    channel_version: str | None = Field(
        default=None,
        description="ASR channel artifact applied to this data. Comparing models trained "
        "under different channel versions is a silent confound, so it is recorded.",
    )


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
    objective: Objective = Objective.MLM
    schedule: ScheduleConfig = ScheduleConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    data: DataConfig = DataConfig()

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
