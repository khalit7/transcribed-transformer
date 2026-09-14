"""Run configuration: one YAML per run under configs/<experiment>/, validated here, logged to wandb resolved."""

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train: Path = Path("data/labelled_data/train.jsonl")
    val: Path = Path("data/labelled_data/val.jsonl")
    # transcript variants to render for training; "labelled" = the one the label was made on
    variants: list[str] = ["labelled"]
    max_seq_len: int = 32768
    val_examples: int = 512  # val-loss subset, fixed by seed
    cache_dir: Path = Path("data/interim/sft")


class ModelConfig(BaseModel):
    base: str = "Qwen/Qwen3-1.7B-Base"
    attn: Literal["flash_attention_2", "sdpa", "flex_attention"] = "flash_attention_2"
    gradient_checkpointing: bool = True
    # E2 flips this: bidirectional attention over the prompt, causal over the answer
    prefix_lm: bool = False


class OptimConfig(BaseModel):
    lr: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    optimizer: Literal["adamw8bit", "adamw"] = "adamw8bit"


class CorpusSpec(BaseModel):
    """A slice of an interim corpus (data/interim/<name>/{train,val}.jsonl) for the adaptation stage."""
    path: Path
    docs: int  # how many documents to sample (seeded, one pass, roughly this many)
    val_docs: int = 50


class AdaptConfig(BaseModel):
    """A short adaptation stage on the training transcripts (no labels) before task fine-tuning.
    causal: next-token prediction with causal attention, loss on every token (the causal arm's own
    objective). mntp: masked next-token prediction with bidirectional attention, LLM2Vec-style: a
    share of tokens is replaced by mask_token and each masked token is predicted from the hidden
    state one position before it, loss on masked positions only."""
    objective: Literal["causal", "mntp"]
    mask_prob: float = 0.2
    mask_token: str = "<|image_pad|>"  # a Qwen3 special token that never occurs in text
    # None: the training transcripts themselves. Otherwise: transcript text from outside the training
    # corpus, one CorpusSpec per interim corpus; documents that are labelled calls are excluded.
    corpora: list[CorpusSpec] | None = None


class TrainConfig(BaseModel):
    experiment: Literal["e1", "e2", "e3", "e4", "e5"]
    name: str
    seed: int = 0
    data: DataConfig
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    adapt: AdaptConfig | None = None  # set: adaptation stage on transcripts instead of task fine-tuning
    epochs: float = 1.0
    batch_sequences: int = 32  # sequences per optimizer step, across all ranks
    micro_tokens: int = 32768  # padded-token budget per micro-batch per rank
    log_every: int = 10
    val_every: int = 200
    checkpoint_minutes: int = 30
    out_dir: Path = Path("checkpoints")
    wandb_project: str = "tt-decoder"
    peak_tflops: float = 209.5  # RTX 5090 dense bf16, spec-sheet figure, used for MFU only
    tags: list[str] = Field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.name

    def wandb_tags(self) -> list[str]:
        local = Path(self.model.base).exists()  # an adapted checkpoint: the original base comes in via tags
        size = [] if local else [self.model.base.split("/")[-1]]
        stage = [f"adapt-{self.adapt.objective}"] if self.adapt else []
        return sorted({self.experiment, self.model.base, *size, *stage, *self.tags})


def load_config(path: Path) -> TrainConfig:
    return TrainConfig.model_validate(yaml.safe_load(path.read_text()))
