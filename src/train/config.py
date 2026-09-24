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
    # encdec: E3 built here, src/train/encdec.py (base may be an EncDec.save dir). hf_encdec: a native Hugging Face
    # encoder-decoder (T5Gemma 2) run through its own forward; same batches as encdec, DDP, full parameters
    # hybrid: E4b, src/train/hybrid.py: a decoder-only model (base) reading the prompt and the answer through its own
    # self-attention as in E1, plus the text encoder of a native encoder-decoder (encoder_base) over the prompt, read
    # through new zero-initialised cross-attention in every layer; DDP, full parameters. base may be a Hybrid.save dir.
    # stitched: E3-mix-and-match, src/train/stitched.py: the text encoder of encoder_base with the decoder of base (both
    # T5Gemma 2, different sizes), joined by an affine stitch on the encoder states fitted beforehand (`stitch`,
    # src/train/fit_stitch.py); everything trains, DDP. base may be a Stitched.save dir.
    arch: Literal["decoder", "encdec", "hf_encdec", "hybrid", "stitched"] = "decoder"
    encoder_base: str | None = None  # hybrid / stitched: the encoder-decoder checkpoint whose text encoder is used
    stitch: Path | None = None  # stitched: the fitted map (fit_stitch.py output); None = a fresh random map
    # parameter-name substrings that train; everything else is frozen (None: every parameter trains). E4c: [encoder, cross]
    trainable: list[str] | None = None
    # E3h (hf_encdec only): the decoder is fed the prompt as well, before the target, so it reads the raw input
    # through self-attention and the encoder's view of it through cross-attention; loss on the target only
    decoder_sees_prompt: bool = False
    lora_r: int = 0  # encdec only: LoRA rank on both towers (0 = full training); cross-attention always trains in full
    lora_alpha: int = 0  # 0 = 2 * lora_r
    # fsdp: shard the weights, gradients and 8-bit optimizer state across the GPUs (FSDP2), bf16 compute; any
    # arch. The only way 3.8B+ parameters train in full on two 32 GB cards; ~35% slower than DDP (weights cross PCIe).
    sharding: Literal["ddp", "fsdp"] = "ddp"
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
    # fp32_master: bf16 model, fp32 master copy owned by the optimizer (E1's recipe; ~5 bytes/param/GPU when
    # sharded). bf16_sr: the bf16 weights are the only copy, updated with stochastic rounding (torchao's 8-bit
    # AdamW; ~3 bytes/param/GPU sharded): what lets 7.5B parameters train in full on two 32 GB cards. fsdp only.
    weights: Literal["fp32_master", "bf16_sr"] = "fp32_master"
    # learning-rate multipliers by parameter-name substring (first match wins), e.g. {"cross": 10} gives the hybrid's new
    # cross-attention ten times the base rate so a fine-tune short enough for the pretrained path has a chance to use it
    lr_mult: dict[str, float] = {}


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
    objective: Literal["causal", "mntp", "seq2seq", "mixed"]  # mixed: seq2seq + MNTP on the encoder (E3)
    mask_prob: float = 0.2
    mntp_weight: float = 1.0  # mixed: weight of the encoder-side masked loss relative to the seq2seq loss
    max_target: int = 2048  # seq2seq: decoder side length cap
    cut: tuple[float, float] = (0.25, 0.75)  # seq2seq: where a document is cut, as a fraction of its length
    mask_token: str = "<|image_pad|>"  # a Qwen3 special token that never occurs in text
    # None: the training transcripts themselves. Otherwise: transcript text from outside the training
    # corpus, one CorpusSpec per interim corpus; documents that are labelled calls are excluded.
    corpora: list[CorpusSpec] | None = None


class TrainConfig(BaseModel):
    experiment: Literal["e1", "e2", "e3", "e4", "e5", "e6"]
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
    keep_checkpoints: int = 2  # resume checkpoints kept on disk; a save needs (keep + 1) × checkpoint size free at the moment of writing
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
