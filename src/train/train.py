"""Fine-tune a decoder on (prompt, target) examples with the loss on the target only.

    torchrun --nproc_per_node 2 -m src.train.train configs/e1/qwen3-1.7b-base-p.yaml [--resume] [--smoke N]

Data parallel (DDP, one process per GPU). The model and its gradients are bf16; the optimizer owns
an fp32 master copy of every parameter and writes the update back after each step (bf16 alone
rounds away updates smaller than a weight's resolution), 8-bit AdamW by default, gradient
checkpointing. Logits are computed only at target positions, so the
vocabulary-sized tensor never spans the whole sequence. Every rank builds the same batching plan
(src/train/data.plan_epoch) and the loss is scaled so gradient accumulation and data parallelism
give the exact mean over the step's target tokens. Checkpoints on a wall-clock interval and at the
end; --resume picks up the latest one (the plan is deterministic, so the data position is the step).
Everything logs to wandb (rank 0), tagged per CLAUDE.md; --smoke N runs N steps without wandb.
"""

import argparse
import contextlib
import datetime
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

from src.train.config import TrainConfig, load_config
from src.train.data import (
    Example,
    Step,
    build_corpus_examples,
    build_examples,
    build_transcript_examples,
    collate,
    collate_encdec,
    labelled_doc_ids,
    mask_encoder_side,
    mntp_mask,
    plan_epoch,
    seq2seq_encoder_count,
    seq2seq_target_count,
)
from src.train.encdec import EncDec
from src.train.hybrid import Hybrid
from src.train.masks import FLEX_KERNEL_OPTIONS, prefix_lm_mask
from src.train.stitched import Stitched


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def setup_dist() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        # rank 0 tokenises the data while the others wait at a barrier; a 500M-token corpus build takes
        # longer than NCCL's default 10-minute timeout
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    return rank, world, device


def load_model(cfg: TrainConfig, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.base)
    attn = cfg.model.attn
    if cfg.model.arch == "encdec":
        if attn == "flash_attention_2":
            try:
                import flash_attn  # type: ignore[import-untyped]
            except ImportError:
                log("flash_attn not importable; falling back to sdpa")
                attn = "sdpa"
        model: torch.nn.Module
        if EncDec.is_encdec_dir(cfg.model.base):
            model = EncDec.load(cfg.model.base, attn, cfg.model.gradient_checkpointing)
        else:
            model = EncDec(cfg.model.base, attn, cfg.model.gradient_checkpointing, cfg.model.lora_r, cfg.model.lora_alpha)
        if cfg.model.sharding == "fsdp":
            if cfg.model.lora_r:
                raise SystemExit("fsdp is for full-parameter training; drop lora_r")
            return tok, shard_encdec(prepare_for_sharding(model, cfg, device)), attn
        return tok, model.to(device), attn
    if cfg.model.arch == "hybrid":
        if cfg.model.lora_r or cfg.model.sharding == "fsdp":
            raise SystemExit("hybrid trains every parameter under DDP; drop lora_r / sharding")
        if attn == "flash_attention_2":
            try:
                import flash_attn  # type: ignore[import-untyped]
            except ImportError:
                log("flash_attn not importable; falling back to sdpa")
                attn = "sdpa"
        if Hybrid.is_hybrid_dir(cfg.model.base):
            model = Hybrid.load(cfg.model.base, attn, cfg.model.gradient_checkpointing)
        else:
            if not cfg.model.encoder_base:
                raise SystemExit("hybrid needs model.encoder_base (the encoder-decoder whose text encoder is attached)")
            model = Hybrid.from_pretrained(cfg.model.base, cfg.model.encoder_base, attn, cfg.model.gradient_checkpointing)
        if getattr(model.config, "final_logit_softcapping", None):
            raise SystemExit("hybrid: target_loss applies lm_head directly; this decoder's logit softcapping would be skipped")
        return tok, model.to(device), attn
    if cfg.model.arch == "stitched":
        if cfg.model.lora_r:
            raise SystemExit("stitched trains every parameter; drop lora_r")
        if attn == "flash_attention_2":
            log("T5Gemma 2 has no flash-attention path in transformers; using sdpa")
            attn = "sdpa"
        if Stitched.is_stitched_dir(cfg.model.base):
            model = Stitched.load(cfg.model.base, attn, cfg.model.gradient_checkpointing)
        else:
            if not cfg.model.encoder_base:
                raise SystemExit("stitched needs model.encoder_base (the encoder donor); model.base is the decoder donor")
            if cfg.model.stitch is None:
                log("model.stitch not set: the stitch starts random (the decoder reads noise at step 0)")
            model = Stitched.from_pretrained(cfg.model.encoder_base, cfg.model.base, attn, cfg.model.gradient_checkpointing, cfg.model.stitch)
        if getattr(model.config, "final_logit_softcapping", None):
            raise SystemExit("stitched: target_loss applies lm_head directly; this decoder's logit softcapping would be skipped")
        if cfg.model.sharding == "fsdp":  # the 4B arms: ~4.2B parameters, 42 GB of weights, gradients, masters and state
            return tok, shard_stitched(prepare_for_sharding(model, cfg, device)), attn
        return tok, model.to(device), attn
    if cfg.model.arch == "hf_encdec":
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
        if cfg.model.lora_r:
            raise SystemExit("hf_encdec trains every parameter (DDP or fsdp); drop lora_r")
        hf_config = AutoConfig.from_pretrained(cfg.model.base)
        cls = AutoModelForSeq2SeqLM._model_mapping[type(hf_config)]
        if attn == "flash_attention_2" and not cls._supports_flash_attn:
            log(f"{cls.__name__} has no flash-attention path in transformers; using sdpa")
            attn = "sdpa"
        if getattr(hf_config.get_text_config(decoder=True), "final_logit_softcapping", None):
            raise SystemExit("hf_encdec: target_loss applies lm_head directly; this model's logit softcapping would be skipped")
        hf = cls.from_pretrained(cfg.model.base, dtype=torch.bfloat16, attn_implementation=attn)
        for n_, m in hf.named_modules():  # the image side of a multimodal encoder: never called, frozen
            if n_.endswith(("vision_tower", "multi_modal_projector")):
                m.requires_grad_(False)
        if cfg.model.gradient_checkpointing:
            hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.config.use_cache = False
        if cfg.model.sharding == "fsdp":
            return tok, shard_hf(prepare_for_sharding(hf, cfg, device)), attn
        return tok, hf.to(device), attn
    if cfg.model.prefix_lm and attn == "flash_attention_2":
        attn = "flex_attention"  # FA2 cannot express the prefix-LM mask
    if attn == "flash_attention_2":
        try:
            import flash_attn  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            log("flash_attn not importable; falling back to sdpa")
            attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base, dtype=torch.bfloat16, attn_implementation=attn)
    for n_, m in model.named_modules():  # a multimodal checkpoint's image side (Gemma 3 4B+): never called, frozen
        if n_.endswith(("vision_tower", "multi_modal_projector")):
            m.requires_grad_(False)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if cfg.model.sharding == "fsdp":
        return tok, shard_hf(prepare_for_sharding(model, cfg, device)), attn
    return tok, model.to(device), attn  # type: ignore[arg-type]


def export_hf(inner, sd: dict[str, torch.Tensor], final: Path) -> None:
    """Write a gathered state dict (module parameter names, bf16, CPU) as a Hugging Face checkpoint through the
    library's own save_pretrained, so the on-disk tensor names follow its conventions (they differ from the
    module names for multimodal classes such as Gemma 3 4B, and vLLM reads the on-disk names). A second copy
    of the model is built on the CPU from the config: 8.6 GB at 4B."""
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    auto = AutoModelForSeq2SeqLM if getattr(inner.config, "is_encoder_decoder", False) else AutoModelForCausalLM
    with torch.device("cpu"):
        cpu = auto.from_config(inner.config, dtype=torch.bfloat16)
    missing, unexpected = cpu.load_state_dict(sd, strict=False)
    cpu.tie_weights()
    tied = getattr(cpu.config.get_text_config(), "tie_word_embeddings", False)  # a tied head is absent from named_parameters
    missing = [k for k in missing if not (tied and k.endswith("lm_head.weight"))]
    if missing or unexpected:
        raise SystemExit(f"export: state dict mismatch, missing {missing[:5]}, unexpected {unexpected[:5]}")
    cpu.save_pretrained(final, safe_serialization=True)
    if getattr(inner, "generation_config", None) is not None:
        inner.generation_config.save_pretrained(final)
    del cpu
    # transformers 5 writes a multimodal checkpoint's image tower as `vision_tower.<x>`; the original Gemma 3 files (and
    # vLLM 0.29, which reads them) use `vision_tower.vision_model.<x>`. The tower is frozen, so only the names change.
    from safetensors.torch import load_file, save_file
    for shard in sorted(final.glob("model*.safetensors")):
        sd_ = load_file(str(shard))
        ren = {k: k.replace("vision_tower.", "vision_tower.vision_model.", 1) for k in sd_ if k.startswith("vision_tower.") and not k.startswith("vision_tower.vision_model.")}
        if ren:
            save_file({ren.get(k, k): v for k, v in sd_.items()}, str(shard), metadata={"format": "pt"})
    idx = final / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        d["weight_map"] = {(k.replace("vision_tower.", "vision_tower.vision_model.", 1) if k.startswith("vision_tower.") and not k.startswith("vision_tower.vision_model.") else k): v
                           for k, v in d["weight_map"].items()}
        idx.write_text(json.dumps(d, indent=1))


def freeze_except(model: torch.nn.Module, trainable: list[str]) -> tuple[int, int]:
    """Freeze every parameter whose name contains none of the substrings; returns (trainable, frozen) counts."""
    n_on = n_off = 0
    for n, p in model.named_parameters():
        on = any(key in n for key in trainable)
        p.requires_grad_(on)
        if on:
            n_on += p.numel()
        else:
            n_off += p.numel()
    return n_on, n_off


def prepare_for_sharding(model, cfg: TrainConfig, device: torch.device):
    """The weights as the optimizer will own them, on the device before sharding: fp32 under the master-copy
    recipe, bf16 under stochastic rounding."""
    if cfg.optim.weights == "fp32_master":
        model = model.float()
    return model.to(device)


def shard_units(model) -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
    """FSDP2 units for a Hugging Face model: every transformer block (each element of a ModuleList named
    `layers`, outside any image tower) and, as one unit each, a frozen image tower and its projector. The
    rest (embeddings, norms, head) belongs to the root. Children are sharded before parents, so a tower's
    own inner layers are not listed separately."""
    towers = [(n, m) for n, m in model.named_modules() if n.endswith(("vision_tower", "multi_modal_projector"))]
    blocks: list[torch.nn.Module] = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and not any(n.startswith(t + ".") for t, _ in towers):
            blocks.extend(m)
    return blocks, [m for _, m in towers]


class ShardedHF(torch.nn.Module):
    """A Hugging Face model behind one forward that computes the task loss. Under FSDP2 the root unit's
    parameters (embeddings, norms, the tied head) are gathered by the call the loop makes, so the loop must
    call the root; calling the inner `.model` directly bypasses the root's hooks (mixed Tensor/DTensor error)."""

    def __init__(self, hf: torch.nn.Module):
        super().__init__()
        self.hf = hf
        self.config = hf.config

    def forward(self, batch: dict, device: torch.device, prefix_lm: str | None = None) -> tuple[torch.Tensor, int]:
        return hf_task_loss(self.hf, batch, device, prefix_lm)


class ShardedStitched(torch.nn.Module):
    """A Stitched model behind one forward that computes the task loss (FSDP2 root unit, as ShardedHF). The
    attribute is named `hf` so the sharded export's prefix handling applies unchanged."""

    def __init__(self, m: Stitched):
        super().__init__()
        self.hf = m
        self.config = m.config

    def forward(self, batch: dict, device: torch.device, prefix_lm: str | None = None) -> tuple[torch.Tensor, int]:
        labels = batch["labels"].to(device, non_blocking=True)
        hidden = self.hf(batch["enc_ids"].to(device, non_blocking=True), batch["enc_mask"].to(device, non_blocking=True),
                         batch["dec_ids"].to(device, non_blocking=True), batch["dec_mask"].to(device, non_blocking=True))
        mask = labels != -100
        return chunked_ce(self.hf.lm_head, hidden[mask], labels[mask]), int(mask.sum())


def shard_stitched(model: Stitched) -> ShardedStitched:
    """FSDP2 over the stitched pair: every encoder and decoder block a unit; the stitch, embeddings, norms and head
    in the root, whose forward is the loss."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for layer in list(model.encoder.layers) + list(model.decoder.layers):
        fully_shard(layer, mp_policy=mp)
    wrapped = ShardedStitched(model)
    fully_shard(wrapped, mp_policy=mp)
    return wrapped


def shard_hf(model) -> ShardedHF:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    blocks, towers = shard_units(model)
    for m in blocks + towers:
        fully_shard(m, mp_policy=mp)
    wrapped = ShardedHF(model)
    fully_shard(wrapped, mp_policy=mp)
    return wrapped


def shard_encdec(model: EncDec) -> EncDec:
    """FSDP2 over the encoder-decoder: every parameter fp32 and sharded, bf16 compute and gradient reduction,
    units at the granularity the code calls (the encoder's layers run through their own forward; the decoder
    loop calls self-attention, MLP and cross-attention modules directly, so those are the units; the rest
    belongs to the root)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for layer in model.encoder.layers:
        fully_shard(layer, mp_policy=mp)
    for layer in model.decoder.layers:
        fully_shard(layer.self_attn, mp_policy=mp)
        fully_shard(layer.mlp, mp_policy=mp)
    for c in model.cross:
        fully_shard(c, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def is_sharded(model) -> bool:
    return any(type(p).__name__ == "DTensor" for p in model.parameters())


def reshard(model) -> None:
    """Put every FSDP2 unit back into its sharded state. The root unit stays unsharded after a forward with no
    backward (validation), so its parameters read as plain tensors until something reshards them."""
    for m in model.modules():
        if hasattr(m, "reshard") and hasattr(m, "unshard"):
            m.reshard()


def target_loss(model, batch: dict, device: torch.device, prefix_lm: str | None = None,
                mntp_weight: float = 1.0, parts: dict | None = None) -> tuple[torch.Tensor, int]:
    """Sum of cross-entropy over target tokens. Hidden states at the positions that predict a target
    token go through lm_head; nothing else does. `prefix_lm` names the attention implementation when
    the prompt is attended bidirectionally (E2); None keeps the causal mask (E1)."""
    raw = model.module if isinstance(model, DDP) else model
    if isinstance(raw, (ShardedHF, ShardedStitched)):
        return raw(batch, device, prefix_lm)  # the root forward: FSDP2 gathers the root's parameters
    labels = batch["labels"].to(device, non_blocking=True)
    if isinstance(raw, (Hybrid, Stitched)):
        hidden = raw(batch["enc_ids"].to(device, non_blocking=True), batch["enc_mask"].to(device, non_blocking=True),
                     batch["dec_ids"].to(device, non_blocking=True), batch["dec_mask"].to(device, non_blocking=True))
        mask = labels != -100
        return chunked_ce(raw.lm_head, hidden[mask], labels[mask]), int(mask.sum())
    if isinstance(raw, EncDec):
        hidden, enc = raw(batch["enc_ids"].to(device, non_blocking=True), batch["enc_mask"].to(device, non_blocking=True),
                          batch["dec_ids"].to(device, non_blocking=True), batch["dec_mask"].to(device, non_blocking=True))
        mask = labels != -100  # decoder input is already shifted: position t predicts labels[t]
        loss = chunked_ce(raw.lm_head, hidden[mask], labels[mask])
        n = int(mask.sum())
        if "enc_labels" in batch:  # mixed objective: MNTP on the encoder's own states, position t predicts token t+1
            el = batch["enc_labels"].to(device, non_blocking=True)[:, 1:]
            em = el != -100
            enc_loss = chunked_ce(raw.lm_head, enc[:, :-1][em], el[em])
            if parts is not None:  # per-component sums and counts, for the val report
                parts["seq2seq"] += loss.detach(); parts["seq2seq_n"] += n
                parts["mntp"] += enc_loss.detach(); parts["mntp_n"] += int(em.sum())
            loss = loss + mntp_weight * enc_loss
            n += int(em.sum())
        return loss, n
    return hf_task_loss(raw, batch, device, prefix_lm)


def hf_task_loss(raw, batch: dict, device: torch.device, prefix_lm: str | None) -> tuple[torch.Tensor, int]:
    """target_loss for a Hugging Face model: its encoder-decoder or decoder forward, chunked CE at the target positions."""
    labels = batch["labels"].to(device, non_blocking=True)
    if getattr(raw.config, "is_encoder_decoder", False):  # hf_encdec: the library's own encoder-decoder forward
        out = raw.model(input_ids=batch["enc_ids"].to(device, non_blocking=True),
                        attention_mask=batch["enc_mask"].to(device, non_blocking=True),
                        decoder_input_ids=batch["dec_ids"].to(device, non_blocking=True),
                        decoder_attention_mask=batch["dec_mask"].to(device, non_blocking=True), use_cache=False)
        mask = labels != -100
        return chunked_ce(raw.lm_head, out.last_hidden_state[mask], labels[mask]), int(mask.sum())
    ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch["attention_mask"].to(device, non_blocking=True)
    base = raw.model
    head = raw.lm_head
    if prefix_lm:
        mask = prefix_lm_mask(batch["prompt_len"].to(device), attn.sum(1), ids.shape[1], prefix_lm)
        extra = {"kernel_options": FLEX_KERNEL_OPTIONS} if prefix_lm == "flex_attention" else {}
        hidden = base(input_ids=ids, attention_mask={"full_attention": mask}, **extra).last_hidden_state
    else:
        hidden = base(input_ids=ids, attention_mask=attn).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    return chunked_ce(head, h, tgt[mask]), int(mask.sum())


CE_CHUNK = 1024
VAL_PARTS: dict = {}  # the last val's mixed-objective halves, for wandb


def chunked_ce(head, h: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Sum of cross-entropy over the rows of h, computing the vocabulary-sized logits CE_CHUNK rows at a
    time under activation checkpointing, so at most one chunk of logits is alive at once. Matters for the
    adaptation stage, where every token is a target (32k rows × 151k vocab is 12.5 GB in fp32)."""
    if h.shape[0] <= CE_CHUNK:
        return F.cross_entropy(head(h).float(), tgt, reduction="sum")

    def piece(hh: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(head(hh).float(), tt, reduction="sum")

    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for i in range(0, h.shape[0], CE_CHUNK):
        total = total + torch.utils.checkpoint.checkpoint(piece, h[i:i + CE_CHUNK], tgt[i:i + CE_CHUNK], use_reentrant=False)
    return total


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    o = cfg.optim
    if step < o.warmup_steps:
        return o.lr * (step + 1) / o.warmup_steps
    p = min(1.0, (step - o.warmup_steps) / max(1, total - o.warmup_steps))
    return o.lr * (o.min_lr_ratio + (1 - o.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def param_groups_for(names: list[str], params: list, cfg: TrainConfig) -> list[dict]:
    """Weight decay off for norms, biases and vectors; a learning-rate multiplier per optim.lr_mult substring
    (first match wins, 1 otherwise). Groups carry `lr_mult`; set_lr applies it."""
    groups: dict[tuple[float, float], list] = {}
    for n, p in zip(names, params):
        wd = 0.0 if (p.ndim < 2 or "norm" in n or "bias" in n) else cfg.optim.weight_decay
        mult = next((m for key, m in cfg.optim.lr_mult.items() if key in n), 1.0)
        groups.setdefault((wd, mult), []).append(p)
    return [{"params": ps, "weight_decay": wd, "lr_mult": mult} for (wd, mult), ps in groups.items()]


class ShardOptimizer:
    """8-bit AdamW over the local shards of an FSDP2 model. fp32_master: the sharded fp32 params are the
    masters, FSDP casts to bf16 for compute (bitsandbytes). bf16_sr: the sharded params are bf16 and torchao's
    kernel applies the fp32-computed update with stochastic rounding, so steps below bf16's resolution survive
    in expectation. Gradients arrive as sharded DTensors and are handed to leaf views of the shards."""

    def __init__(self, model, cfg: TrainConfig):
        import bitsandbytes as bnb
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.locals = [p.to_local().detach().requires_grad_(True) for p in self.params]  # leaves sharing the storage
        groups = param_groups_for(self.names, self.locals, cfg)
        if cfg.optim.weights == "bf16_sr":
            if any(loc.dtype != torch.bfloat16 for loc in self.locals):
                raise SystemExit("optim.weights: bf16_sr expects bf16 sharded parameters")
            from torchao.optim import AdamW8bit
            self.opt = AdamW8bit(groups, lr=cfg.optim.lr, betas=cfg.optim.betas, bf16_stochastic_round=True)
        else:
            self.opt = bnb.optim.AdamW8bit(groups, lr=cfg.optim.lr, betas=cfg.optim.betas)
        self.clip = cfg.optim.grad_clip
        self.model = model

    @property
    def param_groups(self):
        return self.opt.param_groups

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            if isinstance(g["lr"], torch.Tensor):  # torchao keeps lr as a tensor and refuses a float
                g["lr"].fill_(lr * g.get("lr_mult", 1.0))
            else:
                g["lr"] = lr * g.get("lr_mult", 1.0)

    @torch.no_grad()
    def step(self) -> float:
        # global norm in fp32 over the local shards (the DTensor path would accumulate in the gradient dtype)
        sq = torch.zeros((), device=self.locals[0].device, dtype=torch.float32)
        for p in self.params:
            if p.grad is not None:
                sq += p.grad.to_local().float().pow(2).sum()
        if dist.is_initialized():
            dist.all_reduce(sq)
        norm = sq.sqrt()
        scale = torch.clamp(self.clip / (norm + 1e-6), max=1.0)
        for p, loc in zip(self.params, self.locals):
            if p.grad is not None:
                loc.grad = p.grad.to_local().mul_(scale)
            else:
                loc.grad = None
            p.grad = None
        self.opt.step()
        for loc in self.locals:
            loc.grad = None
        return float(norm)

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict()}  # per rank; the model's own (sharded) state carries the masters

    def load_state_dict(self, d: dict) -> None:
        self.opt.load_state_dict(d["opt"])


class MasterOptimizer:
    """fp32 master weights + the configured optimizer over them. step(): move the bf16 gradients to the
    masters (as fp32), clip, update, copy the masters back into the bf16 model."""

    def __init__(self, model, cfg: TrainConfig):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.masters = [p.detach().float().clone() for p in self.params]
        groups = param_groups_for(self.names, self.masters, cfg)
        self.opt: torch.optim.Optimizer
        if cfg.optim.optimizer == "adamw8bit":
            import bitsandbytes as bnb
            self.opt = bnb.optim.AdamW8bit(groups, lr=cfg.optim.lr, betas=cfg.optim.betas)
        else:
            self.opt = torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=cfg.optim.betas, fused=True)
        self.clip = cfg.optim.grad_clip

    @property
    def param_groups(self):
        return self.opt.param_groups

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)

    @torch.no_grad()
    def step(self) -> float:
        for p, m in zip(self.params, self.masters):
            m.grad = p.grad.float() if p.grad is not None else torch.zeros_like(m)
            p.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.masters, self.clip)
        self.opt.step()
        for p, m in zip(self.params, self.masters):
            p.copy_(m)
            m.grad = None
        return float(norm)

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "masters": self.masters}

    def load_state_dict(self, d: dict) -> None:
        self.opt.load_state_dict(d["opt"])
        with torch.no_grad():
            for m, saved in zip(self.masters, d["masters"]):
                m.copy_(saved)
            for p, m in zip(self.params, self.masters):
                p.copy_(m)


def flops_of(n_params: int, n_layers: int, d_model: int, lengths: list[int]) -> float:
    """Model FLOPs for forward+backward, PaLM-style: 6N per token plus 6·L·d·ℓ per token for causal
    attention (half of the full 12·L·d·ℓ). Recompute for checkpointing is not counted, so the MFU
    reported is model-FLOPs utilisation. An approximation, used for MFU only."""
    tokens = sum(lengths)
    return 6 * n_params * tokens + 6 * n_layers * d_model * sum(l * l for l in lengths)


def full_state_dict(model) -> dict:
    """The model's state dict as plain tensors on CPU: gathered from the shards when sharded (every rank must call)."""
    if is_sharded(model):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )
        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    raw = model.module if isinstance(model, DDP) else model
    return raw.state_dict()


def save_checkpoint(path: Path, model, opt, step: int, epoch: int, cfg: TrainConfig, rank: int) -> None:
    tmp = path.with_suffix(".tmp")
    if rank == 0:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    world = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        (tmp / "state.json").write_text(json.dumps({"step": step, "epoch": epoch, "world": world, "config": cfg.model_dump(mode="json")}))
    if is_sharded(model):  # each rank writes its own shards: no gather, no full copy in system memory (30 GB at 7.5B)
        reshard(model)
        plain = [n for n, p in model.named_parameters() if p.requires_grad and type(p).__name__ != "DTensor"]
        if plain:
            raise SystemExit(f"trainable parameters outside any FSDP unit (their gradients would never be reduced): {plain[:8]}")
        shards = {n: p.to_local().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(shards, tmp / f"model-rank{rank}.pt")
        del shards
        torch.save(opt.state_dict(), tmp / f"optim-rank{rank}.pt")
    elif rank == 0:
        torch.save(full_state_dict(model), tmp / "model.pt")
        torch.save(opt.state_dict(), tmp / "optim.pt")
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    if rank == 0:
        if path.exists():
            shutil.rmtree(path)
        tmp.rename(path)
        log(f"checkpoint -> {path}")
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def latest_checkpoint(run_dir: Path) -> Path | None:
    # a step-N.tmp directory is a save that never finished (e.g. a hang mid-checkpoint): not a checkpoint
    cks = sorted((p for p in run_dir.glob("step-*") if p.name.split("-")[1].isdigit()), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


@torch.no_grad()
def val_loss(model, examples: list[Example], idx: list[int], world: int, rank: int, device, pad_id: int,
             micro_tokens: int, prefix_lm: str | None, make_batch, mntp_weight: float = 1.0) -> float:
    model.eval()
    raw = model.module if isinstance(model, DDP) else model
    if isinstance(raw, Hybrid):
        raw.collect_cross_stats(True)
    total = torch.zeros(2, device=device)
    mine = idx[rank::world]
    plan = plan_epoch([examples[i] for i in mine], max(1, len(mine)), micro_tokens, 1, 0, 0) if mine else []
    micros = [micro for step in plan for micro in step.micro[0]]
    n_local = torch.tensor([len(micros)], device=device)
    if is_sharded(model):  # every rank must run the same number of forward passes
        n_all = n_local.clone()
        dist.all_reduce(n_all, op=dist.ReduceOp.MAX)
        micros += [micros[0]] * (int(n_all.item()) - len(micros))
    parts: dict = {"seq2seq": torch.zeros((), device=device), "seq2seq_n": 0, "mntp": torch.zeros((), device=device), "mntp_n": 0}
    for k, micro in enumerate(micros):
        batch = make_batch([examples[i] for i in mine], micro, hash(tuple(micro)) & 0xFFFF)
        loss, n = target_loss(model, batch, device, prefix_lm, mntp_weight, parts if k < int(n_local.item()) else None)
        if k < int(n_local.item()):
            total += torch.tensor([loss.item(), n], device=device)
    comp = torch.tensor([float(parts["seq2seq"]), parts["seq2seq_n"], float(parts["mntp"]), parts["mntp_n"]], device=device)
    if dist.is_initialized():
        dist.all_reduce(total)
        dist.all_reduce(comp)
    model.train()
    if isinstance(raw, Hybrid):  # how much of the residual stream the cross-attention writes (rank 0's share of val)
        raw.collect_cross_stats(False)
        cs = raw.cross_stats()
        log(f"  cross-attention share of the residual: mean over layers {cs.get('cross_ratio_mean', 0):.4f}, max {cs.get('cross_ratio_max', 0):.4f}; "
            f"gate |1+w|: mean {cs.get('cross_gate_mean', 0):.4f}, max {cs.get('cross_gate_max', 0):.4f}")
        VAL_PARTS.update(cs)
    if comp[3] > 0:  # mixed objective: report the halves too
        vs, vm = comp[0].item() / max(1, comp[1].item()), comp[2].item() / max(1, comp[3].item())
        log(f"  val components: seq2seq {vs:.4f} ({int(comp[1].item())} tokens), mntp {vm:.4f} ({int(comp[3].item())} tokens)")
        VAL_PARTS.update({"val_loss_seq2seq": vs, "val_loss_mntp": vm})
    return (total[0] / total[1].clamp(min=1)).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="run N optimizer steps, no wandb, no checkpoint")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rank, world, device = setup_dist()
    torch.manual_seed(cfg.seed)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    tok, model, attn = load_model(cfg, device)
    if cfg.model.trainable:
        if is_sharded(model):
            raise SystemExit("model.trainable: freeze before sharding is not wired for fsdp; use ddp")
        n_on, n_off = freeze_except(model, cfg.model.trainable)
        log(f"trainable {cfg.model.trainable}: {n_on / 1e6:.1f}M parameters train, {n_off / 1e6:.1f}M frozen")
    prefix_lm = attn if cfg.model.prefix_lm else None
    adapt = cfg.adapt
    mntp = adapt is not None and adapt.objective == "mntp"
    if mntp and not prefix_lm:
        raise SystemExit("mntp adaptation needs model.prefix_lm: true (bidirectional attention)")
    mask_id = int(tok.convert_tokens_to_ids(adapt.mask_token)) if adapt is not None and mntp else -1
    mask_prob = adapt.mask_prob if adapt is not None else 0.0

    def prepare(batch: dict, seed: int) -> dict:
        return mntp_mask(batch, mask_prob, mask_id, seed) if mntp else batch

    pad_kw = {"pad_to": 128, "min_len": 256} if prefix_lm else {}
    encdec = cfg.model.arch in ("encdec", "hf_encdec", "hybrid", "stitched")
    # decoder start token: the tokenizer's <bos> for a native encoder-decoder (what T5Gemma 2 was trained with), eos for E3.
    # The hybrid's decoder reads the prompt itself (opening with the tokenizer's own start token, as in E1): no extra one
    hybrid = cfg.model.arch == "hybrid"
    if hybrid and not cfg.model.decoder_sees_prompt:
        raise SystemExit("hybrid: the decoder reads the prompt; set model.decoder_sees_prompt: true")
    start_id = tok.bos_token_id if cfg.model.arch in ("hf_encdec", "stitched") and tok.bos_token_id is not None else tok.eos_token_id
    seq2seq = adapt is not None and adapt.objective in ("seq2seq", "mixed")
    mixed = adapt is not None and adapt.objective == "mixed"
    if seq2seq and not encdec:
        raise SystemExit("seq2seq/mixed adaptation needs model.arch: encdec")
    if mixed and adapt is not None:
        mask_id = int(tok.convert_tokens_to_ids(adapt.mask_token))
        mask_prob = adapt.mask_prob

    def batch_seed(step: int, rank_: int, k: int) -> int:
        return cfg.seed * 1_000_003 + step * 64 + rank_ * 8 + k

    def step_denominator(st: Step, step: int) -> float:
        """Supervised tokens in the step over all ranks: planned target tokens (times the masked share under
        MNTP), or, for seq2seq, the continuation lengths every rank will draw, recomputed from the seeds."""
        if seq2seq and adapt is not None:
            total = 0.0
            for r, rank_micro in enumerate(st.micro):
                for k, micro in enumerate(rank_micro):
                    sd = batch_seed(step, r, k)
                    total += seq2seq_target_count(train, micro, sd, adapt.max_target, adapt.cut)
                    if mixed:  # expected masked encoder tokens
                        total += mask_prob * seq2seq_encoder_count(train, micro, sd, adapt.max_target, adapt.cut)
            return total
        return st.target_tokens * supervised_share

    def make_batch(examples: list[Example], micro: list[int], seed: int) -> dict:
        if encdec:
            b = collate_encdec(examples, micro, pad_id, start_id, seed, seq2seq,
                               adapt.max_target if adapt else 2048, adapt.cut if adapt else (0.25, 0.75),
                               dec_prompt=cfg.model.decoder_sees_prompt, dec_start=not hybrid)
            return mask_encoder_side(b, mask_prob, mask_id, seed + 1) if mixed else b
        return prepare(collate(examples, micro, pad_id, **pad_kw), seed)

    # loss normaliser per step: the group's target tokens, or their expected masked share under MNTP
    # (the realised count differs by a few per cent; using the expectation avoids a collective)
    supervised_share = mask_prob if mntp else 1.0
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tc = model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
    n_layers, d_model = tc.num_hidden_layers, tc.hidden_size
    if encdec:
        n_layers *= 2  # encoder and decoder both run over their inputs; a rough MFU term
    n_flops_params = n_trainable if cfg.model.arch == "hf_encdec" else n_params  # the frozen image tower never runs
    if cfg.optim.lr_mult:
        n_mult = sum(p.numel() for n, p in model.named_parameters() if any(k in n for k in cfg.optim.lr_mult))
        log(f"lr multipliers {cfg.optim.lr_mult}: {n_mult / 1e6:.1f}M parameters")
    log(f"{cfg.model.base}: {n_params / 1e9:.2f}B params ({n_trainable / 1e9:.2f}B trainable), attn={attn}, world={world}")

    def data():
        limit = cfg.batch_sequences * args.smoke * 2 or None
        if cfg.adapt and cfg.adapt.corpora:
            exclude = labelled_doc_ids(cfg.data.train.parent)
            tr = build_corpus_examples(cfg.adapt.corpora, tok, cfg.data.max_seq_len, "train", cfg.seed, exclude, cfg.data.cache_dir, limit)
            va = build_corpus_examples(cfg.adapt.corpora, tok, cfg.data.max_seq_len, "val", cfg.seed, exclude, cfg.data.cache_dir)
            return tr, va
        build = build_transcript_examples if cfg.adapt else build_examples
        tr = build(cfg.data.train, tok, cfg.data.variants, cfg.data.max_seq_len, cfg.data.cache_dir, limit=limit)
        va = build(cfg.data.val, tok, ["labelled"], cfg.data.max_seq_len, cfg.data.cache_dir)
        return tr, va

    if rank == 0:  # rank 0 tokenises and writes the cache; the others wait and read it
        (train, tstats), (val, vstats) = data()
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    if rank != 0:
        (train, tstats), (val, vstats) = data()
    vidx = list(range(len(val)))
    torch.Generator().manual_seed(cfg.seed)
    import random
    random.Random(cfg.seed).shuffle(vidx)
    vidx = sorted(vidx[:cfg.data.val_examples])
    log(f"train {tstats}\nval {vstats} (val loss on {len(vidx)})")

    steps_per_epoch = len(train) // cfg.batch_sequences
    total_steps = int(steps_per_epoch * cfg.epochs)
    if args.smoke:
        total_steps = args.smoke
    log(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    if world > 1 and not is_sharded(model):
        model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
    opt = ShardOptimizer(model, cfg) if is_sharded(model) else MasterOptimizer(model, cfg)
    step0 = 0
    if args.resume and (ck := latest_checkpoint(run_dir)):
        if is_sharded(model):
            saved_world = json.loads((ck / "state.json").read_text()).get("world", world)
            if saved_world != world:
                raise SystemExit(f"checkpoint {ck} holds {saved_world} shards; this run has {world} ranks")
            shards = torch.load(ck / f"model-rank{rank}.pt", map_location=device)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.to_local().copy_(shards[n])
            del shards
            opt.load_state_dict(torch.load(ck / f"optim-rank{rank}.pt", map_location=device))
        else:
            raw = model.module if isinstance(model, DDP) else model
            # loaded to CPU and copied into the live tensors: on the device the saved masters and optimizer state would
            # sit beside the live ones (about 12 GB at 2B parameters), which does not fit next to a 32 GB card's model
            raw.load_state_dict(torch.load(ck / "model.pt", map_location="cpu"))
            opt.load_state_dict(torch.load(ck / "optim.pt", map_location="cpu"))
        step0 = json.loads((ck / "state.json").read_text())["step"]
        log(f"resumed from {ck} at step {step0}")

    use_wandb = rank == 0 and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.name, tags=cfg.wandb_tags(), dir=str(run_dir),
                   config={**cfg.model_dump(mode="json"), "attn_used": attn, "n_params": n_params, "n_trainable": n_trainable,
                           "train_stats": tstats, "val_stats": vstats, "world_size": world},
                   id=(run_dir / "wandb_id").read_text().strip() if (run_dir / "wandb_id").exists() else None,
                   resume="allow")
        assert wandb.run is not None
        (run_dir / "wandb_id").write_text(wandb.run.id)

    model.train()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t_start = t_ck = time.time()
    step = step0
    epoch = step // steps_per_epoch
    plan = plan_epoch(train, cfg.batch_sequences, cfg.micro_tokens, world, cfg.seed, epoch)

    def step_tokens(st: Step) -> int:
        return sum(len(train[i].input_ids) for rank_micro in st.micro for micro in rank_micro for i in micro)

    tokens_seen = epoch * sum(step_tokens(s) for s in plan) + sum(step_tokens(s) for s in plan[:step % steps_per_epoch])
    window_flops, window_tokens, window_t0 = 0.0, 0, time.time()
    window_comp = torch.zeros(4, device=device)  # mixed objective: seq2seq sum, tokens, mntp sum, tokens over the log window
    if step == 0 and vidx and not args.smoke:  # val before any update: the starting point of every curve
        vl = val_loss(model, val, vidx, world, rank, device, pad_id, cfg.micro_tokens, prefix_lm, make_batch,
                      adapt.mntp_weight if adapt else 1.0)
        log(f"step 0: val_loss {vl:.4f}")
        if use_wandb:
            wandb.log({"val_loss": vl, **VAL_PARTS}, step=0)
    while step < total_steps:
        if step // steps_per_epoch != epoch:
            epoch = step // steps_per_epoch
            plan = plan_epoch(train, cfg.batch_sequences, cfg.micro_tokens, world, cfg.seed, epoch)
        st: Step = plan[step % steps_per_epoch]
        lr = lr_at(step, total_steps, cfg)
        opt.set_lr(lr)
        micros = list(st.micro[rank])
        n_dummy = 0
        if is_sharded(model):  # FSDP: every rank must run the same number of forward/backward passes per step
            n_max = max(len(m) for m in st.micro)
            n_dummy = n_max - len(micros)
            micros += [micros[0]] * n_dummy  # extra passes whose loss is zeroed below
        denom = step_denominator(st, step)
        loss_sum = torch.zeros((), device=device)
        parts: dict = {"seq2seq": torch.zeros((), device=device), "seq2seq_n": 0, "mntp": torch.zeros((), device=device), "mntp_n": 0}
        for k, micro in enumerate(micros):
            dummy = k >= len(micros) - n_dummy
            batch = make_batch(train, micro, batch_seed(step, rank, k))
            # DDP: reduce only on the last micro-batch. FSDP: reduce every micro-batch (gradients accumulate in the
            # sharded fp32 .grad; deferring the reduction would hold full-size unsharded gradients instead)
            sync = contextlib.nullcontext() if (k == len(micros) - 1 or world == 1 or is_sharded(model)) else model.no_sync()
            with sync:
                loss, _ = target_loss(model, batch, device, prefix_lm, adapt.mntp_weight if adapt else 1.0,
                                      None if dummy else parts)
                if dummy:
                    loss = loss * 0.0
                (loss * world / denom).backward()
            loss_sum += loss.detach()
        grad_norm = opt.step()
        step += 1
        if dist.is_initialized():
            dist.all_reduce(loss_sum)
        comp = torch.tensor([float(parts["seq2seq"]), parts["seq2seq_n"], float(parts["mntp"]), parts["mntp_n"]], device=device)
        if dist.is_initialized():
            dist.all_reduce(comp)
        window_comp += comp
        # throughput counters come from the plan, which every rank holds in full: no collective needed
        # (E2 showed the float64 all-reduce of these counters returning garbage in the window after a checkpoint:
        # the ranks had checkpointed one step apart, see do_ck above, and the mismatched collectives completed with junk)
        step_lengths = [len(train[i].input_ids) for rank_micro in st.micro for micro in rank_micro for i in micro]
        window_tokens += sum(step_lengths)
        window_flops += flops_of(n_flops_params, n_layers, d_model, step_lengths)
        tokens_seen += sum(step_lengths)
        if step % cfg.log_every == 0 or step == total_steps:
            dt = time.time() - window_t0
            mfu = window_flops / dt / (cfg.peak_tflops * 1e12 * world)
            mem = torch.cuda.max_memory_allocated(device) / 2**30
            rec = {"step": step, "loss": loss_sum.item() / denom, "lr": lr, "grad_norm": grad_norm,
                   "tokens_seen": tokens_seen, "tokens_per_s": window_tokens / dt, "mfu": mfu,
                   f"gpu{rank}_mem_gib": mem, "epoch": step / steps_per_epoch}
            if window_comp[3] > 0:  # the two halves of the mixed objective, each over its own tokens in the window
                rec["loss_seq2seq"] = window_comp[0].item() / max(1.0, window_comp[1].item())
                rec["loss_mntp"] = window_comp[2].item() / max(1.0, window_comp[3].item())
            window_comp.zero_()
            log(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in rec.items()}))
            if use_wandb:
                wandb.log(rec, step=step)
            window_flops, window_tokens, window_t0 = 0.0, 0, time.time()
        if (step % cfg.val_every == 0 or step == total_steps) and vidx:
            vl = val_loss(model, val, vidx, world, rank, device, pad_id, cfg.micro_tokens, prefix_lm, make_batch,
                          adapt.mntp_weight if adapt else 1.0)
            log(f"step {step}: val_loss {vl:.4f}")
            if use_wandb:
                wandb.log({"val_loss": vl, **VAL_PARTS}, step=step)
        do_ck = time.time() - t_ck > cfg.checkpoint_minutes * 60 or step == total_steps
        if dist.is_initialized():  # a wall-clock decision must be the same on every rank: rank 0's clock decides
            flag = torch.tensor([int(do_ck)], device=device)  # (ranks that disagreed by one step sent their
            dist.broadcast(flag, 0)  # checkpoint barriers against the other rank's loss all-reduces: a hang)
            do_ck = bool(flag.item())
        if not args.smoke and do_ck:
            save_checkpoint(run_dir / f"step-{step}", model, opt, step, epoch, cfg, rank)
            for old in sorted((p for p in run_dir.glob("step-*") if p.name.split("-")[1].isdigit()), key=lambda p: int(p.name.split("-")[1]))[:-cfg.keep_checkpoints]:
                if rank == 0:
                    shutil.rmtree(old)
            t_ck = time.time()

    if not args.smoke and is_sharded(model):
        # gather parameter by parameter (collective), rank 0 keeps a bf16 CPU copy: 15 GB at 7.5B, on one rank only
        sd: dict[str, torch.Tensor] = {}
        reshard(model)
        for n, p in model.named_parameters():
            t = p.full_tensor() if type(p).__name__ == "DTensor" else p
            if rank == 0:
                sd[n.removeprefix("hf.")] = t.detach().to(torch.bfloat16).cpu()
            del t
        if rank == 0:
            final = run_dir / "final"
            final.mkdir(parents=True, exist_ok=True)
            if isinstance(model, EncDec):
                torch.save(sd, final / "encdec.pt")
                base = cfg.model.base if not EncDec.is_encdec_dir(cfg.model.base) else json.loads((Path(cfg.model.base) / "encdec.json").read_text())["base"]
                (final / "encdec.json").write_text(json.dumps({"base": base, "attn": attn, "lora_r": 0, "lora_alpha": 0}))
            elif isinstance(model, ShardedStitched):
                torch.save(sd, final / "stitched.pt")
                (final / "stitched.json").write_text(json.dumps({"enc_base": model.hf.enc_base, "dec_base": model.hf.dec_base, "attn": attn}))
            else:
                inner = model.hf if isinstance(model, ShardedHF) else model
                export_hf(inner, sd, final)
            tok.save_pretrained(final)
            (final / "train_config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=1))
            log(f"final bf16 model -> {final} ({(time.time() - t_start) / 3600:.2f} h)")
        del sd
    elif not args.smoke and rank == 0:
        raw = model.module if isinstance(model, DDP) else model
        final = run_dir / "final"
        if isinstance(raw, (EncDec, Hybrid, Stitched)):
            raw.save(final, tok)
        else:
            raw.save_pretrained(final, safe_serialization=True)
            tok.save_pretrained(final)
        (final / "train_config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=1))
        log(f"final bf16 model -> {final} ({(time.time() - t_start) / 3600:.2f} h)")
    if use_wandb:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
