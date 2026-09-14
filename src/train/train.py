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
    labelled_doc_ids,
    mntp_mask,
    plan_epoch,
)
from src.train.masks import FLEX_KERNEL_OPTIONS, prefix_lm_mask


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
    if cfg.model.prefix_lm and attn == "flash_attention_2":
        attn = "flex_attention"  # FA2 cannot express the prefix-LM mask
    if attn == "flash_attention_2":
        try:
            import flash_attn  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            log("flash_attn not importable; falling back to sdpa")
            attn = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.base, dtype=torch.bfloat16, attn_implementation=attn)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return tok, model.to(device), attn  # type: ignore[arg-type]


def target_loss(model, batch: dict, device: torch.device, prefix_lm: str | None = None) -> tuple[torch.Tensor, int]:
    """Sum of cross-entropy over target tokens. Hidden states at the positions that predict a target
    token go through lm_head; nothing else does. `prefix_lm` names the attention implementation when
    the prompt is attended bidirectionally (E2); None keeps the causal mask (E1)."""
    ids = batch["input_ids"].to(device, non_blocking=True)
    attn = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    base = model.module.model if isinstance(model, DDP) else model.model
    head = model.module.lm_head if isinstance(model, DDP) else model.lm_head
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


class MasterOptimizer:
    """fp32 master weights + the configured optimizer over them. step(): move the bf16 gradients to the
    masters (as fp32), clip, update, copy the masters back into the bf16 model."""

    def __init__(self, model, cfg: TrainConfig):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.masters = [p.detach().float().clone() for p in self.params]
        decay: list[torch.Tensor] = []
        no_decay: list[torch.Tensor] = []
        for n, m in zip(self.names, self.masters):
            (no_decay if m.ndim < 2 or "norm" in n or "bias" in n else decay).append(m)
        groups = [{"params": decay, "weight_decay": cfg.optim.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
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


def save_checkpoint(path: Path, model, opt, step: int, epoch: int, cfg: TrainConfig, rank: int) -> None:
    if rank == 0:
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        raw = model.module if isinstance(model, DDP) else model
        torch.save(raw.state_dict(), tmp / "model.pt")
        torch.save(opt.state_dict(), tmp / "optim.pt")
        (tmp / "state.json").write_text(json.dumps({"step": step, "epoch": epoch, "config": cfg.model_dump(mode="json")}))
        if path.exists():
            shutil.rmtree(path)
        tmp.rename(path)
        log(f"checkpoint -> {path}")
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def latest_checkpoint(run_dir: Path) -> Path | None:
    cks = sorted(run_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


@torch.no_grad()
def val_loss(model, examples: list[Example], idx: list[int], world: int, rank: int, device, pad_id: int,
             micro_tokens: int, prefix_lm: str | None, prepare=None) -> float:
    pad_kw = {"pad_to": 128, "min_len": 256} if prefix_lm else {}
    model.eval()
    total = torch.zeros(2, device=device)
    mine = idx[rank::world]
    plan = plan_epoch([examples[i] for i in mine], max(1, len(mine)), micro_tokens, 1, 0, 0) if mine else []
    for step in plan:
        for micro in step.micro[0]:
            batch = collate([examples[i] for i in mine], micro, pad_id, **pad_kw)
            if prepare is not None:
                batch = prepare(batch, hash(tuple(micro)) & 0xFFFF)
            loss, n = target_loss(model, batch, device, prefix_lm)
            total += torch.tensor([loss.item(), n], device=device)
    if dist.is_initialized():
        dist.all_reduce(total)
    model.train()
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

    # loss normaliser per step: the group's target tokens, or their expected masked share under MNTP
    # (the realised count differs by a few per cent; using the expectation avoids a collective)
    supervised_share = mask_prob if mntp else 1.0
    n_params = sum(p.numel() for p in model.parameters())
    n_layers, d_model = model.config.num_hidden_layers, model.config.hidden_size
    log(f"{cfg.model.base}: {n_params / 1e9:.2f}B params, attn={attn}, world={world}")

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

    if world > 1:
        model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
    opt = MasterOptimizer(model, cfg)
    step0 = 0
    if args.resume and (ck := latest_checkpoint(run_dir)):
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(torch.load(ck / "model.pt", map_location=device))
        opt.load_state_dict(torch.load(ck / "optim.pt", map_location=device))
        step0 = json.loads((ck / "state.json").read_text())["step"]
        log(f"resumed from {ck} at step {step0}")

    use_wandb = rank == 0 and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.name, tags=cfg.wandb_tags(), dir=str(run_dir),
                   config={**cfg.model_dump(mode="json"), "attn_used": attn, "n_params": n_params,
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
    while step < total_steps:
        if step // steps_per_epoch != epoch:
            epoch = step // steps_per_epoch
            plan = plan_epoch(train, cfg.batch_sequences, cfg.micro_tokens, world, cfg.seed, epoch)
        st: Step = plan[step % steps_per_epoch]
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        micros = st.micro[rank]
        loss_sum = torch.zeros((), device=device)
        for k, micro in enumerate(micros):
            batch = prepare(collate(train, micro, pad_id, **pad_kw), cfg.seed * 1_000_003 + step * 64 + rank * 8 + k)
            sync = contextlib.nullcontext() if (k == len(micros) - 1 or world == 1) else model.no_sync()
            with sync:
                loss, _ = target_loss(model, batch, device, prefix_lm)
                (loss * world / (st.target_tokens * supervised_share)).backward()
            loss_sum += loss.detach()
        grad_norm = opt.step()
        step += 1
        if dist.is_initialized():
            dist.all_reduce(loss_sum)
        # throughput counters come from the plan, which every rank holds in full: no collective needed
        # (E2 showed the float64 all-reduce of these counters returning garbage in the window after a checkpoint)
        step_lengths = [len(train[i].input_ids) for rank_micro in st.micro for micro in rank_micro for i in micro]
        window_tokens += sum(step_lengths)
        window_flops += flops_of(n_params, n_layers, d_model, step_lengths)
        tokens_seen += sum(step_lengths)
        if step % cfg.log_every == 0 or step == total_steps:
            dt = time.time() - window_t0
            mfu = window_flops / dt / (cfg.peak_tflops * 1e12 * world)
            mem = torch.cuda.max_memory_allocated(device) / 2**30
            rec = {"step": step, "loss": loss_sum.item() / (st.target_tokens * supervised_share), "lr": lr, "grad_norm": grad_norm,
                   "tokens_seen": tokens_seen, "tokens_per_s": window_tokens / dt, "mfu": mfu,
                   f"gpu{rank}_mem_gib": mem, "epoch": step / steps_per_epoch}
            log(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in rec.items()}))
            if use_wandb:
                wandb.log(rec, step=step)
            window_flops, window_tokens, window_t0 = 0.0, 0, time.time()
        if (step % cfg.val_every == 0 or step == total_steps) and vidx:
            vl = val_loss(model, val, vidx, world, rank, device, pad_id, cfg.micro_tokens, prefix_lm, prepare)
            log(f"step {step}: val_loss {vl:.4f}")
            if use_wandb:
                wandb.log({"val_loss": vl}, step=step)
        if not args.smoke and (time.time() - t_ck > cfg.checkpoint_minutes * 60 or step == total_steps):
            save_checkpoint(run_dir / f"step-{step}", model, opt, step, epoch, cfg, rank)
            for old in sorted(run_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))[:-2]:
                if rank == 0:
                    shutil.rmtree(old)
            t_ck = time.time()

    if not args.smoke and rank == 0:
        raw = model.module if isinstance(model, DDP) else model
        final = run_dir / "final"
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
