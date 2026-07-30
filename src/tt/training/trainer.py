"""The training loop. One loop, for every arm.

Config over code: an arm is a YAML plus a registered objective. If launching a new
arm requires editing this file, the abstraction is wrong and that is the thing to
fix, not this loop.

## Resuming is a requirement, not a convenience

Runs here are measured in days on a machine that is also somebody's desktop. It
will be rebooted, it will run out of disk, a driver will update. So a checkpoint
saves everything needed to continue *identically*: model, optimiser, LR
scheduler, step count, and the RNG state of Python, NumPy and Torch on every
rank. Restoring weights alone silently changes the data order and the masking
pattern, which is the kind of thing that quietly makes two runs incomparable
without ever looking broken.

The failure this guards against hardest is a long run crashing and restarting
from step zero unnoticed. Resume is on by default and the loop says loudly which
step it started from.

## What is deliberately not here

No evaluation, no head-specific logic, no data loading. Those are separate so
that this file has one job and stays readable. The objective is a small pluggable
function, so MLM and CLM differ by a loss and a collator rather than a branch in
the loop.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tt.training import hardware
from tt.training.config import Objective, RunConfig
from tt.training.hwmon import HardwareMonitor
from tt.training.tracking import Run, ThroughputMeter, memory_metrics

Batch = dict[str, torch.Tensor]
"""Model kwargs for one micro-step: input ids and labels."""

BatchStream = Iterator[torch.Tensor]
"""Yields raw token ids. The objective turns them into a :data:`Batch`.

Deliberately raw: the same token stream feeds MLM and CLM, and which one is being
trained is a config choice, not a property of the data pipeline."""


def _log(msg: str) -> None:
    if hardware.rank() == 0:
        print(msg, flush=True)


# --- learning rate ----------------------------------------------------------


def lr_at(step: int, cfg: RunConfig) -> float:
    """Learning rate for a step, as a pure function.

    Pure on purpose: resuming recomputes it from the step number rather than
    restoring scheduler internals, so a resumed run cannot drift from an
    uninterrupted one.
    """
    peak = cfg.optim.lr
    warmup = cfg.schedule.warmup_steps
    if warmup and step < warmup:
        return peak * (step + 1) / warmup

    if cfg.schedule.kind == "constant":
        return peak

    progress = 0.0 if cfg.steps <= warmup else (step - warmup) / max(1, cfg.steps - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = peak * cfg.schedule.min_lr_ratio

    if cfg.schedule.kind == "linear":
        return floor + (peak - floor) * (1.0 - progress)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


# --- objectives -------------------------------------------------------------


def mlm_batch(ids: torch.Tensor, cfg: RunConfig) -> Batch:
    """Corrupt a fraction of positions and score the model on recovering them.

    The **input must actually be corrupted**, not just the labels restricted.
    Selecting positions and leaving the tokens in place gives the model the
    answer in its own input, so it learns to copy and the loss collapses towards
    zero while looking like fast convergence. That is a silent failure: a
    multi-day pretraining run would finish having learned nothing.

    Standard BERT corruption: of the selected positions, 80% become the mask
    token, 10% become a random token and 10% are left alone. The last two exist
    because the mask token never appears at fine-tuning time, so a model trained
    only on masks sees an input distribution it will never meet again.
    """
    if cfg.model.mask_token_id is None or cfg.model.vocab_size is None:
        raise ValueError(
            "MLM needs model.mask_token_id and model.vocab_size. Without a mask token "
            "the input is left uncorrupted and the model trains on an identity task "
            "that converges beautifully and teaches nothing."
        )

    labels = ids.clone()
    selected = torch.rand(ids.shape, device=ids.device) < cfg.mlm_probability
    labels[~selected] = -100

    inputs = ids.clone()
    draw = torch.rand(ids.shape, device=ids.device)
    inputs[selected & (draw < 0.8)] = cfg.model.mask_token_id
    randomise = selected & (draw >= 0.8) & (draw < 0.9)
    inputs[randomise] = torch.randint(
        0, cfg.model.vocab_size, ids.shape, device=ids.device, dtype=ids.dtype
    )[randomise]
    return {"input_ids": inputs, "labels": labels}


def clm_batch(ids: torch.Tensor, cfg: RunConfig) -> Batch:  # noqa: ARG001
    """Next-token prediction. The model shifts labels internally.

    ``cfg`` is unused but kept so every objective has one signature and the loop
    never needs to know which it is calling.
    """
    return {"input_ids": ids, "labels": ids.clone()}


OBJECTIVES: dict[Objective, Callable[[torch.Tensor, RunConfig], Batch]] = {
    Objective.MLM: mlm_batch,
    Objective.CLM: clm_batch,
}


# --- checkpointing ----------------------------------------------------------


@dataclass
class TrainState:
    """Everything needed to continue a run identically."""

    step: int = 0
    tokens_seen: int = 0


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


def save_checkpoint(
    directory: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    cfg: RunConfig,
) -> Path:
    """Write a resumable checkpoint. Rank 0 writes weights; every rank writes its RNG.

    Barriered at both ends. The ranks do very unequal work here — rank 0 writes
    hundreds of megabytes while the others write a few kilobytes of RNG state —
    so without a barrier the fast ranks run ahead into the next collective while
    rank 0 is still writing. That does not crash: it desynchronises, and the run
    finishes its work and then spins in NCCL forever instead of exiting, which
    looks exactly like a hang.
    """
    _barrier()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{state.step:08d}"
    path.mkdir(parents=True, exist_ok=True)

    if hardware.rank() == 0:
        inner = model.module if isinstance(model, DistributedDataParallel) else model
        # Written to a temporary name then renamed: a checkpoint half-written when
        # the machine dies must not look loadable.
        tmp = path / "model.pt.tmp"
        torch.save(inner.state_dict(), tmp)
        tmp.rename(path / "model.pt")
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
        (path / "state.json").write_text(
            json.dumps({"step": state.step, "tokens_seen": state.tokens_seen, "name": cfg.name})
        )
        (path / "config.json").write_text(json.dumps(cfg.resolved(), indent=1))

    torch.save(_rng_state(), path / f"rng-rank{hardware.rank()}.pt")
    _barrier()
    return path


def _barrier() -> None:
    """Synchronise ranks, or do nothing when running single-process."""
    if hardware.is_distributed() and dist.is_initialized():
        dist.barrier()


def latest_checkpoint(directory: Path) -> Path | None:
    """The most advanced complete checkpoint, or None."""
    if not directory.exists():
        return None
    candidates = sorted(
        p
        for p in directory.glob("step-*")
        if (p / "model.pt").exists() and (p / "state.json").exists()
    )
    return candidates[-1] if candidates else None


def load_checkpoint(
    path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None
) -> TrainState:
    """Restore a checkpoint in place and return where the run had reached."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    inner.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=False))

    if optimizer is not None and (path / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
        )

    rng_path = path / f"rng-rank{hardware.rank()}.pt"
    if rng_path.exists():
        _restore_rng(torch.load(rng_path, map_location="cpu", weights_only=False))

    raw = json.loads((path / "state.json").read_text())
    return TrainState(step=int(raw["step"]), tokens_seen=int(raw.get("tokens_seen", 0)))


def prune_checkpoints(directory: Path, keep_last: int) -> None:
    if keep_last <= 0 or hardware.rank() != 0:
        return
    kept = sorted(directory.glob("step-*"))
    for stale in kept[:-keep_last]:
        shutil.rmtree(stale, ignore_errors=True)


# --- the loop ---------------------------------------------------------------


def train(
    cfg: RunConfig,
    *,
    model: torch.nn.Module,
    batches: BatchStream,
    flops_per_token: float,
    peak_tflops: float | None = None,
) -> TrainState:
    """Run training to ``cfg.steps``, resuming if a checkpoint exists.

    ``batches`` yields token-id tensors already on the right device; the
    objective turns them into inputs and labels. Keeping data outside this
    function is what lets one loop serve every arm.
    """
    device = torch.device(f"cuda:{hardware.local_rank()}" if torch.cuda.is_available() else "cpu")
    ws = hardware.world_size()
    make_batch = OBJECTIVES[cfg.objective]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        fused=torch.cuda.is_available(),
    )

    state = TrainState()
    ckpt_dir = Path(cfg.checkpoint.dir) / cfg.name
    if cfg.checkpoint.resume:
        found = latest_checkpoint(ckpt_dir)
        if found is not None:
            state = load_checkpoint(found, model=model, optimizer=optimizer)
            _log(f"RESUMED from {found} at step {state.step} ({state.tokens_seen:,} tokens)")
        else:
            _log(f"no checkpoint in {ckpt_dir}, starting from step 0")

    if ws > 1 and not isinstance(model, DistributedDataParallel):
        model = DistributedDataParallel(model, device_ids=[hardware.local_rank()])
    model.train()

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps * ws
    meter = ThroughputMeter(flops_per_token=flops_per_token, peak_tflops=peak_tflops, warmup=5)

    monitor = (
        HardwareMonitor(
            run_name=f"{cfg.name} (train)",
            note=f"{cfg.model.name_or_path}, {cfg.objective.value}, seq {cfg.seq_len}, world {ws}",
        )
        if hardware.rank() == 0
        else None
    )

    extra = {
        "measured/tokens_per_step": tokens_per_step,
        "measured/flops_per_token": flops_per_token,
        "resumed_from_step": state.step,
    }

    from contextlib import nullcontext

    with Run(cfg, extra_config=extra) as wandb_run, monitor or nullcontext():
        if wandb_run.url:
            _log(f"wandb: {wandb_run.url}")

        started = time.perf_counter()
        while state.step < cfg.steps:
            lr = lr_at(state.step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                ids = next(batches).to(device, non_blocking=True)
                out = model(**make_batch(ids, cfg))
                (out.loss / cfg.grad_accum_steps).backward()
                total_loss += float(out.loss.detach())

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

            state.step += 1
            state.tokens_seen += tokens_per_step
            meter.step(tokens_per_step)

            if state.step % cfg.log_every == 0:
                metrics = {
                    "train/loss": total_loss / cfg.grad_accum_steps,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_seen": state.tokens_seen,
                    "train/step": state.step,
                }
                metrics.update(meter.measure())
                metrics.update(memory_metrics())
                wandb_run.log(metrics, step=state.step)

            periodic = cfg.checkpoint.every_steps
            if periodic and state.step % periodic == 0 and state.step < cfg.steps:
                save_checkpoint(ckpt_dir, model=model, optimizer=optimizer, state=state, cfg=cfg)
                prune_checkpoints(ckpt_dir, cfg.checkpoint.keep_last)
                _log(f"checkpoint at step {state.step}")

        save_checkpoint(ckpt_dir, model=model, optimizer=optimizer, state=state, cfg=cfg)
        final = meter.measure()
        final.update(memory_metrics())
        final["train/tokens_seen"] = state.tokens_seen
        final["train/wall_seconds"] = time.perf_counter() - started
        wandb_run.summary(final)
        _log(f"done at step {state.step}, {state.tokens_seen:,} tokens")

    if monitor is not None:
        monitor.append_report()
    return state


def random_batches(cfg: RunConfig, vocab_size: int, device: torch.device) -> BatchStream:
    """Random token ids, for smoke tests only.

    Loss is meaningless on these. Present so the loop can be exercised end to end
    before the real data pipeline exists, and named so nobody mistakes a run on
    them for training.
    """
    while True:
        yield torch.randint(0, vocab_size, (cfg.micro_batch_size, cfg.seq_len), device=device)


def cleanup() -> None:
    """Leave the process group, after making sure every rank has arrived.

    Tearing down while another rank is still inside a collective leaves that rank
    spinning on a peer that no longer exists.
    """
    if hardware.is_distributed() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def setup_distributed() -> torch.device:
    """Join the process group, or return the device if already joined.

    Idempotent: calling it twice in one process is a mistake, but torch answers
    that mistake by aborting every rank with "initialize the default process
    group twice", which surfaces as a bare exit code 1 from a worker and reads
    like a data or model failure. Returning the device instead keeps the failure
    where it belongs.
    """
    if not hardware.is_distributed():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(hardware.local_rank())
    return torch.device(f"cuda:{hardware.local_rank()}")


def seed_everything(seed: int) -> None:
    """Seed per rank, so ranks do not draw identical data."""
    effective = seed + hardware.rank()
    random.seed(effective)
    torch.manual_seed(effective)
    os.environ.setdefault("PYTHONHASHSEED", str(effective))
    try:
        import numpy as np

        np.random.seed(effective)
    except ImportError:
        pass
