"""P0 gate: measure real training throughput on this machine.

The project plan estimates multi-day pretraining runs from an assumed
~100 TFLOP/s effective across both cards. That assumption decides the size of
the from-scratch arm, so it gets measured before anything is planned around it.

This does real forward, backward and optimiser steps on random token ids. It is
a throughput measurement, not a training run: loss is meaningless and no
checkpoint is written.

Run with::

    uv run torchrun --nproc_per_node=2 -m tt.training.throughput \\
        --config training/configs/p0_throughput.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tt.training import hardware
from tt.training.config import RunConfig
from tt.training.tracking import Run, ThroughputMeter, memory_metrics


def _log(msg: str) -> None:
    if hardware.rank() == 0:
        print(msg, flush=True)


def _setup_distributed() -> torch.device:
    if hardware.is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(hardware.local_rank())
        return torch.device(f"cuda:{hardware.local_rank()}")
    return torch.device("cuda:0")


def run(cfg: RunConfig) -> int:
    problems = hardware.check_environment()
    if problems:
        for p in problems:
            _log(f"ENVIRONMENT PROBLEM: {p}")
        return 1

    torch.manual_seed(cfg.seed)
    device = _setup_distributed()
    ws = hardware.world_size()

    from transformers import AutoConfig, AutoModelForMaskedLM

    _log(f"loading {cfg.model.name_or_path}")
    model_cfg = AutoConfig.from_pretrained(cfg.model.name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
    ).to(device)

    if cfg.activation_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = getattr(model_cfg, "num_hidden_layers", 0)
    hidden = getattr(model_cfg, "hidden_size", 0)
    vocab = getattr(model_cfg, "vocab_size", 32000)

    flops_per_token = hardware.model_flops_per_token(
        n_params=n_params,
        n_layers=n_layers,
        hidden=hidden,
        seq_len=cfg.seq_len,
        global_every=cfg.model.global_every,
        local_window=cfg.model.local_window,
    )

    dev = hardware.device_info(hardware.local_rank())
    # Peak is per-device, so the MFU denominator scales with the world size.
    peak = None if dev.peak_bf16_tflops is None else dev.peak_bf16_tflops * ws

    _log(
        f"params {n_params / 1e6:.1f}M | layers {n_layers} | hidden {hidden} | "
        f"seq {cfg.seq_len} | world {ws}"
    )
    _log(f"flops/token {flops_per_token / 1e9:.2f}G | peak {peak} TFLOP/s ({dev.name} x{ws})")

    if ws > 1:
        model = DistributedDataParallel(model, device_ids=[hardware.local_rank()])

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        fused=True,
    )

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps * ws
    meter = ThroughputMeter(flops_per_token=flops_per_token, peak_tflops=peak, warmup=5)

    extra = {
        "measured/n_params": n_params,
        "measured/flops_per_token": flops_per_token,
        "measured/tokens_per_step": tokens_per_step,
    }

    with Run(cfg, extra_config=extra) as wandb_run:
        if wandb_run.url:
            _log(f"wandb: {wandb_run.url}")

        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum_steps):
                ids = torch.randint(0, vocab, (cfg.micro_batch_size, cfg.seq_len), device=device)
                labels = ids.clone()
                # Only masked positions carry loss, as in real MLM training.
                keep = torch.rand_like(labels, dtype=torch.float) < cfg.mlm_probability
                labels[~keep] = -100
                out = model(input_ids=ids, labels=labels)
                (out.loss / cfg.grad_accum_steps).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            meter.step(tokens_per_step)

            if (step + 1) % cfg.log_every == 0:
                metrics = {"train/loss": out.loss.item(), "train/step": step + 1}
                metrics.update(meter.measure())
                metrics.update(memory_metrics())
                wandb_run.log(metrics, step=step + 1)

        final = meter.measure()
        final.update(memory_metrics())
        wandb_run.summary(final)

        tps = final.get("perf/tokens_per_second", 0.0)
        util = final.get("perf/mfu")
        _log("")
        _log("=" * 62)
        _log(f"tokens/s        {tps:>12,.0f}")
        if util is not None:
            _log(f"MFU             {util:>12.1%}")
            _log(f"effective       {flops_per_token * tps / 1e12:>12.1f} TFLOP/s")
        for k, v in sorted(memory_metrics().items()):
            _log(f"{k:<15} {v:>12.2f} GiB")
        _log("=" * 62)

    if ws > 1:
        dist.destroy_process_group()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, help="Override seq_len, for sweeps.")
    ap.add_argument("--steps", type=int, help="Override step count.")
    ap.add_argument("--micro-batch-size", type=int, help="Override micro batch size.")
    ap.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Trade compute for memory. Required beyond 8k on 32GB cards.",
    )
    ap.add_argument("--offline", action="store_true", help="Force WANDB_MODE=offline.")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    overrides: dict[str, object] = {}
    suffix = ""
    if args.seq_len:
        overrides["seq_len"] = args.seq_len
        suffix += f"-seq{args.seq_len}"
    if args.micro_batch_size:
        overrides["micro_batch_size"] = args.micro_batch_size
        suffix += f"-mb{args.micro_batch_size}"
    if args.activation_checkpointing:
        overrides["activation_checkpointing"] = True
        suffix += "-ckpt"
    if suffix:
        overrides["name"] = f"{cfg.name}{suffix}"
    if args.steps:
        overrides["steps"] = args.steps
    if overrides:
        cfg = cfg.model_copy(update=overrides)
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
