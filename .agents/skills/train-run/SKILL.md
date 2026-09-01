---
name: train-run
description: Launch a training run on the dual RTX 5090 machine. Use whenever starting, resuming or debugging any training job (pretraining, continued pretraining, context extension, head fine-tuning). Covers the config, wandb, checkpointing and Blackwell-specific conventions.
---

# Launching a training run

## Before launching

**Check the licence track.** The config declares one track and the data loader asserts it. A run whose data spans both tracks is invalid and its results are unusable. This is the single most expensive mistake available here because it is invisible until someone asks whether a model can ship.

**Check the config is complete.** Configs are YAML in `training/configs/`. Adding a model, dataset or head means adding a class and pointing a config at it. If launching this run requires editing the training loop, stop: the abstraction is wrong and should be fixed instead.

**Smoke first.** Run a handful of steps at the target sequence length before committing to a multi-day job. Confirm loss decreases, memory sits where expected, and a checkpoint writes and reloads. Days of compute have been lost to an OOM at step 40,000 during a context-length ramp.

## Hardware

Dual RTX 5090, Blackwell sm_120, 32GB each, **no NVLink**, PCIe.

- CUDA 12.8+ and a PyTorch build with sm_120 kernels. FlashAttention 2 supports sm_120; **FlashAttention 3 is Hopper-only**, do not reach for it.
- **Prefer DDP.** All-gather crosses PCIe without NVLink, so FSDP costs real throughput. Use FSDP only where memory forces it, and record the measured throughput cost when you do.
- Long context relies on FA2 varlen plus unpadding plus activation checkpointing. Expect batch 1-2 per GPU at 32k with gradient accumulation to reach the target effective batch.
- bf16 mixed precision. This machine is also a desktop, so leave headroom rather than tuning to the last megabyte.

## wandb

Every run logs to wandb. There is no second logging path.

- Project by arm family: `tt-trunk`, `tt-heads`, `tt-scratch`, `tt-encdec`.
- **Log the fully resolved config**, not a path to a YAML. The run must be reproducible from wandb alone.
- **Mandatory tags**: licence track (`track-p` / `track-nc`), arm (`arm-a` … `arm-e`), base checkpoint. Results tables are generated from the wandb API filtered on these tags. An untagged run is invisible to the results pipeline.
- **Always log**: loss, learning rate, grad norm, tokens seen, throughput (tokens/s and MFU), per-GPU memory. Throughput and MFU are how the project's compute estimates become measured facts rather than assumptions.
- Version the tokenizer, ASR channel parameters and dataset build as wandb artifacts so any result traces to the data that produced it.
- `WANDB_API_KEY` from the environment, never committed.

## Long unattended runs

The from-scratch pretraining arms run for days on a shared desktop. Assume interruption.

- `WANDB_MODE=offline` with a later sync.
- Checkpoint on a wall-clock interval, not only on step count, and **verify resume actually works** by killing and resuming once early in the run.
- Resume must restore optimiser state, LR schedule position, data loader position and RNG state. A resume that silently restarts the data order corrupts a matched-corpus ablation without any visible error.
- Launch in the background so the session is not held open.

## After the run

Record it with the `log-experiment` skill: hypothesis, config, result, verdict, cross-linked to the wandb run id. wandb holds the metrics, `experiments/` holds the reasoning about them.

Never transcribe a metric from memory into prose. Read it from wandb.
