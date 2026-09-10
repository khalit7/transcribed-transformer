---
name: train-run
description: Launch a training run on the dual RTX 5090 machine. Use whenever starting, resuming or debugging any training job (continued pretraining, task fine-tuning of any experiment E1 to E5, adaptation stages such as prefix-LM warm-up or encoder-decoder conversion). Covers the config, wandb, checkpointing and Blackwell-specific conventions.
---

# Launching a training run

## Before launching

**Check the licence track.** The config declares one track and the data loader asserts it. A run whose data spans both tracks is invalid and its results are unusable. This is the single most expensive mistake available here because it is invisible until someone asks whether a model can ship.

**Check the comparability rules** (README, "Design"). Every trained arm starts from the same Qwen3 checkpoint at the same size, on the same split and rendering, with loss on the answer only and zero prompt loss, at the same sequence length. An arm differs from E1 in exactly one thing: the attention mask (E2) or the architecture built from the same weights (E3 to E5). If a run needs a second difference to work, that is a finding to record, not a setting to bury.

**Check the config is complete.** Configs are YAML under `configs/`. Adding a model, dataset or experiment means adding a class and pointing a config at it. If launching this run requires editing the training loop, stop: the abstraction is wrong and should be fixed instead.

**Smoke first.** Run a handful of steps at the target sequence length before committing to a multi-day job. Confirm loss decreases, memory sits where expected, and a checkpoint writes and reloads. Confirm that gradient accumulation gives the same curve as the equivalent single step: completion-only loss with packing is exactly the configuration the per-micro-batch normalisation bug corrupts.

## Hardware

Dual RTX 5090, Blackwell sm_120, 32GB each, **no NVLink**, PCIe.

- CUDA 12.8+ and a PyTorch build with sm_120 kernels. FlashAttention 2 builds for sm_120 from source; **FlashAttention 3 is Hopper-only**, do not reach for it. E2's prefix-LM mask cannot be expressed in FlashAttention 2 and runs on FlexAttention; measure its throughput on this machine before relying on it, it is unpublished for sm_120.
- **Prefer DDP.** All-gather crosses PCIe without NVLink, so FSDP costs real throughput. Use FSDP only where memory forces it, and record the measured throughput cost when you do. Do not use context parallelism from Accelerate or TRL: it is SDPA-only and forbids packing.
- Long context relies on FA2 varlen plus unpadding plus activation checkpointing, and on chunked or fused cross-entropy: at 32k tokens the logits tensor, not the weights, is what runs out of memory first.
- bf16 mixed precision. This machine is also a desktop, so leave headroom rather than tuning to the last megabyte.

## wandb

Every run logs to wandb. There is no second logging path.

- Projects: `tt-baselines` (E0), `tt-decoder` (E1, E2), `tt-encdec` (E3, E4, E5), `tt-pretrain` (continued pretraining, adaptation stages, channel-model fitting).
- **Log the fully resolved config**, not a path to a YAML. The run must be reproducible from wandb alone.
- **Mandatory tags**: licence track (`track-p` / `track-nc`), experiment (`e0` to `e5`), base checkpoint (the HF id), size. Results tables are generated from the wandb API filtered on these tags. An untagged run is invisible to the results pipeline.
- **Always log**: loss, learning rate, grad norm, tokens seen, throughput (tokens/s and MFU), per-GPU memory. Throughput and MFU are how the project's compute estimates become measured facts rather than assumptions.
- Version the tokenizer, ASR channel parameters and dataset build (`splits.json` version) as wandb artifacts so any result traces to the data that produced it.
- `WANDB_API_KEY` from the environment, never committed.

## Long unattended runs

Continued pretraining and adaptation stages run for days on a shared desktop. Assume interruption.

- `WANDB_MODE=offline` with a later sync.
- Checkpoint on a wall-clock interval, not only on step count, and **verify resume actually works** by killing and resuming once early in the run.
- Resume must restore optimiser state, LR schedule position, data loader position and RNG state. A resume that silently restarts the data order corrupts a matched comparison without any visible error.
- Launch in the background so the session is not held open.

## After the run

Record it with the `log-experiment` skill: hypothesis, config, result, verdict, cross-linked to the wandb run id. wandb holds the metrics, `experiments/` holds the reasoning about them.

Never transcribe a metric from memory into prose. Read it from wandb.
