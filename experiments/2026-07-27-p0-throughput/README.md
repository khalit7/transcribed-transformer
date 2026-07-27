# P0: Blackwell environment validation and throughput measurement

**Date**: 2026-07-27
**Arm**: P0 (not a research arm)
**wandb**: [`tt-trunk`](https://wandb.ai/khalit7-/tt-trunk), runs tagged `arm-p0`

## Hypothesis

The project plan sizes the from-scratch arm (E) from an assumed **~100 TFLOP/s effective** across both RTX 5090s, giving ~8-9 days for a 400M model over 30B tokens. It also assumes a 32k-context encoder is trainable on 32GB cards at batch 1 with activation checkpointing.

Both are assumptions. If either is materially wrong, arm E drops to 150M and 20B tokens rather than compromising the matched-corpus design.

**Prediction**: ~100 TFLOP/s effective is achievable, and 32k fits in memory.
**Disconfirming result**: effective throughput below ~60 TFLOP/s, or 32k not fitting even with activation checkpointing.

## Setup

- 2x RTX 5090 (sm_120, 31.4 GiB usable each, no NVLink), driver 590.48.01
- torch 2.11.0+cu128, transformers 5.14.1, DDP, bf16, SDPA attention
- `answerdotai/ModernBERT-large`, 395.9M params, 28 layers, hidden 1024
- Real forward, backward, AdamW steps on random token ids at MLM masking rate 0.3. Throughput measurement, not training: loss is meaningless.
- Micro batch 1 per GPU, no gradient accumulation. 30-40 steps, first 5 excluded as warmup.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the checkpointed runs.

## Result

Environment checks pass: `sm_120` present in the torch build's arch list, both devices execute bf16 kernels, flash SDPA backend available. Flash SDPA at 32k sequence length uses 0.13 GiB, confirming attention memory is O(n).

Measured bf16 matmul ceiling (8192-cube, the friendliest possible workload): **234.4 and 229.7 TFLOP/s**, slightly above the 209.5 TFLOP/s vendor dense spec, consistent with boost clocks.

| seq_len | activation ckpt | tokens/s | effective TFLOP/s | MFU | peak mem |
|---:|:---:|---:|---:|---:|---:|
| 8,192 | no | 36,625 | 121.3 | 28.9% | 19.60 GiB |
| 8,192 | yes | 29,451 | 97.5 | 23.3% | 8.94 GiB |
| 16,384 | no | — | — | — | **OOM** |
| 16,384 | yes | 18,464 | 77.9 | 18.6% | 15.05 GiB |
| 32,768 | no | — | — | — | **OOM** |
| 32,768 | yes | 10,481 | 63.2 | 15.1% | 27.63 GiB |

MFU is against the 209.5 TFLOP/s per-card vendor dense bf16 spec, x2 for the world size.

## Verdict

**Confirmed, with two caveats worth carrying forward.**

The ~100 TFLOP/s assumption holds at 8k: 121.3 without checkpointing, 97.5 with. The plan's 8-9 day estimate for arm E at 400M over 30B tokens survives, and is conservative, since arm E trains mostly at 1k context in its first phase where throughput is higher.

**Caveat 1: 32k requires activation checkpointing, and there is very little headroom.** Peak memory is 27.63 GiB of 31.36 GiB at batch 1 with a 395M model. A 1B encoder at 32k will not fit this way. This confirms the plan's stated fallback: ModernBERT-large is the primary long-context trunk, and Ettin-1B is for 8k ablations unless something else changes. Anything at 32k will need gradient accumulation to reach a usable effective batch size, since micro batch is pinned at 1.

**Caveat 2: throughput degrades steeply with context.** 32k runs at 29% of the 8k token rate. Long-context training is roughly 3.5x more expensive per token than the 8k figure used for planning, which needs accounting for when the context-extension phase of arm A is scheduled.

## Correction made during this run

The first measurement reported **57.7% MFU**, which was wrong. `PEAK_BF16_TFLOPS` had been set to 104.8 TFLOP/s, the RTX 5090's FP32 CUDA-core rate, rather than its bf16 tensor rate of 209.5. That inflated every MFU figure by exactly 2x. Caught by measuring the achievable matmul ceiling directly and finding it 2.2x above the supposed peak, which is impossible.

Fixed in `hardware.py`, which now documents both of the wrong numbers that are easy to reach for here (the FP32 rate, and the vendor "AI TOPS" fp4-with-sparsity figure) and provides `python -m tt.training.hardware` to re-measure.

A second bug was fixed in the same pass: `memory_metrics()` queried every device from rank 0, which reports 0 for the other rank's device because each process only sees its own allocations. It read as though GPU1 were idle. It now reports the local device.

## Follow-ups

- Sweep gradient accumulation at 32k to find the effective batch size the context-extension phase can afford.
- Re-measure with FlashAttention 2 proper (the `kernels` extra) rather than torch SDPA, to see whether unpadding recovers throughput at long context. Real transcripts are variable-length, so unpadding should help considerably more than it does on the fixed-length random ids used here.
- Confirm the 1B-at-8k memory figure before committing to Ettin-1B for the ablation arm.
