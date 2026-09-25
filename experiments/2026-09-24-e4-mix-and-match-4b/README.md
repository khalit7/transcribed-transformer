# E4-mix-and-match idea 2: 4B encoder + 270M decoder against 270M encoder + 4B decoder

Written before the run, 2026-09-24. The question: does the E4-mix-and-match result hold at 4B? The same run as `../2026-09-23-e4-mix-and-match/`, with 4B and 270M towers (4B-270M against 270M-4B), full-parameter, no LoRA, provided it fits the hardware.

## Hypothesis

The 1B grid (`experiments/2026-09-23-e4-mix-and-match/`) found that at equal parameter count the capacity is worth far more in the encoder: the 1B encoder with the 270M decoder recovered three quarters of the 1b-1b's margin over the 270m-270m, the 270M encoder with the 1B decoder a sixth. The same pair of unbalanced arms one size up tests whether that holds as the encoder grows past the decoder by a factor of fifteen, and where a 4B encoder with a tiny decoder lands against the two 4B-class decoders already on the benchmark.

## Prediction

- Arm C (4B encoder, 270M decoder, ~4.2B) above arm A (1B encoder, 270M decoder: 0.724 / 0.704 macro-F1, 0.646 / 0.636 unseen, 0.612 / 0.582 evidence) on evidence F1 and unseen-question macro-F1 by more than 0.02 on both variants.
- Arm D (270M encoder, 4B decoder, ~4.2B) within 0.03 of arm B (270M encoder, 1B decoder: 0.673 / 0.659; 0.530 / 0.544; 0.584 / 0.556) on every headline number: a bigger decoder behind a 270M encoder buys little more than the 1B one did.
- Reference points from above: the Gemma 3 4B decoder (0.778 / 0.745; 0.702 / 0.684; 0.635 / 0.604) and SmolLM3 3B (0.774 / 0.752; 0.698 / 0.688; 0.638 / 0.608). Arm C at or above them on evidence would say a 4B encoder with a 270M decoder matches a 4B decoder at a fraction of the decoding cost; the 4b-4b itself is not on this machine (D35).
- **Disconfirmed if** arm D gains as much over arm B as arm C gains over arm A (capacity helps equally on both sides once the towers are large), or if arm C is within 0.02 of arm A everywhere (encoder capacity saturates at 1B for this task).
- Step-400 gate as before: within about 0.05 of the balanced model with the same decoder, read against the 270m-270m (0.932) for arm C; for arm D no same-decoder model exists on this machine (the 4b-4b is not), so its gate is informational.

## Setup

- Exactly the 1B arms' recipe (E1's: lr 1e-5, cosine, one epoch, 32 sequences per step, AdamW 8-bit with fp32 masters, weight decay 0.1; stitch at 10×; regression-fitted stitch from 3M prompt tokens; 16k cap; SDPA), with one hardware difference: ~4.2B parameters need about 42 GB of weights, gradients, masters and optimizer state, so both arms shard under FSDP2 across the two cards (as the Gemma 3 4B and SmolLM3 3B controls did; ~35% throughput cost on PCIe) with 8k micro-batches. No LoRA, no freezing.
- Code: `ShardedStitched` (the FSDP2 root whose forward is the loss) and `shard_stitched` (every encoder and decoder block a unit) in `src/train/train.py`; the sharded export writes the stitched checkpoint; `model.sharding: fsdp` on the stitched arch. Stitches 2560 ↔ 640 (`checkpoints/stitch/4b-to-270m.pt`, `270m-to-4b.pt`).
- Smoke: three steps on the standard slice, then two steps on the 64 longest training records (the memory probe under FSDP at the 16k tail), per arm.
- Queue 20, arm C first (the arm the 1B grid says should matter), then arm D; each: fit, smoke, long smoke, train, generate, score.
- wandb: tt-encdec, run ids in `checkpoints/e4mm-*4b*/wandb_id` (the 270M encoder + 4B decoder arm keeps `e3-stitch-270m-enc-4b-dec` until it finishes).

## Result

- Stitch C (4b encoder → 270m decoder's space, 2560 → 640; 15:06, 415 s): held-out explained variance **0.621** (the 1b → 270m map: 0.644).
- Arm C smoke (FSDP, both cards): 4.15B trainable; three steps on the standard slice, loss 2.42, val 2.33, 5.8k tokens/s, 25.1 GiB; two steps on the 64 longest records (14–16k tokens), loss 3.09, 3.5k tokens/s, **28.9 GiB peak**. Training started 15:13.
- Arm C validation against the 270m-270m (its decoder's balanced model): 2.48 / 2.03 (step 0), 0.960 / 1.015 (200), **0.878 / 0.932 (400): the gate passes, 0.054 below rather than within 0.05 above**; the 1B-encoder arm was at 0.945 here. Then 0.830 / 0.880 (600), 0.792 / 0.842 (800), 0.761 / 0.814 (1000), 0.731 / 0.784 (1200), 0.698 / 0.764 (1400), 0.674 / 0.739 (1600), 0.654 / 0.721 (1800), 0.636 / 0.704 (2000), 0.620 / 0.689 (2200), 0.609 / 0.682 (2400), 0.598 / 0.674 (2600), 0.590 / 0.669 (2800), **0.589 / 0.667 (2858)**: 0.06–0.08 below the 270m-270m throughout, 0.06–0.07 below the 1B-encoder arm (which ended at 0.656), and within 0.01 of the 1B-decoder arm (0.581) from step 1400 on. Throughput 4.7–4.9k tokens/s over the pair under FSDP, 26.6 GiB; training 15:13 → about 06:20.
- **Arm C benchmark** (38,220 outputs; generation 06:34 → 09:49 with a 40k-token batch budget after the first attempt failed to load the FSDP export, see below; scored 09:49), clean / messy, against its neighbours:

| | macro-F1 | unseen-question macro-F1 | evidence F1 | evidence precision | format valid | trainable |
|---|---|---|---|---|---|---|
| 1B encoder + 270M decoder (arm A) | 0.724 / 0.704 | 0.646 / 0.636 | 0.612 / 0.582 | 0.721 / 0.688 | 0.988 / 0.981 | 1.27B |
| **4B encoder + 270M decoder (arm C)** | 0.774 / 0.744 | 0.694 / 0.662 | 0.630 / 0.601 | 0.745 / 0.715 | 0.992 / 0.987 | 4.15B |
| 1b-1b | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.725 / 0.694 | 0.990 / 0.985 | 1.70B |
| Gemma 3 4B decoder (E1) | 0.778 / 0.745 | 0.702 / 0.684 | 0.635 / 0.604 | 0.743 / 0.713 | 0.992 / 0.988 | 3.88B |

  Val split clean: macro-F1 0.802, evidence F1 0.739 (1b-1b 0.773 / 0.707; Gemma 3 4B 0.820 / 0.737). Against the prediction: unseen-question macro-F1 +0.048 / +0.026 over arm A (threshold 0.02, met); evidence F1 +0.018 / +0.019 (threshold 0.02, missed by a thousandth on each variant). Above the 1b-1b on every headline number (+0.030 / +0.022 macro-F1, +0.017 / +0.020 unseen, +0.011 / +0.011 evidence). Level with the Gemma 3 4B decoder on overall answers and evidence (within 0.004) and on apptek, taskmaster and aci_bench; below it on unseen questions messy (0.662 vs 0.684), SPoRC (0.700 vs 0.731) and the messy 8–16k band (0.716 vs 0.739), i.e. on the slices that need the most reading of long, real ASR text.
- Loader fault found by this arm: the sharded export carries the tied head once, under the embedding key; the strict load refused it and the first evaluation attempt scored empty outputs. `Stitched.load` now re-ties the head (test added); the empty evaluation was discarded and rerun.
- Stitch D (270m encoder → 4b decoder's space, 640 → 2560; 10:03, 394 s): **held-out explained variance 0.435** (the 270m → 1b map: 0.476; a 640-wide source spans at most a quarter of the target's dimensions). The first attempt ran out of GPU memory computing that figure (six-gigabyte float64 temporaries beside the encoders); the check is now chunked and the encoders leave the card first. Arm D runs under the shared recipe as the 1B-decoder arm did (D45); its step-400 gate is informational (no same-decoder model on this machine).
- Arm D smoke (FSDP): 4.15B trainable; three steps on the standard slice, loss 1.66, val 1.61, 8.7k tokens/s, 25.2 GiB; two steps on the 64 longest records, loss 2.24, 10.9k tokens/s, 27.0 GiB peak. Training started 10:06.
- Arm D validation: 1.77 (step 0; the 1B-decoder arm started at 2.05), 0.798 (200), **0.733 (400)**, against the 1b-1b's 0.831 / 0.776 and the 1B-decoder arm's 0.928 / 0.857 at the same steps: the informational gate is passed from far below, the loss following the decoder's size as it has at every arm. About 8.4k tokens/s, 26.3 GiB. Later readings and benchmark: TBD.

## Verdict

TBD.

## Follow-ups

TBD.
