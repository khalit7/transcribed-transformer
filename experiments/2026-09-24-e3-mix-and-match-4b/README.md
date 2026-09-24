# E3-mix-and-match idea 2: 4B encoder + 270M decoder against 270M encoder + 4B decoder

Written before the run, 2026-09-24. The question: does the E3 mix-and-match result hold at 4B? The same run as `../2026-09-23-e3-mix-and-match/`, with 4B and 270M towers (4B-270M against 270M-4B), full-parameter, no LoRA, provided it fits the hardware.

## Hypothesis

The 1B grid (`experiments/2026-09-23-e3-mix-and-match/`) found that at equal parameter count the capacity is worth far more in the encoder: the 1B encoder with the 270M decoder recovered three quarters of the 1b-1b's margin over the 270m-270m, the 270M encoder with the 1B decoder a sixth. The same pair of unbalanced arms one size up tests whether that holds as the encoder grows past the decoder by a factor of fifteen, and where a 4B encoder with a tiny decoder lands against the two 4B-class decoders already on the benchmark.

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
- wandb: tt-encdec, run ids in `checkpoints/e3-stitch-*4b*/wandb_id`.

## Result

- Stitch C (4b encoder → 270m decoder's space, 2560 → 640; 15:06, 415 s): held-out explained variance **0.621** (the 1b → 270m map: 0.644).
- Arm C smoke (FSDP, both cards): 4.15B trainable; three steps on the standard slice, loss 2.42, val 2.33, 5.8k tokens/s, 25.1 GiB; two steps on the 64 longest records (14–16k tokens), loss 3.09, 3.5k tokens/s, **28.9 GiB peak**. Training started 15:13.
- Arm C validation against the 270m-270m: TBD.
- Arm C benchmark: TBD.
- Stitch D and arm D: TBD.

## Verdict

TBD.

## Follow-ups

TBD.
