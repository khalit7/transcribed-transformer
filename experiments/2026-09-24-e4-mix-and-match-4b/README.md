# E4-mix-and-match idea 2: 4B encoder + 270M decoder against 270M encoder + 4B decoder

Written before the run, 2026-09-24. The question: does the E4-mix-and-match result hold at 4B? The same run as `../2026-09-23-e4-mix-and-match/`, with 4B and 270M towers (4B-270M against 270M-4B), full-parameter, no LoRA, provided it fits the hardware.

## Hypothesis

The 1B grid (`experiments/2026-09-23-e4-mix-and-match/`) found that at equal parameter count the capacity is worth far more in the encoder: the 1B encoder with the 270M decoder recovered three quarters of the 1b-1b's margin over the 270m-270m, the 270M encoder with the 1B decoder a sixth. The same pair of unbalanced arms one size up tests whether that holds as the encoder grows past the decoder by a factor of fifteen, and where a 4B encoder with a tiny decoder lands against the two 4B-class decoders already on the benchmark.

## Prediction

- The 4B encoder + 270M decoder (~4.2B) above the 1B encoder + 270M decoder (0.724 / 0.704 macro-F1, 0.646 / 0.636 unseen, 0.612 / 0.582 evidence) on evidence F1 and unseen-question macro-F1 by more than 0.02 on both variants.
- The 270M encoder + 4B decoder (~4.2B) within 0.03 of the 270M encoder + 1B decoder (0.673 / 0.659; 0.530 / 0.544; 0.584 / 0.556) on every headline number: a bigger decoder behind a 270M encoder buys little more than the 1B one did.
- Reference points from above: the Gemma 3 4B decoder (0.778 / 0.745; 0.702 / 0.684; 0.635 / 0.604) and SmolLM3 3B (0.774 / 0.752; 0.698 / 0.688; 0.638 / 0.608). The 4B-encoder arm at or above them on evidence would say a 4B encoder with a 270M decoder matches a 4B decoder at a fraction of the decoding cost; the 4b-4b itself is not on this machine (D35).
- **Disconfirmed if** the 4B-decoder arm gains as much over the 1B-decoder arm as the 4B-encoder arm gains over the 1B-encoder arm (capacity helps equally on both sides once the towers are large), or if the 4B-encoder arm is within 0.02 of the 1B-encoder arm everywhere (encoder capacity saturates at 1B for this task).
- Step-400 gate as before: within about 0.05 of the balanced model with the same decoder, read against the 270m-270m (0.932) for the 4B-encoder arm; for the 4B-decoder arm no same-decoder model exists on this machine (the 4b-4b is not), so its gate is informational.

## Setup

- Exactly the 1B arms' recipe (E1's: lr 1e-5, cosine, one epoch, 32 sequences per step, AdamW 8-bit with fp32 masters, weight decay 0.1; stitch at 10×; regression-fitted stitch from 3M prompt tokens; 16k cap; SDPA), with one hardware difference: ~4.2B parameters need about 42 GB of weights, gradients, masters and optimizer state, so both arms shard under FSDP2 across the two cards (as the Gemma 3 4B and SmolLM3 3B controls did; ~35% throughput cost on PCIe) with 8k micro-batches. No LoRA, no freezing.
- Code: `ShardedStitched` (the FSDP2 root whose forward is the loss) and `shard_stitched` (every encoder and decoder block a unit) in `src/train/train.py`; the sharded export writes the stitched checkpoint; `model.sharding: fsdp` on the stitched arch. Stitches 2560 ↔ 640 (`checkpoints/stitch/4b-to-270m.pt`, `270m-to-4b.pt`).
- Smoke: three steps on the standard slice, then two steps on the 64 longest training records (the memory probe under FSDP at the 16k tail), per arm.
- Queue 20, the 4B encoder + 270M decoder first (the arm the 1B grid says should matter), then the 270M encoder + 4B decoder; each: fit, smoke, long smoke, train, generate, score.
- wandb: tt-encdec, run ids in `checkpoints/e4mm-*4b*/wandb_id` (`e4mm-4b-enc-270m-dec`, `e4mm-270m-enc-4b-dec`; the latter ran under the name `e3-stitch-270m-enc-4b-dec` and was renamed on completion).

## Result

- The 4b → 270m stitch (the 4b encoder into the 270m decoder's space, 2560 → 640; 15:06, 415 s): held-out explained variance **0.621** (the 1b → 270m map: 0.644).
- The 4B-encoder arm's smoke (FSDP, both cards): 4.15B trainable; three steps on the standard slice, loss 2.42, val 2.33, 5.8k tokens/s, 25.1 GiB; two steps on the 64 longest records (14–16k tokens), loss 3.09, 3.5k tokens/s, **28.9 GiB peak**. Training started 15:13.
- The 4B-encoder arm's validation against the 270m-270m (its decoder's balanced model): 2.48 / 2.03 (step 0), 0.960 / 1.015 (200), **0.878 / 0.932 (400): the gate passes, 0.054 below rather than within 0.05 above**; the 1B-encoder arm was at 0.945 here. Then 0.830 / 0.880 (600), 0.792 / 0.842 (800), 0.761 / 0.814 (1000), 0.731 / 0.784 (1200), 0.698 / 0.764 (1400), 0.674 / 0.739 (1600), 0.654 / 0.721 (1800), 0.636 / 0.704 (2000), 0.620 / 0.689 (2200), 0.609 / 0.682 (2400), 0.598 / 0.674 (2600), 0.590 / 0.669 (2800), **0.589 / 0.667 (2858)**: 0.06–0.08 below the 270m-270m throughout, 0.06–0.07 below the 1B-encoder arm (which ended at 0.656), and within 0.01 of the 1B-decoder arm (0.581) from step 1400 on. Throughput 4.7–4.9k tokens/s over the pair under FSDP, 26.6 GiB; training 15:13 → about 06:20.
- **The 4B-encoder arm's benchmark** (38,220 outputs; generation 06:34 → 09:49 with a 40k-token batch budget after the first attempt failed to load the FSDP export, see below; scored 09:49), clean / messy, against its neighbours:

| | macro-F1 | unseen-question macro-F1 | evidence F1 | evidence precision | format valid | trainable |
|---|---|---|---|---|---|---|
| 1B encoder + 270M decoder | 0.724 / 0.704 | 0.646 / 0.636 | 0.612 / 0.582 | 0.721 / 0.688 | 0.988 / 0.981 | 1.27B |
| **4B encoder + 270M decoder** | 0.774 / 0.744 | 0.694 / 0.662 | 0.630 / 0.601 | 0.745 / 0.715 | 0.992 / 0.987 | 4.15B |
| 1b-1b | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.725 / 0.694 | 0.990 / 0.985 | 1.70B |
| Gemma 3 4B decoder (E1) | 0.778 / 0.745 | 0.702 / 0.684 | 0.635 / 0.604 | 0.743 / 0.713 | 0.992 / 0.988 | 3.88B |

  Val split clean: macro-F1 0.802, evidence F1 0.739 (1b-1b 0.773 / 0.707; Gemma 3 4B 0.820 / 0.737). Against the prediction: unseen-question macro-F1 +0.048 / +0.026 over the 1B-encoder arm (threshold 0.02, met); evidence F1 +0.018 / +0.019 (threshold 0.02, missed by a thousandth on each variant). Above the 1b-1b on every headline number (+0.030 / +0.022 macro-F1, +0.017 / +0.020 unseen, +0.011 / +0.011 evidence). Level with the Gemma 3 4B decoder on overall answers and evidence (within 0.004) and on apptek, taskmaster and aci_bench; below it on unseen questions messy (0.662 vs 0.684), SPoRC (0.700 vs 0.731) and the messy 8–16k band (0.716 vs 0.739), i.e. on the slices that need the most reading of long, real ASR text.
- Loader fault found by this arm: the sharded export carries the tied head once, under the embedding key; the strict load refused it and the first evaluation attempt scored empty outputs. `Stitched.load` now re-ties the head (test added); the empty evaluation was discarded and rerun.
- The 270m → 4b stitch (the 270m encoder into the 4b decoder's space, 640 → 2560; 10:03, 394 s): **held-out explained variance 0.435** (the 270m → 1b map: 0.476; a 640-wide source spans at most a quarter of the target's dimensions). The first attempt ran out of GPU memory computing that figure (six-gigabyte float64 temporaries beside the encoders); the check is now chunked and the encoders leave the card first. The 4B-decoder arm runs under the shared recipe as the 1B-decoder arm did (D45); its step-400 gate is informational (no same-decoder model on this machine).
- The 4B-decoder arm's smoke (FSDP): 4.15B trainable; three steps on the standard slice, loss 1.66, val 1.61, 8.7k tokens/s, 25.2 GiB; two steps on the 64 longest records, loss 2.24, 10.9k tokens/s, 27.0 GiB peak. Training started 10:06.
- The 4B-decoder arm's validation: 1.77 (step 0; the 1B-decoder arm started at 2.05), 0.798 (200), **0.733 (400)**, against the 1b-1b's 0.831 / 0.776 and the 1B-decoder arm's 0.928 / 0.857 at the same steps: the informational gate is passed from far below, the loss following the decoder's size as it has at every arm. Then 0.697 (600), 0.657 (800), 0.631 (1000), 0.610 (1200), 0.585 (1400), 0.566 (1600), 0.545 (1800), 0.528 (2000), 0.510 (2200), 0.499 (2400), 0.489 (2600), 0.481 (2800), **0.480 (2858)**: the lowest end loss of every stitched arm, below the 1b-1b's 0.530 and above the Gemma 3 4B decoder's 0.428 at the same steps. Training 8.99 h (10:06 → 19:05 on 2026-09-25) at about 8.1k tokens/s over the pair under FSDP, 26.6 GiB peak.
- **The 4B-decoder arm's generation** ran 19:06 → 02:31 (2026-09-26) on both cards, longer than the 4B-encoder arm's 3.25 h for two reasons I should separate: the 4B decoder does the token-by-token work, so decoding is the slow part here, and the run lost time to my own handling. I launched it with the 160k-token batch budget, which the 4B decoder's cache overflows on the longest prompts, so the validation split and the first benchmark shard each ran out of memory near their ends (10,144 of 10,476 and 16,560 of 17,899 done); the remaining shards and the two top-ups ran at the 40k budget, resuming from what was on disk, with about half an hour of one card idle in between. Scored 02:31.
- **The 4B-decoder arm's benchmark** (38,220 outputs), clean / messy, beside the arms it is read against:

| model | macro-F1 | unseen-question macro-F1 | evidence F1 | evidence precision | format valid | params |
|---|---|---|---|---|---|---|
| 270M encoder + 1B decoder | 0.673 / 0.659 | 0.530 / 0.544 | 0.584 / 0.556 | 0.679 / 0.649 | 0.988 / 0.983 | 1.27B |
| **270M encoder + 4B decoder** | 0.733 / 0.705 | 0.585 / 0.573 | 0.601 / 0.574 | 0.710 / 0.681 | 0.988 / 0.986 | 4.15B |
| 1b-1b | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.725 / 0.694 | 0.990 / 0.985 | 2.1B |
| 4B encoder + 270M decoder | 0.774 / 0.744 | 0.694 / 0.662 | 0.630 / 0.601 | 0.745 / 0.715 | 0.992 / 0.987 | 4.15B |
| Gemma 3 4B decoder | 0.778 / 0.745 | 0.702 / 0.684 | 0.635 / 0.604 | 0.687 / 0.660 | 0.992 / 0.988 | 3.9B |

  Val split clean: macro-F1 0.785, evidence F1 0.708 (the 4B-encoder arm 0.802 / 0.739; 1b-1b 0.773 / 0.707). Against the prediction: not within 0.03 of the 1B-decoder arm on macro-F1 (+0.060 / +0.046) or clean unseen questions (+0.055), within it on evidence (+0.017 / +0.018) and messy unseen (+0.029). The disconfirmation fired: the 4B decoder gains over the 1B decoder (+0.060 / +0.046 macro-F1, +0.055 / +0.029 unseen, +0.017 / +0.018 evidence) as much as the 4B encoder gained over the 1B encoder (+0.050 / +0.040, +0.048 / +0.026, +0.018 / +0.019). At the same parameter count the arms still sit far apart in absolute terms: the encoder-heavy one is +0.041 / +0.039 on macro-F1, +0.109 / +0.089 on unseen questions and +0.029 / +0.027 on evidence above the decoder-heavy one, and the decoder-heavy one is below the 1b-1b, half its size, on unseen questions (0.585 vs 0.677) and evidence. Slices where it trails most: unseen questions clean 0.585 (4B decoder 0.702), SPoRC 0.671 (0.731), messy 8–16k 0.694 (0.739). Val loss ranked it first among the stitched arms and above the 1b-1b (0.480 vs 0.530): the sixth mis-ranking on this benchmark.

## Verdict

Half confirmed, half disconfirmed, and the disconfirmed half changes the reading. The 4B encoder with the 270M decoder matches the Gemma 3 4B decoder on answers and evidence overall at a fraction of the decoding cost (its prediction met on unseen questions, missed by 0.001 on evidence). The mirror arm was predicted to gain little from a 4B decoder and gained as much from its 1B → 4B step as the encoder arm did from its own, so the marginal claim from the 1B grid, that capacity buys nothing on the decoder side, does not hold at 4B. What holds is the absolute claim: at equal parameters the encoder-heavy arm is 0.04 ahead on answers and 0.09 to 0.11 ahead on unseen questions, and a 270M encoder caps what any decoder behind it can do on questions it has not seen (0.585 with a 4B decoder, against 0.646 for a 1B encoder behind a 270M decoder). The two sides differ in shape: the encoder pays from the first step and carries zero-shot transfer; the decoder side has a threshold, nothing from 270M → 1B and 0.06 from 1B → 4B, and what it buys is mostly on seen questions. Cost runs against the decoder-heavy arm at inference too: its generation took about twice the wall time of the encoder-heavy arm's, since the large tower is the one that runs once per output token. Caveats as before: one seed per cell; the stitches start from regression fits of 0.435 (this arm) and 0.621; the FSDP recipe is E1's with 8k micro-batches.

## Follow-ups

- The 4B encoder + 1B decoder (`configs/e4/mm-4b-enc-1b-dec-80gb.yaml`) on rented hardware, the exact recipe without frozen embeddings: whether the encoder's gains stack on a decoder that has crossed its threshold.
- A seed replicate of the 4B encoder + 270M decoder before the "level with the 4B decoder" claim goes anywhere.
- The E5 soft-token arm (`experiments/2026-09-25-e5-soft-tokens-t5gemma2-270m-enc-gemma3-1b/`) asks the same-decoder question the grid cannot.
