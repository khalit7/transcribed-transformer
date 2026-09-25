# E4-mix-and-match: unbalanced T5Gemma 2 pairs, 1B encoder + 270M decoder and 270M encoder + 1B decoder

Written before the run, 2026-09-23. The question: at a fixed budget, is it better to spend parameters on the encoder or on the decoder? Compare 1B-270M against 270M-1B, joined by the fitted stitch described below.

## Hypothesis

The balanced pairs (270m-270m, 1b-1b) showed the encoder-decoder beating its own decoder at both sizes and the 1b-1b's extra capacity buying zero-shot question transfer. Where that capacity is best spent is open. With the two existing balanced models, the two unbalanced arms complete a 2 × 2 grid, encoder size × decoder size, from one family under one recipe; the two unbalanced arms are parameter-matched to each other (about 1.3B).

## Prediction

- Encoder size matters more for evidence (locating the supporting lines is the encoder's reading of the transcript); decoder size matters more for unseen-question transfer and answer quality (the decoder carries the question and the format).
- Concretely: arm A (1B encoder, 270M decoder) above the 270m-270m on evidence F1 by more than 0.02 on both variants; arm B (270M encoder, 1B decoder) above the 270m-270m on clean unseen-question macro-F1 by more than 0.02. Each arm below the 1b-1b on the other axis.
- **Disconfirmed if** both arms sit within 0.02 of the 270m-270m on every headline number (the smaller tower is the bottleneck whichever side it is on), or if one arm matches the 1b-1b on everything (that side's size is all that matters).
- Reading rule at step 400 (validation): an arm within about 0.05 of the balanced model with the same decoder continues; further behind, the seq2seq adaptation stage goes first and the arm is re-run.

## Setup

- Model (`src/train/stitched.py`, `model.arch: stitched`): the text encoder of one T5Gemma 2 checkpoint, the decoder and head of another, joined by an affine stitch on the encoder's final states (1152 ↔ 640). The decoder's own key/value projections read the stitched states, as they read its native encoder's; nothing inside the decoder changes.
- Stitch initialisation (`src/train/fit_stitch.py`): both encoders over the same ~3M tokens of training prompts, final states paired token by token, ridge regression from the donor encoder's states onto the decoder-side encoder's states, held-out explained variance reported (`checkpoints/stitch/*.json`). Imitation is the starting point only; the stitch trains at 10× the base rate with everything else.
- Training: E1's recipe as the balanced pairs (lr 1e-5, cosine, one epoch, 32 sequences per step, AdamW 8-bit with fp32 masters), every parameter, DDP, SDPA (no flash path for T5Gemma 2), 16k cap and 16k micro-batch.
- Generation: `generate.py --backend stitched` (encoder pass through the stitch, then the T5Gemma 2 decoder from its start token with a growing cache), benchmark in halves and quarters over both GPUs; scoring as every run.
- Tests: stitched forward and shapes, padded encoder keys never attended, gradients reach stitch, encoder and decoder; the fitting routine recovers a known affine map and the explained variance reads it.
- Queue 19, after queue 17 (the E6 decoder-only + encoder arm): arm A then arm B, each: fit, smoke, train, generate, score. The E6 frozen-decoder arm (queue 18) postponed for this.
- wandb: tt-encdec, run ids in `checkpoints/e4mm-*/wandb_id`.

## Result

- Stitch A (1b encoder → 270m decoder's space; `checkpoints/stitch/1b-to-270m.json`, 2026-09-24 02:59): 3.01M fit tokens, 300k held-out, ridge 1e-2 relative; **held-out explained variance 0.644**. Above the go-ahead threshold (about a half), moderate rather than high: a third of the 270m encoder's representation is not linearly present in the 1b encoder's. A first attempt at the fit was killed by the system's out-of-memory killer at about 50 GB of RAM (every paired state held in memory); the fit now accumulates the regression's moment matrices chunk by chunk.
- Arm A step-0 validation loss: **2.50** (the 270m-270m's own start 2.03; the Qwen-built E3's 3.15): the decoder reads a degraded but recognisable input through the stitch; fine-tune-first proceeds. Smoke: 1.27B trainable, 18k tokens/s over the pair, 17 GiB. Step-400 gate: within ~0.05 of the 270m-270m's 0.932.
- Arm A validation against the 270m-270m: 2.50 / 2.03 (step 0), 1.036 / 1.015 (200), **0.945 / 0.932 (400): the gate passes at a 0.013 gap**, no adaptation stage. Then 0.900 / 0.880 (600), 0.854 / 0.842 (800), 0.827 / 0.814 (1000), 0.791 / 0.784 (1200), 0.766 / 0.764 (1400), 0.741 / 0.739 (1600), 0.721 / 0.721 (1800), 0.700 / 0.704 (2000), 0.682 / 0.689 (2200), 0.672 / 0.682 (2400), 0.664 / 0.674 (2600), 0.658 / 0.669 (2800), **0.656 / 0.667 (2858)**: level by step 1400 and 0.011 below at the end. Training 4.55 h (02:59 → 07:33, ~16k tokens/s over the pair, 17.4 GiB); generation through the stitched backend 2.0 h over both cards; scored 09:31.
- **Arm A benchmark** (clean / messy), against the two balanced models:

| | macro-F1 | unseen-question macro-F1 | evidence F1 | evidence precision | format valid |
|---|---|---|---|---|---|
| arm A: 1B encoder + 270M decoder (1.27B) | 0.724 / 0.704 | 0.646 / 0.636 | 0.612 / 0.582 | 0.721 / 0.688 | 0.988 / 0.981 |
| 270m-270m (0.37B trainable) | 0.660 / 0.641 | 0.509 / 0.510 | 0.576 / 0.545 | 0.667 / 0.634 | 0.982 / 0.973 |
| 1b-1b (1.70B trainable) | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.725 / 0.694 | 0.990 / 0.985 |

  Val split clean: macro-F1 0.743 / evidence 0.698 (270m-270m 0.707 / 0.655). Every slice sits between the two balanced models and much nearer the 1b-1b: apptek messy 0.705 (0.650 / 0.720), SPoRC 0.677 (0.612 / 0.687), messy 8–16k 0.697 (0.618 / 0.702), evidence F1 on messy 8–16k 0.516 (0.470 / 0.539). The prediction for this arm holds (evidence F1 +0.036 / +0.037 over the 270m-270m, threshold 0.02); the unseen-question gain (+0.137 / +0.126, to within 0.03 / 0.006 of the 1b-1b) was not predicted: zero-shot question transfer, which neither small model had, arrives with the larger encoder. Val loss said the opposite (arm A ends at the 270m-270m's level, 0.13 above the 1b-1b): the fourth instance in this project of val loss mis-ranking models on the benchmark.
- Stitch B (270m encoder → 1b decoder's space, 640 → 1152; 09:34): **held-out explained variance 0.476**, at the edge of the go-ahead threshold and lower than A's 0.644 partly for a structural reason: a 640-wide source spans at most 640 of the 1152 target dimensions. Fine-tune-first proceeds; the step-400 gate (within 0.05 of the 1b-1b's 0.776) decides.
- Arm B validation against the 1b-1b: 2.05 / 1.46 (step 0), 0.928 / 0.831 (200), **0.857 / 0.776 (400): the gate as written fails (gap 0.081 against 0.05)**. Decision (D45): the arm runs to the end under the same recipe as arm A, because the gate cannot separate a stitch deficit from the encoder-size effect here (a 270M encoder should trail a 1B encoder on loss even when perfectly adapted, and the 1b-1b is the only same-decoder reference) and because the grid needs both unbalanced arms on one recipe; the adaptation version of arm B is the follow-up if its benchmark lands near the 270m-270m. Then 0.813 / 0.738 (600), 0.772 / 0.708 (800), 0.745 / 0.680 (1000), 0.713 / 0.653 (1200), 0.695 / 0.637 (1400), 0.666 / 0.610 (1600), 0.647 / 0.588 (1800), 0.627 / 0.571 (2000), 0.610 / 0.553 (2200), 0.598 / 0.544 (2400), 0.588 / 0.535 (2600), 0.583 / 0.530 (2800), **0.581 / 0.529 (2858)**: a 0.05 gap that stopped narrowing at step 1400. Note the level: arm B's 0.581 is below arm A's 0.656 and the 270m-270m's 0.667, and above the 1b-1b's 0.529, i.e. the loss follows the decoder's size, as it did for arm A. Training 2.56 h (09:34 → 12:08, ~29k tokens/s over the pair against arm A's 16.6k: the encoder reads every prompt token, so the small-encoder arm is the cheap one), generation 2.4 h, scored 14:35.
- **The grid** (clean / messy; encoder size down, decoder size across):

| | macro-F1 | unseen-question macro-F1 | evidence F1 | evidence precision | format valid | trainable | train tokens/s |
|---|---|---|---|---|---|---|---|
| 270m-270m | 0.660 / 0.641 | 0.509 / 0.510 | 0.576 / 0.545 | 0.667 / 0.634 | 0.982 / 0.973 | 0.37B | ~30k |
| **arm B: 270M encoder + 1B decoder** | 0.673 / 0.659 | 0.530 / 0.544 | 0.584 / 0.556 | 0.679 / 0.649 | 0.988 / 0.983 | 1.27B | 29k |
| **arm A: 1B encoder + 270M decoder** | 0.724 / 0.704 | 0.646 / 0.636 | 0.612 / 0.582 | 0.721 / 0.688 | 0.988 / 0.981 | 1.27B | 17k |
| 1b-1b | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.725 / 0.694 | 0.990 / 0.985 | 1.70B | 16k |

  Val split clean macro-F1 / evidence F1: 0.707 / 0.655, B 0.721 / 0.657, A 0.743 / 0.698, 0.773 / 0.707. Per slice, arm B sits within 0.01–0.03 of the 270m-270m on every dataset and length band (its largest gains: complaint family +0.05 clean, SPoRC evidence +0.02 messy); arm A sits within 0.01–0.03 of the 1b-1b on nearly all of them. Val loss ranked the arms the other way round (B 0.581, A 0.656; 1b-1b 0.529, 270m-270m 0.667): fifth mis-ranking on this benchmark.

## Verdict

The prediction's thresholds are met on both sides (arm A over the 270m-270m on evidence F1 by 0.036 / 0.037; arm B over it on clean unseen-question macro-F1 by 0.021, messy 0.034), neither disconfirmation fired, and the finding is stronger than the prediction: at equal parameter count (1.27B), the capacity is worth far more in the encoder. Growing the encoder from 270M to 1B while keeping the 270M decoder recovers three quarters of the 1b-1b's margin over the 270m-270m on answers (0.064 of 0.084 clean) and evidence (0.036 of 0.043), and four fifths on unseen questions (0.137 of 0.168); growing the decoder instead recovers about a sixth on answers (0.013), a fifth on evidence (0.008) and an eighth on unseen questions (0.021). The part I had assigned to the decoder, zero-shot transfer to unseen questions, travels with the encoder too. The cost runs the other way: the encoder reads every prompt token, so the useful configuration is the expensive one (17k against 29k tokens/s in training) and the cheap one is the weak one. Caveats: one seed per cell; the stitched arms start from a regression fit and a 2.8k-step fine-tune, so a stitch or adaptation deficit could depress either unbalanced arm, and arm B's failed step-400 gate (D45) leaves that open for the decoder-side arm in particular, though its benchmark sits so close to the 270m-270m on every slice that a deficit would have to be large and uniform to change the reading.

## Follow-ups

- Idea 2, now directed: grow the encoder (4B encoder with the 1B or 270M decoder), not the decoder. Memory: a 4B encoder plus a 1B decoder is about 5B parameters and does not fit full training with fp32 masters on two 32 GB cards; a 4B encoder with the 270M decoder (~4.3B) is borderline under FSDP; freezing or LoRA on the encoder, or rented hardware, are the options.
- Arm B with a seq2seq adaptation stage (and the 270m-270m adapted alongside), to close the question D45 left open: whether the decoder-side arm was held back by the stitch rather than by its small encoder.
- Parked as an ablation: no fitted stitch; a random map learned through the fine-tune alone, or through a short adaptation first. Tells whether the initialisation matters.
- Idea 2: grow one tower to 4B if an asymmetry appears (memory: 4B + 1B does not fit full training with fp32 masters on two 32 GB cards; freeze, LoRA or rented hardware).
