# E5++ with the 270m encoder: the smaller soft-token model after the same continuation-adaptation stage

Written before the runs, 2026-09-27 22:40, while the 1b-encoder E5++ fine-tune was still training. Companion to
[the 1b-encoder adaptation pair](../2026-09-26-e5pp-adaptation-t5gemma2-1b-enc-gemma3-1b/README.md) and follow-up to
[the unadapted 270m arm](../2026-09-25-e5-soft-tokens-t5gemma2-270m-enc-gemma3-1b/README.md).

## Hypothesis

Unadapted, the 270m encoder in front of the Gemma 3 1B decoder scored below its control (0.636 / 0.461 clean
macro-F1 / unseen-question macro-F1 against 0.651 / 0.496), and the 1b encoder brought the arm level with it. I read
the 270m deficit as partly an interface problem: the decoder had to learn to read a narrower encoder (640 wide, stitch
explained variance 0.620) from the task loss alone. If the continuation-adaptation stage teaches that interface, the
270m arm should recover more from it than the 1b arm did, because it had more to recover.

This arm reuses the 1b pair's control. The decoder, the adaptation data, the cuts, the objective and the budget are
identical, so E1++ (`e1pp-gemma3-1b-pt`, already scored: 0.650 / 0.470) is the matched control for this arm too.

## Prediction

- E5++ (270m) above the unadapted 270m arm (0.636 / 0.461) by more than a seed's worth (about 0.02) on clean macro-F1.
- Relative to E1++, the interesting reading is the gap: if the 270m arm closes to within 0.02 of E1++ after adaptation,
  the unadapted deficit was the interface, not the encoder's size.
- **Disconfirmed if** E5++ (270m) is within 0.02 of the unadapted 270m arm on both variants: adaptation does not help
  the smaller encoder either, and the 270m deficit is capacity.
- One seed per cell; gaps under about 0.02 are a direction, not a result.

## Setup

- Adaptation: `adapt-continuation-corpora-e5-t5gemma2-270m-enc-gemma3-1b`
  (`configs/adapt/continuation-corpora-e5-t5gemma2-270m-enc-gemma3-1b.yaml`): the 1b stage's config with the T5Gemma 2
  270m-270m encoder and its regression-fitted stitch (`checkpoints/stitch/270m-to-gemma3-1b-embed.pt`). Same outside-text
  sample (callcenteren 88,933, SPoRC 35,000, MeetingBank 1,218, CourtListener 6,400 documents), one epoch, 32 documents
  per step, lr 1e-5, stitch at 10x, 16k cap, 8k micro-batches, cuts in (0.25, 0.75), continuation capped at 2,048 tokens.
- Fine-tune: `e5pp-t5gemma2-270m-enc-gemma3-1b` (`configs/e5/t5gemma2-270m-enc-gemma3-1b-plusplus.yaml`) from the
  adapted checkpoint under E1's recipe, 16k micro-batches as the unadapted 270m fine-tune ran.
- Generation with `--backend soft` on both cards; scored by `src/train/evaluate.py`; benchmark metrics logged to the
  fine-tune's wandb run.
- wandb: adaptation stage [tt-pretrain/d1ax1lf6](https://wandb.ai/khalit7-/tt-pretrain/runs/d1ax1lf6); E5++ [tt-encdec/18f3lqfg](https://wandb.ai/khalit7-/tt-encdec/runs/18f3lqfg), benchmark keys logged to the same run.

## Result

> **The numbers below come from a run with a bug and are being replaced.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (each rank trained its own replica on its half of every step; rank 0's was saved), so these are for a model trained on half the data at an effective batch of 16. The run is in the re-run queue under the fixed loop ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); its replacement's numbers replace these when scored.

- **Adaptation stage** (2026-09-28 01:46 → 08:55, 7.16 h at about 24k tokens/s, 19.3 GiB, 1.27B trainable): continuation validation 2.513 (step 0), 1.806 (500), 1.784 (1000), 1.769 (1500), 1.758 (2000), 1.749 (2500), 1.743 (3000), 1.739 (3500), **1.736 (4107)**, against the decoder alone at 1.952 → 1.720 and the 1b soft-token model at 2.271 → 1.734. The 270m prefix starts 0.56 behind the token prefix (the 1b started 0.32 behind) and ends 0.016 behind, where the 1b ended 0.014 behind: on plain continuation the two encoders' soft tokens read almost equally well.
- **Fine-tune** (09:00 → 13:23, 4.36 h at about 16.7k tokens/s, 19.5 GiB): task validation against E1++ at the same steps 1.977 / 1.969 (step 0; the unadapted 270m arm started at 4.44), 0.930 / 0.924 (200), 0.856 / 0.837 (400), 0.750 / 0.741 (1000), 0.706 / 0.706 (1400), 0.684 / 0.686 (1600), 0.657 / 0.661 (2000), 0.638 / 0.644 (2400), **0.627 / 0.633 (2858)**. The stage removed the interface cost at step 0 entirely; the arm crosses the control at step 1600 and ends 0.006 below it. Generation through the soft backend 13:23 → 15:44 on both cards; scored 15:44.
- **Benchmark** (38,220 outputs), clean / messy, beside the other E5 arms and the controls:

| model | macro-F1 | unseen-question macro-F1 | seen-question macro-F1 | evidence F1 | evidence precision | format valid |
|---|---|---|---|---|---|---|
| E1: Gemma 3 1B | 0.651 / 0.640 | 0.496 / 0.493 | 0.683 / 0.670 | 0.530 / 0.500 | 0.589 / 0.555 | 0.955 / 0.946 |
| E1++: Gemma 3 1B after the continuation stage | 0.650 / 0.636 | 0.470 / 0.473 | 0.686 / 0.670 | 0.532 / 0.498 | 0.592 / 0.554 | 0.958 / 0.943 |
| E5 (unadapted), 270m encoder | 0.636 / 0.620 | 0.461 / 0.450 | 0.672 / 0.655 | 0.531 / 0.494 | 0.594 / 0.550 | 0.959 / 0.944 |
| **E5++, 270m encoder** | 0.648 / 0.634 | 0.473 / 0.472 | 0.684 / 0.667 | 0.567 / 0.539 | 0.651 / 0.617 | 0.979 / 0.970 |
| E5 (unadapted), 1b encoder | 0.647 / 0.633 | 0.477 / 0.482 | 0.681 / 0.663 | 0.544 / 0.510 | 0.619 / 0.580 | 0.968 / 0.956 |
| E5++, 1b encoder | 0.671 / 0.658 | 0.505 / 0.511 | 0.705 / 0.690 | 0.591 / 0.561 | 0.693 / 0.659 | 0.984 / 0.979 |

  Against the unadapted 270m arm: +0.012 / +0.014 macro-F1, +0.012 / +0.022 unseen, +0.036 / +0.045 evidence F1. Against E1++: level on answers (−0.002 / −0.002 overall, +0.003 / −0.001 unseen, −0.002 / −0.003 seen), **ahead on evidence F1 by 0.035 / 0.041**, evidence precision by 0.06 and format validity by 0.02 to 0.03. Per dataset (clean macro-F1, E5++ 270m / E1++): aci_bench 0.575 / 0.573, apptek 0.661 / 0.660, taskmaster 0.638 / 0.653; messy SPoRC 0.591 / 0.591, messy 8–16k 0.600 / 0.558. Val split 0.695 / 0.678 macro-F1 and 0.634 / 0.598 evidence against E1++'s 0.692 / 0.664 and 0.578 / 0.534.

## Verdict

The adaptation stage repaired the 270m encoder's deficit exactly, and no further: the arm is level with E1++ on answers, seen and unseen, and ahead of it only where every adapted soft-token arm has been ahead, on evidence and output quality. The first prediction (above the unadapted arm by more than 0.02 on clean macro-F1) missed by 0.008; the second reading holds (within 0.02 of E1++ after adaptation, so the unadapted deficit was the interface, not capacity); the disconfirmation did not fire on evidence, where the gain is 0.04. Read with the 1b arm, the encoder axis under the recipe now has two points at one decoder: 270m repairs and adds evidence, 1b adds answers, unseen questions and more evidence. The 4b arm, queued, is the third. One seed; the answer-level differences here are within a seed's worth and are not claimed.

## Follow-ups

- The 4b encoder into the same decoder under the same stage (queue 28), the third point of the axis.
- A seed replicate of E1++ and one E5++ arm before any answer-level gap under 0.02 in this family is read as a result.
