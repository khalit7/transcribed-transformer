# E5: the T5Gemma 2 1b encoder as soft tokens for the Gemma 3 1B decoder

Written before the run, 2026-09-26 10:15; result and verdict added 19:45. Follow-up to
[the 270m-encoder arm](../2026-09-25-e5-soft-tokens-t5gemma2-270m-enc-gemma3-1b/README.md), which landed below the control.

## Hypothesis

The 270m arm left three readings open: the encoder was too small, the stitched input cost the decoder more than the
encoder gave, or soft tokens are the wrong interface for a causal decoder. This run changes one thing, the encoder: the
T5Gemma 2 1b-1b text encoder (1152 wide, so the stitch is square) into the same Gemma 3 1B decoder, same recipe. It
follows the strongest signal in the project so far, that capacity in the encoder is where the benchmark moves in the
native and stitched pairs, and it costs nothing new. Adaptation of the interface is the next run if this one fails,
with a matched adapted control, since an adapted arm that wins could otherwise not be attributed to the interface
rather than to extra training. A next-token adaptation objective is out in any case: the bidirectional encoder has
seen the target, so it would have to be masked reconstruction or held-out continuation.

The stitch fitted onto the decoder's embeddings explains barely more of them than the 270m one did (held-out 0.636
against 0.620) despite mapping 1152 to 1152 rather than 640 to 1152, so I expect a similar starting distance from E1's
own input; the run's step-0 validation loss is that measurement.

## Related work

The mechanism is not new. [LangBridge (Yoon et al., ACL 2024)](https://aclanthology.org/2024.acl-long.405/) maps an mT5 encoder's hidden states through a trainable linear layer into a frozen decoder-only model's input-embedding space as soft prompts, trained with a language-modelling objective, for cross-lingual transfer. [Dolphin (Chen et al., 2024)](https://arxiv.org/abs/2408.15518) projects a small decoder's encoding of a long context into a 7B decoder's input space, borrowing the vision-language projector recipe; [E2LLM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.970.pdf), [xRAG (NeurIPS 2024)](https://arxiv.org/pdf/2405.13792) and [ICAE (ICLR 2024)](https://arxiv.org/abs/2307.06945) project a text encoder's output into the decoder's input at various compression ratios with an alignment stage before task tuning. What this record adds is the controlled measurement: the decoder can read the input as tokens anyway, so the question is whether a bidirectional read of the same tokens is worth anything on top of it; the unadapted arm is run as a control; and the adapted arm is read against the same decoder given the identical adaptation data, cuts and budget with the prefix as tokens, so a gain is attributable to the encoder rather than to the extra training. Same-family encoder and decoder, both towers trained, the stitch initialised by regression onto the decoder's own embeddings, and an encoder-size axis at one decoder are the other differences.

## Prediction

- Clearly above the control (`e1-gemma3-1b-pt`: clean macro-F1 0.651, unseen-question 0.496, evidence F1 0.530),
  especially on unseen questions: soft tokens can help this causal decoder without adaptation, and the 270m failure was
  a size or representation limit. That would not isolate capacity alone, since the larger encoder also lifts
  representation quality and removes the 640-wide rank ceiling.
- Above the 270m arm (0.636 / 0.461) but level with the control: scaling repairs the deficit and there is still no
  benefit from an encoder in front.
- **Disconfirmed if** at or below the control on both variants: unadapted replacement is deprioritised at this size.
  A second negative would not show that soft tokens fail in general or that reading the encoder at every layer is necessary; two encoder
  sizes under one recipe cannot separate the interface from the objective. Adaptation of the 270m arm is then next.
- One seed; a gap under about 0.02 is a direction, not a result.

## Setup

- As the 270m arm except: encoder the text encoder of `google/t5gemma-2-1b-1b`; stitch `nn.Linear(1152, 1152)` from
  `checkpoints/stitch/1b-to-gemma3-1b-embed.pt` (ridge regression onto the decoder's scaled token embeddings over 3M
  prompt tokens, held-out explained variance 0.636, fit 118 s); micro-batch 8k tokens (2.00B trainable with fp32
  masters, the size at which the E6 hybrid ran out of memory at 16k). Config `configs/e5/t5gemma2-1b-enc-gemma3-1b.yaml`.
- Recipe otherwise E1's: lr 1e-5, 50 warm-up steps, weight decay 0.1, 8-bit AdamW with fp32 masters, one epoch, 32
  sequences per step, 16k cap, loss on the answer only, DDP; stitch at 10× the base rate.
- Generation `--backend soft`, scored by `src/train/evaluate.py` as every arm. Control `e1-gemma3-1b-pt`.
- wandb: [tt-encdec/bbiurpxu](https://wandb.ai/khalit7-/tt-encdec/runs/bbiurpxu); benchmark keys logged to the same run.

## Result

> **The numbers below come from a run with a bug and are being replaced.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (each rank trained its own replica on its half of every step; rank 0's was saved), so these are for a model trained on half the data at an effective batch of 16. The run is in the re-run queue under the fixed loop ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); its replacement's numbers replace these when scored.

- Smoke (10:14, both cards): 2.00B trainable; three steps loss 2.20, val 2.25 at step 3 (the 270m arm: 4.16 and 3.45), 13.0k tokens/s, 26.2 GiB at 8k micro-batches; two steps on the 64 longest records 27.2 GiB peak.
- Validation against the control, same steps: **2.81 / 1.73 (step 0)**, 0.938 / 0.921 (200), 0.857 / 0.837 (400), 0.810 / 0.797 (600), 0.775 / 0.764 (800), 0.748 / 0.742 (1000), 0.728 / 0.717 (1200), 0.711 / 0.705 (1400), 0.689 / 0.685 (1600), 0.677 / 0.672 (1800), 0.658 / 0.659 (2000), 0.646 / 0.649 (2200), 0.637 / 0.642 (2400), 0.631 / 0.637 (2600), 0.627 / 0.634 (2800), **0.624 / 0.632 (2858)**. The stitched input starts 1.1 nats above E1's own (the 270m arm's 2.7) despite a stitch that fits the embeddings no better (0.636 against 0.620): the larger encoder's states are far more readable to this decoder. Level with the control from step 2000 and 0.008 below at the end, the first E5 arm to cross it. The pre-clip gradient norm spiked between steps 1110 and 1290 (mostly 26–47, once 303, once 150) with no movement in the loss, and sat at 6–10 otherwise. Training 6.31 h (10:17 → 16:36) at about 11.5k tokens/s over the pair, 27.1 GiB peak.
- After the export and the wandb sync both ranks hung in the trainer's final barrier (zero CPU, memory held, the same lines the 270m arm passed in seconds); I killed them at 16:44 with the final model complete on disk (4.0 GB) and ran generation and scoring by hand. Generation through the soft backend 16:44 → 19:26 on both cards, 160k batch budget, no memory trouble; scored 19:27.
- **Benchmark** (38,220 outputs), clean / messy:

| model | macro-F1 | unseen-question macro-F1 | seen-question macro-F1 | evidence F1 | evidence precision | format valid |
|---|---|---|---|---|---|---|
| control: Gemma 3 1B (E1) | 0.651 / 0.640 | 0.496 / 0.493 | 0.683 / 0.670 | 0.530 / 0.500 | 0.589 / 0.555 | 0.955 / 0.946 |
| E5: 270m encoder into Gemma 3 1B | 0.636 / 0.620 | 0.461 / 0.450 | 0.672 / 0.655 | 0.531 / 0.494 | 0.594 / 0.550 | 0.959 / 0.944 |
| **E5: 1b encoder into Gemma 3 1B** | 0.647 / 0.633 | 0.477 / 0.482 | 0.681 / 0.663 | 0.544 / 0.510 | 0.619 / 0.580 | 0.968 / 0.956 |
| native 1B encoder + 270M decoder (E4-mix-and-match) | 0.724 / 0.704 | 0.646 / 0.636 | | 0.612 / 0.582 | 0.721 / 0.688 | 0.988 / 0.981 |

  Against the control: answers level (−0.004 / −0.007 overall, −0.002 / −0.007 on seen questions), unseen questions below (−0.019 / −0.011), evidence F1 above (+0.014 / +0.010) with precision up 0.03 / 0.025, format valid up 0.013 / 0.010. Against the 270m arm: +0.011 / +0.013 overall, +0.016 / +0.032 unseen, +0.013 / +0.016 evidence. Per dataset (clean macro-F1, E5 / control): aci_bench 0.602 / 0.582, apptek 0.652 / 0.654, taskmaster 0.645 / 0.666; messy SPoRC 0.574 / 0.592, messy 8–16k 0.576 / 0.580. Val split: clean / messy macro-F1 0.691 / 0.670 against 0.687 / 0.670, evidence 0.597 / 0.559 against 0.578 / 0.536.

## Verdict

The middle outcome: scaling the encoder repaired the 270m arm's deficit, and there is still no gain on answers. The pre-registered disconfirmation (at or below the control on both variants) fires on overall macro-F1 by 0.004 and 0.007, inside what one seed could move, so I read it as level rather than as a loss; unseen questions trail by 0.01–0.02 and evidence F1 leads by 0.01, with the cleaner outputs (format and evidence precision) the one consistent improvement. The comparison that decides the reading is the same 1B encoder behind the native 270M decoder: 0.724 / 0.646 with a decoder a quarter the size, against 0.647 / 0.477 here. The encoder carries the information; a causal decoder given it as soft tokens, with no attention adapted to read it, recovers its own performance and little more. Two encoder sizes under one recipe cannot separate the interface from the objective, so this does not show that soft tokens fail in general or that the encoder's states are needed at every layer rather than at the input, but it does settle that unadapted replacement is not the route at this size, and together with the 270m arm and the E6 gate result it is the third time a decoder-only model has failed to profit from a bidirectional encoder under a 2.8k-step fine-tune, against native and stitched pairs that do. One seed.

## Follow-ups

- If E5 continues, an adaptation stage for the interface (masked reconstruction or held-out continuation, not next-token, since the encoder has seen the target) beside an adapted Gemma 3 1B control; otherwise the design is parked and the encoder-decoder lines continue on rented hardware.
- A seed replicate of the control before any 0.01 gap in this family is read as a result.
- The end-of-run barrier hang: the trainer's final `dist.barrier()` after the export blocked both ranks with the model saved; worth a look before the next long run, since a queue script reads it as a failed stage.
