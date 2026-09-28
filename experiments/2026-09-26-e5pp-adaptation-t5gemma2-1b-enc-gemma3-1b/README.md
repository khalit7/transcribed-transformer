# E5++ and E1++: the soft-token model and its decoder after a matched continuation-adaptation stage at corpora scale

Written before the runs, 2026-09-26 23:10; re-scoped from the training transcripts ("+") to the outside-text corpora ("++") at 23:30, before the first fine-tune. Follow-up to
[the unadapted 1b-encoder arm](../2026-09-26-e5-soft-tokens-t5gemma2-1b-enc-gemma3-1b/README.md).

## Hypothesis

Two E5 arms have shown that a causal decoder given a bidirectional encoder as soft tokens recovers its own performance
and little more (270m encoder 0.636 / 0.461, 1b encoder 0.647 / 0.477, against the control's 0.651 / 0.496 on clean
macro-F1 / unseen-question macro-F1), while the same encoders behind native decoders score far higher. The untested
lever is the interface: in both runs the decoder spent the early fine-tune learning to read its new input (validation
loss 2.8 and 4.4 at step 0 against the control's 1.7) with only the task loss to learn from. This pair asks whether an
adaptation stage that teaches the decoder to read the encoder before the task fine-tune changes the answer.

The adaptation objective is continuation: a document is cut at a seeded point in its middle half, the prefix is read
and the model predicts up to 2,048 tokens of the continuation, loss on the continuation only. It cannot
be next-token prediction over the whole document, because the bidirectional encoder would have seen every target. The
matched control is the decoder alone under the identical objective, cuts and budget, reading the prefix as tokens, so
the two adapted arms differ in exactly one thing: how the prefix enters. Without that control an adapted E5 that
gained could not be attributed to the interface rather than to extra training on transcript text.

## Related work

The mechanism is not new. [LangBridge (Yoon et al., ACL 2024)](https://aclanthology.org/2024.acl-long.405/) maps an mT5 encoder's hidden states through a trainable linear layer into a frozen decoder-only model's input-embedding space as soft prompts, trained with a language-modelling objective, for cross-lingual transfer. [Dolphin (Chen et al., 2024)](https://arxiv.org/abs/2408.15518) projects a small decoder's encoding of a long context into a 7B decoder's input space, borrowing the vision-language projector recipe; [E2LLM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.970.pdf), [xRAG (NeurIPS 2024)](https://arxiv.org/pdf/2405.13792) and [ICAE (ICLR 2024)](https://arxiv.org/abs/2307.06945) project a text encoder's output into the decoder's input at various compression ratios with an alignment stage before task tuning. What this record adds is the controlled measurement: the decoder can read the input as tokens anyway, so the question is whether a bidirectional read of the same tokens is worth anything on top of it; the unadapted arm is run as a control; and the adapted arm is read against the same decoder given the identical adaptation data, cuts and budget with the prefix as tokens, so a gain is attributable to the encoder rather than to the extra training. Same-family encoder and decoder, both towers trained, the stitch initialised by regression onto the decoder's own embeddings, and an encoder-size axis at one decoder are the other differences.

## Prediction

- E1++ (the adapted control) within a seed's worth of E1 (0.651 / 0.496): the Qwen E1+ moved 0.003 on 20M tokens of
  transcripts, and outside text at 16k did little for the Qwen arms either.
- E5++ above E1++ on unseen-question macro-F1 and evidence F1 if the interface was the bottleneck; the step-0 validation
  loss of the E5++ fine-tune, against the unadapted arm's 2.81, measures how much of the interface the stage taught.
- **Disconfirmed if** E5++ is at or below E1++ on both variants: with the interface adapted at scale and the encoder at
  1B, soft tokens into a causal decoder do not help on this task at this budget, and E5 is closed.
- One seed per cell; gaps under about 0.02 are a direction, not a result.

## Setup

- Adaptation data: the outside-text sample the Qwen E1++/E2++ stages used (callcenteren 88,933 documents, SPoRC 35,000,
  MeetingBank 1,218, CourtListener 6,400; labelled calls excluded), one epoch, 32 documents per step (about 4.1k steps),
  lr 1e-5, 50 warm-up steps, 16k cap; cut in (0.25, 0.75) of the document, continuation capped at 2,048 tokens, the
  same seeded cut points in both arms.
- `adapt-continuation-corpora-gemma3-1b-pt` (`configs/adapt/continuation-corpora-gemma3-1b-pt.yaml`): Gemma 3 1B PT,
  decoder-only, `collate_continuation` (labels on the continuation only).
  `adapt-continuation-corpora-e5-t5gemma2-1b-enc-gemma3-1b` (`configs/adapt/continuation-corpora-e5-t5gemma2-1b-enc-gemma3-1b.yaml`):
  the E5 model from the regression-fitted stitch, `collate_encdec(seq2seq=True, dec_prompt=True, dec_start=False)`: the
  encoder reads the prefix, the decoder's prefix positions carry the stitched states, everything trains, stitch at 10×,
  8k micro-batches.
- Fine-tunes: `e1pp-gemma3-1b-pt` (`configs/e1/gemma3-1b-pt-plusplus.yaml`) and `e5pp-t5gemma2-1b-enc-gemma3-1b`
  (`configs/e5/t5gemma2-1b-enc-gemma3-1b-plusplus.yaml`) start from the adapted checkpoints under E1's recipe unchanged.
- Generation: vLLM for E1++ as every decoder-only arm, `--backend soft` for E5++; scored by `src/train/evaluate.py`.
- The small-data ("+") version, 2 epochs of the 3,164 training transcripts, was built first (`configs/adapt/continuation-gemma3-1b-pt.yaml`,
  `continuation-e5-t5gemma2-1b-enc-gemma3-1b.yaml`, `e1/gemma3-1b-pt-plus.yaml`, `e5/t5gemma2-1b-enc-gemma3-1b-plus.yaml`).
  Its control stage ran (0.18 h, 196 steps, continuation val 2.17 → 1.97) and is kept under
  `checkpoints/adapt-continuation-gemma3-1b-pt/`; its E5 stage was smoked (three steps, val 2.24 at step 3 against the
  control's 2.16 at the same point, 26.8 GiB) and the pair was re-scoped to the corpora before any fine-tune ran.
- Code: `collate_continuation` in `src/train/data.py`; the seq2seq objective now accepts a decoder-only model and, with
  `decoder_sees_prompt`, puts the prefix on the decoder side for the soft arch (28 tests + 1 GPU skip).
- wandb: adaptation stages [tt-pretrain/o9hagkv2](https://wandb.ai/khalit7-/tt-pretrain/runs/o9hagkv2) (decoder) and [tt-pretrain/lj5v5iah](https://wandb.ai/khalit7-/tt-pretrain/runs/lj5v5iah) (soft-token model); E1++ [tt-decoder/xck6pj1v](https://wandb.ai/khalit7-/tt-decoder/runs/xck6pj1v); E5++ [tt-encdec/t3qprip2](https://wandb.ai/khalit7-/tt-encdec/runs/t3qprip2).

## Result

> **The numbers below come from a run with a bug and are being replaced.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (each rank trained its own replica on its half of every step; rank 0's was saved), so these are for a model trained on half the data at an effective batch of 16. The run is in the re-run queue under the fixed loop ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); its replacement's numbers replace these when scored.

- **Adaptation stages** (131,426 documents, 594M tokens, 4,107 steps, 16k cap; 158 held-out documents). Continuation validation loss, decoder / soft-token model: 1.952 / 2.271 (step 0), 1.783 / 1.803 (500), 1.763 / 1.781 (1000), 1.751 / 1.767 (1500), 1.740 / 1.756 (2000), 1.732 / 1.747 (2500), 1.728 / 1.742 (3000), 1.723 / 1.737 (3500), 1.720 / 1.734 (4000), **1.720 / 1.734 (4107)**. The soft-token prefix starts 0.32 behind the token prefix, is within 0.02 by step 500 and ends 0.014 behind: for predicting continuations the encoder's view of the prefix is worth no more than the tokens, and costs almost nothing. Decoder stage 4.33 h at about 38k tokens/s, 15.9 GiB; soft-token stage 9.93 h at about 17k tokens/s, 27.9 GiB at 8k micro-batches. Both exited cleanly.
- **E1++ fine-tune** (2026-09-27 13:50 → 16:40, 2.83 h at about 25.7k tokens/s, 17.7 GiB): task validation against E1 at the same steps 1.969 / 1.732 (step 0: the continuation stage moved the decoder's prior away from the answer format), 0.924 / 0.921 (200), 0.837 / 0.837 (400), 0.741 / 0.742 (1000), 0.706 / 0.705 (1400), 0.661 / 0.659 (2000), **0.633 / 0.632 (2858)**: level from step 200 on. vLLM generation 7 min on both cards; scored 16:48.
- **E1++ benchmark** (38,220 outputs), clean / messy:

| model | macro-F1 | unseen-question macro-F1 | seen-question macro-F1 | evidence F1 | evidence precision | format valid |
|---|---|---|---|---|---|---|
| E1: Gemma 3 1B | 0.651 / 0.640 | 0.496 / 0.493 | 0.683 / 0.670 | 0.530 / 0.500 | 0.589 / 0.555 | 0.955 / 0.946 |
| **E1++: Gemma 3 1B after the continuation stage** | 0.650 / 0.636 | 0.470 / 0.473 | 0.686 / 0.670 | 0.532 / 0.498 | 0.592 / 0.554 | 0.958 / 0.943 |
| E5 (unadapted): 1b encoder into Gemma 3 1B | 0.647 / 0.633 | 0.477 / 0.482 | 0.681 / 0.663 | 0.544 / 0.510 | 0.619 / 0.580 | 0.968 / 0.956 |
| **E5++: 1b encoder into Gemma 3 1B after the continuation stage** | **0.671 / 0.658** | **0.505 / 0.511** | **0.705 / 0.690** | **0.591 / 0.561** | **0.693 / 0.659** | **0.984 / 0.979** |

  Level with E1 overall, on seen questions, evidence and format; **below on unseen questions by 0.026 / 0.020**. The prediction (within a seed's worth of E1) holds everywhere except the unseen-question cell, where 0.02 to 0.03 is the size of gap this project has been reading as real elsewhere; with one seed I note it rather than claim it. Messy SPoRC 0.591 / 0.592 and messy 8–16k 0.558 / 0.580 (E1++ / E1). Val split 0.692 / 0.664 against 0.687 / 0.670. Continuation adaptation on 594M tokens of outside transcript text did nothing for this decoder on the task and may have cost it some zero-shot question transfer.
- **E5++ fine-tune** (2026-09-27 17:03 → 23:19, 6.27 h at about 11.9k tokens/s, 27.0 GiB; a first launch at 16:48 died before step 0 on a wandb tag over 64 characters, fixed in `TrainConfig.wandb_tags`): task validation at step 0 1.940 against the unadapted E5's 2.812 and E1++'s 1.969, so the stage taught the interface; then below every other arm from step 200 (0.903 against E1 0.921, E1++ 0.924, E5 0.938), 0.638 at step 2000 against 0.659 / 0.661 / 0.658, and **0.609 at the end** against E1 0.632, E1++ 0.633, E5 0.624. Soft-backend generation 2 h 26 min on both cards; scored 01:45.
- **E5++ benchmark** (38,220 outputs; row in the table above): above E1++ everywhere. Clean / messy gaps: macro-F1 +0.021 / +0.022, unseen-question macro-F1 +0.035 / +0.038, seen +0.019 / +0.020, evidence F1 +0.059 / +0.063, evidence precision +0.101 / +0.105, format valid +0.026 / +0.036. Messy SPoRC 0.615 (E1++ 0.591), messy 8–16k 0.616 (E1++ 0.558). Val split 0.708 / 0.703 against E1++'s 0.692 / 0.664. Against the unadapted E5 the stage added +0.024 / +0.025 macro-F1 and +0.047 / +0.051 evidence F1.

## Verdict

**Confirmed, with one seed.** The same continuation stage that did nothing for the decoder alone (E1++ level with E1, a little behind on unseen questions) moved the soft-token model above it on every metric I report, clean and messy: about 0.02 on macro-F1, 0.035 to 0.038 on unseen questions and 0.06 on evidence F1. The evidence gain is the largest and comes mostly from precision (+0.10). This is the first arm in which a causal decoder with a bidirectional encoder in front beats the same decoder without one, and the comparison is matched: same decoder, same adaptation data, cuts, objective and budget, differing only in how the prefix enters.

What it does not show: the answer-level gaps are about the size I have been treating as the edge of seed noise, so the macro-F1 claim needs a second seed before it is load-bearing; the evidence and unseen-question gaps are larger and point the same way. The arm also costs more than E1++ (a 2.0B model at 8k micro-batches, about half the control's throughput in both stages). The unadapted E5 runs read as "the interface, not the encoder, was the bottleneck", and this is consistent with that.

## Follow-ups

- The same stage for the 270m encoder (queued 2026-09-27, [its own record](../2026-09-27-e5pp-adaptation-t5gemma2-270m-enc-gemma3-1b/README.md)): does adaptation close the smaller encoder's deficit too.
- A second seed of E5++ and E1++ fine-tunes from the same adapted checkpoints, to settle the answer-level gap.
- Summary faithfulness (LLM-as-a-judge) for E5++ and E1++: TBD.
