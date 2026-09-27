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
- wandb: adaptation stages in tt-pretrain, E1++ in tt-decoder, E5++ in tt-encdec; run ids TBD.

## Result

TBD.

## Verdict

TBD.

## Follow-ups

TBD.
