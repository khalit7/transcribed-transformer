# E2++: the mask flip after a larger adaptation on outside transcript text

Follow-up to [E1+/E2+](../2026-09-11-e1plus-e2plus-adapted-qwen3-1.7b-base/README.md), written 2026-09-11 while E2+ was at step 530 (val loss 0.685 at step 400 against E1's 0.648 and E2's 0.715), before its result. Runs whether or not E2+ beats E1+; its interpretation depends on that.

## Hypothesis

The E2+ adaptation stage was stopped mid-repair: its masked-token val loss was still falling 0.05 per 50 steps at the end (9.13 → 3.03 → 2.26 → 2.08 → 2.03), while the causal control's loss was flat (2.167 → 2.160 → 2.157). The stages were matched in tokens, not in convergence. A larger masked-next-token stage on transcript text from outside the training corpus finishes the repair, and the fine-tuned prefix-LM then reaches or beats the causal control. I chose outside text over more epochs on the training transcripts: real new speech rather than repetition, and E1++ on the same text keeps the pair matched if it is needed.

## Prediction

- Masked-token val loss at the end of the stage below 1.9 on the outside-corpus val documents (not directly comparable to E2+'s 2.03 on training transcripts; the stage also logs nothing on those).
- Fine-tuning val loss at step 200 within 0.02 of E1's 0.704 (E2: 0.815; E2+: 0.763), and at or below E1's 0.503 at the end.
- Benchmark unseen-question macro-F1 above E1+ by at least 0.02 on both variants; then E1++ is run on the same text to confirm the comparison is matched.
- **Disconfirmed if** E2++ is at or below E1+ on unseen-question macro-F1 on both variants. With a converged adaptation, "under-adapted" is no longer available as an explanation, and hypothesis 1 is disconfirmed at 1.7B for this task.

## Setup

- Adaptation (`configs/adapt/mntp-corpora-qwen3-1.7b-base.yaml`): masked next-token prediction, bidirectional attention, 20% masking, on about 131,500 documents, one pass: all of CallCenterEN (88,933 docs, real call-centre ASR, undiarised, rendered one sentence per numbered line), a seeded sample of 35,000 SPoRC episodes (diarised, numbered speaker lines; every labelled episode excluded), all of MeetingBank (1,218) and CourtListener (6,400) (undiarised, sentence lines), documents truncated at 32k tokens. Measured at build (2026-09-12 06:20): 131,426 documents (CallCenterEN 88,933, SPoRC 34,875, CourtListener 6,400, MeetingBank 1,218), **661.5M tokens**, 1,804 documents truncated at 32k, 4,107 steps; 34× E2+'s stage (the 500M estimate undershot because SPoRC episodes are longer than the published median implied); I chose this over a 150M-token sample (too little) and 1B (almost all podcast text) on 2026-09-11. 32 documents per step, lr 1e-5, 50 warm-up steps, 16k-token micro-batches.
- Fine-tune (`configs/e2/qwen3-1.7b-base-adapted-corpora.yaml`): E2's recipe from the adapted checkpoint. Generation with the prefix-LM loop, scoring as before.
- E1++ (`configs/adapt/causal-corpora-qwen3-1.7b-base.yaml`, `configs/e1/qwen3-1.7b-base-adapted-corpora.yaml`): the same text, causal objective, run only if E2++ beats E1+.
- Note on hypothesis 2: this stage does put outside transcript text into one arm. The E2++ vs E1+ comparison therefore mixes "bidirectional" with "more transcript text"; the E1++ control, if run, separates them. Hypothesis 2's own experiment (continued pretraining for both arms, with a no-pretraining control) remains separate.

## wandb runs

- Adaptation: [tt-pretrain/jgix6wya](https://wandb.ai/khalit7-/tt-pretrain/runs/jgix6wya), 2026-09-12 06:17 → 2026-09-13 ~08:20 (17.7 h of training after a resume from step 1305; the first attempt crashed at step 1335 on a FlexAttention compile failure for a micro-batch of very short calls, fixed by padding prefix-LM batches to a 256-token floor). 4,107 steps, 661.5M tokens seen, ~132M supervised (20% masked). Val masked-token loss on held-out outside documents: 1.475 (500) / 1.321 (1000) / 1.253 (1500) / 1.217 (2000) / 1.196 (2500) / 1.180 (3000) / 1.172 (3500) / 1.166 (4000) / **1.165** (end): converged, the prediction's "below 1.9" met by a wide margin (not comparable to E2+'s 2.03, which was on training-transcript text). Train loss 8.8 at step 10 → about 1.1–1.2.
- Fine-tune: [tt-decoder/c82ujj67](https://wandb.ai/khalit7-/tt-decoder/runs/c82ujj67), 2026-09-13 08:22–14:34 (6.20 h). Val loss 0.740 / 0.667 / 0.632 / 0.607 / 0.587 / 0.572 / 0.559 / 0.545 / 0.537 / 0.527 / 0.519 / 0.512 / 0.509 / 0.507 / **0.5056** at the end; E1 0.704 → 0.5027, E2+ 0.763 → 0.5114, E2 0.815 → 0.5189 at the same steps. The gap to E1 went 0.036 → 0.019 → 0.012 → 0.007 → 0.004 → 0.003 and stayed a hair behind; the prediction's "within 0.02 at step 200" missed (0.036) and "at or below E1 at the end" missed by 0.003.

## Result

Read from wandb `tt-decoder/c82ujj67` (`bench/*` keys; full tables under `checkpoints/e2pp-qwen3-1.7b-base/eval/benchmark/`). Generation 14:35–22:35 through the prefix-LM loop (8.0 h). Against E1+ (the causal control, which equals E1 to ±0.005) and E2+:

| slice | variant | n | E1+ macro-F1 / evidence F1 | E2+ | E2++ |
|---|---|---|---|---|---|
| overall | clean | 17,520 | 0.738 / 0.602 | 0.731 / 0.613 | 0.731 / 0.619 |
| overall | messy | 20,700 | 0.715 / 0.567 | 0.705 / 0.577 | 0.713 / 0.584 |
| **unseen questions** | clean | 3,260 | 0.650 / 0.575 | 0.637 / 0.580 | **0.655 / 0.586** |
| **unseen questions** | messy | 3,860 | 0.645 / 0.545 | 0.632 / 0.554 | **0.649 / 0.555** |
| seen questions | clean | 14,260 | 0.757 / 0.609 | 0.751 / 0.620 | 0.747 / 0.626 |
| seen questions | messy | 16,840 | 0.730 / 0.572 | 0.720 / 0.582 | 0.727 / 0.591 |

Answers: E2++ is level with E1+: +0.005 / +0.004 on unseen questions, −0.007 / −0.002 overall, all inside single-run noise; the +0.02 predicted did not appear. The long-input deficit that E2 and E2+ showed is mostly gone (8–16k messy 0.707 vs 0.711; SPoRC 0.689 vs 0.698; 4–8k 0.678 vs 0.696 is the one slice still clearly behind). Taskmaster is the one corpus where E2++ leads clearly (0.736 vs 0.719). Evidence: E2++ leads E1+ by 0.017 F1 overall on both variants (0.619 / 0.584 vs 0.602 / 0.567) and by 0.010 on unseen questions, through precision (0.722 / 0.690 vs 0.698 / 0.661, +0.024 / +0.029) with recall equal (0.608 / 0.578 vs 0.603 / 0.577). Fewest runaway generations of any arm (526, 1.38%; E1+ 694). Format valid 0.988 / 0.984, the best of the five. Rare-event families within noise (complaint 0.755 vs 0.776, vulnerability 0.769 vs 0.786, eod 0.805 vs 0.799 on 300–640 pairs).

Across the three bidirectional arms the answer gap to the causal control shrank monotonically with adaptation tokens (unseen, clean: E2 −0.031, E2+ −0.013, E2++ +0.005 against their controls) while the evidence-precision lead grew (+0.009, +0.019, +0.024 clean).

Val (dev only): macro-F1 0.753 / 0.729, evidence F1 0.686 / 0.643 (E1 0.762 / 0.735, 0.662 / 0.618).

## Verdict

**Inconclusive.** E2++ neither beats E1+ by the predicted 0.02 on unseen-question macro-F1 nor falls below it (the disconfirmation condition); it ties, +0.005 and +0.004. What is not a tie: evidence precision (+0.024 / +0.029) and evidence F1 (+0.017 overall, +0.010 on unseen questions), the third consecutive bidirectional arm to lead on evidence and the largest lead yet, with the fewest malformed generations.

**Caveat, which is why the verdict stays inconclusive rather than "confirmed on evidence": the matched control E1++ was not run.** E2++ saw 661M tokens of outside transcript text and E1+ saw 19M of training transcripts, so both the answer parity and the evidence gain could be the extra text rather than bidirectional attention. E1++ (the same 661M tokens, causal objective, then the E1 fine-tune; configs `configs/adapt/causal-corpora-qwen3-1.7b-base.yaml` and `configs/e1/qwen3-1.7b-base-adapted-corpora.yaml`) was started on 2026-09-13 at 22:37 and stopped at 22:49 when I decided to move on; it can be run later if the evidence result is worth settling. Until then: at 1.7B, bidirectional attention plus a large transcript adaptation ties the causal control on answers and leads on evidence precision, and the two ingredients are not separated.

Costs unchanged: the prefix-LM arm trains 33% slower, decodes 25× slower (8 h against 20 min for the benchmark), and gives up KV-cache sharing across the questions asked of one call.

## Follow-ups

- E1++ (skipped; configs ready): the matched causal control on the same 661M tokens. If E1++ also gains evidence precision, the effect is the text; if not, it is the attention. The one run that would turn this verdict into a confirmed or disconfirmed one.
- Error analysis of the evidence gain: which lines the causal model cites that the bidirectional one does not.
- The inference path for prefix-LM if it survives E1++: static cache and GQA-aware decode at least.
- Summary faithfulness is still unscored for every arm.
