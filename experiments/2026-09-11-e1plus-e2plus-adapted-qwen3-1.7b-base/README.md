# E1+ and E2+: the mask flip after a matched adaptation stage

The follow-up that [E2](../2026-09-10-e2-prefix-lm-qwen3-1.7b-base/README.md) was planned to have if the pure flip trailed [E1](../2026-09-10-e1-qwen3-1.7b-base/README.md). Written before either run, on 2026-09-10 while E2 was at step 1600 with val loss 0.564 against E1's 0.541 at the same step (gap closing from 0.11 at step 200, but flattening).

## Hypothesis

E2's handicap is representational, not architectural: the LLM2Vec diagnostic showed the flip alone drops Qwen3-1.7B-Base's mid-network representations to 0.49 cosine against causal, and one epoch of answer-only loss (8.5M supervised tokens) is too little to repair them. A short self-supervised stage with bidirectional attention repairs them (LLM2Vec: ~16M tokens of masked next-token prediction was enough for Llama-family models), after which the bidirectional prompt encoding should pay off. To keep the comparison to one variable, the causal arm gets a matched stage with its own objective on the same text.

- **Adaptation, both arms**: the 3,164 training transcripts rendered exactly as the prompt's TRANSCRIPT block (numbered lines), 2 epochs, 32 sequences per step, about 198 steps and 20M tokens, lr 1e-5 cosine, 20 warm-up steps. No new text: the transcripts are the ones both arms already see inside the fine-tuning prompts.
  - **E1+ stage**: next-token prediction, causal attention, loss on every token (`configs/adapt/causal-qwen3-1.7b-base.yaml`).
  - **E2+ stage**: masked next-token prediction, bidirectional attention: 20% of tokens replaced by an unused special token, each masked token predicted from the position before it, loss on masked positions only (`configs/adapt/mntp-qwen3-1.7b-base.yaml`).
- **Fine-tuning**: E1's recipe from the causal-adapted checkpoint (`configs/e1/qwen3-1.7b-base-adapted.yaml`), E2's recipe from the MNTP-adapted checkpoint (`configs/e2/qwen3-1.7b-base-adapted.yaml`). Same data, steps, optimiser and schedule as E1 and E2.

The two objectives supervise different numbers of positions (all tokens vs 20%); that is inherent to giving each arm its native objective and is recorded, not equalised.

## Prediction

- E2+ closes E2's val-loss gap to E1 (0.024 at step 1600) within the first 400 fine-tuning steps and finishes at or below E1+'s val loss.
- On the unseen-question cell of the benchmark, E2+ macro-F1 above E1+ by at least 0.02, on both transcript variants, and E2+ evidence F1 above E1+.
- E1+ itself lands within 0.01 macro-F1 of E1: 20M tokens of causal next-token prediction on text the model already sees should change little.
- **Disconfirmed if** E2+ is at or below E1+ on unseen-question macro-F1 on both variants after the adaptation. That would mean bidirectional attention over the transcript does not help at 1.7B with this data, which disconfirms hypothesis 1 at this size and removes the case for building E3.
- **Inconclusive if** the MNTP stage itself fails to train (val loss on masked tokens not falling) or E1+ moves far from E1, which would mean the adaptation confounded the comparison.

## Setup

Queued behind E2 (`scratchpad/queue_e1plus_e2plus.sh`): E2 generation and scoring, then the two adaptation stages (short), then the two fine-tunes. Adaptation runs log to `tt-pretrain`, fine-tunes to `tt-decoder`, all tagged `adapted` and with the original base. Generation and scoring as E1 (vLLM) for E1+ and as E2 (`--backend prefixlm`) for E2+.

## wandb runs

- Adaptation, causal: [tt-pretrain/hb712u6f](https://wandb.ai/khalit7-/tt-pretrain/runs/hb712u6f), 2026-09-11 09:59–10:22.
- Adaptation, MNTP: [tt-pretrain/nsrb2e3u](https://wandb.ai/khalit7-/tt-pretrain/runs/nsrb2e3u), 10:23–10:55.
- E1+ fine-tune: [tt-decoder/99heynpr](https://wandb.ai/khalit7-/tt-decoder/runs/99heynpr), 10:55–15:33 (4.62 h); final val loss 0.5034 (E1: 0.5027; the two curves match to ±0.001 at every check from step 1400).
- E2+ fine-tune: [tt-decoder/397r6da5](https://wandb.ai/khalit7-/tt-decoder/runs/397r6da5), 15:33–21:42 (6.13 h); val loss 0.763 / 0.685 / 0.645 / 0.619 / 0.593 / 0.579 / 0.565 / 0.551 / (1800) / 0.534 / 0.526 / 0.519 / 0.515 / … / **0.5114** at the end, against E1's 0.704 → 0.5027 and E2's 0.815 → 0.5189 at the same steps: the adaptation removed half of E2's deficit at every point of the curve, and the remaining 0.009 gap to E1 stopped closing after step 2000.

## Adaptation stage results (read before the fine-tunes)

Both stages: 196 steps, 19.4M tokens, 3,164 transcripts, 2 epochs, 16k-token micro-batches, peak 24 GiB.

- **Causal**: val next-token loss 2.25 → 2.167 (step 100) → 2.160 → **2.157**; train loss 2.41 → 2.11; 14.2k tokens/s, MFU 0.42, 23 min. A small gain on text the model already reads well, as predicted.
- **MNTP**: val masked-token loss **9.13 before training** (the base model under bidirectional attention with an unknown mask token) → 3.03 (step 50) → 2.26 (100) → 2.08 (150) → **2.03** (196), still falling; train loss 6.47 at step 10 → 1.99; 11.4k tokens/s, MFU 0.33, 32 min. The repair the diagnostic called for happened within the first 50 steps and continued to the end; a longer stage would likely go further.


## Result

**E1+ (read from wandb `tt-decoder/99heynpr`, `bench/*` keys; full tables under `checkpoints/e1plus-qwen3-1.7b-base/eval/benchmark/`).** Benchmark macro-F1 clean 0.738 / messy 0.715 (E1: 0.735 / 0.719); unseen questions 0.650 / 0.645 (E1: 0.654 / 0.652); evidence F1 0.602 / 0.567 (E1: 0.603 / 0.570); format valid 0.985 / 0.978 (E1: 0.985 / 0.979). Every slice within ±0.005 of E1 except the small vulnerability cell (0.758 / 0.786 vs 0.729 / 0.770, n = 420 / 480). The prediction that the causal stage would change little holds: E1+ is E1.

**E2+ (wandb `tt-decoder/397r6da5`; full tables under `checkpoints/e2plus-qwen3-1.7b-base/eval/benchmark/`).** Generation 22:03–06:10 through the prefix-LM loop (8.1 h). Against E1+ (which is E1 to ±0.005):

| slice | variant | n | E1+ macro-F1 / evidence F1 | E2+ macro-F1 / evidence F1 |
|---|---|---|---|---|
| overall | clean | 17,520 | 0.738 / 0.602 | 0.731 / 0.613 |
| overall | messy | 20,700 | 0.715 / 0.567 | 0.705 / 0.577 |
| **unseen questions** | clean | 3,260 | **0.650** / 0.575 | **0.637** / 0.580 |
| **unseen questions** | messy | 3,860 | **0.645** / 0.545 | **0.632** / 0.554 |
| seen questions | clean | 14,260 | 0.757 / 0.609 | 0.751 / 0.620 |
| seen questions | messy | 16,840 | 0.730 / 0.572 | 0.720 / 0.582 |

Answers: E2+ trails E1+ by 0.013 on unseen questions on both variants and by 0.007 / 0.010 overall; the deficit is concentrated on long inputs (8–16k messy 0.681 vs 0.711; 4–8k 0.662 vs 0.696; SPoRC 0.667 vs 0.698) and absent on short ones (≤2k 0.725 vs 0.735; Taskmaster 0.716 vs 0.719). Evidence: E2+ leads by 0.010 F1 on every headline slice, driven by precision (0.717 / 0.681 vs 0.698 / 0.661) with recall equal, and produces the fewest runaway generations of any arm (549, 1.44%, vs E1+'s 694). Rare-event families (n = 300 to 640): eod 0.832 vs 0.799, complaint 0.746 vs 0.776, vulnerability 0.758 vs 0.786; mixed and within noise. Format validity 0.987 / 0.983 vs 0.985 / 0.978. Val (dev only): 0.751 / 0.727 vs E1's 0.762 / 0.735.

Relative to E2 (no adaptation): E2+ gained 0.010 / 0.002 overall and 0.014 / 0.007 on unseen questions, so the adaptation recovered about half of E2's gap to E1, matching what the training curves showed.

## Verdict

**Disconfirmed for the adapted flip at this budget.** E2+ is below E1+ on unseen-question macro-F1 on both variants (0.637 vs 0.650, 0.632 vs 0.645), which is the entry's disconfirmation condition; the predicted +0.02 did not appear and the sign is wrong. The adaptation did what it was meant to (masked-token loss 9.13 → 2.03; half of E2's deficit removed, fine-tuning val loss 0.511 vs E2's 0.519 and E1's 0.503), and the control did nothing (E1+ = E1), so the comparison is clean. Two qualifications keep this short of a final verdict on hypothesis 1 at 1.7B: the adaptation stage was still improving when it stopped (0.05 per 50 steps at the end), which is why E2++ with a 25× larger stage on outside text is already queued; and the one place E2+ consistently wins, evidence precision (+0.02) and fewer runaway generations, is the half of the task that matters most for a compliance reviewer and was not in the prediction.

What can be said now: at 1.7B, with 20M tokens of adaptation, bidirectional attention over the transcript buys better evidence selection and slightly worse answers, loses most on the longest inputs where it was expected to help most, costs 33% more training time and a 25× slower decode path. If E2++ shows the same shape with a converged adaptation, hypothesis 1 is disconfirmed at this size.

## Follow-ups

- E2++ (queued, `../2026-09-11-e2pp-outside-text-adaptation/`): the same flip after ~500M tokens of masked adaptation on outside transcript text. E1++ on the same text only if E2++ beats E1+.
- The evidence-precision gain is consistent across E2 and E2+ (+0.01 and +0.02). Worth an error analysis of which citations the causal model adds that the bidirectional one does not (near-miss adjacent lines vs unrelated lines).
- The long-input deficit: whether it is the flip's representations (would shrink with E2++) or the FlexAttention block sizes / prefix mask at length (would not).
- Summary faithfulness is unscored for every arm.
