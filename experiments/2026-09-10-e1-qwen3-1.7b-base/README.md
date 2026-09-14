# E1: causal SFT of Qwen3-1.7B-Base on every labelled record

The control arm of the README's "Design" table. Entry written 2026-09-10 after launch and after the training curve, before any benchmark number (the hypothesis below was the one the run was launched under; nothing in it was changed after scoring).

## Hypothesis

A 1.7B causal decoder, fine-tuned with the loss on the answer only, learns the task format and enough of the judgements to beat the majority-class predictor on questions it has never seen, from ~90k labelled pairs and one epoch. This is the floor every other arm is measured against; E2 (prefix-LM, same checkpoint, same data, mask only) is the first comparison.

## Prediction

- Format validity (strict JSON, answer byte-equal to an option, evidence an in-range ascending integer list) above 0.95 on the benchmark.
- Answer macro-F1 on the unseen-question cell above the majority-class predictor's macro-F1 on the same cell, on both transcript variants.
- Evidence precision against the gold key above 0.5 on the clean variant.
- **Disconfirmed if** macro-F1 on the unseen-question cell does not exceed the majority predictor: the model would then have learnt the label distribution per question, not the task, and zero-shot generality (the project's central claim) would be unsupported at this size.

## Setup

- Experiment E1. Base `Qwen/Qwen3-1.7B-Base`, full fine-tuning, bf16 model with fp32 master weights, 8-bit AdamW, lr 1e-5 cosine to 1e-6 after 50 warm-up steps, weight decay 0.1, clip 1.0, 32 sequences per step, 1 epoch, 32k max sequence length, FlashAttention 2, gradient checkpointing, DDP over 2× RTX 5090.
- Data: `data/labelled_data/train.jsonl` at the 2026-09-08 freeze (`splits.json` seed 0), every record whatever its licence track (CLAUDE.md "Dataset licences", clarified 2026-09-10): 91,501 examples (one SPoRC pair over 32k tokens dropped), 241.6M prompt tokens, 8.5M target tokens. Each example rendered from the variant its label was made on. Prompt: `src/train/data.task_prompt`; target: the label as JSON (evidence, answer, summary, tags where the question has them). Prompt loss zero.
- Generation: vLLM 0.29, greedy, max 512 new tokens, both transcript variants. Scoring: `src/train/evaluate.py` per the eval-bench skill.
- Config: `configs/e1/qwen3-1.7b-base.yaml`; full resolved config in the wandb run.

## wandb run

[tt-decoder/nxxkzq91](https://wandb.ai/khalit7-/tt-decoder/runs/nxxkzq91)

## Result

Read from wandb `tt-decoder/nxxkzq91` (training summary and the `bench/*` keys logged by `src/train/evaluate.py`; full tables in `checkpoints/e1-qwen3-1.7b-base/eval/benchmark/results.md`).

Training: 2,859 steps, 4.61 h on two GPUs, 15.1k tokens/s, MFU 0.42, val loss 1.18 → 0.503 (monotone, still falling at the end of the epoch).

Benchmark, 20,700 pairs, both transcript variants (clean has no SPoRC, which is real ASR only):

| slice | variant | n | format valid | accuracy | macro-F1 | majority macro-F1 | evidence precision | evidence recall | evidence F1 |
|---|---|---|---|---|---|---|---|---|---|
| overall | clean | 17,520 | 0.985 | 0.802 | 0.735 | 0.592 | 0.697 | 0.604 | 0.603 |
| overall | messy | 20,700 | 0.979 | 0.787 | 0.719 | 0.594 | 0.663 | 0.578 | 0.570 |
| **unseen questions** | clean | 3,260 | 0.976 | 0.725 | **0.654** | 0.582 | 0.681 | 0.573 | 0.573 |
| **unseen questions** | messy | 3,860 | 0.977 | 0.723 | **0.652** | 0.572 | 0.656 | 0.556 | 0.550 |
| seen questions | clean | 14,260 | 0.987 | 0.819 | 0.752 | 0.591 | 0.701 | 0.611 | 0.609 |
| seen questions | messy | 16,840 | 0.980 | 0.801 | 0.733 | 0.597 | 0.665 | 0.584 | 0.574 |

By corpus (messy): AppTek macro-F1 0.716, ACI-Bench 0.718, Taskmaster 0.719, SPoRC 0.700; majority 0.52 to 0.60. By family (messy): general_qa 0.715, complaint 0.782, vulnerability 0.770, eod 0.782, each 0.18 to 0.26 above its majority baseline. By length (messy): macro-F1 0.734 at ≤2k tokens, 0.705 at 2–4k, 0.695 at 4–8k, 0.721 at 8–16k; evidence F1 falls faster, 0.626 → 0.539 → 0.495 → 0.492. Clean vs messy on the same AppTek calls: macro-F1 0.728 vs 0.716, evidence F1 0.566 vs 0.546. Empty-evidence gold matched by an empty prediction: 0.89 (clean) / 0.87 (messy). Tag Jaccard on vulnerability: 0.78 / 0.76.

Format: 1.72% of the 38,220 generations (656) hit the 512-token cap and are scored invalid: 393 runaway evidence lists that enumerate every line, 263 repetition loops in the summary. Beyond those, the answer is byte-equal to an option in 98.0 to 98.6% of outputs and there are no recovered-but-inexact answers of note.

Priority families on AppTek (messy, n=300 / 500 / 200): vulnerability macro-F1 0.662, complaint 0.764, eod 0.778, against majority 0.52 / 0.51 / 0.48; the confusion matrices show the residual errors are mostly pass↔fail on the positives, not NA confusion. Taskmaster complaint is trivial (60/60 pass, majority 1.0). ACI-Bench eod on messy is 0.474 (3 gold fails, all predicted pass).

## Verdict

**Confirmed.** All three predictions hold: format validity 0.98 to 0.99 (above 0.95); unseen-question macro-F1 0.654 / 0.652 against majority 0.582 / 0.572 on both variants; evidence precision 0.70 on clean (above 0.5). The model generalises to held-out question lineages, with a cost: 0.10 macro-F1 between seen and unseen questions, which is the number E2 and the larger sizes have to close. Evidence is the weaker half of the task (F1 0.57 to 0.60 overall, 0.49 above 4k tokens), consistent with the literature's warning that citation degrades fastest with length.

Two caveats on reading the numbers. The gold is model-produced (qwen3.8 and gemma4 agreement, Opus adjudication; Sonnet for the families), so agreement with it is agreement with those labellers, not with a human. And the run was one epoch at one learning rate, unswept; val loss was still falling.

## Follow-ups

- E2 (prefix-LM) on this exact checkpoint, data and recipe: the comparison this run exists for.
- Runaway outputs: 1.7% of generations loop or enumerate every line. Score stays on raw output per the eval-bench rules; the fix, if wanted, is a training-side one (more epochs, or a repetition penalty applied uniformly to every arm and recorded).
- A second epoch or a higher learning rate (val loss still falling at the end).
- Summary faithfulness is unscored (needs the LLM-as-a-judge step).
- E0 (API baseline) on the same prompt, so the "beat the baseline on quality and cost" gate has its other side.
