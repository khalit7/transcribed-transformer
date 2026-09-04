---
name: eval-bench
description: Run the compliance-QA benchmark and produce the results table. Use when evaluating any checkpoint or baseline, comparing arms, or regenerating the results table in the README. Covers the metric definitions, the two-track reporting rule and the label-provenance rules.
---

# Running the benchmark

## What gets measured

Four things per question, scored separately. A model that gets the answer right for the wrong reason, or right in a form the consuming system cannot parse, is a failure this benchmark exists to catch.

**Answer** — accuracy and macro-F1 over the values the question supplied, plus per-class support. There is no fixed label set: each question carries its own `AnswerOption` list, so the metric is computed against that question's vocabulary rather than a global one.

Macro-F1 is the headline, not accuracy. Screening tasks of this shape tend to produce skewed answer distributions, and on a skewed set accuracy flatters badly. **The majority-class predictor is a mandatory baseline**: an arm that does not beat "always answer the most common label" on macro-F1 has not demonstrated anything, however good its accuracy looks. Report the realised class distribution alongside every result, so the skew is visible rather than inferred, and the confusion matrix for the three priority families (vulnerability, complaint, dissatisfaction) separately from the aggregate.

**Format validity** — scored strictly, and reported whether or not it looks embarrassing.

| Outcome | Credit | Meaning |
|---|---|---|
| `answer_exact` | 1.0 | Byte-equal to one supplied option value |
| `answer_recovered` | 0.3 | Wrong on the wire, but recoverable by matching the option's grading rule back to its value |
| `answer_invalid` | 0.0 | Not recoverable |

`answer_recovered` is **a failure with partial credit, not a success.** The reason it is scored separately at all: a downstream normalisation layer that repairs these silently makes the raw error rate invisible, so the benchmark scores what the model emitted rather than what a repair layer rescued. Score the raw output.

Evidence carries hard gates, any of which is a failed generation: a JSON array of JSON integers (never strings, never floats, never a bare scalar), 1-based, every element within `[1, N]`, de-duplicated, ascending, and `[]` when nothing was found rather than a sentinel like `-1`, `0` or `"NA"`. **Check types, not just key presence.** Evidence returned as the string `"12, 15"` rather than the list `[12, 15]` satisfies any check that only asks whether the key exists, and then yields nothing useful downstream; a scorer that does not verify the type will record that as a success.

**Evidence** — **precision against a partial key** is the primary metric, plus **sufficiency** and **comprehensiveness** from the rationale-faithfulness literature: does the answer survive on the selected lines alone, and does it change when they are removed.

Report set-level recall and F1 **only where the key is marked exhaustive** (`Assessment.evidence_exhaustive`). Human evidence keys are normally partial: a question can have several genuinely correct supporting lines and an annotator marks the ones they noticed. Scoring recall against a partial key penalises a model for finding a correct line the annotator missed, which understates every model including the baselines and makes the comparison useless. This is why set-F1 is not the headline.

**Summary** — faithfulness to the cited evidence, judged against a rubric. A summary that is fluent, correct-sounding and not entailed by the lines it cites scores zero. Hallucinated evidence is the most damaging failure mode available here, because it makes a wrong answer look well-supported.

**Pin the judge model version (the LLM-as-a-judge that scores summaries; never confuse it with the labeller that produced the benchmark's labels) and record it as a wandb artifact** alongside the benchmark version. Judge drift silently invalidates every score taken before it, and nothing in the numbers reveals that it happened. LLM judges have other known failure modes here too, notably truncation on long inputs, which quietly flatters models that fail late in a transcript; verify the judge sees the whole input before trusting any ranking.

## Reporting slices

Every headline table is sliced three ways, because each slice answers a different question and the aggregate hides all three.

**Seen vs unseen criteria, and the unseen column is the headline.** Zero-shot generality over questions is the project's central claim, so it gets the primary number. The split is **by question family, not by question and never by case**: questions within a family are often near paraphrases, so holding out individual questions from a family still in training leaks the answer, and holding out cases for questions that were trained on measures memorisation and will read as success.

**By length bucket.** Cases binned by rendered token length, with accuracy *and* cost per bucket. Real transcript collections vary enormously in length; a single aggregate over a benchmark where every case is maximal misrepresents both quality and cost, and the long tail dominates the cost side.

**By licence track.** See below.

## Reporting rules

**Both licence tracks, always.** Every headline result appears twice: Track P (permissive, commercially portable) and Track NC (research-only, better data). Reporting only the better number hides the cost of the licence restriction, which is one of the project's actual findings.

**Gold slice is the headline, and gold is immutable.** Silver labels come from LLM consensus and are dev signal only. Beating a frontier model on labels that a frontier model produced is partly circular, and the human-annotated gold slice is the number that answers the question honestly. State inter-annotator agreement alongside it.

**Never tune on gold.** Iterate on the silver/dev split; gold is looked at to report, not to decide. A gold set that has been used for model selection has become a dev set, silently, and there is no way to undo it or to detect it later.

**Cost is a gate, not a footnote.** Report tokens/s and cost per case per length bucket, and require an arm to be **at least as cheap per case as the baseline it beats** before the result is written up as a win. Efficiency is half the reason for preferring a smaller encoder over a prompted frontier model, so a quality-only gate would certify a checkpoint that is slower and more expensive than the thing it replaces, which is not a result anyone can use.

## Mechanics

- Evaluation logs to the **same wandb run id** as the training job that produced the checkpoint, so quality and training curve stay attached.
- Results tables are **generated from the wandb API** filtered on the run tags, not hand-written. This is what stops the numbers in the README drifting from the numbers that were actually measured.
- Record the benchmark version (a wandb artifact) with every result. Comparing a model scored on benchmark v1 against one scored on v2 is meaningless and easy to do by accident.
- Baselines are re-run whenever the benchmark version changes. Do not carry a baseline number forward across benchmark revisions.

## Honesty

If a checkpoint loses to its baseline, that is the result. Write it down, log it, and say so plainly. The architecture hypothesis being wrong is a publishable finding; a quietly dropped negative result is not.
