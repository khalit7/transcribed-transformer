---
name: eval-bench
description: Run the compliance-QA benchmark and produce the results table. Use when evaluating any checkpoint or baseline, comparing arms, or regenerating the results table in the README. Covers the metric definitions, the reporting slices and the label-provenance rules.
---

# Running the benchmark

## What gets measured

Four things per question, scored separately. A model that gets the answer right for the wrong reason, or right in a form the consuming system cannot parse, is a failure this benchmark exists to catch.

**Answer** — accuracy and macro-F1 over the values the question supplied, plus per-class support. There is no fixed label set: each question carries its own `AnswerOption` list, so validity is checked against that question's vocabulary. Macro-F1 is pooled: one F1 per answer value over every row of the slice, the classes being every option any question in the slice supplied plus any gold value, averaged over classes (`macro_f1` in `src/train/evaluate.py`); it is not a per-question macro-F1 averaged over questions.

Macro-F1 is the headline, not accuracy. Screening tasks of this shape tend to produce skewed answer distributions, and on a skewed set accuracy flatters badly. **The majority-class predictor is a mandatory baseline**: an arm that does not beat "always answer the most common label" on macro-F1 has not demonstrated anything, however good its accuracy looks. Report the realised class distribution alongside every result, so the skew is visible rather than inferred, and the confusion matrix for the three priority families (vulnerability, complaint, dissatisfaction) separately from the aggregate.

**Format validity** — scored strictly, and reported whether or not it looks embarrassing.

| Outcome | Credit | Meaning |
|---|---|---|
| `answer_exact` | 1.0 | Byte-equal to one supplied option value |
| `answer_recovered` | 0.3 | A case, whitespace or hyphen variant of an option value; the evaluator recovers only these and never matches grading-rule text |
| `answer_invalid` | 0.0 | Not recoverable |

`answer_recovered` is **a failure with partial credit, not a success.** The reason it is scored separately at all: a downstream normalisation layer that repairs these silently makes the raw error rate invisible, so the benchmark scores what the model emitted rather than what a repair layer rescued. Score the raw output.

Evidence carries hard gates, any of which is a failed generation: a JSON array of JSON integers (never strings, never floats, never a bare scalar), 1-based, every element within `[1, N]`, de-duplicated, ascending, and `[]` when nothing was found rather than a sentinel like `-1`, `0` or `"NA"`. **Check types, not just key presence.** Evidence returned as the string `"12, 15"` rather than the list `[12, 15]` satisfies any check that only asks whether the key exists, and then yields nothing useful downstream; a scorer that does not verify the type will record that as a success.

**Evidence** — precision, recall, F1 and exact match of the cited line set against the gold key, with the empty-means-empty rule below. Sufficiency and comprehensiveness tests from the rationale-faithfulness literature (does the answer survive on the selected lines alone; does it change when they are removed) are not implemented in the evaluator; if they are added they are reported beside, not instead of, the set metrics.

**Empty means empty (decided 2026-09-09).** A gold key of `[]` is a real key, not a missing one: the answer rests on something being absent from the transcript and the expected evidence is the empty list. Score such pairs as evidence-correct only when the model also emits `[]`; any cited line is a false positive. Do not exclude empty-key pairs from the evidence metric and do not fill them with near-miss lines. Report the empty-key share of the benchmark alongside the metric, per gold (under gold = teacher 4,597 of 20,700 keys are empty). Known limitation: labellers were not instructed on this, so some absence answers in the gold do cite near-miss lines; the gold is what the labeller wrote.

Evidence keys are one labeller's selection, not an exhaustive list of every supporting line, so recall and F1 understate every model alike; they are reported for every pair (there is no exhaustive-key flag), and precision is the number to read when a model finds a correct line the labeller did not cite.

**Summary** — not scored today beyond format validity (the summary must be a string); faithfulness to the cited evidence, judged against a rubric, is the target. A summary that is fluent, correct-sounding and not entailed by the lines it cites scores zero. Hallucinated evidence is the most damaging failure mode available here, because it makes a wrong answer look well-supported.

**Pin the judge model version (the LLM-as-a-judge that scores summaries; never confuse it with the labeller that produced the benchmark's labels) and record it as a wandb artifact** alongside the benchmark version. Judge drift silently invalidates every score taken before it, and nothing in the numbers reveals that it happened. LLM judges have other known failure modes here too, notably truncation on long inputs, which quietly flatters models that fail late in a transcript; verify the judge sees the whole input before trusting any ranking.

## Reporting slices

Every headline table is sliced three ways, because each slice answers a different question and the aggregate hides all three.

**Seen against unseen questions, and the no-twin unseen slice is the zero-shot number.** The split is by call (and by podcast and encounter), with 51 question ids held out of training (`splits.json`); twenty of those have a near-verbatim or strong semantic twin among the training questions, so `cell=unseen_q` overstates zero-shot transfer and `cell=unseen_q_no_twin` (31 questions) is the number to report as zero-shot. A future split holds out concept clusters, not question ids.

**By length bucket.** Cases binned by rendered token length (each arm's own tokenizer, so cross-family length slices are not the same pairs), with accuracy per bucket; cost per bucket is a target the evaluator does not yet log. Real transcript collections vary enormously in length; a single aggregate over a benchmark where every case is maximal misrepresents both quality and cost, and the long tail dominates the cost side.

**By corpus.** Each source is a slice of its own, so the contribution of every corpus, and of the non-commercial ones in particular, can be read off.

## Reporting rules

**Training and evaluation are research-only and use every record regardless of licence track** (CLAUDE.md, "Dataset licences"). Results are reported once; the per-corpus slices make the non-commercial sources' share visible. What the tracks restrict is the release of labelled data, never what a model is trained or scored on.

**Two golds, named in every table, and nothing else (D61, 2026-09-29).** `gold = teacher` is the headline: the training labeller's own label on every benchmark pair (qwen3.8 for general_qa, Sonnet for the rare-event families), scored over all 20,700 pairs (17,520 of them with a clean rendering); a number against it reads as imitation of the teacher, and it is read against the majority-answer floor. `gold = opus` is Opus's label on the 4,012 pairs Opus labelled (the general_qa pairs where the teacher and a second local labeller disagreed, four general_qa pairs that had no second opinion, and the family pairs the teacher called `fail`), a selected subset whose numbers compare arms with each other and are never compared with the teacher numbers; it carries two floors, the Opus majority and the teacher's majority applied to those pairs. There is no human gold slice; the gold is model-generated and the DATASHEET's limitations say so. No model that wrote training labels is ever evaluated on this benchmark.

**Never tune on the benchmark.** Iterate on `val.jsonl`; the benchmark is looked at to report, not to decide. A benchmark that has been used for model selection has become a dev set, silently, and there is no way to undo it or to detect it later.

**Cost is a gate, not a footnote (a target, not yet enforced: the evaluator logs no cost, and the records state wall-clock and throughput beside a win rather than cost per case).** Report tokens/s and cost per case per length bucket, and require an arm to be **at least as cheap per case as the control it beats** before the result is written up as a win. Efficiency is half the reason for preferring a smaller encoder over a prompted frontier model, so a quality-only gate would certify a checkpoint that is slower and more expensive than the thing it replaces, which is not a result anyone can use.

## Mechanics

- Evaluation logs to the **same wandb run id** as the training job that produced the checkpoint, so quality and training curve stay attached.
- Results tables are **verified against the wandb API**: every reported number is read from the run's `metrics.json` and checked against its `bench/` summary (the current practice is a verification script over both; generated tables are the target). This is what stops the numbers in the README drifting from the numbers that were actually measured.
- Record the benchmark version with every result (`bench/gold_version` in the wandb summary today; a versioned artifact is the target). Comparing a model scored on benchmark v1 against one scored on v2 is meaningless and easy to do by accident.
- Baselines are re-run whenever the benchmark version changes. Do not carry a baseline number forward across benchmark revisions.

## Honesty

If a checkpoint loses to its baseline, that is the result. Write it down, log it, and say so plainly. The architecture hypothesis being wrong is a publishable finding; a quietly dropped negative result is not.
