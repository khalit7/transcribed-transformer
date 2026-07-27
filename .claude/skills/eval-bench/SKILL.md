---
name: eval-bench
description: Run the compliance-QA benchmark and produce the results table. Use when evaluating any checkpoint or baseline, comparing arms, or regenerating the results table in the README. Covers the metric definitions, the two-track reporting rule and the label-provenance rules.
---

# Running the benchmark

## What gets measured

Three outputs per question, scored separately. A model that gets the answer right for the wrong reason is a failure mode this benchmark exists to catch.

**Answer** — 4-way over `pass | fail | partial_pass | NA`. Report accuracy and macro-F1. Macro-F1 matters because the classes are heavily imbalanced and accuracy alone flatters a model that always predicts the majority class. Report the confusion matrix for the three priority question families (vulnerability, complaint, dissatisfaction) separately from the aggregate.

**Evidence** — set-level precision, recall and F1 over line indices. Additionally report **sufficiency** and **comprehensiveness** from the rationale-faithfulness literature: does the answer survive on the selected lines alone, and does it change when they are removed. These catch a model that produces plausible-looking evidence unconnected to its own decision.

**Summary** — faithfulness to the cited evidence, judged against a rubric. A summary that is fluent, correct-sounding and not entailed by the lines it cites scores zero. LLM judges have known failure modes here, notably truncation on long inputs silently flattering models that fail late in a transcript; verify the judge sees the whole input before trusting any ranking.

## Reporting rules

**Both licence tracks, always.** Every headline result appears twice: Track P (permissive, commercially portable) and Track NC (research-only, better data). Reporting only the better number hides the cost of the licence restriction, which is one of the project's actual findings.

**Gold slice is the headline.** Silver labels come from LLM consensus and are dev signal only. Beating a frontier model on labels that a frontier model produced is partly circular, and the human-annotated gold slice is the number that answers the question honestly. State inter-annotator agreement alongside it.

**Held-out questions separately.** Zero-shot generality is the claim, so questions never seen in training are reported as their own column. Aggregate numbers that mix seen and unseen questions overstate the capability.

**Cost alongside quality.** Report tokens/s and cost per case next to accuracy. A model that wins on quality while being slower and more expensive than the incumbent has not necessarily won.

## Mechanics

- Evaluation logs to the **same wandb run id** as the training job that produced the checkpoint, so quality and training curve stay attached.
- Results tables are **generated from the wandb API** filtered on the run tags, not hand-written. This is what stops the numbers in the README drifting from the numbers that were actually measured.
- Record the benchmark version (a wandb artifact) with every result. Comparing a model scored on benchmark v1 against one scored on v2 is meaningless and easy to do by accident.
- Baselines are re-run whenever the benchmark version changes. Do not carry a baseline number forward across benchmark revisions.

## Honesty

If a checkpoint loses to its baseline, that is the result. Write it down, log it, and say so plainly. The architecture hypothesis being wrong is a publishable finding; a quietly dropped negative result is not.
