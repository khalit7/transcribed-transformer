---
name: log-experiment
description: Record an experiment in experiments/ with its hypothesis, config, result and verdict, cross-linked to its wandb run. Use after any training or evaluation run completes, or when setting up a run so the hypothesis is committed before the result is known.
---

# Recording an experiment

wandb holds the metrics. `experiments/` holds the reasoning about them. Neither is sufficient alone: a wandb workspace six months from now is a wall of curves with no memory of what anyone was trying to find out.

The tech report is assembled from these entries, so write them as if someone else will read them.

## Write the hypothesis before the run

This is the part that gets skipped and the part that matters. A hypothesis recorded after seeing the result is not a hypothesis. Create the entry at launch with the hypothesis and prediction filled in, and the result left empty.

State what would count as the hypothesis being **wrong**. If no outcome would disconfirm it, the experiment is not measuring anything.

## Entry format

One directory per run, `experiments/<date>-<short-slug>/`, containing a `README.md` with:

- **Hypothesis** — what is believed, and why it might be true
- **Prediction** — the specific measurable outcome expected, and what result would disconfirm it
- **Setup** — arm, base checkpoint, licence track, dataset build, key hyperparameters. Point at the wandb run for the full config rather than duplicating it.
- **wandb run** — project and run id, as a link
- **Result** — what actually happened, with numbers read from wandb, not recalled
- **Verdict** — confirmed, disconfirmed, or inconclusive, and what changes as a result
- **Follow-ups** — what this makes worth trying next

## Rules

**Numbers come from wandb.** Never transcribe a metric from memory or from an earlier message in a conversation. Open the run and read it.

**Inconclusive is a legitimate verdict.** A run that crashed, was misconfigured, or produced noise indistinguishable from the baseline gets recorded as inconclusive with the reason. These entries are how the same mistake stops being repeated.

**Disconfirmation gets the same treatment as confirmation.** The two central hypotheses in this project (bidirectional attention helps; transcript pretraining helps) may turn out to be false. Those entries are among the most valuable in the repository and are written up with the same care.

**Do not retro-edit a hypothesis to match a result.** If the original hypothesis was wrong, that is what the verdict field records. Rewriting it destroys the only evidence of what was actually predicted.

## Cross-linking

Link related experiments to each other. A matched-corpus ablation is only interpretable next to its control, and the two entries should each point at the other.
