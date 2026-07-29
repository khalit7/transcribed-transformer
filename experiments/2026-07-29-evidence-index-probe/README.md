# P1a: can a generative model emit valid evidence line indices?

**Date**: 2026-07-29 (hypothesis registered; not yet run)
**Arm**: P1a (a probe, not a research arm)
**wandb**: `tt-heads`, runs tagged `arm-p1a`
**Status**: hypothesis only. No result. Nothing below the Setup section is filled in.

## Why this exists

The README makes three structural arguments for an encoder trunk. The second of them,
that per-line evidence tagging avoids the arithmetic a generative model has to do, is the
only one that could turn out to be worth little, and it is cheap to test. So it gets
tested before any arm is trained on the assumption that it holds.

The task: a transcript is rendered as numbered lines, and the model must return the line
numbers that support an answer. For an encoder this is a per-line binary decision on
pooled line representations, with no counting anywhere. For a decoder it is a
copy-and-count problem: locate the relevant lines, recover their integer labels, and emit
them as well-formed JSON, over a document that may be tens of thousands of tokens long.

If small open decoders do this reliably, the argument is weak and should be softened in
the README. If they degrade with length, the encoder's tagging head is doing real work and
the probe produces the public measured number that says so.

## Hypothesis

Evidence-index validity for a generative model **degrades measurably with transcript
length**, and does so faster for a small open decoder than for a frontier model. An
encoder-style per-line tagger has no length-dependent failure of this kind, because it
never represents a line number as a token it has to produce.

**Prediction**: at short transcripts (under ~1k tokens) both decoders emit in-range,
well-typed, de-duplicated indices in the large majority of generations. By ~8k tokens the
small decoder's validity rate has fallen substantially, with out-of-range indices the
dominant failure. The frontier model degrades more slowly but not negligibly.

**Disconfirming result**: the small decoder holds a high validity rate flat across the
whole length range. That would mean index emission is not a real obstacle at this scale,
and the second structural argument for the encoder should be withdrawn from the README
rather than restated more carefully.

Recording this now, before running anything, because a prediction written after seeing the
numbers is not a prediction. Either outcome gets written up here.

## Setup

Not yet run. Planned:

- Public Track P transcripts only, sampled across a wide length range, rendered with
  `Transcript.render()` so the numbering matches exactly what the rest of the project uses.
- All three render styles (`colon`, `bracket`, `dotted`) so the result is not an artefact
  of one surface form.
- A small open decoder served locally, and a frontier API model, on identical inputs.
- Questions drawn from the draft question bank, with their answer options supplied as
  input, in the same shape the benchmark will use.
- Scored with the format-validity gates from the `eval-bench` skill: JSON array of JSON
  integers, 1-based, within `[1, N]`, de-duplicated, ascending, `[]` rather than a sentinel
  when nothing is found. Judgement quality is **not** scored here; this probe is only about
  whether the output is well formed and in range.
- Reported as validity rate against transcript length, broken down by failure mode
  (out of range, wrong type, duplicated, unsorted, sentinel value, unparseable).

## Result

Not yet run.

## Verdict

Not yet run.
