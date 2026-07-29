# P1a: can a generative model emit valid evidence line indices?

**Date**: 2026-07-29
**Arm**: P1a (a probe, not a research arm)
**wandb**: [`tt-heads`](https://wandb.ai/khalit7-/tt-heads), runs tagged `arm-p1a`
**Status**: Qwen3-1.7B and Qwen3-8B complete, on identical items. Frontier baseline **not run** (no API key available).

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

*Registered before the run.*

Evidence-index validity for a generative model **degrades measurably with transcript
length**, and does so faster for a small open decoder than for a frontier model. An
encoder-style per-line tagger has no length-dependent failure of this kind, because it
never represents a line number as a token it has to produce.

**Prediction**: at short transcripts (under ~1k tokens) both decoders emit in-range,
well-typed, de-duplicated indices in the large majority of generations. By ~8k tokens the
small decoder's validity rate has fallen substantially, **with out-of-range indices the
dominant failure**. The frontier model degrades more slowly but not negligibly.

**Disconfirming result**: the small decoder holds a high validity rate flat across the
whole length range. That would mean index emission is not a real obstacle at this scale,
and the second structural argument for the encoder should be withdrawn from the README
rather than restated more carefully.

## Setup

- **Corpus**: AMI (Track P, CC BY 4.0), human verbatim transcripts.
- **Length is manipulated, not sampled.** Each meeting is truncated at whole-turn
  boundaries to a target token budget, so the same source material appears at several
  lengths. Sampling naturally-short and naturally-long meetings instead would confound
  length with whatever else differs between them.
- 7 length buckets × 8 items = **56 generations per model**, seed 0. Both models saw the
  **identical item set**, asserted rather than assumed.
- All three render styles (`colon`, `bracket`, `dotted`) rotated across items, so a result
  cannot be an artefact of one surface form.
- **Qwen3-1.7B** and **Qwen3-8B**, bf16, greedy decoding, `max_new_tokens=512`,
  `enable_thinking=False`. Neither emitted a thinking block.
- Scored on the format-validity gates from the `eval-bench` skill: bare JSON object, no
  fence, evidence a JSON array of JSON integers, 1-based, within `[1, N]`, de-duplicated,
  ascending, `[]` rather than a sentinel when empty, at most 15 items.
- **Judgement quality is not scored.** This probe asks only whether the output is well
  formed and in range.
- Generation time: 1.7 min (1.7B), 3.3 min (8B).

## Result

Fully-valid rate, and the longest consecutive run of line numbers in any output:

| target tokens | mean lines | 1.7B valid | 8B valid | 1.7B max run | 8B max run |
|---:|---:|---:|---:|---:|---:|
| 512 | 28 | **100%** | 75% | 8 | 23 |
| 1,024 | — | 88% | 75% | 13 | 38 |
| 2,048 | 84 | 75% | 62% | 31 | 86 |
| 4,096 | — | 62% | 50% | 52 | 43 |
| 8,192 | 322 | 38% | 25% | 90 | 46 |
| 16,384 | — | 38% | 25% | 3 | 50 |
| 28,000 | 415 | **38%** | **50%** | 51 | 58 |
| **overall** | | **62%** (35/56) | **52%** (29/56) | | |

**Indices were in range 100% of the time, at every length, for both models. Not one
out-of-range index in 112 generations.**

Validity by render style (1.7B): colon 14/21, bracket 12/21, dotted 9/14. No style is
obviously worse; the sample is far too small to rank them.

### The actual failure mode, which is not the predicted one

The model does not lose track of the index space. It **stops selecting lines and starts
enumerating them**, emitting a contiguous run counting upward:

```
"evidence": [112, 113, 114, 116, 117, 119, 121, ... 162, 163, 164]     (164 = last line)
```

The longest consecutive run in an output grows with input length, from a mean of 2.5 at
512 tokens to 24.2 at 8k, with a maximum of 90. Coverage reaches 65% of all lines in the
transcript. These outputs are well-typed, ascending, de-duplicated and in range, so
**every gate except cardinality passes them**. Only the run-length diagnostic catches it.

From there the mechanism is mechanical, and confirmed rather than inferred: **all 13 JSON
failures hit the 512-token output cap exactly.** The enumeration overruns the output
budget and the JSON is cut off mid-integer:

```
... 148, 149, 150, 151, 1
```

So the two visible failure classes are one failure. Whether a degenerate enumeration shows
up as `evidence_cardinality` or as `valid_json` depends only on whether it happened to
finish inside the budget.

### Scale did not help

The 8B model is **not better than the 1.7B model** on this task. It is slightly worse
overall (52% vs 62%), worse in six of seven buckets, and it enumerates *more*: 18
cardinality failures against 8, and longer runs at short inputs (max run 23 at 512 tokens,
where the 1.7B managed 8). 4.7x the parameters bought nothing here.

The direction is within noise at n=8 per bucket, so the honest claim is not "8B is worse"
but the weaker and still useful one: **this does not improve with scale across the range
tested.** Whatever is happening is not obviously a capacity problem.

## Verdict

**Hypothesis confirmed in direction. Prediction wrong on mechanism. The size claim is
unsupported.**

Validity falls monotonically from 100% at 512 tokens to 38% at 8k and stays there out to
28k for the 1.7B, and from 75% to 25% for the 8B. The disconfirming result did not occur:
validity is emphatically not flat for either model.

The registered hypothesis also claimed degradation would be *faster for a smaller model
than a frontier one*. Against the only comparison available, that is not supported: the
larger of the two open models degraded slightly faster, not slower. The frontier half of
that comparison was never run.

But the predicted mechanism was wrong, and it is worth being precise about how. I expected
the model to emit indices pointing at lines that do not exist. It never did that once. The
index space is not what degrades. What degrades is the model's ability to *select*, and its
fallback is to enumerate, which then collides with the output budget.

That distinction matters for what to build:

- The encoder's per-line tagging head still avoids this, but for a reason worth stating
  correctly. It is not that tagging gets the arithmetic right where generation gets it
  wrong. It is that tagging emits a **fixed-size** output, one decision per line, so there
  is no variable-length list to run past a budget, and no fallback of counting upward when
  selection fails.
- **Constrained decoding would not fix this.** A grammar forcing a JSON array of integers
  in `[1, N]` accepts `[112, 113, ..., 164]` happily. The failure is semantic, not
  syntactic. This is a concrete answer to an open question the plan flagged, and the answer
  is that the fine-tune, or the architecture, has to carry it.
- The cardinality cap is doing more work than it looks. Without it, a degenerate
  enumeration that fits in the budget scores as fully valid.

## Caveats, and they are substantial

1. **Two models, one family.** Both are Qwen3. A shared quirk of one model family's
   post-training would look exactly like this. Testing a second family is the cheapest way
   to find out and has not been done.
2. **No frontier baseline.** No API key was available. Without it there is no reference for
   how much of this is inherent to generation and how much a stronger model would absorb,
   which was half the designed comparison.
3. **56 generations per model.** Per-bucket n is 8. The monotone trend across seven buckets
   is more convincing than any single cell, but the cells are noisy: the 16k bucket behaves
   differently from its neighbours, and 28k is non-monotone for both models. Do not read
   individual cells.
4. **AMI is multi-party meetings**, 4–5 speakers, not two-party advisor/customer calls.
5. **`max_new_tokens=512` is a chosen budget**, matching the project's stated output budget.
   A larger budget would convert some JSON failures into cardinality failures. It would not
   make the enumeration behaviour go away, which is the finding.
6. This measures format validity only. A model could pass every gate here and still cite
   entirely irrelevant lines.

## Follow-ups

- **Test a second model family.** The single most valuable next run, because it separates
  "generative models do this" from "Qwen3 does this", and that distinction decides how much
  weight the README argument can carry.
- Add a frontier baseline when a key is available.
- Report the run-length diagnostic as a standard evidence metric in `eval-bench`, since it
  catches a degenerate output that every other gate passes.
- Consider whether the benchmark's evidence cap should be stated in the prompt at all, or
  enforced at decode time, and measure both.
- Re-run at a larger `max_new_tokens` to confirm the prediction implied here: the JSON
  failures should convert into cardinality failures, leaving the enumeration behaviour
  unchanged. If they do not, the mechanism above is wrong.
