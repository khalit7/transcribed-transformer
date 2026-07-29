# P1a: can a generative model emit valid evidence line indices?

**Date**: 2026-07-29
**Arm**: P1a (a probe, not a research arm)
**wandb**: [`tt-heads`](https://wandb.ai/khalit7-/tt-heads), runs tagged `arm-p1a`
**Status**: Qwen3-1.7B, Qwen3-8B and Mistral-7B-Instruct-v0.3 complete, on identical items. Frontier baseline **not run** (no API key available).

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

- **Corpus**: AMI (Track P, CC BY 4.0).

> **Methodology correction, 2026-07-29.** The first three runs are **not** comparable to
> each other and the earlier version of this file wrongly said they were. AMI's default
> transcript layer was changed from `manual` to `asr` partway through the session, between
> the two Qwen runs and the Mistral run. That changed which meetings the item builder
> sampled, so Mistral saw a **different item set** *and* a different transcript layer. Two
> confounds at once.
>
> Everything below is therefore the **re-run on the ASR layer for all three models**, which
> is also the correct experiment: ASR output is the tier 1 target distribution, and the
> earlier runs used human-verbatim text the project treats as a substitute for it. The
> superseded manual-layer results are kept in `data/processed/index_probe/` and the ASR
> results in `data/processed/index_probe_asr/`.
>
> One result survives the correction untouched, because it does not depend on items being
> matched: **zero out-of-range indices in 168 generations** across the original three runs.
- **Length is manipulated, not sampled.** Each meeting is truncated at whole-turn
  boundaries to a target token budget, so the same source material appears at several
  lengths. Sampling naturally-short and naturally-long meetings instead would confound
  length with whatever else differs between them.
- 7 length buckets × 8 items = **56 generations per model**, seed 0, **168 in total**. All
  three models saw the identical item set, asserted in code rather than assumed.
- All three render styles (`colon`, `bracket`, `dotted`) rotated across items, so a result
  cannot be an artefact of one surface form.
- **Qwen3-1.7B**, **Qwen3-8B** and **Mistral-7B-Instruct-v0.3**, bf16, greedy decoding,
  `max_new_tokens=512`, `enable_thinking=False` where supported. No thinking blocks emitted.
- Scored on the format-validity gates from the `eval-bench` skill: bare JSON object, no
  fence, evidence a JSON array of JSON integers, 1-based, within `[1, N]`, de-duplicated,
  ascending, `[]` rather than a sentinel when empty, at most 15 items.
- **Judgement quality is not scored.** This probe asks only whether the output is well
  formed and in range.
- Generation time: roughly 2, 4 and 5 minutes respectively.

## Result

All three models on **identical items**, on the ASR layer (asserted, not assumed):

| target tokens | Qwen3-1.7B | Qwen3-8B | Mistral-7B-Instr |
|---:|---:|---:|---:|
| 512 | 88% | 88% | 88% |
| 1,024 | 75% | 62% | 62% |
| 2,048 | 50% | 62% | 62% |
| 4,096 | 62% | 38% | 62% |
| 8,192 | **0%** | 25% | 38% |
| 16,384 | **0%** | 38% | 62% |
| 28,000 | **0%** | 50% | 50% |
| **overall** | **39%** (22/56) | **52%** (29/56) | **61%** (34/56) |

**Out-of-range indices: zero.** Again. 168 generations here, plus 168 in the superseded
manual-layer runs, three models, two families, two transcript layers, and not one index
pointing at a line that does not exist.

Every JSON failure hit the 512-token output cap exactly, in all three models: 24 of 24,
18 of 18, 11 of 11.

### The failure mode

The models do not lose track of the index space. They **stop selecting lines and start
enumerating them**, emitting a contiguous run counting upward:

```
"evidence": [112, 113, 114, 116, 117, 119, 121, ... 162, 163, 164]     (164 = last line)
```

Those outputs are well-typed, ascending, de-duplicated and in range, so **every gate except
cardinality passes them**. Only the run-length diagnostic catches it. The enumeration then
overruns the output budget and the JSON is cut off mid-integer:

```
... 148, 149, 150, 151, 1
```

So the two visible failure classes are one failure, and which one you see depends only on
whether the enumeration finished inside the budget.

### ASR text is markedly harder than human transcription

This is the finding that only appeared after the methodology correction, and it matters
because ASR is the distribution the project actually targets.

Qwen3-1.7B scores 38% at 8k, 16k and 28k on human-verbatim text. On ASR text of the same
corpus it scores **0% at all three**. Not degraded — extinguished. Its overall rate falls
from 62% to 39%.

Unpunctuated, misrecognised, finely-segmented text is simply harder to count positions in.
Any estimate of this task's difficulty taken from clean transcripts is optimistic.

### Scale does help, contradicting the earlier run

On the manual layer the three models were indistinguishable (62%, 52%, 61%) and this file
previously concluded that scale bought nothing. **That was an artefact of the easier text.**

On ASR text there is a clean ordering: Mistral-7B 61% > Qwen3-8B 52% > Qwen3-1.7B 39%, and
the gap widens with length precisely where it matters. The earlier conclusion is withdrawn.

## Verdict

**Hypothesis confirmed. Prediction wrong on mechanism. The size claim reverses on the
distribution that matters.**

Validity degrades sharply with length for every model tested, and the disconfirming result
did not occur: nothing held flat. On the target distribution the smallest model fails
completely beyond 8k.

The predicted mechanism was wrong, and precisely so. I expected indices pointing at
non-existent lines. In 336 generations that never happened once. The index space does not
degrade; the *selection* does, and the fallback is enumeration.

What this means for the architecture argument:

- The encoder's per-line tagging head still avoids this, but for a reason worth stating
  correctly. Not that tagging gets the arithmetic right where generation gets it wrong, but
  that tagging emits a **fixed-size** output, one decision per line, so there is no
  variable-length list to overrun a budget and no counting fallback when selection fails.
- **Constrained decoding would not fix this.** A grammar forcing a JSON array of integers
  in `[1, N]` accepts `[112, 113, ..., 164]` happily. The failure is semantic, not
  syntactic, so the architecture or the fine-tune has to carry it.
- The cardinality cap is load-bearing. Without it, a degenerate enumeration that happens to
  fit the budget scores as fully valid.

## Caveats, and they are substantial

1. **Three models, two families** — done, and the result held. What is still untested is
   whether a *frontier* model behaves differently; all three here are small open models.
2. **No frontier baseline.** No API key was available. Without it there is no reference for
   how much of this is inherent to generation and how much a stronger model would absorb,
   which was half the designed comparison.
3. **56 generations per model, 168 total.** Per-bucket n is 8, which is too few to read any
   individual cell. The zero-out-of-range result is solid because it is 168 for 168; the
   per-bucket validity rates are not.
4. **AMI is multi-party meetings**, 4–5 speakers, not two-party advisor/customer calls.
5. **`max_new_tokens=512` is a chosen budget**, matching the project's stated output budget.
   A larger budget would convert some JSON failures into cardinality failures. It would not
   make the enumeration behaviour go away, which is the finding.
6. This measures format validity only. A model could pass every gate here and still cite
   entirely irrelevant lines.

## Follow-ups

- Add a frontier baseline when a key is available. This is now the only untested axis that
  could still overturn the conclusion.
- Report the run-length diagnostic as a standard evidence metric in `eval-bench`, since it
  catches a degenerate output that every other gate passes.
- Consider whether the benchmark's evidence cap should be stated in the prompt at all, or
  enforced at decode time, and measure both.
- Re-run at a larger `max_new_tokens` to confirm the prediction implied here: the JSON
  failures should convert into cardinality failures, leaving the enumeration behaviour
  unchanged. If they do not, the mechanism above is wrong.
