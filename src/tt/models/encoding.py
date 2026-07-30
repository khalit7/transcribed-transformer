"""Laying out a case, a question and its answer options as one model input.

The whole point of the encoder design is that this is **one forward pass**. The
question, every permitted answer with its grading rule, and the transcript are
concatenated and read together, so an option is scored in the context of the
conversation rather than compared to a separately-embedded summary of it. That is
the GLiClass shape, applied to a setting where the label set changes per example
rather than per dataset.

Two things have to survive the layout, and both are easy to lose:

**Where each option is.** Options are scored by reading the hidden state at a
marker token placed before each one. Lose those positions and there is nothing to
score. Arity varies per question, so the count is data, not a constant.

**Which tokens belong to which transcript line.** Evidence is per line, so the
evidence head pools token states within a line's span. If the span boundaries are
wrong the head is trained to tag the wrong lines, and nothing about the loss curve
would reveal it.

Truncation is the sharp edge. A transcript that does not fit must lose *whole
lines from the end*, never half a line, and the lines that survive must keep their
original numbers so evidence labels still refer to the same text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tt.data.schema import ComplianceQuestion, RenderStyle, Transcript

OPTION_MARKER = "[unused0]"
"""Token placed before each answer option, whose hidden state scores that option.

A dedicated marker rather than reusing the option's first content token: the
first token varies with the label text, so its representation would carry the
label's identity into the score and make short labels behave differently from
long ones.
"""


@dataclass(frozen=True)
class EncodedCase:
    """One model input, with the positions the heads need to find things again."""

    input_ids: list[int]
    option_positions: list[int]
    """Token index of each option's marker, in the question's option order."""

    line_spans: list[tuple[int, int]]
    """``(start, end)`` token range per surviving transcript line, end exclusive."""

    line_numbers: list[int]
    """1-based original line number for each span. Not the same as its position
    in ``line_spans`` once truncation has dropped lines."""

    n_truncated_lines: int = 0
    """Lines dropped to fit. Nonzero means the model cannot cite them at all."""

    @property
    def n_options(self) -> int:
        return len(self.option_positions)


def encode_case(
    transcript: Transcript,
    question: ComplianceQuestion,
    tokenizer: Any,
    *,
    max_length: int,
    style: RenderStyle = "colon",
) -> EncodedCase:
    """Lay out question, options and transcript as one sequence.

    Layout is ``[CLS] question description <marker> value: criteria ... [SEP]
    transcript [SEP]``. The question comes first so that truncation, which removes
    transcript lines from the end, can never remove the question or an option: a
    truncated input that has lost an option would silently change the label set
    the model is choosing from.
    """
    marker_id = _marker_id(tokenizer)
    cls_id, sep_id = _special_ids(tokenizer)

    ids: list[int] = [cls_id] if cls_id is not None else []
    ids += _encode(tokenizer, question.text)
    if question.description:
        ids += _encode(tokenizer, question.description)

    option_positions: list[int] = []
    for option in question.options:
        option_positions.append(len(ids))
        ids.append(marker_id)
        gloss = f"{option.value}: {option.criteria}"
        if option.example:
            gloss += f" e.g. {option.example}"
        ids += _encode(tokenizer, gloss)

    if sep_id is not None:
        ids.append(sep_id)

    # One token slot reserved for the trailing separator.
    budget = max_length - (1 if sep_id is not None else 0)
    if len(ids) >= budget:
        raise ValueError(
            f"question {question.id!r} needs {len(ids)} tokens before the transcript "
            f"but max_length is {max_length}. Truncating here would drop answer options "
            "and silently change the label set the model chooses from."
        )

    line_spans: list[tuple[int, int]] = []
    line_numbers: list[int] = []
    truncated = 0

    rendered = transcript.render(style=style).split("\n")
    for number, line in enumerate(rendered, start=1):
        line_ids = _encode(tokenizer, line)
        if len(ids) + len(line_ids) > budget:
            # Whole lines only. Half a line would give the evidence head a span
            # covering text the model never saw in full.
            truncated = len(rendered) - number + 1
            break
        line_spans.append((len(ids), len(ids) + len(line_ids)))
        line_numbers.append(number)
        ids += line_ids

    if sep_id is not None:
        ids.append(sep_id)

    return EncodedCase(
        input_ids=ids,
        option_positions=option_positions,
        line_spans=line_spans,
        line_numbers=line_numbers,
        n_truncated_lines=truncated,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    out: list[int] = tokenizer(text, add_special_tokens=False)["input_ids"]
    return out


def _marker_id(tokenizer: Any) -> int:
    """Id of the option marker, falling back to a rarely-used special token."""
    for candidate in (OPTION_MARKER, "[unused1]", tokenizer.sep_token, tokenizer.cls_token):
        if candidate is None:
            continue
        ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError(
        "no single-token option marker available in this tokenizer. Option scores are "
        "read from the marker's hidden state, so a multi-token marker would make the "
        "score depend on which sub-token happened to be picked."
    )


def _special_ids(tokenizer: Any) -> tuple[int | None, int | None]:
    return (
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    )
