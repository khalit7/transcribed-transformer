"""Format validity: is a model's output well formed, separately from whether it is right.

Scored strictly and deliberately without repair. A normalisation layer downstream
of a model hides its raw error rate, so anything that has to be rescued is
recorded as a rescue, never as a success.

Two failure classes are kept apart because they have different fixes:

- **Structural** — not JSON, wrapped in a code fence, missing a required key,
  evidence that is a string rather than a list of integers. A grammar-constrained
  decoder makes these unreachable.
- **Referential** — well-typed integers that do not point at a line that exists,
  or duplicate, or arrive out of order. Constrained decoding does *not* fix
  these: the output is valid JSON either way. This is the class the evidence-index
  probe exists to measure, because it is the one an encoder's per-line tagging
  head avoids by construction rather than by learning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum

_FENCE_RE = re.compile(r"^\s*```")


class AnswerOutcome(StrEnum):
    """How an emitted answer string relates to the permitted values."""

    EXACT = "answer_exact"
    """Byte-equal to a permitted value. The only full credit."""

    RECOVERED = "answer_recovered"
    """Only recoverable by matching a gloss back to its value. A failure, with partial credit."""

    INVALID = "answer_invalid"
    """Not recoverable at all."""


ANSWER_CREDIT: dict[AnswerOutcome, float] = {
    AnswerOutcome.EXACT: 1.0,
    AnswerOutcome.RECOVERED: 0.3,
    AnswerOutcome.INVALID: 0.0,
}

SENTINELS = frozenset({"-1", "0", "na", "n/a", "none", "null", ""})
"""Values models reach for instead of an empty list. All are failures."""


@dataclass(frozen=True)
class EvidenceReport:
    """The outcome of parsing one evidence field."""

    parsed: list[int] | None
    """The integers, if the field was structurally valid. ``None`` if it was not."""

    failures: list[str] = field(default_factory=list)
    """Gate names that failed, in the order checked."""

    in_range_fraction: float | None = None
    """Fraction of emitted indices pointing at a line that exists. ``None`` if nothing parsed."""

    @property
    def structurally_valid(self) -> bool:
        return self.parsed is not None

    @property
    def fully_valid(self) -> bool:
        return not self.failures


def parse_json_object(raw: str) -> tuple[dict[str, object] | None, list[str]]:
    """Parse a model's raw output as a bare JSON object.

    No fence stripping and no repair, on purpose: the question is what the model
    emitted, not what a tolerant parser could rescue.
    """
    failures: list[str] = []
    if _FENCE_RE.match(raw):
        failures.append("no_fence")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        failures.append("valid_json")
        return None, failures
    if not isinstance(value, dict):
        failures.append("valid_json")
        return None, failures
    return value, failures


def score_evidence(value: object, n_lines: int, *, cap: int | None = None) -> EvidenceReport:
    """Apply the evidence gates to one ``evidence`` field.

    ``n_lines`` is the number of rendered lines, so valid indices are 1-based
    within ``[1, n_lines]``.
    """
    failures: list[str] = []

    # A bare scalar, or a string that merely looks like a list, is a common and
    # silent failure: a presence-only check passes it and it then yields nothing.
    if not isinstance(value, list):
        failures.append("evidence_typed")
        if isinstance(value, str) and value.strip().lower() in SENTINELS:
            failures.append("evidence_no_sentinel")
        return EvidenceReport(parsed=None, failures=failures)

    # bool is a subclass of int in Python and is never a line number.
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        failures.append("evidence_typed")
        if any(isinstance(x, str) and x.strip().lower() in SENTINELS for x in value):
            failures.append("evidence_no_sentinel")
        return EvidenceReport(parsed=None, failures=failures)

    items: list[int] = list(value)

    if any(str(x).lower() in SENTINELS for x in items):
        failures.append("evidence_no_sentinel")

    in_range = [1 <= x <= n_lines for x in items]
    fraction = (sum(in_range) / len(in_range)) if in_range else 1.0
    if not all(in_range):
        failures.append("evidence_in_range")

    if items != sorted(set(items)):
        failures.append("evidence_dedup_sorted")

    if cap is not None and len(items) > cap:
        failures.append("evidence_cardinality")

    return EvidenceReport(parsed=items, failures=failures, in_range_fraction=fraction)


def score_answer(
    raw_answer: object, permitted: list[str], glosses: dict[str, str]
) -> AnswerOutcome:
    """Classify an emitted answer against the values the question supplied.

    ``glosses`` maps each permitted value to its grading rule, which is what makes
    ``RECOVERED`` detectable: emitting the rule instead of the label it describes
    is the near miss worth measuring separately.
    """
    if not isinstance(raw_answer, str):
        return AnswerOutcome.INVALID

    if raw_answer in permitted:
        return AnswerOutcome.EXACT

    stripped = raw_answer.strip().strip("\"'").rstrip(".")
    lowered = stripped.lower()

    for value in permitted:
        if lowered == value.strip().lower():
            return AnswerOutcome.RECOVERED
        # "Answer (Fail): ..." and "<value>: <criteria>" wrappers.
        if lowered.startswith(f"answer ({value.lower()})") or lowered.startswith(
            f"{value.lower()}:"
        ):
            return AnswerOutcome.RECOVERED

    for value, gloss in glosses.items():
        if gloss and lowered.startswith(gloss.strip().lower().rstrip(".")[:40]):
            return AnswerOutcome.RECOVERED
        if value.lower() in lowered:
            return AnswerOutcome.RECOVERED

    return AnswerOutcome.INVALID
