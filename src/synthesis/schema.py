"""Benchmark record types.

On disk the benchmark is `labelled_data.jsonl` (one LabelledRecord per label: the
transcript, the question, the label, and how it was generated) plus
`questions.jsonl`, which is derived from it (the bank in code + every question
written for a call). `Case` is the in-memory build type the generator works on.

A transcript is one call in one or more line-aligned variants: every variant
has the same number of lines and line i is the same turn in each, so one
evidence key (1-based line numbers) serves all of them. Speaker tags are
`SPEAKER_NN`; the role behind each tag is hidden metadata, never rendered.

Evidence keys are never guaranteed complete, injected labels included: the
inserted lines certainly support the answer, but the original call may contain
further supporting lines that were not checked for. Score evidence as precision
against the key; recall is not measurable on this benchmark.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Answer = Literal["pass", "fail", "partial_pass", "NA"]
Family = Literal["vulnerability", "complaint_and_eod", "general_qa"]
Mode = Literal["as_is", "injected"]
Track = Literal["track-p", "track-nc"]


class Option(BaseModel):
    value: Answer
    criteria: str


class Question(BaseModel):
    id: str
    source: Literal["bank", "written"]  # reusable bank question, or written for one call
    family: Family
    text: str
    description: str = ""
    options: list[Option]

    @model_validator(mode="after")
    def _at_least_pass_fail(self) -> "Question":
        values = self.values
        if "pass" not in values or "fail" not in values:
            raise ValueError(f"{self.id}: options must include pass and fail")
        if len(values) != len(set(values)):
            raise ValueError(f"{self.id}: duplicate option values")
        return self

    @property
    def values(self) -> list[str]:
        return [o.value for o in self.options]


class Variant(BaseModel):
    kind: Literal["clean", "messy"]
    origin: str  # which layer produced these lines, e.g. "verbatim", "whisper large-v3 time-aligned"
    lines: list[str]  # "SPEAKER_NN: text", one turn per line


class Transcript(BaseModel):
    variants: list[Variant]
    speaker_roles: dict[str, str]  # hidden: SPEAKER_NN -> agent | customer | other
    tag_policy: str  # how tags were assigned to roles: random | agent_first

    @model_validator(mode="after")
    def _aligned(self) -> "Transcript":
        n = {len(v.lines) for v in self.variants}
        if len(n) != 1:
            raise ValueError(f"variants are not line-aligned: {n}")
        return self

    @property
    def n_lines(self) -> int:
        return len(self.variants[0].lines)

    def lines(self, kind: str = "clean") -> list[str]:
        v = next((v for v in self.variants if v.kind == kind), self.variants[0])
        return v.lines


class Case(BaseModel):
    """A call ready for labelling (in memory only)."""
    id: str
    track: Track
    source_id: str  # exactly where the call came from: corpus/config/locale/document
    transcript: Transcript
    meta: dict = {}


class Label(BaseModel):
    answer: Answer
    evidence: list[int]  # 1-based, ascending, de-duplicated; empty when nothing supports
    summary: str

    @model_validator(mode="after")
    def _evidence_shape(self) -> "Label":
        if self.evidence != sorted(set(self.evidence)):
            raise ValueError("evidence must be ascending and de-duplicated")
        if any(e < 1 for e in self.evidence):
            raise ValueError("evidence line numbers are 1-based")
        return self


class Generation(BaseModel):
    name: str  # model passed to claude -p
    mode: Mode  # injected: turns were written into the call so the answer holds (by construction);
    #             as_is: the call was judged as it stood (LLM judgement)
    cost_usd: float
    timestamp: str


class LabelledRecord(BaseModel):
    id: str  # "<call_id>::<question_id>"
    source_id: str
    track: Track
    question: Question
    transcript: Transcript
    label: Label
    generation_info: Generation
    meta: dict = {}

    @classmethod
    def build(cls, case: Case, q: Question, label: Label, gen: Generation) -> "LabelledRecord":
        return cls(id=f"{case.id}::{q.id}", source_id=case.source_id, track=case.track, question=q,
                   transcript=case.transcript, label=label, generation_info=gen, meta=case.meta)
