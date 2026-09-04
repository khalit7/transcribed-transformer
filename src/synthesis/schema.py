"""Labelled-data record types.

On disk, `data/labelled_data/labelled_data.jsonl` holds one LabelledRecord per
label (the transcript, the question, the label, and how it was generated) and
`questions.jsonl` is derived from the bank in code. The same records serve
training and benchmarking; how they are split is decided downstream, not here.
`Case` is the in-memory build type the generator works on.

A transcript is one call in one or more line-aligned variants: every variant
has the same number of lines and line i is the same turn in each, so one
evidence key (1-based line numbers) serves all of them. Every line is
`<role>: text` where the role is the corpus's own speaker label, verbatim
(AppTek `agent`/`customer`, Taskmaster `assistant`/`user`, ACI-Bench
`doctor`/`patient`); a corpus without role labels cannot be built until a policy
for it is decided (cases.NoSpeakerRoles).

Labels are produced by LLM labelling of real calls (the labeller answers; a
separate LLM-as-a-judge step, when run, assesses the labels). Evidence keys are never guaranteed
complete: score evidence as precision against the key; a label's optional
`ablation` record says whether its cited lines were necessary and sufficient
for the answer, and its optional `verification` record holds a second labeller's
blind verdict.
"""

from typing import Literal

from pydantic import BaseModel, model_validator

Answer = Literal["pass", "fail", "partial_pass", "NA"]
Family = Literal["vulnerability", "complaint", "eod", "general_qa"]  # eod = expression of dissatisfaction
Track = Literal["track-p", "track-nc"]


class Option(BaseModel):
    value: Answer
    criteria: str


class Question(BaseModel):
    id: str
    source: Literal["bank", "written"] = "bank"  # only bank questions exist in the current flow
    family: Family
    text: str
    description: str = ""
    options: list[Option]
    # Optional closed vocabulary the answer must be qualified with (e.g. the
    # vulnerability characteristics present). Empty = the question takes no tags.
    tags: list[str] = []
    # Datasets (source names as in cases.BUILDERS) this question may be asked of.
    # Service-call conduct questions list the agent/customer corpora; role-agnostic
    # conversation questions list podcasts too.
    dataset_allow_list: list[str]

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
    lines: list[str]  # "<role>: text", one turn per line, role verbatim from the corpus


class Transcript(BaseModel):
    variants: list[Variant]
    speakers: list[str]  # the role labels that occur, verbatim from the corpus, in order of first appearance
    role_source: str  # where the labels come from, e.g. "corpus role annotation (agent/customer)"

    @model_validator(mode="after")
    def _aligned(self) -> "Transcript":
        n = {len(v.lines) for v in self.variants}
        if len(n) != 1:
            raise ValueError(f"variants are not line-aligned: {n}")
        seen = {ln.partition(": ")[0] for v in self.variants for ln in v.lines}
        if not seen <= set(self.speakers):
            raise ValueError(f"lines carry speaker labels not declared in speakers: {seen - set(self.speakers)}")
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
    dataset: str  # source name as in cases.BUILDERS: apptek | taskmaster | aci_bench | sporc
    track: Track
    source_id: str  # exactly where the call came from: corpus/config/locale/document
    transcript: Transcript
    meta: dict = {}


class Label(BaseModel):
    answer: Answer
    evidence: list[int]  # 1-based, ascending, de-duplicated; empty when nothing supports
    summary: str
    tags: list[str] = []  # qualifiers drawn from the question's tag vocabulary, when it has one
    confidence: float | None = None  # the labeller's own 0-1 confidence; routes low values to human audit

    @model_validator(mode="after")
    def _evidence_shape(self) -> "Label":
        if self.evidence != sorted(set(self.evidence)):
            raise ValueError("evidence must be ascending and de-duplicated")
        if any(e < 1 for e in self.evidence):
            raise ValueError("evidence line numbers are 1-based")
        return self


class Verification(BaseModel):
    """A second labeller's blind answer to the same (call, question)."""
    model: str
    answer: Answer
    evidence: list[int]
    tags: list[str] = []
    agrees: bool  # answer matches the primary label


class Ablation(BaseModel):
    """Does the cited evidence carry the answer? Re-labelled with the cited lines
    removed (necessary if the answer changes) and with only the cited lines kept
    (sufficient if the answer holds)."""
    model: str
    necessary: bool | None
    sufficient: bool | None


class Generation(BaseModel):
    name: str  # backend:model, e.g. claude:sonnet or ollama:qwen3:32b
    labelled_variant: str  # which transcript variant the labeller saw: clean | messy
    cost_usd: float
    timestamp: str
    claude_account: str | None = None  # "p" | "w" for claude: labellers (which account was billed); None otherwise


class LabelledRecord(BaseModel):
    id: str  # "<call_id>::<question_id>"
    dataset: str
    source_id: str
    track: Track
    question: Question
    transcript: Transcript
    label: Label
    generation_info: Generation
    verification: Verification | None = None
    ablation: Ablation | None = None
    meta: dict = {}
