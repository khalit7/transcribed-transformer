"""Canonical data types.

Every corpus loader emits these types. Nothing downstream ever sees a
corpus-specific format. If a corpus does not fit, extend the schema
deliberately rather than special-casing the consumer.

Three invariants matter more than the rest, because breaking any of them is
silent:

**Line indices are 0-based and contiguous.** Evidence labels are line indices,
so a loader that renumbers, merges or drops turns invalidates every label
attached to that transcript. ``Transcript`` validates this on construction.

**Licence tracks never mix.** A ``Case`` whose transcripts span both tracks is
rejected, so a mixed-track training batch fails loudly at load time rather than
being discovered later when someone asks whether a model can ship.

**Answers are runtime data, not an enum.** A question carries its own list of
:class:`AnswerOption`, varying in arity, wording and polarity between question
authors. There is deliberately no ``Answer`` enum and no fixed label set
anywhere in this module: a model that has one cannot answer a question written
after it was trained, which is the whole zero-shot claim. An
:class:`Assessment`'s answer is checked byte-exact against the options its
question supplied.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Track(StrEnum):
    """Licence track. Ambiguity always resolves to ``NC``."""

    P = "track-p"
    """Permissive: commercial use allowed and derivatives redistributable."""

    NC = "track-nc"
    """Non-commercial, research-only, or unclear."""


class Role(StrEnum):
    """Speaker role, where the corpus supports the distinction.

    Most public corpora do not label an advisor/customer split, so ``UNKNOWN``
    is expected and normal. Loaders must not guess.
    """

    ADVISOR = "advisor"
    CUSTOMER = "customer"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class AnswerOption(BaseModel):
    """One permitted answer to a question, supplied as input at inference time.

    Deliberately not an enum member. Question authors differ on how many
    answers a question has, what they are called, and which side of the
    judgement is the interesting one: a two-way ``{yes, no}``, a four-way
    ``{pass, partial pass, fail, NA}`` and a coded ``{01, 02, 03}`` are all
    valid vocabularies for the same underlying task, and a question written
    tomorrow may use none of them.

    ``value`` is what a model must emit, byte for byte. ``criteria`` is the
    grading rule that makes it answerable without training on the question.
    """

    model_config = ConfigDict(frozen=True)

    value: str = Field(
        min_length=1,
        description="The label itself. Emitted verbatim, compared byte-exact, never normalised.",
    )
    criteria: str = Field(
        min_length=1, description="What has to be true of the case for this answer to apply."
    )
    example: str = Field(default="", description="Optional illustration. Often absent.")


class CaseSemantics(StrEnum):
    """How a question aggregates over the transcripts in a case.

    Held as benchmark ground truth for analysis. It is **not** given to the
    model: inferring this from the question text is part of the task.
    """

    ANY = "any"
    """Passes if any single transcript satisfies the question."""

    ALL = "all"
    """Passes only if every transcript satisfies the question."""


class Provenance(StrEnum):
    """Where a label came from. Never compare across provenances without saying so."""

    GOLD_HUMAN = "gold_human"
    """Human annotated. The headline metric."""

    SILVER_CONSENSUS = "silver_consensus"
    """Multi-model consensus. Dev signal only, and partly circular against LLM baselines."""

    MODEL_PREDICTION = "model_prediction"
    """A model's output being scored."""


RenderStyle = Literal["colon", "bracket", "dotted"]
"""Surface form of the line number in a rendered transcript."""

_RENDER_TEMPLATES: dict[RenderStyle, str] = {
    "colon": "{n}: {speaker}: {text}",
    "bracket": "[{n}] {speaker}: {text}",
    "dotted": "{n}. {speaker}: {text}",
}


class Turn(BaseModel):
    """One speaker turn. The unit that evidence line numbers refer to."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="0-based position within the transcript.")
    speaker: str = Field(description="Corpus speaker id, kept verbatim.")
    role: Role = Role.UNKNOWN
    text: str = Field(description="Verbatim text. Disfluencies are signal, never stripped.")


class Transcript(BaseModel):
    """A single call, as speaker-labelled turns."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str = Field(description="Corpus name, matching its DATASHEET entry.")
    track: Track
    turns: list[Turn]
    is_asr: bool = Field(
        description="True if produced by ASR, False if human verbatim. "
        "Only human-verbatim transcripts with audio can calibrate the ASR channel model."
    )
    asr_system: str | None = Field(
        default=None, description="Which ASR system, when is_asr is True."
    )
    channel_version: str | None = Field(
        default=None,
        description="ASR channel-model artifact version, when this text is synthetic. "
        "Comparing models trained under different channel versions is a silent confound.",
    )
    meta: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_line_indices(self) -> Transcript:
        expected = list(range(len(self.turns)))
        actual = [t.index for t in self.turns]
        if actual != expected:
            raise ValueError(
                f"transcript {self.id!r}: turn indices must be 0-based and contiguous; "
                f"got {actual[:8]}{'...' if len(actual) > 8 else ''} "
                f"for {len(self.turns)} turns. Evidence labels are line indices, so "
                f"renumbering silently invalidates them."
            )
        return self

    def render(self, *, one_based: bool = True, style: RenderStyle = "colon") -> str:
        """Serialise to the model input format.

        The single place a transcript becomes text, so every consumer agrees on
        line numbering. Defaults to 1-based on the surface because that is what
        a model is asked to emit as evidence, while indices stay 0-based
        internally. Use :meth:`line_to_index` to convert back.

        ``style`` selects the surface form of the line number. Numbered
        transcripts are rendered differently by different systems, and a model
        that only works under one of them is brittle for no reason worth
        paying for, so training and evaluation vary this rather than fixing it.
        The numbering itself is identical across styles.
        """
        off = 1 if one_based else 0
        template = _RENDER_TEMPLATES[style]
        return "\n".join(
            template.format(n=t.index + off, speaker=t.speaker, text=t.text) for t in self.turns
        )

    @staticmethod
    def line_to_index(line: int, *, one_based: bool = True) -> int:
        """Convert a rendered line number back to a 0-based turn index."""
        return line - 1 if one_based else line

    @property
    def n_turns(self) -> int:
        return len(self.turns)


class Case(BaseModel):
    """The full interaction with one customer: the unit a question is asked about."""

    model_config = ConfigDict(frozen=True)

    id: str
    transcripts: list[Transcript]
    meta: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_single_track(self) -> Case:
        tracks = {t.track for t in self.transcripts}
        if len(tracks) > 1:
            raise ValueError(
                f"case {self.id!r} mixes licence tracks {sorted(t.value for t in tracks)}. "
                "Tracks must never mix: a model trained on both is research-only, "
                "and the contamination is invisible once training has started."
            )
        if not self.transcripts:
            raise ValueError(f"case {self.id!r} has no transcripts")
        return self

    @property
    def track(self) -> Track:
        return self.transcripts[0].track

    def transcript(self, transcript_id: str) -> Transcript:
        for t in self.transcripts:
            if t.id == transcript_id:
                return t
        raise KeyError(f"case {self.id!r} has no transcript {transcript_id!r}")


class ComplianceQuestion(BaseModel):
    """A question plus the answers it permits and what each one means.

    All of it is model input, not documentation. The options are what make a
    question answerable without having been trained on it.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    description: str = Field(
        default="",
        description="Elaboration beyond the one-line question, where the author gave one.",
    )
    options: list[AnswerOption] = Field(
        description="The permitted answers, in the order the author supplied them. "
        "Arity and wording vary by question; nothing here is a fixed vocabulary."
    )
    family: str = Field(
        description="Grouping for per-family reporting, e.g. 'vulnerability', "
        "'complaint', 'dissatisfaction'. Also the unit the train/test split holds "
        "out: questions within a family are often near paraphrases, so splitting "
        "below this level leaks."
    )
    semantics: CaseSemantics = Field(
        description="Ground truth for analysis only. Never shown to the model."
    )
    held_out: bool = Field(
        default=False,
        description="If True, never seen in training. Set by a family-level splitter "
        "rather than by hand, and reported as its own column: mixing seen and unseen "
        "questions overstates zero-shot generality.",
    )

    @model_validator(mode="after")
    def _check_options(self) -> ComplianceQuestion:
        if len(self.options) < 2:
            raise ValueError(
                f"question {self.id!r} has {len(self.options)} answer option(s); "
                "at least two are needed for there to be a judgement to make."
            )
        values = [o.value for o in self.options]
        stripped = [v.strip() for v in values]
        if any(not v for v in stripped):
            raise ValueError(
                f"question {self.id!r} has an answer option whose value is blank once "
                "stripped. Values are compared byte-exact, so whitespace-only labels "
                "are unanswerable."
            )
        duplicates = {v for v in values if values.count(v) > 1}
        if duplicates:
            raise ValueError(
                f"question {self.id!r} has duplicate answer values {sorted(duplicates)}. "
                "Options are identified by value, so a duplicate makes the answer ambiguous."
            )
        return self

    @property
    def values(self) -> tuple[str, ...]:
        """The permitted answer strings, in author order."""
        return tuple(o.value for o in self.options)


class Evidence(BaseModel):
    """A reference to one line of one transcript within a case."""

    model_config = ConfigDict(frozen=True)

    transcript_id: str
    index: int = Field(ge=0, description="0-based turn index.")


class Assessment(BaseModel):
    """One question answered about one case: the task's output triple."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question_id: str
    answer: str = Field(
        description="Byte-exact copy of one of the question's option values. "
        "Not normalised, not case-folded: a near miss is the failure being measured."
    )
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_exhaustive: bool = Field(
        default=False,
        description="True only if this lists *every* line that supports the answer. "
        "Human evidence keys are normally partial, because a question can have "
        "several genuinely correct supporting lines and an annotator marks the one "
        "they noticed. Scoring recall against a partial key understates every model, "
        "so the scorer reads this flag before computing recall at all. False is the "
        "safe default and the usual case.",
    )
    summary: str = ""
    provenance: Provenance
    meta: dict[str, str] = Field(default_factory=dict)

    def validate_against(self, case: Case, question: ComplianceQuestion) -> None:
        """Check this assessment is well formed against the case and question it answers.

        Not a pydantic validator because it needs both. Call it wherever
        assessments meet their inputs: at benchmark build time, and when parsing
        model output, where an out-of-range line number and an answer that is a
        near miss for a permitted value are both known failure modes.

        The answer check is byte-exact by design. Accepting ``"Pass "`` or
        ``"pass"`` for ``"Pass"`` here would hide exactly the behaviour the
        benchmark exists to measure; a scorer that wants to award partial credit
        for a recoverable near miss does so explicitly, and records it as a
        failure rather than a success.
        """
        if self.case_id != case.id:
            raise ValueError(f"assessment targets case {self.case_id!r}, got {case.id!r}")
        if self.question_id != question.id:
            raise ValueError(
                f"assessment targets question {self.question_id!r}, got {question.id!r}"
            )
        if self.answer not in question.values:
            raise ValueError(
                f"answer {self.answer!r} is not one of the values permitted by question "
                f"{question.id!r}: {list(question.values)}. The answer must be a byte-exact "
                "copy of a supplied option value, not its grading rule, not a paraphrase, "
                "and not a wrapper around it."
            )
        for ev in self.evidence:
            transcript = case.transcript(ev.transcript_id)
            if ev.index >= transcript.n_turns:
                raise ValueError(
                    f"evidence line {ev.index} is out of range for transcript "
                    f"{ev.transcript_id!r} ({transcript.n_turns} turns)"
                )
