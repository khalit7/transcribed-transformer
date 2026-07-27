"""Canonical data types.

Every corpus loader emits these types. Nothing downstream ever sees a
corpus-specific format. If a corpus does not fit, extend the schema
deliberately rather than special-casing the consumer.

Two invariants matter more than the rest, because breaking either is silent:

**Line indices are 0-based and contiguous.** Evidence labels are line indices,
so a loader that renumbers, merges or drops turns invalidates every label
attached to that transcript. ``Transcript`` validates this on construction.

**Licence tracks never mix.** A ``Case`` whose transcripts span both tracks is
rejected, so a mixed-track training batch fails loudly at load time rather than
being discovered later when someone asks whether a model can ship.
"""

from __future__ import annotations

from enum import StrEnum

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


class Answer(StrEnum):
    """The four possible answers to a compliance question."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL_PASS = "partial_pass"
    NA = "NA"


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

    def render(self, *, one_based: bool = True) -> str:
        """Serialise to the model input format.

        The single place a transcript becomes text, so every consumer agrees on
        line numbering. Defaults to 1-based on the surface because that is what
        a model is asked to emit as evidence, while indices stay 0-based
        internally. Use :meth:`line_to_index` to convert back.
        """
        off = 1 if one_based else 0
        return "\n".join(f"{t.index + off}: {t.speaker}: {t.text}" for t in self.turns)

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
    """A question plus the definitions of what each answer means.

    The definitions are model input, not documentation. They are what makes a
    question answerable zero-shot.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    definitions: dict[Answer, str] = Field(
        description="What constitutes each answer. All four required."
    )
    family: str = Field(
        description="Grouping for per-family reporting, e.g. 'vulnerability', "
        "'complaint', 'dissatisfaction'."
    )
    semantics: CaseSemantics = Field(
        description="Ground truth for analysis only. Never shown to the model."
    )
    held_out: bool = Field(
        default=False,
        description="If True, never seen in training. Reported as its own column, "
        "since mixing seen and unseen questions overstates zero-shot generality.",
    )

    @model_validator(mode="after")
    def _check_all_definitions(self) -> ComplianceQuestion:
        missing = set(Answer) - set(self.definitions)
        if missing:
            raise ValueError(
                f"question {self.id!r} is missing definitions for "
                f"{sorted(a.value for a in missing)}. A question without all four "
                "definitions cannot be answered zero-shot."
            )
        return self


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
    answer: Answer
    evidence: list[Evidence] = Field(default_factory=list)
    summary: str = ""
    provenance: Provenance
    meta: dict[str, str] = Field(default_factory=dict)

    def validate_against(self, case: Case) -> None:
        """Check every evidence reference resolves within ``case``.

        Not a pydantic validator because it needs the case. Call it wherever
        assessments meet cases: at benchmark build time and when parsing model
        output, where out-of-range line numbers are a known failure mode.
        """
        if self.case_id != case.id:
            raise ValueError(f"assessment targets case {self.case_id!r}, got {case.id!r}")
        for ev in self.evidence:
            transcript = case.transcript(ev.transcript_id)
            if ev.index >= transcript.n_turns:
                raise ValueError(
                    f"evidence line {ev.index} is out of range for transcript "
                    f"{ev.transcript_id!r} ({transcript.n_turns} turns)"
                )
