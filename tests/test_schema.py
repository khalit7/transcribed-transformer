"""Schema invariants. These are the failures that would otherwise be silent."""

import pytest
from pydantic import ValidationError

from tt.data.schema import (
    Answer,
    Assessment,
    Case,
    CaseSemantics,
    ComplianceQuestion,
    Evidence,
    Provenance,
    Role,
    Track,
    Transcript,
    Turn,
)


def _turns(n: int) -> list[Turn]:
    return [
        Turn(index=i, speaker=f"spk{i % 2}", role=Role.UNKNOWN, text=f"um so line {i}")
        for i in range(n)
    ]


def _transcript(tid: str = "t1", n: int = 4, track: Track = Track.P) -> Transcript:
    return Transcript(id=tid, source="fixture", track=track, turns=_turns(n), is_asr=False)


def test_line_indices_must_be_contiguous() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        Transcript(
            id="bad",
            source="fixture",
            track=Track.P,
            turns=[Turn(index=0, speaker="a", text="x"), Turn(index=2, speaker="b", text="y")],
            is_asr=False,
        )


def test_render_is_one_based_and_round_trips() -> None:
    t = _transcript(n=3)
    lines = t.render().split("\n")
    assert lines[0].startswith("1: ")
    assert lines[2].startswith("3: ")
    # A model emitting "3" must resolve to the last turn.
    assert Transcript.line_to_index(3) == t.turns[-1].index


def test_render_preserves_disfluencies() -> None:
    assert "um so" in _transcript().render()


def test_case_rejects_mixed_licence_tracks() -> None:
    with pytest.raises(ValidationError, match="mixes licence tracks"):
        Case(
            id="c1",
            transcripts=[_transcript("a", track=Track.P), _transcript("b", track=Track.NC)],
        )


def test_case_track_propagates() -> None:
    assert Case(id="c1", transcripts=[_transcript(track=Track.NC)]).track is Track.NC


def test_question_requires_all_four_definitions() -> None:
    with pytest.raises(ValidationError, match="missing definitions"):
        ComplianceQuestion(
            id="q1",
            text="Did the advisor communicate clearly?",
            definitions={Answer.PASS: "yes", Answer.FAIL: "no"},
            family="clarity",
            semantics=CaseSemantics.ALL,
        )


def test_evidence_out_of_range_is_rejected() -> None:
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    a = Assessment(
        case_id="c1",
        question_id="q1",
        answer=Answer.PASS,
        evidence=[Evidence(transcript_id="t1", index=9)],
        provenance=Provenance.MODEL_PREDICTION,
    )
    with pytest.raises(ValueError, match="out of range"):
        a.validate_against(case)


def test_valid_evidence_passes() -> None:
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    Assessment(
        case_id="c1",
        question_id="q1",
        answer=Answer.PASS,
        evidence=[Evidence(transcript_id="t1", index=3)],
        provenance=Provenance.GOLD_HUMAN,
    ).validate_against(case)
