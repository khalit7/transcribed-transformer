"""Schema invariants. These are the failures that would otherwise be silent."""

import pytest
from pydantic import ValidationError

from tt.data.schema import (
    AnswerOption,
    Assessment,
    Case,
    CaseSemantics,
    ComplianceQuestion,
    Evidence,
    Provenance,
    RenderStyle,
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


def _question(
    qid: str = "q1",
    options: list[AnswerOption] | None = None,
) -> ComplianceQuestion:
    return ComplianceQuestion(
        id=qid,
        text="Did the advisor communicate clearly?",
        options=options
        or [
            AnswerOption(value="pass", criteria="The advisor was clear throughout."),
            AnswerOption(value="fail", criteria="The advisor was unclear."),
        ],
        family="clarity",
        semantics=CaseSemantics.ALL,
    )


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


@pytest.mark.parametrize(
    ("style", "first_line"),
    [
        ("colon", "1: spk0: um so line 0"),
        ("bracket", "[1] spk0: um so line 0"),
        ("dotted", "1. spk0: um so line 0"),
    ],
)
def test_render_styles_differ_only_in_surface_form(style: RenderStyle, first_line: str) -> None:
    """Surface form varies; the numbering a model is asked to cite does not.

    A model trained under one style and evaluated under another must still be
    pointing at the same turns, so line numbers cannot drift with the template.
    """
    t = _transcript(n=5)
    lines = t.render(style=style).split("\n")
    assert lines[0] == first_line
    assert len(lines) == t.n_turns
    for n in range(1, t.n_turns + 1):
        assert Transcript.line_to_index(n) == t.turns[n - 1].index
    assert lines[-1].endswith("um so line 4")


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


def test_answer_vocabularies_vary_in_arity_and_wording() -> None:
    """Two-way, four-way and coded vocabularies are all valid for the same task.

    This is the property the whole zero-shot claim rests on: nothing in the
    schema privileges one label set, so a question written after training works.
    """
    two_way = _question(
        options=[
            AnswerOption(value="yes", criteria="It happened."),
            AnswerOption(value="no", criteria="It did not."),
        ]
    )
    four_way = _question(
        options=[
            AnswerOption(value="Pass", criteria="Fully met."),
            AnswerOption(value="partial pass", criteria="Partly met."),
            AnswerOption(value="Fail", criteria="Not met."),
            AnswerOption(value="NA", criteria="Question does not apply to this case."),
        ]
    )
    coded = _question(
        options=[
            AnswerOption(value="01", criteria="Compliant."),
            AnswerOption(value="02", criteria="Non-compliant."),
            AnswerOption(value="03", criteria="Not applicable."),
        ]
    )
    assert two_way.values == ("yes", "no")
    assert four_way.values == ("Pass", "partial pass", "Fail", "NA")
    assert coded.values == ("01", "02", "03")


def test_question_needs_at_least_two_options() -> None:
    with pytest.raises(ValidationError, match="at least two"):
        _question(options=[AnswerOption(value="pass", criteria="Fine.")])


def test_question_rejects_duplicate_option_values() -> None:
    with pytest.raises(ValidationError, match="duplicate answer values"):
        _question(
            options=[
                AnswerOption(value="pass", criteria="Fully met."),
                AnswerOption(value="pass", criteria="Partly met."),
                AnswerOption(value="fail", criteria="Not met."),
            ]
        )


def test_question_rejects_whitespace_only_option_value() -> None:
    with pytest.raises(ValidationError, match="blank once stripped"):
        _question(
            options=[
                AnswerOption(value="   ", criteria="Fully met."),
                AnswerOption(value="fail", criteria="Not met."),
            ]
        )


def test_answer_must_be_the_value_not_the_grading_rule() -> None:
    """The specific near miss this schema exists to catch.

    Returning the option's ``criteria`` instead of its ``value`` is a plausible
    and easily-made mistake, and a scorer that quietly repaired it would hide
    the behaviour being measured. It fails here, loudly.
    """
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    question = _question()
    gloss = question.options[0].criteria
    a = Assessment(
        case_id="c1",
        question_id="q1",
        answer=gloss,
        provenance=Provenance.MODEL_PREDICTION,
    )
    with pytest.raises(ValueError, match="not one of the values permitted"):
        a.validate_against(case, question)


def test_answer_comparison_is_byte_exact() -> None:
    """Case and whitespace variants are failures, not near-enough matches."""
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    question = _question(
        options=[
            AnswerOption(value="Pass", criteria="Fully met."),
            AnswerOption(value="Fail", criteria="Not met."),
        ]
    )
    for near_miss in ("pass", "Pass ", "PASS", '"Pass"'):
        a = Assessment(
            case_id="c1",
            question_id="q1",
            answer=near_miss,
            provenance=Provenance.MODEL_PREDICTION,
        )
        with pytest.raises(ValueError, match="not one of the values permitted"):
            a.validate_against(case, question)


def test_evidence_out_of_range_is_rejected() -> None:
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    a = Assessment(
        case_id="c1",
        question_id="q1",
        answer="pass",
        evidence=[Evidence(transcript_id="t1", index=9)],
        provenance=Provenance.MODEL_PREDICTION,
    )
    with pytest.raises(ValueError, match="out of range"):
        a.validate_against(case, _question())


def test_valid_evidence_passes() -> None:
    case = Case(id="c1", transcripts=[_transcript(n=4)])
    Assessment(
        case_id="c1",
        question_id="q1",
        answer="pass",
        evidence=[Evidence(transcript_id="t1", index=3)],
        provenance=Provenance.GOLD_HUMAN,
    ).validate_against(case, _question())


def test_evidence_keys_are_partial_by_default() -> None:
    """Recall is only meaningful against an exhaustive key, so the flag defaults off."""
    a = Assessment(
        case_id="c1",
        question_id="q1",
        answer="pass",
        evidence=[Evidence(transcript_id="t1", index=0)],
        provenance=Provenance.GOLD_HUMAN,
    )
    assert a.evidence_exhaustive is False
