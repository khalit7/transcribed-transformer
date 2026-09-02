import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.synthesis.generate as gen
from src.synthesis.generate import insert_turns, prompt
from src.synthesis.question_bank import QUESTIONS
from src.synthesis.schema import (Case, Generation, Label, LabelledRecord, Question, Transcript,
                                  Variant)


def _case():
    clean = ["SPEAKER_00: Hello, thanks for calling.", "SPEAKER_01: Hi, I need help.", "SPEAKER_00: Sure."]
    messy = ["SPEAKER_00: Hello thanks for calling.", "SPEAKER_01: Hi I need help.", "SPEAKER_00: Sure."]
    return Case(id="t-1", track="track-p", source_id="test/1",
                transcript=Transcript(variants=[Variant(kind="clean", origin="v", lines=clean),
                                                Variant(kind="messy", origin="m", lines=messy)],
                                      speaker_roles={"SPEAKER_00": "agent", "SPEAKER_01": "customer"},
                                      tag_policy="random"))


def test_bank_is_valid():
    assert len(QUESTIONS) >= 3
    assert len({x.id for x in QUESTIONS}) == len(QUESTIONS)
    for q in QUESTIONS:
        assert q.source == "bank"
        assert "pass" in q.values and "fail" in q.values
        assert q.family in ("vulnerability", "complaint_and_eod", "general_qa")


def test_question_requires_pass_and_fail():
    with pytest.raises(ValueError):
        Question(id="x", source="bank", family="general_qa", text="?",
                 options=[{"value": "pass", "criteria": "a"}, {"value": "NA", "criteria": "b"}])


def test_transcript_rejects_misaligned_variants():
    with pytest.raises(ValueError):
        Transcript(variants=[Variant(kind="clean", origin="v", lines=["SPEAKER_00: a", "SPEAKER_01: b"]),
                             Variant(kind="messy", origin="m", lines=["SPEAKER_00: a"])],
                   speaker_roles={"SPEAKER_00": "agent"}, tag_policy="random")


def test_prompt_has_the_right_blocks():
    q = QUESTIONS[0]
    p = prompt(_case(), "as_is", q, None, None)
    assert q.text in p and "Cite every line" in p and "insert_after_line" not in p
    p = prompt(_case(), "injected", q, None, "pass")
    assert 'Prefer "pass"' in p and '"turns"' in p and "Cite every line" not in p
    p = prompt(_case(), "injected", None, "vulnerability", None)
    assert '"vulnerability" family' in p and '"question"' in p and '"turns"' in p


def test_insert_turns_keeps_alignment_and_numbers_lines(monkeypatch):
    monkeypatch.setattr(gen, "noise_line", lambda text, seed: text.lower())
    turns = [{"speaker": "SPEAKER_01", "text": "I've been unwell recently."},
             {"speaker": "SPEAKER_00", "text": "Sorry to hear that."}]
    case, lines = insert_turns(_case(), turns, 1, "k")
    assert lines == [2, 3] and case.transcript.n_lines == 5
    assert case.transcript.variants[0].lines[1] == "SPEAKER_01: I've been unwell recently."
    assert case.transcript.variants[1].lines[1] == "SPEAKER_01: i've been unwell recently."
    assert case.transcript.variants[0].lines[3] == "SPEAKER_01: Hi, I need help."


def test_insert_turns_rejects_unknown_speaker():
    with pytest.raises(ValueError):
        insert_turns(_case(), [{"speaker": "SPEAKER_09", "text": "x"}], 0, "k")


def test_label_evidence_must_be_sorted_unique_one_based():
    with pytest.raises(ValueError):
        Label(answer="pass", evidence=[3, 1], summary="")
    with pytest.raises(ValueError):
        Label(answer="pass", evidence=[0], summary="")


def test_labelled_record_build():
    q = QUESTIONS[0]
    g = Generation(name="test", mode="injected", cost_usd=0.01, timestamp="2026-01-01T00:00:00+00:00")
    r = LabelledRecord.build(_case(), q, Label(answer="pass", evidence=[2], summary="s"), g)
    assert r.id == f"t-1::{q.id}" and r.track == "track-p" and r.source_id == "test/1"
    assert r.question.source == "bank" and r.generation_info.mode == "injected"
    assert r.transcript.tag_policy == "random" and r.label.evidence == [2]
