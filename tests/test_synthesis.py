import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthesis.label import _parse, prompt, schema_for
from src.synthesis.llm import LLMError, split_model
from src.synthesis.question_bank import QUESTIONS
from src.synthesis.schema import Case, Label, Question, Transcript, Variant


def _case():
    clean = ["agent: Hello, thanks for calling.", "customer: Hi, I need help.", "agent: Sure."]
    messy = ["agent: Hello thanks for calling.", "customer: Hi I need help.", "agent: Sure."]
    return Case(id="t-1", dataset="apptek", track="track-p", source_id="test/1",
                transcript=Transcript(variants=[Variant(kind="clean", origin="v", lines=clean),
                                                Variant(kind="messy", origin="m", lines=messy)],
                                      speakers=["agent", "customer"], role_source="test"))


def test_bank_is_valid():
    assert len(QUESTIONS) >= 50
    assert len({x.id for x in QUESTIONS}) == len(QUESTIONS)
    fams = {q.family for q in QUESTIONS}
    assert fams == {"vulnerability", "complaint", "eod", "general_qa"}
    for q in QUESTIONS:
        assert "pass" in q.values and "fail" in q.values
    vul = next(q for q in QUESTIONS if q.id == "vul-01-present")
    assert len(vul.tags) >= 20
    for q in QUESTIONS:
        assert q.dataset_allow_list and set(q.dataset_allow_list) <= {"apptek", "taskmaster", "aci_bench", "sporc"}
    assert all("sporc" in q.dataset_allow_list for q in QUESTIONS if q.id.startswith("spk-"))
    assert not any("sporc" in q.dataset_allow_list for q in QUESTIONS if q.id.startswith("gen-"))


def test_question_requires_pass_and_fail():
    with pytest.raises(ValueError):
        Question(id="x", family="general_qa", text="?", dataset_allow_list=["apptek"],
                 options=[{"value": "pass", "criteria": "a"}, {"value": "NA", "criteria": "b"}])


def test_transcript_rejects_misaligned_variants():
    with pytest.raises(ValueError):
        Transcript(variants=[Variant(kind="clean", origin="v", lines=["agent: a", "customer: b"]),
                             Variant(kind="messy", origin="m", lines=["agent: a"])],
                   speakers=["agent", "customer"], role_source="test")


def test_transcript_rejects_undeclared_speaker_labels():
    with pytest.raises(ValueError, match="not declared"):
        Transcript(variants=[Variant(kind="clean", origin="v", lines=["agent: a", "SPEAKER_03: b"])],
                   speakers=["agent"], role_source="test")


def test_prompt_is_evidence_first_and_schema_matches_question():
    q = next(x for x in QUESTIONS if x.id == "vul-01-present")
    p = prompt(_case().transcript.lines(), ["agent", "customer"], q)
    assert p.index('"evidence"') < p.index('"answer"') < p.index('"summary"')
    assert "tags chosen only from this list" in p
    sch = schema_for(q)
    assert sch["properties"]["answer"]["enum"] == q.values and "tags" in sch["required"]
    q2 = next(x for x in QUESTIONS if x.family == "general_qa")
    assert "tags" not in schema_for(q2)["properties"]


def test_parse_clips_evidence_filters_tags_and_bounds_confidence():
    q = next(x for x in QUESTIONS if x.id == "vul-01-present")
    lab = _parse({"answer": "fail", "evidence": [3, 1, 99, 1], "summary": " s ", "confidence": 1.7,
                  "tags": ["bereavement", "not-a-tag"]}, q, n_lines=5)
    assert lab.evidence == [1, 3] and lab.tags == ["bereavement"] and lab.confidence == 1.0 and lab.summary == "s"
    with pytest.raises(ValueError):
        _parse({"answer": "partial_pass", "evidence": [], "summary": ""}, q, 5)  # not one of vul-01's options


def test_label_evidence_must_be_sorted_unique_one_based():
    with pytest.raises(ValueError):
        Label(answer="pass", evidence=[3, 1], summary="")
    with pytest.raises(ValueError):
        Label(answer="pass", evidence=[0], summary="")


def test_model_strings():
    assert split_model("ollama:qwen3:32b") == ("ollama", "qwen3:32b")
    assert split_model("claude:sonnet") == ("claude", "sonnet")
    with pytest.raises(LLMError):
        split_model("sonnet")


def test_labeller_sees_verbatim_roles_and_roleless_corpora_raise(monkeypatch):
    from src.synthesis.cases import BUILDERS, NoSpeakerRoles

    lines = ["doctor: Hello.", "patient: Hi.", "patient_guest: Also here."]
    p = prompt(lines, ["doctor", "patient", "patient_guest"], QUESTIONS[0])
    assert "1: doctor: Hello." in p and "3: patient_guest: Also here." in p
    assert "as recorded by the source (doctor, patient, patient_guest)" in p
    from src.synthesis import identify_speakers as ids

    monkeypatch.setattr(ids, "load_cache", dict)  # no cached hosts, identification forbidden -> must raise
    with pytest.raises(NoSpeakerRoles):
        next(iter(BUILDERS["sporc"](identify_model="")))


def test_claude_limit_is_waited_out_not_failed(monkeypatch):
    """A usage/rate limit pauses the call (sleeping until the reported reset) instead of burning retries."""
    import json
    import types

    from src.synthesis import llm

    reset = int(llm.time.time()) + 90
    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=f"Claude AI usage limit reached|{reset}")
        if calls["n"] == 2:  # error envelope with exit 0
            return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                {"is_error": True, "result": "API Error: 429 rate limit exceeded"}))
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"result": '{"answer": "pass"}', "total_cost_usd": 0.01}))

    sleeps: list[float] = []
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    out, cost = llm.ask_json("p", "claude:sonnet")
    assert out == {"answer": "pass"} and cost == 0.01 and calls["n"] == 3
    assert 90 <= sleeps[0] <= 140  # slept until the reported reset (+15 s, up to 20% jitter)
    assert 30 <= sleeps[1] <= 36  # no reset given: first exponential step


def test_claude_other_errors_still_fail_fast(monkeypatch):
    import types

    from src.synthesis import llm

    monkeypatch.setattr(llm.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    monkeypatch.setattr(llm.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))
    with pytest.raises(LLMError, match="boom"):
        llm.ask_json("p", "claude:sonnet", retries=1)
