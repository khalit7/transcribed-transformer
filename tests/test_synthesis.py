import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.synthesis.benchmark import two_golds
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
    from src.synthesis import cases

    monkeypatch.setattr(cases, "sporc_hosts", dict)  # no identified hosts -> no SPoRC case is ever built
    monkeypatch.setattr(cases, "sporc_episodes", lambda max_lines=160: iter([("x", [("SPEAKER_00", "hi")], {})]))
    assert next(iter(BUILDERS["sporc"]()), None) is None
    assert NoSpeakerRoles  # still the error for a role-bearing corpus with an unattributed turn


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


# --- split and the benchmark's two golds -------------------------------------------------------

def test_lineage_maps_taskmaster_variants_to_their_base_question():
    from src.synthesis.split import lineage
    assert lineage("tm-16-read-back") == "gen-16-read-back"
    assert lineage("tm-vul-01-present") == "vul-01-present"
    assert lineage("tm-101-own") == "tm-101-own"  # hand-written Taskmaster question: its own lineage
    assert lineage("spk-01-speakers-introduced") == "spk-01-speakers-introduced"


def test_held_out_lineages_never_include_families_and_hit_the_target_share():
    import random

    from src.synthesis.split import HELD_OUT_FRACTION, held_out_lineages, lineage
    held = held_out_lineages(random.Random(0))
    by_id = {q.id: q for q in QUESTIONS}
    assert all(by_id[l].family == "general_qa" for l in held)
    for ds in ("apptek", "taskmaster", "aci_bench", "sporc"):
        allowed = {lineage(q.id) for q in QUESTIONS if q.family == "general_qa" and ds in q.dataset_allow_list}
        share = len(allowed & set(held)) / len(allowed)
        assert HELD_OUT_FRACTION - 0.03 <= share <= HELD_OUT_FRACTION + 0.08, (ds, share)


def test_assign_drops_held_out_questions_off_benchmark_only():
    from src.synthesis.split import assign
    splits = {"calls": {"c-bench": "benchmark", "c-val": "val", "c-x": "excluded"}, "held_out_questions": ["gen-9"]}
    rec = lambda cid, qid: {"id": f"{cid}::{qid}", "question": {"id": qid}}
    assert assign(rec("c-bench", "gen-9"), splits) == "benchmark"
    assert assign(rec("c-val", "gen-9"), splits) == "drop"
    assert assign(rec("c-val", "gen-1"), splits) == "val"
    assert assign(rec("c-new", "gen-1"), splits) == "train"  # labelled after the freeze
    assert assign(rec("c-new", "gen-9"), splits) == "drop"
    assert assign(rec("c-x", "gen-1"), splits) == "drop"


def test_two_golds_never_alters_the_teacher_label_and_names_the_opus_labeller():
    ledger = {"id": "x", "label": {"answer": "pass", "evidence": [1, 2], "summary": "s", "tags": []}, "generation_info": {"name": "ollama:qwen3.8"}}
    agree = {"label": {"answer": "pass", "evidence": [2, 3], "summary": "s", "tags": []}, "generation_info": {"name": "ollama:gemma4:12b"}}
    differ = {"label": {"answer": "fail", "evidence": [3], "summary": "s", "tags": []}, "generation_info": {"name": "ollama:gemma4:12b"}}
    opus_rec = {"label": {"answer": "fail", "evidence": [4], "summary": "s", "tags": []}, "generation_info": {"name": "claude:opus"}}
    teacher, opus, prov = two_golds(ledger, agree, None)
    assert teacher.evidence == [1, 2] and opus is None and prov.method == "teacher"  # no union: the teacher's key stays its own
    teacher, opus, prov = two_golds(ledger, differ, opus_rec)
    assert teacher.answer == "pass" and opus.answer == "fail" and opus.evidence == [4] and prov.opus_labeller == "claude:opus"
    assert prov.labels == {"ollama:qwen3.8": "pass", "ollama:gemma4:12b": "fail", "claude:opus": "fail"}


def test_assign_routes_post_freeze_podcast_episodes_by_their_podcast():
    """A SPoRC episode labelled after the freeze follows its podcast's side: a sibling of a benchmark or excluded episode is
    dropped, a sibling of a val episode goes to val, anything else to train (2026-10-05 audit finding)."""
    from src.synthesis.split import assign
    splits = {"calls": {"sporc-a": "benchmark"}, "held_out_questions": [],
              "groups": {"pod:Bench Show": "benchmark", "pod:Val Show": "val", "pod:Side Show": "excluded"}}
    def rec(cid, pod):
        return {"id": f"{cid}::gen-1", "dataset": "sporc", "question": {"id": "gen-1"}, "meta": {"pod_title": pod}}
    assert assign(rec("sporc-new1", "Bench Show"), splits) == "drop"
    assert assign(rec("sporc-new2", "Side Show"), splits) == "drop"
    assert assign(rec("sporc-new3", "Val Show"), splits) == "val"
    assert assign(rec("sporc-new4", "Fresh Show"), splits) == "train"
    assert assign({"id": "tm-new::gen-1", "dataset": "taskmaster", "question": {"id": "gen-1"}, "meta": {"subset": "x"}}, splits) == "train"


def test_transfer_questions_bring_their_own_vocabulary_but_compliance_questions_do_not():
    q = Question(id="t", family="transfer", text="?", dataset_allow_list=["ecthr"],
                 options=[{"value": "A", "criteria": "a"}, {"value": "B", "criteria": "b"}])
    assert q.values == ["A", "B"]
    with pytest.raises(ValueError):
        Question(id="t", family="transfer", text="?", dataset_allow_list=["ecthr"], options=[{"value": "A", "criteria": "a"}])
    with pytest.raises(ValueError):
        Question(id="x", family="general_qa", text="?", dataset_allow_list=["apptek"],
                 options=[{"value": "pass", "criteria": "a"}, {"value": "fail", "criteria": "b"},
                          {"value": "violated", "criteria": "c"}])


def test_labelled_record_rejects_an_answer_outside_its_options():
    from src.synthesis.schema import Generation, LabelledRecord
    c = _case()
    q = Question(id="q", family="general_qa", text="?", dataset_allow_list=["apptek"],
                 options=[{"value": "pass", "criteria": "a"}, {"value": "fail", "criteria": "b"}])
    g = Generation(name="t:m", labelled_variant="clean", cost_usd=0.0, timestamp="t")
    LabelledRecord(id="t-1::q", dataset="apptek", source_id="s", track="track-p", question=q, transcript=c.transcript,
                   label=Label(answer="pass", evidence=[], summary=""), generation_info=g)
    with pytest.raises(ValueError):
        LabelledRecord(id="t-1::q", dataset="apptek", source_id="s", track="track-p", question=q, transcript=c.transcript,
                       label=Label(answer="NA", evidence=[], summary=""), generation_info=g)


def test_claude_command_default_is_unchanged_and_lean_is_lean(monkeypatch):
    from src.synthesis import llm
    monkeypatch.setattr(llm, "_claude_env", {"CLAUDE_CONFIG_DIR": "/x"})
    cmd, env = llm.claude_command("opus")
    assert cmd == ["claude", "-p", "--model", "opus", "--output-format", "json"] and env == {"CLAUDE_CONFIG_DIR": "/x"}
    cmd, env = llm.claude_command("opus", "Be brief.")
    assert cmd[:6] == ["claude", "-p", "--model", "opus", "--output-format", "json"]
    assert cmd[6:] == ["--tools", "", "--system-prompt", "Be brief."]
    assert env == {"CLAUDE_CONFIG_DIR": "/x", "DISABLE_PROMPT_CACHING": "1"}


def test_system_prompt_is_refused_off_the_claude_backend():
    from src.synthesis import llm
    try:
        llm.ask_json_full("p", "ollama:x", system="s")
    except llm.LLMError as e:
        assert "claude backend" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_the_clis_own_limit_envelope_is_waited_out(monkeypatch):
    """The refusal as `claude -p` prints it in 2026 (captured from the pilot's log), whatever the exit code."""
    import json
    import types

    from src.synthesis import llm

    envelope = {"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
                "api_error_origin": "first_party", "api_error": "usage_limit_reached",
                "result": "You've hit your 5-hour limit · resets 3:40am (Europe/London)", "total_cost_usd": 0}
    for code in (0, 1):
        monkeypatch.setattr(llm.subprocess, "run", lambda *a, code=code, **k: types.SimpleNamespace(
            returncode=code, stdout=json.dumps(envelope), stderr=""))
        try:
            llm._claude("p", "opus", 5, system="s")
        except llm.LLMLimit as e:
            assert e.retry_after is not None and 0 <= e.retry_after <= 86400
        else:
            raise AssertionError("expected LLMLimit")
    for text in ("You've hit your weekly limit · resets Nov 5, 3pm", "You've hit your Opus limit · resets 3pm"):
        assert llm._LIMIT.search(text) and llm._reset_after(text) is not None


def test_a_dated_reset_is_read_as_that_day():
    from src.synthesis import llm
    now = llm.time.localtime(llm.time.time() + 3 * 86400)
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][now.tm_mon - 1]
    secs = llm._reset_after(f"resets {mon} {now.tm_mday}, 3pm")
    assert secs is not None and 1.5 * 86400 < secs < 4.5 * 86400


def test_a_malformed_model_reply_is_a_retry_not_a_limit(monkeypatch):
    """A long reply that fails to parse (its evidence may hold line 429, its summaries 'capacity') must not be waited
    out as a limit; a lean reply with stray text around the object still parses."""
    import json
    import types

    from src.synthesis import llm

    bad = '{"q1": {"evidence": [429, 529], "answer": "x", "summary": "the venue was at capacity"' + " pad" * 400
    monkeypatch.setattr(llm.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps({"result": bad, "total_cost_usd": 0.3}), stderr=""))
    try:
        llm._claude("p", "opus", 5, system="s")
    except llm.LLMLimit as e:
        raise AssertionError("a model reply was read as a limit") from e
    except llm.LLMError as e:
        assert e.cost == 0.3
    good = 'Here are the answers:\n{"q1": {"answer": "x"}}\nDone.'
    monkeypatch.setattr(llm.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps({"result": good, "total_cost_usd": 0.3}), stderr=""))
    assert llm._claude("p", "opus", 5, system="s")[0] == {"q1": {"answer": "x"}}
    try:  # the default route keeps its strict parse
        llm._claude("p", "opus", 5)
    except llm.LLMError:
        pass
    else:
        raise AssertionError("default route should not loosen its parse")


def test_a_weekly_limit_is_waited_out_until_its_reset(monkeypatch):
    """A reset days away is slept until in full (not capped at the 6 h backoff budget), then the call succeeds."""
    import json
    import types

    from src.synthesis import llm

    calls = {"n": 0}

    def fake_run(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return types.SimpleNamespace(returncode=1, stderr="", stdout=json.dumps(
                {"is_error": True, "api_error_status": 429, "api_error": "usage_limit_reached",
                 "result": "You've hit your weekly limit · resets Nov 5, 3pm"}))
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"result": '{"a": 1}', "total_cost_usd": 0.1}))

    sleeps: list[float] = []
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm, "_reset_after", lambda msg: 3 * 86400.0)
    assert llm.ask_json_full("p", "claude:opus", system="s")[0] == {"a": 1}
    assert len(sleeps) == 1 and sleeps[0] >= 3 * 86400


def test_a_reset_just_past_is_retried_soon_not_tomorrow():
    from src.synthesis import llm
    t = llm.time.localtime(llm.time.time() - 600)  # ten minutes ago
    if t.tm_mday != llm.time.localtime().tm_mday:  # just after midnight "ten minutes ago" is yesterday's clock time
        return
    h = t.tm_hour % 12 or 12
    secs = llm._reset_after(f"resets {h}:{t.tm_min:02d}{'pm' if t.tm_hour >= 12 else 'am'}")
    assert secs == 60.0


def test_short_fenced_or_prefixed_replies_and_usage_counts_are_not_limits(monkeypatch):
    import json
    import types

    from src.synthesis import llm

    for result in ('```json\n{"answer": "pass", "summary": "in her capacity as agent"', 'Sure: {"answer": "pass", "evidence": [429]'):
        monkeypatch.setattr(llm.subprocess, "run", lambda *a, r=result, **k: types.SimpleNamespace(
            returncode=0, stdout=json.dumps({"result": r, "total_cost_usd": 0.01}), stderr=""))
        try:
            llm._claude("p", "sonnet", 5)
        except llm.LLMLimit as e:
            raise AssertionError(f"read as a limit: {result}") from e
        except llm.LLMError:
            pass
    monkeypatch.setattr(llm.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stderr="", stdout=json.dumps({"is_error": True, "api_error": "max_output_tokens", "result": "",
                                                    "usage": {"output_tokens": 529}, "total_cost_usd": 0.2})))
    try:
        llm._claude("p", "opus", 5, system="s")
    except llm.LLMLimit as e:
        raise AssertionError("a usage count was read as a limit") from e
    except llm.LLMError as e:
        assert e.cost == 0.2


def test_prose_replies_and_empty_error_envelopes_are_retries_not_limits(monkeypatch):
    import json
    import types

    from src.synthesis import llm

    cases = [(0, {"result": "I cannot do this in my capacity as a labeller; try again later.", "total_cost_usd": 0.01}),
             (0, {"is_error": True, "result": "", "usage": {"output_tokens": 529}, "total_cost_usd": 0.01})]
    for code, env in cases:
        monkeypatch.setattr(llm.subprocess, "run", lambda *a, c=code, e=env, **k: types.SimpleNamespace(
            returncode=c, stdout=json.dumps(e), stderr=""))
        try:
            llm._claude("p", "opus", 5, system="s")
        except llm.LLMLimit as e:
            raise AssertionError(f"read as a limit: {env}") from e
        except llm.LLMError:
            pass
    monkeypatch.setattr(llm.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout=json.dumps({"result": "Claude AI usage limit reached", "total_cost_usd": 0}), stderr=""))
    try:
        llm._claude("p", "opus", 5)
    except llm.LLMLimit:
        pass
    else:
        raise AssertionError("the CLI's own limit notice must still be a limit")


def test_a_past_or_new_year_dated_reset_is_a_short_retry():
    from src.synthesis import llm
    past = llm.time.localtime(llm.time.time() - 2 * 86400)
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][past.tm_mon - 1]
    assert llm._reset_after(f"resets {mon} {past.tm_mday}, 3pm") == 60.0
