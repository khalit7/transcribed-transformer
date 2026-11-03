import json

from src.synthesis_v2 import agent_label, cases, question_bank


def _case(doc_id: str = "ep1", n_lines: int = 12) -> cases.Case:
    text = "\n".join(f"SPEAKER_0{i % 2}: this is turn number {i} about Nashville and the Grand Ole Opry" for i in range(n_lines))
    return cases.build(doc_id, text, ["SPEAKER_00"], {"stratum": "2-4k"})


def test_package_writes_the_models_rendering_and_the_questions(tmp_path):
    run = tmp_path / "run"
    qs = question_bank.BANK[:3]
    agent_label.write_task(run)
    d = agent_label.package(_case(), qs, run)
    assert (run / "TASK.md").read_text().startswith("# Labelling task")
    lines = (d / "transcript.txt").read_text().splitlines()
    assert lines[0] == "1. host_1:this is turn number 0 about Nashville and the Grand Ole Opry" and len(lines) == 12
    qj = json.loads((d / "questions.json").read_text())
    assert [q["id"] for q in qj] == [q.id for q in qs] and qj[0]["options"][0]["criteria"]
    assert qj[0]["description"].startswith(question_bank.VOCAB)


def test_ingest_episode_validates_answers_and_evidence(tmp_path):
    run = tmp_path / "run"
    qs = question_bank.BANK[:4]
    c = _case()
    d = agent_label.package(c, qs, run)
    labels = {qs[0].id: {"evidence": [2, "5", 99, True], "answer": qs[0].values[0], "summary": "ok", "confidence": 0.9},
              qs[1].id: {"evidence": [], "answer": "not an option", "summary": "", "confidence": 0.5},
              qs[2].id: {"evidence": "3-4", "answer": qs[2].values[-1], "summary": "", "confidence": 0.5},
              "v2-unknown": {"evidence": [], "answer": "x", "summary": "", "confidence": 0}}
    (d / "labels.json").write_text(json.dumps(labels))
    recs, problems = agent_label.ingest_episode(run, "ep1", c, qs, "agent:opus", "agent-7", "batch-2", 12345)
    assert [r.question.id for r in recs] == [qs[0].id]
    assert recs[0].label.evidence == [2, 5] and recs[0].meta["evidence_dropped"] == 2 and recs[0].meta["agent"] == "agent-7"
    assert recs[0].meta["batch_tokens"] == 12345
    assert recs[0].generation_info.name == "agent:opus" and recs[0].meta["method"] == "claude-code-subagent"
    assert recs[0].meta["flags"] == ["2 evidence entries dropped"]
    assert any("not in" in p for p in problems) and any("missing" in p for p in problems) and any("unknown" in p for p in problems)
    assert any("evidence is not a list" in p for p in problems)


def test_ingest_episode_coerces_and_flags(tmp_path):
    run = tmp_path / "run"
    qs = question_bank.BANK[:3]
    c = _case()
    d = agent_label.package(c, qs, run)
    labels = {qs[0].id: {"evidence": [2, 3], "answer": " NONE ", "summary": "", "confidence": "0.8"},
              qs[1].id: {"answer": qs[1].values[0], "summary": "x", "confidence": 1.7},
              qs[2].id: {"evidence": ["³", "4"], "answer": qs[2].values[0], "summary": "y", "confidence": float("nan")}}
    (d / "labels.json").write_text(json.dumps(labels))
    recs, problems = agent_label.ingest_episode(run, "ep1", c, qs, "agent:opus", "a", "b", None)
    by = {r.question.id: r for r in recs}
    assert by[qs[0].id].label.answer == "none" and "none with evidence" in by[qs[0].id].meta["flags"]
    assert "answer coerced from ' NONE '" in by[qs[0].id].meta["flags"] and "blank summary" in by[qs[0].id].meta["flags"]
    assert by[qs[0].id].label.confidence == 0.8 and "confidence coerced from string" in by[qs[0].id].meta["flags"]
    assert by[qs[1].id].label.confidence == 1.0 and "evidence key missing" in by[qs[1].id].meta["flags"]
    assert "non-none without evidence" in by[qs[1].id].meta["flags"]
    assert by[qs[2].id].label.evidence == [4] and by[qs[2].id].label.confidence is None
    assert "confidence unusable" in by[qs[2].id].meta["flags"] and len(problems) >= 7


def test_ingest_refuses_a_changed_transcript_and_uses_the_package_questions(tmp_path):
    run = tmp_path / "run"
    qs = question_bank.BANK[:1]
    c = _case()
    shown = [qs[0].model_copy(update={"text": qs[0].text + " (as worded when packaged)"})]
    d = agent_label.package(c, shown, run)
    pq = agent_label.package_questions(run, "ep1")
    assert pq[0].text == shown[0].text and pq[0].options == shown[0].options
    (d / "labels.json").write_text(json.dumps({qs[0].id: {"evidence": [1], "answer": qs[0].values[0], "summary": "s",
                                                          "confidence": 1}}))
    (d / "transcript.txt").write_text("1: host: changed\n")
    recs, problems = agent_label.ingest_episode(run, "ep1", c, pq, "agent:opus", "a", "b", None)
    assert recs == [] and "differs from the current rendering" in problems[0]


def test_ingest_episode_handles_missing_or_broken_file(tmp_path):
    run = tmp_path / "run"
    qs = question_bank.BANK[:1]
    c = _case()
    agent_label.package(c, qs, run)
    recs, problems = agent_label.ingest_episode(run, "ep1", c, qs, "agent:opus", "a", "b", None)
    assert recs == [] and "unreadable" in problems[0]
    (run / "episodes" / "ep1" / "labels.json").write_text("[1, 2]")
    recs, problems = agent_label.ingest_episode(run, "ep1", c, qs, "agent:opus", "a", "b", None)
    assert recs == [] and "not an object" in problems[0]


def test_batches_group_by_stratum_and_size():
    manifest = {"episodes": {f"e{i}": {"stratum": ["2-4k", "4-8k", "8-16k"][i % 3]} for i in range(10)}}
    groups = agent_label.batches([f"e{i}" for i in range(10)], manifest, 2)
    assert all(len(g) <= 2 for g in groups) and sum(len(g) for g in groups) == 10
    assert all(len({manifest["episodes"][d]["stratum"] for d in g}) == 1 for g in groups)


def test_question_ids_rejects_unknown_and_removed_ids_and_ingest_skips_removed_questions(tmp_path):
    import pytest

    assert agent_label.question_ids(None) is None
    assert [q.id for q in agent_label.question_ids("v2-qa-01-host-question-unanswered, v2-seg-01-structure")] == [
        "v2-qa-01-host-question-unanswered", "v2-seg-01-structure"]
    with pytest.raises(SystemExit, match="removed from the bank"):
        agent_label.question_ids("v2-qa-01-host-question-unanswered,v2-top-01-own-topic")
    with pytest.raises(SystemExit, match="unknown question ids"):
        agent_label.question_ids("v2-nope")
    run = tmp_path / "run"
    qs = question_bank.BANK[:2]
    c = _case()
    d = agent_label.package(c, qs, run)
    entries = json.loads((d / "questions.json").read_text())
    entries.append({**entries[0], "id": "v2-top-01-own-topic"})  # a package written before the cut names a removed question
    (d / "questions.json").write_text(json.dumps(entries))
    assert [q.id for q in agent_label.package_questions(run, "ep1")] == [q.id for q in qs]
