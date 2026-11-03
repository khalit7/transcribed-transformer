import json

from src.synthesis_v2 import cases, label_episodes, question_bank


def _case(n_lines: int = 12) -> cases.Case:
    text = "\n".join(f"SPEAKER_0{i % 2}: this is turn number {i} about the Grand Ole Opry and Nashville" for i in range(n_lines))
    return cases.build("ep1", text, ["SPEAKER_00"], {"stratum": "2-4k"})


def test_prompt_lists_every_question_and_keys_the_json_by_id():
    qs = question_bank.BANK[:3]
    p = label_episodes.prompt(_case(), qs)
    assert p.startswith("You answer questions about the transcript of a podcast episode. One turn per line, numbered.")
    assert "1. host_1:this is turn number 0" in p and "12. SPEAKER_01:this is turn number 11" in p
    for q in qs:
        assert f"QUESTION {q.id}: {q.text}" in p and f'"{q.id}": {{"evidence"' in p
    assert "Answer each of the following 3 questions" in p


def test_parse_validates_answer_and_evidence_range():
    q = question_bank.BANK[0]
    lab, dropped = label_episodes.parse({q.id: {"evidence": [3, "1", 99, 1, True, 2.5, 4.0], "answer": q.values[0],
                                                "summary": None, "confidence": 1.4}}, q, 10)
    assert lab.evidence == [1, 3, 4] and lab.answer == q.values[0] and lab.summary == "" and lab.confidence == 1.0
    assert dropped == 3  # 99 out of range, True, 2.5
    for bad in ({}, {q.id: {"answer": "nope"}}, {q.id: "x"}, {q.id: {"answer": q.values[0], "evidence": "3-5"}}):
        try:
            label_episodes.parse(bad, q, 10)
        except (ValueError, TypeError):
            continue
        raise AssertionError(bad)


def test_batches_are_seeded_and_cover_every_question_once():
    qs = question_bank.BANK
    b1 = label_episodes.batches(qs, 7, "0:sporc-a")
    b2 = label_episodes.batches(qs, 7, "0:sporc-a")
    b3 = label_episodes.batches(qs, 7, "1:sporc-a")
    assert [q.id for b in b1 for q in b] == [q.id for b in b2 for q in b] != [q.id for b in b3 for q in b]
    assert sorted(q.id for b in b1 for q in b) == sorted(q.id for q in qs) and all(len(b) <= 7 for b in b1)
    assert len(label_episodes.batches(qs, 1, "s")) == len(qs)


def test_label_case_writes_one_record_per_question_and_splits_cost(monkeypatch, tmp_path):
    qs = question_bank.BANK[:4]
    answers = {q.id: {"evidence": [2], "answer": q.values[-1], "summary": "none seen", "confidence": 0.8} for q in qs}
    answers[qs[1].id]["answer"] = "not an option"  # one bad answer must not lose the other three
    monkeypatch.setattr(label_episodes, "ask_json_full",
                        lambda prompt, model, schema, **kw: (answers, 0.40, {"attempts": 1, "cost_last": 0.4, "cost_total": 0.4, "usage": {"cache_read_input_tokens": 5}}))
    log = tmp_path / "x.calls.jsonl"
    recs, failed = label_episodes.label_case(_case(), qs, "claude:opus", 2, 0, "2-4k", calls_log=log)
    assert failed == 1 and len(recs) == 3 and {r.id for r in recs} <= {f"sporc-ep1::{q.id}" for q in qs}
    assert all(abs(r.generation_info.cost_usd - 0.20) < 1e-9 for r in recs) and recs[0].meta["k"] == 2
    assert recs[0].meta["stratum"] == "2-4k" and recs[0].question.family == "podcast"
    assert {r.meta["batch_size"] for r in recs} == {2} and {r.meta["batch_index"] for r in recs} == {0, 1}
    calls = [json.loads(line) for line in log.open()]
    assert len(calls) == 2 and sum(c["cost_usd"] for c in calls) == 0.8 and calls[0]["usage"]["cache_read_input_tokens"] == 5
    json.loads(recs[0].model_dump_json())  # serialisable
    # a reply that is not an object fails the whole batch without raising
    monkeypatch.setattr(label_episodes, "ask_json_full", lambda prompt, model, schema, **kw: ([1, 2], 0.1, {"attempts": 1, "cost_last": 0.1, "cost_total": 0.1, "usage": {}}))
    recs2, failed2 = label_episodes.label_case(_case(), qs, "claude:opus", 4, 0, "2-4k")
    assert recs2 == [] and failed2 == 4


def test_prompt_lean_prints_the_shared_definitions_once():
    qs = question_bank.BANK[:5]
    std, lean = label_episodes.prompt(_case(), qs), label_episodes.prompt_lean(_case(), qs)
    assert std.count(question_bank.VOCAB) == 5 and lean.count(question_bank.VOCAB) == 1
    assert lean.index(question_bank.VOCAB) < lean.index("QUESTION ")  # before the questions
    assert std.split("TRANSCRIPT")[1].split("Answer each")[0] == lean.split("TRANSCRIPT")[1].split("Answer each")[0]
    for q in qs:
        assert f"QUESTION {q.id}: {q.text}" in lean and f'"{q.id}": {{"evidence"' in lean
        for o in q.options:
            assert f"- {o.value}: {o.criteria}" in lean
        assert f"Context: {q.description}" in lean  # each question shows its own description
    assert std.replace(question_bank.VOCAB + " ", "").replace(question_bank.VOCAB, "").split() == \
        lean.replace(question_bank.VOCAB, "").split()  # nothing else differs but whitespace


def test_lean_label_case_asks_every_question_in_one_lean_call(monkeypatch):
    qs = question_bank.BANK
    seen = []

    def fake(prompt, model, schema, system=None):
        seen.append((prompt, system))
        return ({q.id: {"evidence": [1], "answer": q.values[-1], "summary": "s", "confidence": 0.5} for q in qs}, 0.44,
                {"attempts": 1, "cost_last": 0.44, "cost_total": 0.44, "usage": {}})
    monkeypatch.setattr(label_episodes, "ask_json_full", fake)
    recs, failed = label_episodes.label_case(_case(), qs, "claude:opus", 7, 0, "0-4k", lean=True)
    assert failed == 0 and len(recs) == len(qs) and len(seen) == 1
    prompt, system = seen[0]
    assert system == label_episodes.prompts.text(label_episodes.SYSTEM_LEAN) and system
    assert prompt.count(question_bank.VOCAB) == 1 and f"Answer each of the following {len(qs)} questions" in prompt
    assert {r.meta["format"] for r in recs} == {"lean"} and {r.meta["k"] for r in recs} == {len(qs)}
    assert all(abs(r.generation_info.cost_usd - 0.01) < 1e-9 for r in recs)
    assert [r.question for r in recs] == [question_bank.by_id()[r.question.id] for r in recs]  # the bank's question as stored
    assert not any(r.question.description.startswith(question_bank.VOCAB) for r in recs)


def test_existing_streams_ids_and_skips_a_cut_line(tmp_path):
    out = tmp_path / "labels.jsonl"
    out.write_text('{"id": "a::q1", "x": 1}\n\n{"id": "b::q2"}\n{"id": "c::q3", "x": [1, 2')  # last line cut by a kill
    assert label_episodes.existing(out) == {"a::q1", "b::q2"}
    assert label_episodes.existing(tmp_path / "missing.jsonl") == set()


def test_append_repairs_a_cut_line_beyond_one_block(tmp_path):
    from src.synthesis_v2 import append_jsonl
    out = tmp_path / "big.jsonl"
    out.write_text('{"id": "a"}\n' + '{"id": "cut", "pad": "' + "x" * (3 << 20))  # a 3 MB fragment with no newline
    append_jsonl(out, {"id": "b"})
    assert out.read_text() == '{"id": "a"}\n{"id": "b"}\n'
    out.write_text('{"id": "no newline at all"')
    append_jsonl(out, {"id": "c"})
    assert out.read_text() == '{"id": "c"}\n'


def test_existing_survives_a_cut_character_and_skips_an_unfinished_line(tmp_path):
    out = tmp_path / "labels.jsonl"
    out.write_bytes('{"id": "a::q1", "s": "café"}\n{"id": "b::q2"}'.encode() + b"\n" + '{"id": "c::q3", "s": "caf'.encode() + b"\xc3")
    assert label_episodes.existing(out) == {"a::q1", "b::q2"}
    out.write_text('{"id": "a::q1"}\n{"id": "b::q2"}')  # valid JSON but no newline: unfinished, the next append cuts it
    assert label_episodes.existing(out) == {"a::q1"}


def test_pilot_report_pairs_old_and_new_storage_of_the_same_question(tmp_path):
    from src.synthesis_v2 import pilot_report
    q = question_bank.BANK[0]
    old = question_bank.with_definitions(q)
    for name, qq in (("old.jsonl", old), ("new.jsonl", q)):
        (tmp_path / name).write_text(json.dumps({
            "id": f"sporc-e::{q.id}", "question": qq.model_dump(), "label": {"answer": q.values[0], "evidence": []},
            "transcript": {"variants": [{"lines": ["a"]}]}, "meta": {"stratum": "0-4k"},
            "generation_info": {"cost_usd": 0.0}}) + "\n")
    a, b = pilot_report.load(tmp_path / "old.jsonl"), pilot_report.load(tmp_path / "new.jsonl")
    assert a[f"sporc-e::{q.id}"]["text"] == b[f"sporc-e::{q.id}"]["text"]
