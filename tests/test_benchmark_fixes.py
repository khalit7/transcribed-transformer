"""The 2026-09-29 benchmark-accuracy fixes: empty labelling directory, the two golds (teacher and Opus), the AppTek word alignment,
ACI-Bench continuation lines, the exporter's billing strip and notice, the evaluator's shared-pair view."""

import json
import subprocess

import pytest

from src.synthesis.benchmark import two_golds
from src.synthesis.cases import aci_subsets, aci_turns, assign_words
from src.synthesis.export import release_record
from src.train.evaluate import evaluate


def test_assign_words_sends_gap_words_to_the_nearer_turn():
    spans = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    words = [{"word": "a", "start": 0.2, "end": 0.4}, {"word": "gap-early", "start": 1.1, "end": 1.3},
             {"word": "gap-late", "start": 1.7, "end": 1.9}, {"word": "b", "start": 2.5, "end": 2.6},
             {"word": "tail", "start": 3.05, "end": 3.15}, {"word": "c", "start": 4.5, "end": 4.6},
             {"word": "after", "start": 5.5, "end": 5.7}]
    assert assign_words(words, spans) == [["a", "gap-early"], ["gap-late", "b", "tail"], ["c", "after"]]
    # the old rule sent every gap word to the preceding turn
    assert assign_words([{"word": "x", "start": 1.9, "end": 1.95}], spans) == [[], ["x"], []]


def test_aci_turns_keeps_untagged_continuations_and_reads_the_corpus_subsets(tmp_path):
    turns, dropped = aci_turns("[doctor] How are you?\n[patient] Fine.\nany other questions for me?\n[doctor] No.")
    assert turns == [("doctor", "How are you?"), ("patient", "Fine. any other questions for me?"), ("doctor", "No.")]
    assert dropped == 0
    turns, dropped = aci_turns("hi , susan\n[doctor] hi , susan , how are you ?\n[patient] i'm okay .\n[doctor] good .")
    assert dropped == 1 and len(turns) == 3
    (tmp_path / "x_metadata.csv").write_text("dataset,encounter_id,id\nvirtassist,D2N088,VA045\naci,D2N001,A1\n")
    assert aci_subsets(tmp_path) == {"D2N088": "virtassist", "D2N001": "aci"}


def _rec(name, answer, evidence=(1,)):
    return {"generation_info": {"name": name}, "label": {"answer": answer, "evidence": list(evidence), "summary": "s", "tags": []}}


def test_two_golds_keeps_the_teacher_label_and_carries_the_opus_label_separately():
    r = {**_rec("claude:sonnet", "pass", (1,)), "id": "x"}
    teacher, opus, prov = two_golds(r, None, None)
    assert teacher.answer == "pass" and opus is None and prov.method == "teacher" and prov.opus_labeller is None
    teacher, opus, prov = two_golds(r, _rec("ollama:x", "fail", (4,)), _rec("claude:opus", "fail", (2,)))
    assert teacher.answer == "pass" and teacher.evidence == [1]  # the teacher's label is never merged or overridden
    assert opus is not None and opus.answer == "fail" and opus.evidence == [2] and prov.opus_labeller == "claude:opus"
    assert prov.labels == {"claude:sonnet": "pass", "ollama:x": "fail", "claude:opus": "fail"}
    teacher, opus, prov = two_golds(r, None, _rec("claude:opus", "fail"), blind=_rec("claude:opus", "pass", (3,)))
    assert opus.answer == "pass" and opus.evidence == [3] and prov.opus_labeller == "claude:opus blind"  # a blind Opus label outranks an adjudication
    assert prov.labels["claude:opus"] == "fail" and prov.labels["claude:opus blind"] == "pass"


def test_release_record_strips_billing_fields_and_marks_the_split():
    r = {"id": "a", "track": "track-p", "generation_info": {"name": "claude:sonnet", "labelled_variant": "clean", "cost_usd": 0.12,
                                                             "claude_account": "w", "timestamp": "t"}}
    out = release_record(r, "benchmark")
    assert out["split"] == "benchmark" and out["generation_info"] == {"name": "claude:sonnet", "labelled_variant": "clean", "timestamp": "t"}
    assert "claude_account" not in json.dumps(out) and "cost_usd" not in json.dumps(out)


def test_claude_calls_start_from_an_empty_directory_and_time_out_cleanly(monkeypatch, tmp_path):
    from src.synthesis import llm
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(llm, "CLAUDE_CWD", tmp_path / "empty")
    monkeypatch.setattr(llm.subprocess, "run", fake_run)
    with pytest.raises(llm.LLMError):
        llm._claude("p", "sonnet", 5)
    assert seen["cwd"] == tmp_path / "empty" and (tmp_path / "empty").is_dir() and not any((tmp_path / "empty").iterdir())


def test_evaluate_reports_clean_against_messy_on_the_shared_pairs(tmp_path):
    q = {"id": "q1", "family": "general_qa", "text": "?", "options": [{"value": "pass", "criteria": ""}, {"value": "fail", "criteria": ""}], "tags": []}
    def rec(i, kinds, dataset):
        return {"id": i, "dataset": dataset, "track": "track-p", "cell": "seen_q", "question": q,
                "label": {"answer": "pass", "evidence": [1], "summary": "", "tags": []},
                "transcript": {"variants": [{"kind": k, "origin": "", "lines": ["a: x", "b: y"]} for k in kinds], "speakers": ["a", "b"], "role_source": ""}}
    split = tmp_path / "b.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in [rec("both", ["clean", "messy"], "d1"), rec("messy-only", ["messy"], "d2")]) + "\n")
    good = json.dumps({"evidence": [1], "answer": "pass", "summary": "s"})
    out = tmp_path / "o.jsonl"
    out.write_text("\n".join(json.dumps(o) for o in [{"id": "both", "variant": "clean", "text": good, "prompt_tokens": 10},
                                                    {"id": "both", "variant": "messy", "text": good, "prompt_tokens": 10},
                                                    {"id": "messy-only", "variant": "messy", "text": "", "prompt_tokens": 10}]) + "\n")
    res = evaluate(split, out)
    assert res["variants"]["messy"]["overall"]["n"] == 2 and res["variants"]["clean"]["overall"]["n"] == 1
    assert res["shared"]["n_pairs"] == 1
    assert res["shared"]["variants"]["clean"]["overall"]["n"] == 1 and res["shared"]["variants"]["messy"]["overall"]["n"] == 1
    assert res["shared"]["variants"]["messy"]["overall"]["accuracy"] == 1.0  # the messy-only failure is outside the shared view


def test_evaluate_scores_the_teacher_gold_everywhere_and_the_opus_gold_on_its_pairs(tmp_path):
    q1 = {"id": "q1", "family": "general_qa", "text": "?", "options": [{"value": "pass", "criteria": ""}, {"value": "fail", "criteria": ""}], "tags": []}
    def rec(i, ans, opus=None, cell="seen_q"):
        return {"id": i, "dataset": "d", "track": "track-p", "cell": cell, "question": q1, "provenance": {"method": "teacher", "labels": {}},
                "label": {"answer": ans, "evidence": [1], "summary": "", "tags": []},
                "opus_label": None if opus is None else {"answer": opus, "evidence": [2], "summary": "", "tags": []},
                "transcript": {"variants": [{"kind": "clean", "origin": "", "lines": ["a: x", "b: y"]}], "speakers": ["a", "b"], "role_source": ""}}
    recs = [rec("1", "pass"), rec("2", "pass"), rec("3", "pass", opus="fail"), rec("4", "fail", opus="fail", cell="unseen_q")]
    split = tmp_path / "b.jsonl"; split.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    out = tmp_path / "o.jsonl"
    out.write_text("\n".join(json.dumps({"id": r["id"], "variant": "clean", "prompt_tokens": 5,
                                          "text": json.dumps({"evidence": [1], "answer": "pass", "summary": "s"})}) for r in recs) + "\n")
    res = evaluate(split, out)
    t = res["variants"]["clean"]
    assert res["gold"] == "teacher" and t["overall"]["n"] == 4 and t["overall"]["accuracy"] == 0.75  # teacher gold: three pass, one fail
    assert t["floor=majority_answer"]["accuracy"] == 0.75 and t["floor=majority_answer"]["format_valid"] == 1.0
    assert t["cell=unseen_q_no_twin"]["n"] == 1
    o = res["opus"]
    assert o["n_pairs"] == 2 and o["variants"]["clean"]["overall"]["n"] == 2  # only the pairs Opus labelled
    assert o["variants"]["clean"]["overall"]["accuracy"] == 0.0 and o["variants"]["clean"]["overall"]["ev_f1"] == 0.0  # Opus says fail, evidence [2]
    assert o["variants"]["clean"]["floor=teacher_majority_answer"]["n"] == 2  # the teacher's majority applied to the Opus pairs
    assert "gold_method=agreement" not in t  # no pipeline-method strata any more


def test_majority_tie_break_is_deterministic():
    from src.train.evaluate import _majority
    assert _majority(["b", "a", "a", "b"]) == "a" and _majority(["fail", "pass", "pass"]) == "pass"


def test_evaluate_refuses_a_pre_two_gold_benchmark_and_fills_every_variant(tmp_path):
    q1 = {"id": "q1", "family": "general_qa", "text": "?", "options": [{"value": "pass", "criteria": ""}, {"value": "fail", "criteria": ""}], "tags": []}
    def rec(i, method="teacher", kinds=("clean", "messy")):
        return {"id": i, "dataset": "d", "track": "track-p", "cell": "seen_q", "question": q1, "provenance": {"method": method, "labels": {}},
                "label": {"answer": "pass", "evidence": [1], "summary": "", "tags": []}, "opus_label": None,
                "transcript": {"variants": [{"kind": k, "origin": "", "lines": ["a: x", "b: y"]} for k in kinds], "speakers": ["a", "b"], "role_source": ""}}
    stale = tmp_path / "old.jsonl"; stale.write_text(json.dumps(rec("1", "agreement")) + "\n")
    out = tmp_path / "o.jsonl"; out.write_text(json.dumps({"id": "1", "variant": "clean", "prompt_tokens": 5, "text": ""}) + "\n")
    with pytest.raises(SystemExit):
        evaluate(stale, out)
    split = tmp_path / "b.jsonl"; split.write_text(json.dumps(rec("1")) + "\n" + json.dumps(rec("2", kinds=("messy",))) + "\n")
    good = json.dumps({"evidence": [1], "answer": "pass", "summary": "s"})
    out.write_text(json.dumps({"id": "1", "variant": "clean", "prompt_tokens": 5, "text": good}) + "\n"
                   + json.dumps({"id": "2", "variant": "clean", "prompt_tokens": 5, "text": good}) + "\n")  # record 2 has no clean variant: ignored
    res = evaluate(split, out)
    assert res["variants"]["clean"]["overall"]["n"] == 1 and res["variants"]["messy"]["overall"]["n"] == 2  # messy never generated: all empty
    assert res["variants"]["messy"]["overall"]["accuracy"] == 0.0 and res["missing_outputs"] == 2
    assert res["shared"]["n_pairs"] == 1


def test_two_golds_refuses_an_opus_label_from_another_labeller():
    r = {**_rec("ollama:qwen3.8", "pass"), "id": "x"}
    with pytest.raises(SystemExit):
        two_golds(r, None, _rec("ollama:gemma4:12b", "fail"))
