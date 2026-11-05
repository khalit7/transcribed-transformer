import json
import re

from src.synthesis import question_bank as v1
from src.synthesis_v2 import question_bank as v2


def test_bank_ids_are_unique_well_formed_and_disjoint_from_v1():
    ids = [q.id for q in v2.BANK]
    assert len(ids) == len(set(ids)) == 44  # D80: 32 trained, 12 held out
    assert all(re.fullmatch(r"v2-[a-z]+-\d\d-[a-z0-9-]+", i) for i in ids)
    assert not set(ids) & {q.id for q in v1.all_questions(labelled_path=v1.OUT_DIR / "nonexistent.jsonl")}
    assert {q.family for q in v2.BANK} == {"podcast"} and all(q.dataset_allow_list == ["sporc"] for q in v2.BANK)


def test_options_have_distinct_values_and_criteria():
    for q in v2.BANK:
        assert 3 <= len(q.options) <= 4, q.id
        assert q.description and not q.description.startswith(v2.VOCAB), q.id  # stored once, not per question
        values = [o.value for o in q.options]
        if "two_plus" in values:  # counts run in ascending order, none first
            assert values[0] == "none" and values[-1] == "two_plus", q.id
        else:
            assert values[-1] == "none" or q.id == "v2-seg-01-structure", q.id
        assert len({o.value for o in q.options}) == len(q.options), q.id
        assert all(o.criteria.strip() and len(o.criteria) > 20 for o in q.options), q.id


def test_held_out_and_rare_event_lists_name_bank_questions():
    ids = {q.id for q in v2.BANK}
    # rare-event detectors stay declared after a cut so the pilot reports that used them can be regenerated
    assert set(v2.HELD_OUT) <= ids and set(v2.RARE_EVENT) <= ids | set(v2.REMOVED)
    assert len(ids) == 44 and len(v2.HELD_OUT) == 12  # round four (D80): 32 trained, 12 held out
    # the pilot's cuts left the bank and their ids are not reused
    assert not set(v2.REMOVED) & ids and len(v2.REMOVED) == 37  # 23 cut by the pilots, 14 argument templates
    assert not set(v2.REMOVED) & set(v2.HELD_OUT) and not {q.id for q in v2.PROVISIONAL} & (ids | set(v2.REMOVED))
    assert set(v2.PROVISIONAL_HELD_OUT) <= {q.id for q in v2.PROVISIONAL}
    assert not any(i.startswith(d) for i in ids for d in v2.DROPPED_DRAFTS)  # dropped draft ids are never reused
    by = v2.by_id()  # the shared definition sentences are not forked
    for qid in ("v2-num-01-figure-sourced", "v2-num-02-figure-judged"):
        assert v2.NUMBER in by[qid].description, qid
    for qid in ("v2-fam-01-family-returned",):
        assert v2.LINES3 in by[qid].description, qid


def test_no_question_refers_to_line_ranges_or_takes_an_argument():
    for q in v2.BANK + v2.PROVISIONAL:
        for s in [q.text, *(o.criteria for o in q.options)]:
            assert not re.search(r"\{[a-zA-Z_]+\}", s), (q.id, s)  # no per-episode argument slot (D78)
            assert not re.search(r"\blines? \d|\bline \{", s), (q.id, s)


def test_write_questions_splits_approved_and_provisional(tmp_path):
    assert v2.write_questions(tmp_path) == (len(v2.BANK), len(v2.PROVISIONAL))
    rows = [json.loads(line) for line in (tmp_path / "questions.jsonl").open()]
    assert [r["id"] for r in rows] == [q.id for q in v2.BANK] and sum(r["held_out"] for r in rows) == len(v2.HELD_OUT)
    assert "template" not in rows[0]
    prov = tmp_path / "questions_provisional.jsonl"
    assert prov.exists() == bool(v2.PROVISIONAL)
    if v2.PROVISIONAL:
        assert len(prov.read_text().splitlines()) == len(v2.PROVISIONAL)


def test_write_questions_removes_a_stale_provisional_file(tmp_path, monkeypatch):
    stale = tmp_path / "questions_provisional.jsonl"
    stale.write_text("{}\n")
    monkeypatch.setattr(v2, "PROVISIONAL", [])
    v2.write_questions(tmp_path)
    assert not stale.exists() and (tmp_path / "questions.jsonl").exists()


def test_bank_is_every_round_less_removed():
    every = v2.ROUNDS_1_TO_3 + v2.ROUND4 + v2.ROUND5
    assert [q.id for q in v2.BANK] == [q.id for q in every if q.id not in v2.REMOVED]
    held = set(v2.HELD_OUT)
    assert len(v2.BANK) - len(held) == 32 and len(held) == 12
    assert held == ((set(v2.ROUNDS_1_TO_3_HELD_OUT) | set(v2.ROUND4_HELD_OUT)) - set(v2.REMOVED)
                    - set(v2.MOVED_TO_TRAINING)) | set(v2.MOVED_TO_HELD_OUT)
    assert "v2-per-03-story-answered" not in held and "v2-agr-01-agreement-stated" in held  # the twin audit (D85)
    assert not set(v2.MOVED_TO_HELD_OUT) & set(v2.REMOVED) and set(v2.MOVED_TO_HELD_OUT) <= {q.id for q in v2.BANK}


def test_definitions_are_added_back_once_and_only_to_bank_questions():
    q = v2.BANK[0]
    full = v2.with_definitions(q)
    assert full.description == f"{v2.VOCAB} {q.description}" and v2.with_definitions(full) == full  # idempotent
    assert v2.without_definitions(full) == q and v2.without_definitions(q) == q
    other = q.model_copy(update={"id": "x-podcast"})
    assert v2.with_definitions(other) == other  # not a bank question
