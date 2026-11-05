import collections
import json
from pathlib import Path

import pytest

from src.synthesis_v2 import benchmark, cases, label_episodes, question_bank, split
from src.synthesis_v2.episodes import STRATA

SIZES = {"train": 6, "val": 2, "benchmark": 4}
QIDS = ["q1", "q2"]


def _eps() -> dict[str, dict]:
    """12 episodes per band: e0..e7 eligible singletons, e8/e9 one shared podcast, e10 incomplete, e11 a design episode."""
    eps = {}
    for b, band in enumerate(STRATA):
        for i in range(12):
            ep = f"{b}e{i}"
            answers = {"q1": "rare" if i == 0 else "common", "q2": "x" if i % 2 else "y"}
            if i == 10:
                del answers["q2"]
            eps[ep] = {"band": band, "podcast": f"{b}p8" if i in (8, 9) else f"{b}p{i}", "show": f"show {ep}",
                       "title": f"episode {ep}", "duration": 100.0 * i, "passages": {hash(ep)},
                       "category": "comedy" if i % 3 else "news", "cell": f"0,{i % 2}", "answers": answers}
    return eps


def _ep(podcast: str, show: str, title: str, duration: float, passages: set[int] | None = None) -> dict:
    return {"podcast": podcast, "show": show, "title": title, "duration": duration, "passages": passages}


def _build(seed: int = 0) -> dict:
    eps = _eps()
    design = {f"{b}e11" for b in range(len(STRATA))}
    return split.build(eps, QIDS, {"q2"}, design, seed, sizes=SIZES, draws=20, swaps=200)


def test_sizes_are_exact_per_band_and_only_eligible_episodes_reach_val_or_benchmark():
    s = _build()
    for b, band in enumerate(STRATA):
        sides = collections.Counter(side for ep, side in s["episodes"].items() if ep.startswith(f"{b}e"))
        assert sides == {"train": 6, "val": 2, "benchmark": 4}, band
        for i in (8, 9, 10, 11):  # two-episode podcast, incomplete labels, bank-design episode
            assert s["episodes"][f"{b}e{i}"] == "train"
        assert s["report"][band]["design_episodes_in_train"] == 1
    assert s["held_out_questions"] == ["q2"]


def test_rare_answer_floor_pulls_the_only_rare_episode_into_the_benchmark_and_draws_are_seeded():
    eps = _eps()
    for band in STRATA:  # a class of two in the pool gets a floor of one
        eps[f"{list(STRATA).index(band)}e1"]["answers"]["q1"] = "rare"
    s = split.build(eps, QIDS, {"q2"}, set(), 0, sizes=SIZES, draws=5, swaps=500)
    for b, band in enumerate(STRATA):
        assert "rare" in {eps[ep]["answers"]["q1"] for ep, side in s["episodes"].items()
                          if side == "benchmark" and ep.startswith(f"{b}e")}
        assert s["report"][band]["rare_floors_short"] == {}
    assert _build(3)["episodes"] == _build(3)["episodes"]


def test_build_refuses_a_band_of_the_wrong_size_or_too_small_a_pool():
    eps = _eps()
    del eps["0e0"]
    with pytest.raises(SystemExit):
        split.build(eps, QIDS, set(), set(), 0, sizes=SIZES, draws=5, swaps=5)
    with pytest.raises(SystemExit):  # 9 eligible per band (no design set) cannot give 4 benchmark + 6 val
        split.build(_eps(), QIDS, set(), set(), 0, sizes={"train": 2, "val": 6, "benchmark": 4}, draws=5, swaps=5)


def test_group_links_feeds_show_titles_reposted_episodes_and_shared_transcripts():
    words = [f"w{i}" for i in range(200)]
    lines = ["SPEAKER_00: " + " ".join(words)]
    shared = split.passages(lines)
    assert shared and split.passages(["host_1: " + " ".join(words)]) == shared  # speaker labels do not matter
    eps = {
        "a": _ep("f1", "Inspiring Minds", "x", 10, set()), "b": _ep("f2", "inspiring minds.", "y", 99, set()),
        "c": _ep("f3", "Visions In The Making", "The Life, Of A Mother", 3053, set()),
        "d": _ep("f4", "Visions in the making podcast", "z", 1, set()),
        "e": _ep("f5", "A Slob Comes Clean", "255: Decluttering", 2825, set()),
        "f": _ep("f6", "podcasts Archives - Dana K. White", "255 decluttering", 2824, set()),
        "g": _ep("f7", "Bronwyn Allen", "q", 432, shared), "h": _ep("f8", "Joel Riley", "r", 445, shared | {1, 2}),
        "i": _ep("f9", "Unrelated", "255: Decluttering", 2000, set()),  # same title, other duration: not linked
        "j": _ep("f10", "Alone", "s", 5, set(range(100))),
        "k": {**_ep("f11", "Select Cuts", "u", 7, set()), "owners": ["armstrong getty"]},
        "m": {**_ep("f12", "One More Thing", "v", 8, set()), "owners": ["armstrong getty"]},
    }
    roots, links = split.group(eps)
    same = lambda a, b: roots[a] == roots[b]
    assert same("a", "b") and same("c", "d") and same("e", "f") and same("g", "h")
    assert split.norm_show("Master Your Finances(Official 107.7 The Bronc Podcast)") == split.norm_show("Master Your Finances")
    assert split.norm_show("[CLIC] Connect") == "clic connect" and split.norm_show("(Re)Watch") == "re watch"
    assert split.norm_show("The (not so) Great Outdoors") != split.norm_show("The podcast")
    assert split.norm_show("The From Imposter to Empowered Podcast") == split.norm_show("From Imposter to Empowered")
    assert split.norm_show("The Podcast") == "the" and split.norm_show("The") == "the"  # never emptied
    assert split.sporc_rows(set()) == {}
    assert not same("e", "i") and len({roots[x] for x in "acegij"}) == 6
    assert same("k", "m")
    assert set(links) == {"show title", "episode title and duration", "transcript overlap", "itunes owner or author"}


def test_a_design_episode_outside_the_selection_keeps_its_show_out_of_val_and_benchmark():
    eps = _eps()
    eps["0e0"]["show"] = "Designed Show"
    outside = {"zz": _ep("other-feed", "designed show", "t", 1.0)}
    s = split.build(eps, QIDS, {"q2"}, {"zz"}, 0, outside=outside, sizes=SIZES, draws=5, swaps=50)
    assert s["episodes"]["0e0"] == "train" and "zz" not in s["episodes"]
    assert s["grouping"]["groups_of_two_or_more"] == 4  # the 8/9 feed pair in each band; zz is not a selected episode


def test_load_episodes_rejects_unknown_episodes_band_mismatches_and_duplicates(tmp_path):
    sel = tmp_path / "selection.jsonl"
    sel.write_text(json.dumps({"doc_id": "a1", "stratum": "0-4k", "podcast_id": "p", "pod_title": "s", "category": "c",
                               "cell": "0,0", "duration_s": 1.0}) + "\n")
    meta = tmp_path / "meta.jsonl"
    meta.write_text(json.dumps({"doc_id": "a1", "ep_title": "t"}) + "\n")
    rec = lambda ep, q, band="0-4k": json.dumps({"id": f"sporc-{ep}::{q}", "meta": {"stratum": band},
                                                 "question": {"id": q}, "label": {"answer": "x"},
                                                 "transcript": {"variants": [{"lines": ["host_1: a b c"]}]}})
    led = tmp_path / "ledger.jsonl"
    for bad in ([rec("zz", "q")], [rec("a1", "q", "4-8k")], [rec("a1", "q"), rec("a1", "q")]):
        led.write_text("\n".join(bad) + "\n")
        with pytest.raises(SystemExit):
            split.load_episodes(led, sel, meta)
    led.write_text(rec("a1", "q") + "\n")
    e = split.load_episodes(led, sel, meta)["a1"]
    assert e["answers"] == {"q": "x"} and e["title"] == "t" and e["passages"] == set()


def test_main_refuses_to_overwrite_a_frozen_split(tmp_path):
    out = tmp_path / "splits.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        split.main(["--out", str(out)])
    assert out.read_text() == "{}"


def test_assign_drops_held_out_questions_off_the_benchmark_and_refuses_unknown_episodes():
    s = {"episodes": {"a": "train", "b": "val", "c": "benchmark"}, "held_out_questions": ["h"]}
    rec = lambda ep, q: {"id": f"sporc-{ep}::{q}", "question": {"id": q}}
    assert [split.assign(rec(ep, "h"), s) for ep in "abc"] == ["drop", "drop", "benchmark"]
    assert [split.assign(rec(ep, "t"), s) for ep in "abc"] == ["train", "val", "benchmark"]
    with pytest.raises(KeyError):
        split.assign(rec("z", "t"), s)


def test_benchmark_build_writes_three_files_with_cells_and_a_single_gold(monkeypatch, tmp_path):
    qs = question_bank.BANK[:2]
    answers = {q.id: {"evidence": [1], "answer": q.values[-1], "summary": "s", "confidence": 0.9} for q in qs}
    monkeypatch.setattr(label_episodes, "ask_json_full",
                        lambda prompt, model, schema, **kw: (answers, 0.1, {"attempts": 1, "cost_last": 0.1, "cost_total": 0.1, "usage": {}}))
    ledger = tmp_path / "labelled_data.jsonl"
    with ledger.open("w") as f:
        for ep in ("ea", "eb", "ec"):
            text = "\n".join(f"SPEAKER_0{i % 2}: turn {i} of a long conversation here" for i in range(6))
            recs, failed = label_episodes.label_case(cases.build(ep, text, ["SPEAKER_00"], {"stratum": "0-4k"}),
                                                     qs, "claude:opus", 2, 0, "0-4k")
            assert failed == 0
            for r in recs:
                f.write(r.model_dump_json() + "\n")
    splits = {"episodes": {"ea": "train", "eb": "val", "ec": "benchmark"}, "held_out_questions": [qs[1].id],
              "questions": [q.id for q in qs]}
    splits["_held"] = {qs[1].id}
    outs = {s: tmp_path / f"{s}.jsonl" for s in ("train", "val", "benchmark")}
    counts = benchmark.build(ledger, splits, outs, _check=False)
    assert counts == {"benchmark/0-4k/seen_q": 1, "benchmark/0-4k/unseen_q": 1, "drop/0-4k": 2,
                      "train/0-4k": 1, "val/0-4k": 1}
    bench = [json.loads(line) for line in outs["benchmark"].open()]
    assert {b["cell"] for b in bench} == {"seen_q", "unseen_q"}
    assert all(b["provenance"] == {"method": "teacher", "labels": {"claude:opus": b["label"]["answer"]},
                                   "opus_labeller": None, "adjudicator": None} and b["opus_label"] is None for b in bench)
    assert [json.loads(line)["question"]["id"] for line in outs["train"].open()] == [qs[0].id]
    # meta.held_out comes from the split, not from the flag written at labelling time
    assert all(b["meta"]["held_out"] == (b["question"]["id"] == qs[1].id) for b in bench)
    assert [json.loads(line)["meta"]["held_out"] for line in outs["val"].open()] == [False]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())
    # a ledger out of step with the split stops the build and leaves the files as they were
    before = outs["train"].read_text()
    lines = ledger.read_text().splitlines()
    last = json.loads(lines[-1])
    stray = lines[-1].replace(last["question"]["id"], "v2-not-a-question")  # the same count, one wrong question id
    for bad in (lines + lines[-1:], lines[:-1], lines[:-1] + [stray]):  # duplicate id; one label short; a stray question id
        ledger.write_text("\n".join(bad) + "\n")
        with pytest.raises(SystemExit):
            benchmark.build(ledger, splits, outs, _check=False)
        assert outs["train"].read_text() == before and not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_check_bank_refuses_a_split_or_a_questions_file_out_of_step_with_the_bank():
    qids = [q.id for q in question_bank.BANK]
    held = set(question_bank.HELD_OUT)
    splits = {"questions": qids, "held_out_questions": sorted(held)}
    benchmark.check_bank(splits, qids, held)
    swapped = (held - {"v2-agr-01-agreement-stated"}) | {"v2-per-03-story-answered"}  # the pre-D85 held-out set
    for s_q, s_h, f_q, f_h in ((qids, swapped, qids, held),            # a split frozen before the swap
                               (qids, swapped, qids, swapped),         # ... and a questions.jsonl never regenerated
                               (qids[:-1], held, qids, held),          # a question missing from the split
                               (qids[::-1], held, qids[::-1], held)):  # a reordered file
        with pytest.raises(SystemExit):
            benchmark.check_bank({"questions": s_q, "held_out_questions": sorted(s_h)}, f_q, f_h)


def test_replay_sample_keeps_a_superset_of_what_the_adaptation_sampler_took(tmp_path):
    import random
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"doc_id": f"d{i}", "text": f"SPEAKER_00: words {i}",
                                        "meta": {"podcast_id": f"p{i}", "pod_title": f"s{i}", "duration_s": 1.0}}) + "\n"
                            for i in range(400)))
    want, total, seed, exclude = 20, 200, 0, {"d3", "d5"}
    rng = random.Random(f"{seed}|{path}")  # the loop of src.train.data.build_corpus_examples, side "train"
    taken_ids, kept = set(), set()
    for i in range(400):
        if rng.random() > want / total:
            continue
        kept.add(f"d{i}")
        if f"d{i}" not in exclude and len(taken_ids) < want:
            taken_ids.add(f"d{i}")
    nodes, extra = split.replay_sample(path, want, total, seed, also={"d399", "d0"})
    assert set(nodes) == kept and taken_ids < kept  # every draw kept to the end of the file, the excluded ones too
    assert set(extra) == {"d399", "d0"} and extra["d399"]["podcast"] == "p399"  # v1 documents come from the same pass
    assert split.replay_sample(path, want, total, seed)[0] == nodes  # asking for extra documents changes no draw
    first = min(nodes)
    assert nodes[first]["podcast"] == f"p{first[1:]}" and nodes[first]["show"] == f"s{first[1:]}"


def test_adaptation_seen_episodes_and_shared_hosts_stay_out_of_val_and_benchmark():
    eps = _eps()
    eps["0e0"]["hosts"] = ["Jonathan Ellsworth"]
    eps["0e1"]["hosts"] = ["jonathan ellsworth", "Ann"]
    eps["0e2"]["hosts"] = ["Ann"]  # a first name alone links nothing
    seen = {"1e0": {**eps["1e0"]},  # a selected episode the adaptation read
            "x1": _ep("feed-x", "show 2e3", "t", 1.0, set()),  # another episode of 2e3's show, read under its own id
            "x2": _ep("feed-y", "other", "t", 2.0, eps["3e4"]["passages"])}  # 3e4's transcript under another id
    s = split.build(eps, QIDS, {"q2"}, set(), 0, sizes=SIZES, draws=5, swaps=50, seen=seen)
    assert s["episodes"]["0e0"] == s["episodes"]["0e1"] == "train"
    assert s["episodes"]["1e0"] == s["episodes"]["2e3"] == s["episodes"]["3e4"] == "train"
    assert "x1" not in s["episodes"] and s["report"]["4-8k"]["trained_on_in_train"] == 1
    assert s["trained_on"] == {"replayed": [], "selected_episodes_trained_on": 1, "selected_episodes_in_a_trained_on_group": 3}


def test_adaptation_seen_reads_configs_like_the_trainer_and_refuses_gaps(tmp_path, monkeypatch):
    import random
    monkeypatch.chdir(tmp_path)  # configs name the corpus by a relative path, and the sampler seeds on that string
    corpus = tmp_path / "data" / "interim" / "sporc"
    corpus.mkdir(parents=True)
    for side, n in (("train", 300), ("val", 30)):
        (corpus / f"{side}.jsonl").write_text("".join(json.dumps(
            {"doc_id": f"{side}{i}", "text": "SPEAKER_00: x", "meta": {"podcast_id": f"p{i}", "pod_title": "s",
                                                                       "duration_s": 1.0}}) + "\n" for i in range(n)))
    (corpus / "summary.json").write_text(json.dumps({"docs": {"train": 300, "val": 30}}))
    cfgs = tmp_path / "configs" / "adapt"
    cfgs.mkdir(parents=True)
    (cfgs / "a.yaml").write_text("seed: 3\nadapt:\n  corpora:\n    - {path: data/interim/sporc/train.jsonl, docs: 30, val_docs: 5}\n"
                                 "    - {path: data/interim/other/train.jsonl, docs: 9}\n")
    monkeypatch.setattr(split, "_with_rows", lambda nodes, what: None)
    seen, replayed = split.adaptation_seen(Path("configs/adapt"), v1_docs={"train7"})
    rng = random.Random("3|data/interim/sporc/train.jsonl")  # the trainer's seed string
    kept = {f"train{i}" for i in range(300) if rng.random() <= 30 / 300}
    assert kept | {"train7"} <= set(seen) and [r.get("path") for r in replayed[:2]] == [
        "data/interim/sporc/train.jsonl", "data/interim/sporc/val.jsonl"]
    with pytest.raises(SystemExit):  # a v1 episode missing from the interim files
        split.adaptation_seen(Path("configs/adapt"), v1_docs={"nowhere"})
    with pytest.raises(SystemExit):  # no config samples SPoRC
        split.adaptation_seen(tmp_path / "empty")


def test_with_rows_refuses_when_metadata_is_mostly_missing(monkeypatch):
    monkeypatch.setattr(split, "sporc_rows", lambda ids: {})
    with pytest.raises(SystemExit):
        split._with_rows({"a": {}, "b": {}}, "nodes")
