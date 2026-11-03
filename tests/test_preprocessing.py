import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import common
from src.preprocessing.common import Document, Writer, is_val, render_turns


def test_render_turns_one_per_line():
    text = render_turns([("SPEAKER_00", "hi"), ("SPEAKER_01", "hello, um, yes")])
    assert text == "SPEAKER_00: hi\nSPEAKER_01: hello, um, yes"


def test_is_val_is_deterministic_and_roughly_two_percent():
    ids = [f"call-{i}" for i in range(20_000)]
    a = [is_val(i) for i in ids]
    assert a == [is_val(i) for i in ids]
    frac = sum(a) / len(a)
    assert 0.015 < frac < 0.025


def test_document_rejects_bad_track_and_tier():
    ok = {"source": "x", "doc_id": "1", "asr_system": None, "has_speakers": False,
          "n_turns": 0, "n_words": 1, "text": "a"}
    Document(track="track-p", tier=1, **ok)
    with pytest.raises(ValueError):
        Document(track="commercial", tier=1, **ok)
    with pytest.raises(ValueError):
        Document(track="track-p", tier=4, **ok)


def test_writer_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "INTERIM", tmp_path)
    w = Writer("demo")
    for i in range(50):
        w.add(Document(source="demo", doc_id=str(i), track="track-p", tier=1, asr_system="t",
                       has_speakers=True, n_turns=2, n_words=3, text="SPEAKER_00: a b\nSPEAKER_01: c"),
              container_id=f"c{i}")
    s = w.close()
    assert s["docs"]["train"] + s["docs"]["val"] == 50
    rows = [json.loads(l) for l in (tmp_path / "demo" / "train.jsonl").open()]
    assert rows[0]["text"].startswith("SPEAKER_00: ")
    assert {r["track"] for r in rows} == {"track-p"}


# --- transfer dataset (ECtHR): labelled-data records with the dataset's own gold ---

def _ecthr_case(n: int, labels: list[int]) -> dict:
    return {"text": [f"{n}.  The applicant was born in 19{n:02d}.", "7.  Facts continue."], "labels": labels}


def test_ecthr_loader_decomposes_labels_and_holds_out_articles(tmp_path):
    from src.preprocessing import ecthr
    from src.synthesis.schema import BenchmarkRecord, LabelledRecord
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in ("train", "validation", "test"):
        for task in ("a", "b"):
            (raw / f"ecthr_{task}.{split}.jsonl").write_text(
                json.dumps(_ecthr_case(1, [3, 9])) + "\n" + json.dumps(_ecthr_case(2, [])) + "\n")
    s = ecthr.main(raw=raw, tasks=("a", "b"), held_out=frozenset({"9"}), out_dir=tmp_path / "out")
    # 2 cases x 2 tasks x 10 articles, minus the held-out article's 2 questions per case per task off train and val
    assert s["records"] == {"train": 36, "val": 36, "benchmark": 40}
    rows = [LabelledRecord.model_validate(json.loads(l)) for l in (tmp_path / "out" / "train.jsonl").open()]
    assert {r.question.id for r in rows} == {f"ecthr-{t}-{c}" for t in "ab" for c, _ in ecthr.ARTICLES if c != "9"}
    by_id = {r.id: r for r in rows}
    assert by_id["ecthr-train-00000::ecthr-a-6"].label.answer == "violated"
    assert by_id["ecthr-train-00000::ecthr-a-2"].label.answer == "not_violated"
    assert by_id["ecthr-train-00000::ecthr-b-6"].label.answer == "alleged"
    assert by_id["ecthr-train-00001::ecthr-a-6"].label.answer == "not_violated"
    assert by_id["ecthr-train-00000::ecthr-a-6"].meta["label_set"] == "6,P1-1"
    assert rows[0].transcript.lines()[0] == "text: 1. The applicant was born in 1901."  # double space collapsed
    b = [BenchmarkRecord.model_validate(json.loads(l)) for l in (tmp_path / "out" / "benchmark.jsonl").open()]
    assert {x.cell for x in b if x.question.id.endswith("-9")} == {"unseen_q"}
    assert {x.cell for x in b if not x.question.id.endswith("-9")} == {"seen_q"}
    assert {x.provenance.method for x in b} == {"dataset"}


def test_document_lines_never_contain_newlines():
    from src.preprocessing.transfer import document
    t = document(["10.  The court stated:\n\"quoted\n  passage\"", "  11.  Next. "], "t")
    assert t.lines() == ['text: 10. The court stated: "quoted passage"', "text: 11. Next."]
