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
