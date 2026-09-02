"""Shared pieces for the per-corpus preprocessing modules.

Every module turns one raw corpus under data/raw/<name>/ into
data/interim/<name>/{train,val}.jsonl, one Document per line, and prints the
counts that go into DATASHEET.md. Splits are by container (call, episode,
argument, meeting), never by turn, using a deterministic hash so re-runs agree.

The canonical document: `text` is the training-ready rendering. Where a corpus
has speaker turns it is one turn per line as `SPEAKER_NN: utterance`, the shape
diarised commercial ASR produces. Where a corpus ships flat text (CourtListener,
CallCenterEN) the text stays flat and `has_speakers` is False rather than
inventing turn boundaries.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW = REPO_ROOT / "data" / "raw"
INTERIM = REPO_ROOT / "data" / "interim"
VAL_FRACTION = 0.02


class Document(BaseModel):
    source: str
    doc_id: str
    track: str = Field(pattern=r"^track-(p|nc)$")
    tier: int = Field(ge=1, le=3)
    asr_system: str | None
    has_speakers: bool
    n_turns: int
    n_words: int
    text: str
    meta: dict[str, str | int | float | bool | None] = {}


def is_val(container_id: str, fraction: float = VAL_FRACTION) -> bool:
    h = int(hashlib.sha1(container_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def render_turns(turns: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in turns)


class Writer:
    """Streams documents into train/val JSONL and keeps the datasheet counts."""

    def __init__(self, name: str):
        self.dir = INTERIM / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {s: open(self.dir / f"{s}.jsonl", "w") for s in ("train", "val")}
        self.docs = {"train": 0, "val": 0}
        self.words = {"train": 0, "val": 0}
        self.turns = 0
        self.lengths: list[int] = []

    def add(self, doc: Document, container_id: str) -> None:
        split = "val" if is_val(container_id) else "train"
        self._files[split].write(doc.model_dump_json() + "\n")
        self.docs[split] += 1
        self.words[split] += doc.n_words
        self.turns += doc.n_turns
        self.lengths.append(doc.n_words)

    def close(self) -> dict:
        for f in self._files.values():
            f.close()
        ls = sorted(self.lengths)
        n = len(ls)
        pct = lambda p: ls[min(n - 1, int(p * n))] if n else 0  # noqa: E731
        summary = {
            "docs": self.docs,
            "words": self.words,
            "turns": self.turns,
            "words_per_doc": {
                "min": ls[0] if n else 0, "p50": pct(0.5), "p90": pct(0.9),
                "p99": pct(0.99), "max": ls[-1] if n else 0,
            },
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
        return summary
