"""Shared pieces for the transfer datasets (question 3: does the capacity finding hold off the compliance task?).

A transfer dataset is a public long-input, short-output task with its own gold (ECtHR). Its loader emits
records in the labelled-data schema, `data/labelled_data/<name>/{train,val,benchmark}.jsonl`, so the fine-tuning,
generation and scoring code read them unchanged. Where the compliance records differ, the transfer records say so:

- the input is a written document, not a conversation, so there are no speaker roles. Every line is one paragraph and
  carries the fixed label `text` where a role would stand (`transcript.role_source` records this). The rendering a model
  sees is still `<n>: text: <paragraph>`, which keeps the line numbers meaningful should evidence ever be labelled;
- the question's `family` is `transfer` and its options are the dataset's own vocabulary (the schema relaxes the
  pass/fail requirement for this family only);
- the gold ships with the dataset: `label.evidence` is empty and `label.summary` blank (neither is annotated), the
  benchmark records carry `provenance.method = "dataset"`, and there is no second gold (`opus_label` is null).

Raw files are read from the gitignored `data/raw/<name>/`; nothing here downloads.
"""

import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from src.preprocessing.common import REPO_ROOT, VAL_FRACTION, is_val
from src.synthesis.schema import (
    BenchmarkRecord,
    Generation,
    Label,
    LabelledRecord,
    Provenance,
    Question,
    Transcript,
    Variant,
)

LABELLED = REPO_ROOT / "data" / "labelled_data"
ROLE = "text"
ROLE_SOURCE = "none: written document, one paragraph per line; the fixed label `text` stands where a speaker role would"
_WS = re.compile(r"\s+")


def document(paras: Iterable[str], origin: str) -> Transcript:
    """One line per paragraph. Whitespace inside a paragraph is collapsed to single spaces: an embedded newline (ECtHR
    quotes passages across lines) would otherwise break the one-paragraph-per-numbered-line rendering."""
    lines = [f"{ROLE}: {_WS.sub(' ', p).strip()}" for p in paras]
    return Transcript(variants=[Variant(kind="clean", origin=origin, lines=lines)], speakers=[ROLE],
                      role_source=ROLE_SOURCE)


def gold(answer: str) -> Label:
    return Label(answer=answer, evidence=[], summary="", tags=[], confidence=None)


def generation(dataset: str) -> Generation:
    return Generation(name=f"dataset:{dataset}", labelled_variant="clean", cost_usd=0.0,
                      timestamp=datetime.now(UTC).isoformat(timespec="seconds"))


def record(*, dataset: str, doc_id: str, source_id: str, track: str, question: Question, transcript: Transcript,
           answer: str, meta: dict, benchmark: bool, cell: str = "seen_q") -> LabelledRecord:
    """One (document, question) pair. Benchmark records carry the dataset's own gold as their provenance."""
    base = {"id": f"{doc_id}::{question.id}", "dataset": dataset, "source_id": source_id, "track": track,
            "question": question, "transcript": transcript, "label": gold(answer), "generation_info": generation(dataset),
            "meta": meta}
    if not benchmark:
        return LabelledRecord(**base)
    return BenchmarkRecord(**base, cell=cell, provenance=Provenance(method="dataset", labels={"dataset": answer}),
                           opus_label=None)


class SplitWriter:
    """Streams records into train/val/benchmark JSONL under data/labelled_data/<name>/ and keeps the datasheet counts.

    `split_of(container_id)` decides train against val for the records the caller does not route to the benchmark,
    by the same deterministic hash the pretraining loaders use, so re-runs agree.
    """

    def __init__(self, name: str, out_dir: Path | None = None):
        self.dir = out_dir or LABELLED / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._files = {s: open(self.dir / f"{s}.jsonl", "w") for s in ("train", "val", "benchmark")}  # noqa: SIM115
        self.records = {"train": 0, "val": 0, "benchmark": 0}
        self.docs: dict[str, set[str]] = {"train": set(), "val": set(), "benchmark": set()}
        self.lines: dict[str, list[int]] = {"train": [], "val": [], "benchmark": []}
        self.answers: dict[str, dict[str, int]] = {"train": {}, "val": {}, "benchmark": {}}

    @staticmethod
    def split_of(container_id: str, val_fraction: float = VAL_FRACTION) -> str:
        return "val" if is_val(container_id, val_fraction) else "train"

    def add(self, rec: LabelledRecord, split: str) -> None:
        self._files[split].write(rec.model_dump_json() + "\n")
        self.records[split] += 1
        doc = rec.id.split("::")[0]
        if doc not in self.docs[split]:
            self.docs[split].add(doc)
            self.lines[split].append(rec.transcript.n_lines)
        a = self.answers[split]
        a[rec.label.answer] = a.get(rec.label.answer, 0) + 1

    def close(self, build: dict | None = None) -> dict:
        """`build`: the arguments the records were built with (tasks, held-out articles), kept in summary.json so a
        rebuild with other arguments is detectable."""
        for f in self._files.values():
            f.close()

        def dist(xs: list[int]) -> dict:
            ls = sorted(xs)
            n = len(ls)
            pct = lambda p: ls[min(n - 1, int(p * n))] if n else 0
            return {"min": ls[0] if n else 0, "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                    "max": ls[-1] if n else 0}

        summary = {
            "build": build or {},
            "records": self.records,
            "docs": {s: len(d) for s, d in self.docs.items()},
            "lines_per_doc": {s: dist(xs) for s, xs in self.lines.items()},
            "answers": self.answers,
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
        return summary
