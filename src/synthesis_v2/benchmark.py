"""Build train.jsonl, val.jsonl and benchmark.jsonl for labelled data v2 from the ledger and the frozen split.

    uv run python -m src.synthesis_v2.benchmark

train / val: the ledger records `split.assign` sends there (held-out questions dropped), unchanged except `meta.held_out`,
which every output record takes from the frozen split (the flag written at labelling time predates the twin audit, D85).

benchmark: every benchmark pair, with `cell` (seen_q: a trained question; unseen_q: one of the 12 held out) and a
provenance naming the single gold. v2 has one labeller for every side (D67): Opus through the lean `claude -p` route
(D82), so `label` is Opus's label, `provenance.method` is `teacher` (the training labeller's own label, as in v1 since
D61) and `opus_label` stays empty: v1 used that field for a second, independent Opus pass on a selected subset, and v2
has no such pass.

The ledger must match the frozen split exactly: a duplicate label id, an episode the split does not know, or an episode
without one label per question in `splits["questions"]` stops the build before any file is replaced. Outputs go to
temporary files renamed on success, and `benchmark_summary.json` (beside the outputs) records the counts by band, side
and cell, the numbers the record and the DATASHEET quote.
"""

import argparse
import collections
import json
from pathlib import Path

from src.synthesis.label import dumps
from src.synthesis.schema import BenchmarkRecord, Label, LabelledRecord, Provenance
from src.synthesis_v2 import OUT_DIR, file_sha1
from src.synthesis_v2.split import (
    LEDGER,
    QUESTIONS,
    SPLITS,
    assign,
    episode_of,
    load_questions,
    load_splits,
)

OUTS = {s: OUT_DIR / f"{s}.jsonl" for s in ("train", "val", "benchmark")}


def benchmark_record(r: dict, held: set[str]) -> BenchmarkRecord:
    label = Label.model_validate(r["label"])
    prov = Provenance(method="teacher", labels={r["generation_info"]["name"]: label.answer})
    cell = "unseen_q" if r["question"]["id"] in held else "seen_q"
    return BenchmarkRecord(**{**r, "label": label, "cell": cell, "provenance": prov})


def build(ledger: Path, splits: dict, outs: dict[str, Path] = OUTS, _check: bool = True) -> dict:
    if _check:  # only tests pass False (a two-question ledger); check_bank has its own test
        check_bank(splits, *load_questions())
    held = splits["_held"]
    counts: collections.Counter = collections.Counter()
    seen: set[str] = set()
    per_episode: dict[str, set[str]] = collections.defaultdict(set)
    tmp = {s: outs[s].with_suffix(".jsonl.tmp") for s in outs}
    files = {s: tmp[s].open("w") for s in outs}
    try:
        for line in ledger.open():
            r = json.loads(line)
            if r["id"] in seen:
                raise SystemExit(f"{r['id']}: duplicate label id in the ledger")
            seen.add(r["id"])
            side = assign(r, splits)
            per_episode[episode_of(r["id"])].add(r["question"]["id"])
            r["meta"]["held_out"] = r["question"]["id"] in held  # the split's held-out set wins over the flag stored at labelling time
            band = r["meta"]["stratum"]
            if side == "drop":
                counts[f"drop/{band}"] += 1
                continue
            if side == "benchmark":
                rec = benchmark_record(r, held)
                counts[f"benchmark/{band}/{rec.cell}"] += 1
                files[side].write(dumps(rec) + "\n")
            else:
                counts[f"{side}/{band}"] += 1
                files[side].write(dumps(LabelledRecord.model_validate(r)) + "\n")
        want = set(splits["questions"])
        wrong = {ep: sorted(per_episode[ep] ^ want)[:3] for ep in splits["episodes"] if per_episode[ep] != want}
        if wrong:
            raise SystemExit(f"{len(wrong)} episodes' question ids differ from the split's {len(want)}: {dict(list(wrong.items())[:3])}")
    except BaseException:
        for f in files.values():
            f.close()
        for t in tmp.values():
            t.unlink(missing_ok=True)
        raise
    for f in files.values():
        f.close()
    for s, path in outs.items():
        tmp[s].replace(path)
    return dict(sorted(counts.items()))


def check_bank(splits: dict, qids: list[str], held: set[str]) -> None:
    """Refuse a split frozen against a different question bank (`questions.jsonl` as given, and the bank in
    `question_bank` it is written from): a stale held-out set would send a held-out question's labels into train and
    val without any other sign."""
    from src.synthesis_v2 import question_bank as bank
    if [q.id for q in bank.BANK] != qids or set(bank.HELD_OUT) != held:
        raise SystemExit("questions.jsonl differs from question_bank.py; regenerate it (python -m src.synthesis_v2.question_bank)")
    if splits["questions"] != qids or set(splits["held_out_questions"]) != held:
        raise SystemExit("the split's questions or held-out set differ from questions.jsonl; rebuild the split with --force "
                         f"(only in split: {sorted(set(splits['held_out_questions']) - held)}, only in bank: "
                         f"{sorted(held - set(splits['held_out_questions']))}, question lists equal: {splits['questions'] == qids})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--splits", type=Path, default=SPLITS)
    args = p.parse_args()
    splits = load_splits(args.splits)
    counts = build(args.ledger, splits)
    import hashlib

    summary = {"splits_created": splits["created"], "seed": splits["seed"], "pairs": counts,
               "questions_sha1": hashlib.sha1(QUESTIONS.read_bytes()).hexdigest(),
               "files_sha1": {p.name: file_sha1(p) for p in OUTS.values()},  # training checks these (src.train.data.check_questions)
               "episodes": dict(collections.Counter(splits["episodes"].values()))}
    (OUTS["benchmark"].parent / "benchmark_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
