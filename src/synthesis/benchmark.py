"""Build train.jsonl, val.jsonl and benchmark.jsonl from the ledger, the split and the benchmark re-labels.

    uv run python -m src.synthesis.benchmark

train / val: the ledger records assigned there by splits.json, minus pairs with a held-out question
(split.assign). One label each, as labelled.

benchmark: every benchmark pair carries two labels (D61, 2026-09-29):
- `label`, the teacher's: the ledger label exactly as the training labeller produced it (qwen3.8 for
  general_qa, Sonnet for the rare-event families), answer, evidence and summary untouched. It is the
  headline gold: the same function the training objective asked the arms to learn, so a number against
  it reads as imitation of the teacher.
- `opus_label`, where one exists: the label Opus gave the pair as a second, independent labeller
  (benchmark/adjudicate_*.jsonl and benchmark/blind_*.jsonl, file names from the earlier scheme; a blind_ file
  wins over an adjudicate_ file for the same pair).
  Nothing else counts as Opus's label. The pairs that have one are a selected subset, not a sample: for
  general_qa the pairs where the teacher and a second local labeller disagreed (plus four with no second
  opinion), for the families the pairs the teacher called `fail`; numbers against it compare arms with
  each other on that subset and are not comparable with the teacher numbers.
Every record keeps every labeller's answer in `provenance.labels` (the second local opinion included,
though it is no longer part of any gold), `provenance.opus_labeller` names Opus's labeller string when there is one, and
`cell` says whether the question was in the training question set (seen_q) or held out (unseen_q).
Outputs are written to temporary files and renamed on success.

Also writes benchmark_summary.json (counts by dataset, family, cell and Opus coverage; how often Opus
agrees with the teacher), the numbers the README and DATASHEET quote.
"""

import argparse
import collections
import json
from pathlib import Path

from src.synthesis.label import dumps
from src.synthesis.question_bank import OUT_DIR
from src.synthesis.schema import BenchmarkRecord, Label, LabelledRecord, Provenance
from src.synthesis.split import LEDGER, SPLITS, assign, load_splits

BENCH_DIR = OUT_DIR / "benchmark"
OUTS = {s: OUT_DIR / f"{s}.jsonl" for s in ("train", "val", "benchmark")}
SUMMARY = OUT_DIR / "benchmark_summary.json"


def load_relabels(pattern: str) -> dict[str, dict]:
    """id -> record, over every file matching the pattern (later files win on duplicate ids)."""
    out: dict[str, dict] = {}
    for path in sorted(BENCH_DIR.glob(pattern)):
        for line in path.open():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def two_golds(r: dict, second: dict | None, adjudicated: dict | None, blind: dict | None = None
              ) -> tuple[Label, Label | None, Provenance]:
    """(teacher label, Opus label or None, provenance) for a benchmark pair. The teacher label is the ledger label as
    is. The Opus label comes from Opus's files under benchmark/ (`blind_*.jsonl`, or else `adjudicate_*.jsonl`; the file
    names date from the earlier scheme and are only where Opus's labels are stored); `provenance.opus_labeller` names it."""
    teacher = Label.model_validate(r["label"])
    labels = {r["generation_info"]["name"]: teacher.answer}
    if second is not None:
        labels[second["generation_info"]["name"]] = second["label"]["answer"]
    opus_rec = blind if blind is not None else adjudicated
    if adjudicated is not None:
        labels[adjudicated["generation_info"]["name"]] = adjudicated["label"]["answer"]
    if blind is not None:
        labels[blind["generation_info"]["name"] + " blind"] = blind["label"]["answer"]
    if opus_rec is None:
        return teacher, None, Provenance(method="teacher", labels=labels)
    name = opus_rec["generation_info"]["name"] + (" blind" if blind is not None else "")
    if opus_rec["generation_info"]["name"] != "claude:opus":  # only Opus is ever `opus_label`; a misnamed file must not become gold
        raise SystemExit(f"{r['id']}: the label in an adjudicate/blind file comes from {opus_rec['generation_info']['name']!r}, not Opus")
    return teacher, Label.model_validate(opus_rec["label"]), Provenance(method="teacher", labels=labels, opus_labeller=name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    args = p.parse_args()
    splits = load_splits(SPLITS)
    held = set(splits["held_out_questions"])
    second = load_relabels("second_*.jsonl")
    adjudicated = load_relabels("adjudicate_*.jsonl")
    blind = load_relabels("blind_*.jsonl")
    counts: collections.Counter = collections.Counter()
    summary: dict = {"records": collections.Counter(), "benchmark": collections.Counter(), "opus": collections.Counter()}
    tmp = {s: OUTS[s].with_suffix(".jsonl.tmp") for s in OUTS}
    files = {s: tmp[s].open("w") for s in OUTS}
    try:
        for line in args.ledger.open():
            r = json.loads(line)
            side = assign(r, splits)
            counts[side] += 1
            if side == "drop":
                continue
            if side != "benchmark":
                files[side].write(dumps(LabelledRecord.model_validate(r)) + "\n")
                continue
            teacher, opus, prov = two_golds(r, second.get(r["id"]), adjudicated.get(r["id"]), blind.get(r["id"]))
            cell = "unseen_q" if r["question"]["id"] in held else "seen_q"
            rec = BenchmarkRecord(**{**r, "label": teacher, "opus_label": opus, "cell": cell, "provenance": prov})
            files["benchmark"].write(dumps(rec) + "\n")
            fam = r["question"]["family"]
            summary["benchmark"][f"{r['dataset']}/{fam}/{cell}/{'with-opus-label' if opus else 'teacher-only'}"] += 1
            if opus is not None:
                summary["opus"][f"{fam}: {prov.opus_labeller} " + ("agrees with" if opus.answer == teacher.answer else "differs from") + " the teacher"] += 1
    finally:
        for f in files.values():
            f.close()
    summary["records"] = dict(counts)
    summary["benchmark"] = dict(sorted(summary["benchmark"].items()))
    summary["opus"] = dict(sorted(summary["opus"].items()))
    for s, path in OUTS.items():
        tmp[s].replace(path)
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(f"records by side: {dict(counts)} -> {', '.join(str(o) for o in OUTS.values())}")
    print(f"benchmark: {json.dumps(summary['benchmark'], indent=1)}")
    print(f"opus: {json.dumps(summary['opus'], indent=1)}")


if __name__ == "__main__":
    main()
