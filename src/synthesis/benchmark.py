"""Build train.jsonl, val.jsonl and benchmark.jsonl from the ledger, the split and the benchmark re-labels.

    uv run python -m src.synthesis.benchmark [--allow-pending]

train / val: the ledger records assigned there by splits.json, minus pairs with a held-out question
(split.assign). One label each, as labelled.

benchmark: every benchmark pair with a reconciled gold label and its provenance:
- general_qa: the ledger label (qwen3.8) and the second opinion (benchmark/second_*.jsonl) agree on
  the answer -> `agreement`, gold answer as agreed, gold evidence the union of both keys; they
  disagree, or the second opinion is missing -> the adjudicator's label (benchmark/adjudicate_*.jsonl,
  Opus) is the gold label -> `adjudicated`; no adjudication yet -> `pending` (the build refuses unless
  --allow-pending, and then keeps the ledger label).
- families (Sonnet): the Opus spot-check covers the rare-event positives; where it exists the same
  agreement / adjudicated rule applies, otherwise `single`.
Every record keeps every labeller's answer in `provenance.labels`, and `cell` says whether the
question was in the training question set (seen_q) or held out (unseen_q).

Also writes benchmark_summary.json (counts by dataset, family, cell and method; agreement rates),
the numbers the README and DATASHEET quote.
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


def reconcile(r: dict, second: dict | None, adjudicated: dict | None) -> tuple[Label, Provenance]:
    primary = Label.model_validate(r["label"])
    labels = {r["generation_info"]["name"]: primary.answer}
    if second is not None:
        labels[second["generation_info"]["name"]] = second["label"]["answer"]
    if adjudicated is not None:
        labels[adjudicated["generation_info"]["name"]] = adjudicated["label"]["answer"]
    if second is not None and second["label"]["answer"] == primary.answer:
        gold = primary.model_copy(update={"evidence": sorted(set(primary.evidence) | set(second["label"]["evidence"]))})
        return gold, Provenance(method="agreement", labels=labels)
    if adjudicated is not None:
        return Label.model_validate(adjudicated["label"]), Provenance(
            method="adjudicated", labels=labels, adjudicator=adjudicated["generation_info"]["name"])
    if second is None:
        return primary, Provenance(method="single", labels=labels)
    return primary, Provenance(method="pending", labels=labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--allow-pending", action="store_true", help="build even if disagreements await adjudication")
    p.add_argument("--ledger", type=Path, default=LEDGER)
    args = p.parse_args()
    splits = load_splits(SPLITS)
    held = set(splits["held_out_questions"])
    second = load_relabels("second_*.jsonl")
    adjudicated = load_relabels("adjudicate_*.jsonl")
    counts: collections.Counter = collections.Counter()
    summary: dict = {"records": collections.Counter(), "benchmark": collections.Counter(),
                     "pending": [], "agreement": collections.Counter()}
    files = {s: OUTS[s].open("w") for s in OUTS}
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
            gold, prov = reconcile(r, second.get(r["id"]), adjudicated.get(r["id"]))
            cell = "unseen_q" if r["question"]["id"] in held else "seen_q"
            rec = BenchmarkRecord(**{**r, "label": gold, "cell": cell, "provenance": prov})
            files["benchmark"].write(dumps(rec) + "\n")
            fam = r["question"]["family"]
            summary["benchmark"][f"{r['dataset']}/{fam}/{cell}/{prov.method}"] += 1
            if prov.method == "pending":
                summary["pending"].append(r["id"])
            if len(prov.labels) >= 2:
                primary = r["generation_info"]["name"]
                others = {n: a for n, a in prov.labels.items() if n != primary}
                if prov.adjudicator:
                    # who did the adjudicator side with? (the ledger labeller, the second opinion, or neither)
                    sided = [n for n, a in prov.labels.items() if n != prov.adjudicator and a == prov.labels[prov.adjudicator]]
                    summary["agreement"][f"{fam}: {prov.adjudicator} sided with " + (", ".join(sorted(sided)) or "neither")] += 1
                else:
                    summary["agreement"][f"{fam}: {' vs '.join(sorted(others))} vs {primary}: agree"] += 1
    finally:
        for f in files.values():
            f.close()
    summary["records"] = dict(counts)
    summary["benchmark"] = dict(sorted(summary["benchmark"].items()))
    summary["agreement"] = dict(sorted(summary["agreement"].items()))
    n_pending = len(summary["pending"])
    summary["pending"] = n_pending
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(f"records by side: {dict(counts)} -> {', '.join(str(o) for o in OUTS.values())}")
    print(f"benchmark: {json.dumps(summary['benchmark'], indent=1)}")
    print(f"agreement: {json.dumps(summary['agreement'], indent=1)}")
    if n_pending and not args.allow_pending:
        for path in OUTS.values():
            path.unlink()
        raise SystemExit(f"{n_pending} benchmark pairs await adjudication (run relabel.py claude:opus --disagree-with); "
                         f"outputs removed. --allow-pending to build with the ledger label for those.")


if __name__ == "__main__":
    main()
