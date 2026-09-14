"""Score generated outputs against a split file, per the eval-bench skill.

    uv run python -m src.train.evaluate <split.jsonl> <outputs.jsonl> <out_dir> [--wandb-run ID --wandb-project P]

Scores the raw text (no repair). Per pair: parse (one JSON object), answer status (exact: byte-equal
to a supplied option value, credit 1; recovered: equal to a value after trimming/case/space
normalisation, credit 0.3, a failure; invalid: 0), evidence gates (a JSON array of integers, 1-based,
within [1, N], ascending, de-duplicated; anything else fails the gates and scores 0), evidence
precision/recall/F1/exact against the gold key with empty-means-empty (a gold `[]` is matched only by
`[]`), tag Jaccard where the question has a tag vocabulary. Slices, within each rendered variant:
overall, dataset, family, cell (seen_q / unseen_q), length bucket by prompt tokens; confusion
matrices for the three priority families. The majority-class predictor (per question, from the
split's own gold) is reported next to every macro-F1. Summary faithfulness needs a judge and is TBD.
Writes metrics.json and results.md; --wandb-run logs the headline numbers to that training run.
"""

import argparse
import collections
import json
import re
from pathlib import Path

PRIORITY = ("vulnerability", "complaint", "eod")
BUCKETS = [(0, 2048, "<=2k"), (2048, 4096, "2-4k"), (4096, 8192, "4-8k"), (8192, 16384, "8-16k"), (16384, 10**9, ">16k")]
INVALID = "<invalid>"


def parse(text: str) -> dict | None:
    t = text.strip()
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t and "}" in t else ""):
        if not cand:
            continue
        try:
            o = json.loads(cand)
            if isinstance(o, dict):
                return o
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*?\}", t, re.DOTALL)
    if m:
        try:
            o = json.loads(m.group(0))
            return o if isinstance(o, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def score_pair(r: dict, out: dict) -> dict:
    q = r["question"]
    values = [o["value"] for o in q["options"]]
    n_lines = len(r["transcript"]["variants"][0]["lines"])
    gold_ev = set(r["label"]["evidence"])
    gold_tags = set(r["label"].get("tags", []))
    o = parse(out["text"])
    ans = o.get("answer") if o else None
    if isinstance(ans, str) and ans in values:
        status, pred = "exact", ans
    elif isinstance(ans, str) and _norm(ans) in {_norm(v): v for v in values}:
        status, pred = "recovered", {_norm(v): v for v in values}[_norm(ans)]
    else:
        status, pred = "invalid", INVALID
    ev = o.get("evidence") if o else None
    ev_valid = (isinstance(ev, list) and all(isinstance(e, int) and not isinstance(e, bool) for e in ev)
                and all(1 <= e <= n_lines for e in ev) and ev == sorted(set(ev)))
    pev = set(ev) if ev_valid and isinstance(ev, list) else None
    if pev is None:
        prec = rec = f1 = exact = 0.0
    else:
        inter = len(pev & gold_ev)
        prec = (inter / len(pev)) if pev else (1.0 if not gold_ev else 0.0)
        rec = (inter / len(gold_ev)) if gold_ev else (1.0 if not pev else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        exact = float(pev == gold_ev)
    tag_j = None
    if q.get("tags"):
        pt = o.get("tags") if o else None
        if isinstance(pt, list) and all(isinstance(t, str) for t in pt):
            ps = set(pt) & set(q["tags"])
            tag_j = 1.0 if not ps and not gold_tags else len(ps & gold_tags) / len(ps | gold_tags)
        else:
            tag_j = 0.0
    return {"id": r["id"], "variant": out["variant"], "dataset": r["dataset"], "track": r["track"], "family": q["family"],
            "qid": q["id"], "cell": r.get("cell", "seen_q"), "prompt_tokens": out.get("prompt_tokens"),
            "gold": r["label"]["answer"], "pred": pred, "status": status, "parsed": o is not None,
            "credit": {"exact": 1.0, "recovered": 0.3, "invalid": 0.0}[status],
            "ev_valid": bool(ev_valid), "ev_prec": prec, "ev_rec": rec, "ev_f1": f1, "ev_exact": exact,
            "gold_empty": not gold_ev, "tag_jaccard": tag_j, "format_valid": status == "exact" and bool(ev_valid)}


def macro_f1(rows: list[dict], key: str = "pred") -> float:
    classes = sorted({r["gold"] for r in rows})
    f1s = []
    for c in classes:
        tp = sum(1 for r in rows if r["gold"] == c and r[key] == c)
        fp = sum(1 for r in rows if r["gold"] != c and r[key] == c)
        fn = sum(1 for r in rows if r["gold"] == c and r[key] != c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(rows: list[dict]) -> dict:
    ne = [r for r in rows if not r["gold_empty"]]
    ge = [r for r in rows if r["gold_empty"]]
    return {"n": len(rows), "parsed": mean(rows, "parsed"), "format_valid": mean(rows, "format_valid"),
            "answer_exact": mean(rows, "credit") and sum(r["status"] == "exact" for r in rows) / len(rows),
            "answer_recovered": sum(r["status"] == "recovered" for r in rows) / len(rows),
            "answer_invalid": sum(r["status"] == "invalid" for r in rows) / len(rows),
            "answer_credit": mean(rows, "credit"),
            "accuracy": sum(r["status"] == "exact" and r["pred"] == r["gold"] for r in rows) / len(rows),
            "macro_f1": macro_f1(rows), "majority_macro_f1": macro_f1(rows, "majority"),
            "ev_valid": mean(rows, "ev_valid"), "ev_precision": mean(rows, "ev_prec"), "ev_recall": mean(rows, "ev_rec"),
            "ev_f1": mean(rows, "ev_f1"), "ev_exact": mean(rows, "ev_exact"),
            "gold_empty_share": len(ge) / len(rows), "gold_empty_matched": mean(ge, "ev_exact"),
            "ev_precision_nonempty": mean(ne, "ev_prec"), "ev_recall_nonempty": mean(ne, "ev_rec"),
            "tag_jaccard": mean(rows, "tag_jaccard"),
            "pred_distribution": dict(collections.Counter(r["pred"] for r in rows)),
            "gold_distribution": dict(collections.Counter(r["gold"] for r in rows))}


def bucket(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return next(name for lo, hi, name in BUCKETS if lo < tokens <= hi)


def confusion(rows: list[dict]) -> dict:
    m: dict = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        m[r["gold"]][r["pred"]] += 1
    return {g: dict(c) for g, c in m.items()}


def evaluate(split: Path, outputs: Path) -> dict:
    gold = {json.loads(l)["id"]: json.loads(l) for l in split.open()}
    rows = []
    for line in outputs.open():
        o = json.loads(line)
        if o["id"] in gold:
            rows.append(score_pair(gold[o["id"]], o))
    majority = {qid: collections.Counter(g["label"]["answer"] for g in gold.values() if g["question"]["id"] == qid).most_common(1)[0][0]
                for qid in {g["question"]["id"] for g in gold.values()}}
    for r in rows:
        r["majority"] = majority[r["qid"]]
    result: dict = {"split": str(split), "outputs": str(outputs), "pairs_in_split": len(gold), "scored": len(rows), "variants": {}}
    for variant in sorted({r["variant"] for r in rows}):
        vr = [r for r in rows if r["variant"] == variant]
        slices: dict = {"overall": summarise(vr)}
        for key in ("dataset", "family", "cell"):
            for val in sorted({r[key] for r in vr}):
                slices[f"{key}={val}"] = summarise([r for r in vr if r[key] == val])
        for _, _, name in BUCKETS:
            sub = [r for r in vr if bucket(r["prompt_tokens"]) == name]
            if sub:
                slices[f"length={name}"] = summarise(sub)
        for fam in PRIORITY:
            for ds in sorted({r["dataset"] for r in vr if r["family"] == fam}):
                sub = [r for r in vr if r["family"] == fam and r["dataset"] == ds]
                slices[f"priority={fam}/{ds}"] = {**summarise(sub), "confusion": confusion(sub)}
        result["variants"][variant] = slices
    return result


HEADLINE = ["n", "format_valid", "answer_exact", "accuracy", "macro_f1", "majority_macro_f1", "ev_precision", "ev_recall",
            "ev_f1", "ev_exact", "gold_empty_matched", "tag_jaccard"]


def markdown(result: dict) -> str:
    head = (f"# Benchmark results\n\nsplit `{result['split']}`, outputs `{result['outputs']}`, "
            f"{result['scored']} outputs scored over {result['pairs_in_split']} pairs.\n")
    lines = [head]
    for variant, slices in result["variants"].items():
        header = "| slice | " + " | ".join(HEADLINE) + " |\n|---|" + "---|" * len(HEADLINE)
        lines.append(f"\n## Variant `{variant}`\n\n{header}")
        for name, s in slices.items():
            if name.startswith("priority="):
                continue
            cells = [str(s["n"])] + [("" if s.get(k) is None else f"{s[k]:.3f}") for k in HEADLINE[1:]]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        for name, s in slices.items():
            if name.startswith("priority="):
                lines.append(f"\n**{name}** (n={s['n']}, macro-F1 {s['macro_f1']:.3f}, majority {s['majority_macro_f1']:.3f}); "
                             f"confusion gold→pred: `{json.dumps(s['confusion'])}`")
    lines.append("\nSummary faithfulness (LLM-as-a-judge): TBD.\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path)
    ap.add_argument("outputs", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--wandb-project", default="tt-decoder")
    args = ap.parse_args()
    result = evaluate(args.split, args.outputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    md = markdown(result)
    (args.out_dir / "results.md").write_text(md)
    print(md)
    if args.wandb_run:
        import wandb
        run = wandb.init(project=args.wandb_project, id=args.wandb_run, resume="must")
        flat = {}
        for variant, slices in result["variants"].items():
            for name, s in slices.items():
                for k in HEADLINE:
                    if s.get(k) is not None:
                        flat[f"bench/{variant}/{name}/{k}"] = s[k]
        run.log(flat)
        run.finish()


if __name__ == "__main__":
    main()
