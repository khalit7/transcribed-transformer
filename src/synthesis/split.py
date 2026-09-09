"""Split the labelled data into train, val and benchmark, by transcript and by question.

    uv run python -m src.synthesis.split [--seed 0] [--out data/labelled_data/splits.json] [--force]

Reads the ledger (labelled_data.jsonl) and writes splits.json: an assignment of every call id to
train | val | benchmark, plus the general_qa question lineages held out of training. The file is
frozen once written (--force to overwrite); everything labelled after the freeze joins train.

Rules (decided 2026-09-08, see src/synthesis/README.md "Train, val and benchmark"):

- Split by transcript, never by (transcript, question) pair. SPoRC groups by podcast: every episode
  of one podcast lands on the same side.
- Only fully labelled calls (every allowed question answered) can be benchmark or val; calls that
  lost a pair to a labeller failure, and calls with family labels only, go to train.
- Benchmark calls are chosen as the best of many seeded random draws, scored on: proportional
  coverage of the dataset's strata (AppTek domain and locale; SPoRC category; ACI-Bench source
  split), a floor on rare-event positives per family (answer `fail` on a family question), and,
  for SPoRC, over-representation of the long tail (Qwen3 tokens of the rendered transcript).
- About 20% of general_qa question lineages are held out of training per dataset (a lineage is a
  question plus its Taskmaster word-substituted variant, which is the same question). Pairs with a
  held-out question are dropped from train and val; on benchmark calls they form the unseen-question
  cell, the one the zero-shot claim rests on. The vulnerability, complaint and eod questions are
  never held out (decision: they stay in training).
- Val is a small in-distribution slice of the remaining fully labelled calls (same question set as
  train) for early stopping and model selection only; it is never reported.

`assign(record, splits)` returns train | val | benchmark | drop for a ledger record, and is what
every consumer (benchmark build, training data export) uses, so there is one place the rule lives.
"""

import argparse
import collections
import datetime as dt
import json
import random
import re
import statistics
from pathlib import Path

from src.synthesis.question_bank import OUT_DIR, QUESTIONS

LEDGER = OUT_DIR / "labelled_data.jsonl"
SPLITS = OUT_DIR / "splits.json"
TOKENIZER = "Qwen/Qwen3-1.7B"  # provisional project tokenizer (configs/mixtures); only used for length strata

# benchmark / val calls per dataset
BENCH = {"apptek": 100, "aci_bench": 40, "taskmaster": 60, "sporc": 60}
VAL = {"apptek": 40, "aci_bench": 10, "taskmaster": 15, "sporc": 15}
HELD_OUT_FRACTION = 0.20  # of general_qa question lineages, per dataset allow list
POSITIVE_FLOOR = 30  # rare-event positive calls per family in the benchmark, where the pool allows
DRAWS = 3000  # candidate benchmark samples scored per dataset
SWAPS = 20000  # single-group swap attempts that refine the best draw
LONG_TAIL_QUANTILE = 0.9  # SPoRC: share of benchmark episodes above this pool quantile is forced up
LONG_TAIL_MIN = 15  # ... to at least this many of the 60


def lineage(qid: str) -> str:
    """Taskmaster variants (tm-NN-x from gen-NN-x, tm-vul-… from vul-…) belong to their base question."""
    m = re.match(r"^tm-(\d\d-.*)$", qid)
    if m:
        return f"gen-{m.group(1)}"
    m = re.match(r"^tm-(vul|cmp|eod)-(.*)$", qid)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return qid


def group_key(dataset: str, call_id: str, meta: dict) -> str:
    return f"pod:{meta['pod_title']}" if dataset == "sporc" else call_id


def load_calls(ledger: Path = LEDGER) -> dict[str, dict]:
    """One entry per call: dataset, meta, group, lines, the question ids answered and the positives."""
    calls: dict[str, dict] = {}
    for line in ledger.open():
        r = json.loads(line)
        cid = r["id"].split("::")[0]
        c = calls.get(cid)
        if c is None:
            c = calls[cid] = {"dataset": r["dataset"], "meta": r["meta"], "track": r["track"],
                              "group": group_key(r["dataset"], cid, r["meta"]),
                              "lines": r["transcript"]["variants"][0]["lines"], "answered": set(), "positives": set()}
        q = r["question"]
        c["answered"].add(q["id"])
        if q["family"] != "general_qa" and r["label"]["answer"] == "fail":
            c["positives"].add(q["family"])
    return calls


def complete(calls: dict[str, dict]) -> set[str]:
    allowed = collections.defaultdict(set)
    for q in QUESTIONS:
        for ds in q.dataset_allow_list:
            allowed[ds].add(q.id)
    return {cid for cid, c in calls.items() if allowed[c["dataset"]] <= c["answered"]}


def token_lengths(calls: dict[str, dict], ids: list[str]) -> dict[str, int]:
    from tokenizers import Tokenizer

    from src.synthesis.label import numbered
    tok = Tokenizer.from_pretrained(TOKENIZER)
    return {cid: len(tok.encode(numbered(calls[cid]["lines"])).ids) for cid in ids}


def strata_of(dataset: str, meta: dict) -> list[tuple[str, str]]:
    if dataset == "apptek":
        return [("domain", meta["domain"]), ("locale", meta["locale"])]
    if dataset == "sporc":
        return [("category", meta["category"])]
    if dataset == "aci_bench":
        return [("split_file", meta["split_file"])]
    return []


def _deviation(pool: list[str], chosen: list[str], strata: dict[str, list[tuple[str, str]]]) -> float:
    """Sum over strata of |share in chosen - share in pool|."""
    dev = 0.0
    for name in sorted({n for cid in pool for n, _ in strata[cid]}):
        pool_c = collections.Counter(v for cid in pool for n, v in strata[cid] if n == name)
        chosen_c = collections.Counter(v for cid in chosen for n, v in strata[cid] if n == name)
        dev += sum(abs(chosen_c[v] / len(chosen) - pool_c[v] / len(pool)) for v in pool_c)
    return dev


def sample_benchmark(dataset: str, pool: list[str], calls: dict[str, dict], n: int, rng: random.Random,
                     lengths: dict[str, int] | None) -> tuple[list[str], dict]:
    """Best of DRAWS seeded draws of whole groups summing to n calls."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for cid in pool:
        groups[calls[cid]["group"]].append(cid)
    strata = {cid: strata_of(dataset, calls[cid]["meta"]) for cid in pool}
    families = sorted({f for cid in pool for f in calls[cid]["positives"]})
    pool_pos = {f: sum(1 for cid in pool if f in calls[cid]["positives"]) for f in families}
    floors = {f: min(POSITIVE_FLOOR, pool_pos[f] // 2) for f in families}
    tail_cut = None
    if lengths:
        ls = sorted(lengths[cid] for cid in pool)
        tail_cut = ls[int(LONG_TAIL_QUANTILE * (len(ls) - 1))]
    def score_of(chosen: list[str]) -> float:
        score = _deviation(pool, chosen, strata)
        for f in families:  # a missing positive costs far more than any stratum imbalance
            score += 1.0 * max(0, floors[f] - sum(1 for cid in chosen if f in calls[cid]["positives"]))
        if tail_cut is not None and lengths is not None:
            score += 1.0 * max(0, LONG_TAIL_MIN - sum(1 for cid in chosen if lengths[cid] > tail_cut))
        return score

    best: tuple[float, list[str]] | None = None
    keys = sorted(groups)
    for _ in range(DRAWS):
        rng.shuffle(keys)
        chosen: list[str] = []
        for k in keys:
            if len(chosen) + len(groups[k]) <= n:
                chosen += groups[k]
            if len(chosen) == n:
                break
        if len(chosen) != n:
            continue
        score = score_of(chosen)
        if best is None or score < best[0]:
            best = (score, sorted(chosen))
    assert best is not None, f"{dataset}: cannot draw {n} calls from {len(pool)}"
    # hill-climb: swap one chosen group for one unchosen group of the same size while the score improves
    score, chosen = best
    in_groups = {calls[cid]["group"] for cid in chosen}
    for _ in range(SWAPS):
        g_out = rng.choice(sorted(in_groups))
        g_in = rng.choice(keys)
        if g_in in in_groups or len(groups[g_in]) != len(groups[g_out]):
            continue
        cand = sorted([cid for cid in chosen if calls[cid]["group"] != g_out] + groups[g_in])
        s2 = score_of(cand)
        if s2 < score:
            score, chosen = s2, cand
            in_groups = (in_groups - {g_out}) | {g_in}
    best = (score, chosen)
    assert best is not None, f"{dataset}: cannot draw {n} calls from {len(pool)}"
    chosen = best[1]
    report = {
        "score": round(best[0], 3),
        "positives": {f: [sum(1 for cid in chosen if f in calls[cid]["positives"]), pool_pos[f], floors[f]] for f in families},
    }
    if lengths:
        report["tokens"] = {"tail_cut": tail_cut, "above_tail": sum(1 for cid in chosen if lengths[cid] > (tail_cut or 0)),
                            "median": statistics.median(lengths[cid] for cid in chosen),
                            "max": max(lengths[cid] for cid in chosen)}
    for name in sorted({n for cid in pool for n, _ in strata[cid]}):
        report[name] = dict(collections.Counter(v for cid in chosen for n, v in strata[cid] if n == name))
    return chosen, report


def held_out_lineages(rng: random.Random) -> list[str]:
    """~HELD_OUT_FRACTION of the general_qa lineages allowed on each dataset, drawn from one shuffle."""
    by_lineage: dict[str, set[str]] = collections.defaultdict(set)  # lineage -> datasets it is asked of
    for q in QUESTIONS:
        if q.family == "general_qa":
            by_lineage[lineage(q.id)].update(q.dataset_allow_list)
    order = sorted(by_lineage)
    rng.shuffle(order)
    target = {ds: round(HELD_OUT_FRACTION * sum(1 for l in by_lineage.values() if ds in l)) for ds in BENCH}
    held: set[str] = set()
    for ds in BENCH:  # datasets in fixed order; lineages in seeded order
        have = sum(1 for l in held if ds in by_lineage[l])
        for l in order:
            if have >= target[ds]:
                break
            if l not in held and ds in by_lineage[l]:
                held.add(l)
                have += 1
    return sorted(held)


def build(seed: int) -> dict:
    rng = random.Random(seed)
    calls = load_calls()
    full = complete(calls)
    by_ds: dict[str, list[str]] = collections.defaultdict(list)
    for cid in sorted(calls):
        by_ds[calls[cid]["dataset"]].append(cid)
    assignment: dict[str, str] = {}
    reports: dict[str, dict] = {}
    for ds, n_bench in BENCH.items():
        pool = [cid for cid in by_ds[ds] if cid in full]
        lengths = token_lengths(calls, pool) if ds == "sporc" else None
        bench, rep = sample_benchmark(ds, pool, calls, n_bench, rng, lengths)
        bench_groups = {calls[cid]["group"] for cid in bench}
        rest = [cid for cid in pool if calls[cid]["group"] not in bench_groups]
        rng.shuffle(rest)
        val: list[str] = []
        val_groups: set[str] = set()
        for cid in rest:  # whole groups, until VAL[ds] reached
            if len(val) >= VAL[ds]:
                break
            g = calls[cid]["group"]
            if g not in val_groups:
                val_groups.add(g)
                val += [x for x in rest if calls[x]["group"] == g]
        for cid in by_ds[ds]:
            if cid in bench:
                assignment[cid] = "benchmark"
            elif cid in val:
                assignment[cid] = "val"
            elif calls[cid]["group"] in bench_groups:
                assignment[cid] = "excluded"  # a partially labelled sibling (same podcast) of a benchmark call
            elif calls[cid]["group"] in val_groups:
                assignment[cid] = "val"  # partially labelled siblings of val calls stay out of train
            else:
                assignment[cid] = "train"
        rep["calls"] = dict(collections.Counter(assignment[cid] for cid in by_ds[ds]))
        rep["calls"]["fully_labelled"] = len(pool)
        reports[ds] = rep
    held = held_out_lineages(rng)
    held_ids = sorted(q.id for q in QUESTIONS if lineage(q.id) in held)
    return {
        "version": 1, "seed": seed, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "policy": {"benchmark_per_dataset": BENCH, "val_per_dataset": VAL, "held_out_fraction": HELD_OUT_FRACTION,
                   "positive_floor": POSITIVE_FLOOR, "long_tail": [LONG_TAIL_QUANTILE, LONG_TAIL_MIN],
                   "tokenizer": TOKENIZER, "families_never_held_out": ["vulnerability", "complaint", "eod"]},
        "held_out_lineages": held, "held_out_questions": held_ids,
        "report": reports, "calls": assignment,
    }


def load_splits(path: Path = SPLITS) -> dict:
    s = json.loads(path.read_text())
    s["_held"] = set(s["held_out_questions"])
    return s


def assign(record: dict, splits: dict) -> str:
    """train | val | benchmark | drop for one ledger record (drop = held-out question on a train/val call, or a
    partially labelled sibling of a benchmark call)."""
    side = splits["calls"].get(record["id"].split("::")[0], "train")
    if side == "excluded":
        return "drop"
    if side != "benchmark" and record["question"]["id"] in splits.get("_held", set(splits["held_out_questions"])):
        return "drop"
    return side


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=SPLITS)
    p.add_argument("--force", action="store_true", help="overwrite an existing (frozen) split")
    args = p.parse_args()
    if args.out.exists() and not args.force:
        p.error(f"{args.out} exists and is frozen; --force to rebuild (this changes the benchmark)")
    s = build(args.seed)
    args.out.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {collections.Counter(s['calls'].values())}")
    print(f"held-out general_qa lineages: {len(s['held_out_lineages'])} ({len(s['held_out_questions'])} question ids)")
    for ds, rep in s["report"].items():
        print(f"{ds}: {json.dumps(rep, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
