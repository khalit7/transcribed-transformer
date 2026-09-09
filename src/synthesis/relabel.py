"""Re-label benchmark pairs with another labeller: a second opinion, or an adjudication of disagreements.

    uv run python -m src.synthesis.relabel <model> --families general_qa --out <file> [options]

The benchmark's ground truth must be better than one local labeller's say-so, so every benchmark pair
gets a second, independent label and the pairs the two labellers disagree on go to a stronger
adjudicating labeller (src/synthesis/README.md "Train, val and benchmark"). This module runs either
pass: it takes the benchmark pairs from the ledger (splits.json decides which calls are benchmark),
labels them with <model> and writes LabelledRecords to --out, one file per labeller. Nothing here
judges; it is LLM labelling with a different labeller, and benchmark.py reconciles the files.

Options:
  --families F,G      question families to label (default general_qa)
  --disagree-with F   only pairs whose answer in file F differs from the ledger's, or is missing from F
  --answers a,b       only pairs whose ledger answer is one of these (e.g. fail: the rare-event positives)
  --limit N           stop after N pairs (cost check)
  --workers N         concurrent calls (claude: 16; ollama: one per GPU server, started on demand)
  --claude-account p|w
  --dry-run           list the pairs and stop
Pairs already in --out are skipped, so a run resumes.
"""

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import time
from pathlib import Path

from src.synthesis.label import dumps, label
from src.synthesis.llm import (
    OLLAMA_URLS,
    LLMError,
    set_claude_account,
    set_ollama_urls,
    split_model,
)
from src.synthesis.ollama_servers import ensure_servers
from src.synthesis.question_bank import OUT_DIR
from src.synthesis.schema import Case, Generation, LabelledRecord, Question
from src.synthesis.split import LEDGER, assign, load_splits
from src.synthesis.synth_data import DEFAULT_WORKERS, existing_pairs

BENCH_DIR = OUT_DIR / "benchmark"


def case_of(r: dict) -> Case:
    """The ledger embeds the transcript, so a case is rebuilt without touching the corpora."""
    return Case(id=r["id"].split("::")[0], dataset=r["dataset"], track=r["track"], source_id=r["source_id"],
                transcript=r["transcript"], meta=r["meta"])


def benchmark_records(families: set[str], ledger: Path = LEDGER) -> list[dict]:
    splits = load_splits()
    return [r for r in map(json.loads, ledger.open())
            if r["question"]["family"] in families and assign(r, splits) == "benchmark"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--families", default="general_qa")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--disagree-with", type=Path, default=None)
    p.add_argument("--answers", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--claude-account", "--claude_account", dest="claude_account", choices=["p", "w"], default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    set_claude_account(args.claude_account)
    backend = split_model(args.model)[0]
    if backend == "ollama" and not args.dry_run:
        set_ollama_urls(ensure_servers())
    if args.workers is None:
        args.workers = DEFAULT_WORKERS if backend == "claude" else len(OLLAMA_URLS)

    recs = benchmark_records({f.strip() for f in args.families.split(",")})
    if args.answers:
        keep = set(args.answers.split(","))
        recs = [r for r in recs if r["label"]["answer"] in keep]
    if args.disagree_with:
        other = {json.loads(l)["id"]: json.loads(l)["label"]["answer"] for l in args.disagree_with.open()}
        recs = [r for r in recs if other.get(r["id"]) != r["label"]["answer"]]
    done_pairs = existing_pairs(args.out)
    recs = [r for r in recs if r["id"] not in done_pairs]
    if args.limit:
        recs = recs[:args.limit]
    # group by call so a server works through one transcript's questions with a warm prefix cache
    by_call: dict[str, list[dict]] = collections.OrderedDict()
    for r in recs:
        by_call.setdefault(r["id"].split("::")[0], []).append(r)
    call_index = {cid: i for i, cid in enumerate(by_call)}
    calls = list(by_call.values())
    jobs: list[dict] = []
    for i in range(0, len(calls), args.workers):
        chunk = calls[i:i + args.workers]
        for k in range(max(len(c) for c in chunk)):
            jobs += [c[k] for c in chunk if k < len(c)]
    print(f"{len(jobs)} pairs over {len(calls)} calls to label with {args.model}, workers={args.workers} -> {args.out}",
          flush=True)
    if args.dry_run:
        for r in jobs:
            print(f"[dry] {r['id']} (ledger: {r['label']['answer']})")
        return

    def work(r: dict) -> tuple[LabelledRecord, float]:
        case = case_of(r)
        q = Question.model_validate(r["question"])
        variant = r["generation_info"]["labelled_variant"]
        lab, cost = label(case, q, args.model, variant, route=call_index[case.id])
        rec = LabelledRecord(
            id=r["id"], dataset=r["dataset"], source_id=r["source_id"], track=r["track"], question=q,
            transcript=case.transcript, label=lab, meta=r["meta"],
            generation_info=Generation(name=args.model, labelled_variant=variant, cost_usd=round(cost, 5),
                                       timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                                       claude_account=args.claude_account if backend == "claude" else None),
        )
        return rec, cost

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = failures = 0
    total_cost = 0.0
    agree: collections.Counter = collections.Counter()
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, args.out.open("a") as f:
        futures = {ex.submit(work, r): r for r in jobs}
        for fut in concurrent.futures.as_completed(futures):
            r = futures[fut]
            try:
                rec, cost = fut.result()
            except (LLMError, ValueError, KeyError) as e:
                failures += 1
                print(f"  {r['id']}: label failed: {str(e)[:120]}", flush=True)
                continue
            f.write(dumps(rec) + "\n")
            f.flush()
            done += 1
            total_cost += cost
            agree[rec.label.answer == r["label"]["answer"]] += 1
            print(f"[{done}/{len(jobs)}] {r['id']} -> {rec.label.answer} (ledger {r['label']['answer']}) "
                  f"ev={rec.label.evidence} cost=${total_cost:.3f}", flush=True)
    print(f"done: {done} labels, {failures} failures, agree with ledger {agree[True]}/{done}, ${total_cost:.3f}, "
          f"{time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
