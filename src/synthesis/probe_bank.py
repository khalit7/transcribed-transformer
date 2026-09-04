"""Probe the answer distribution of every bank question over a sample of real calls.

A question whose answer is the same (or NA) on nearly every call is useless for
both training and benchmarking. This asks a labeller to answer ALL
bank questions of a family for each sampled call, answer only, and reports the
per-question answer distribution so skewed questions can be rewritten or
dropped before any labelled data is generated.

    uv run python -m src.synthesis.probe_bank --model ollama:qwen3:32b --calls 40 --family general_qa
    uv run python -m src.synthesis.probe_bank --model claude:sonnet --calls 40 --seed 1 --workers 4 --save data/labelled_data/probes/sonnet.json
"""

import argparse
import collections
import concurrent.futures
import itertools
import json
import random

from src.synthesis.cases import BUILDERS, ID_STREAMS
from src.synthesis.llm import ask_json
from src.synthesis.question_bank import QUESTIONS
from src.synthesis.schema import Case


def ask(model: str, prompt: str) -> tuple[dict, float]:
    return ask_json(prompt, model)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="ollama:qwen3:32b")
    p.add_argument("--calls", type=int, default=40)
    p.add_argument("--family", default="general_qa", help="a family, or 'all'")
    p.add_argument("--sources", default="apptek:20,taskmaster:10,aci_bench:10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="write per-question distributions to this JSON file")
    p.add_argument("--ids", default=None, help="comma-separated question ids to probe (default: whole family)")
    p.add_argument("--dataset", default=None, help="only questions whose dataset_allow_list includes this dataset")
    p.add_argument("--workers", type=int, default=1, help="concurrent labeller calls (use >1 with claude:, not ollama:)")
    args = p.parse_args()

    qs = [q for q in QUESTIONS if args.family == "all" or q.family == args.family]
    if args.ids:
        wanted = set(args.ids.split(","))
        qs = [q for q in qs if q.id in wanted]
    if args.dataset:
        qs = [q for q in qs if args.dataset in q.dataset_allow_list]
    qtext = "\n".join(
        f"{q.id}: {q.text} Options: " + "; ".join(f"{o.value} = {o.criteria}" for o in q.options) for q in qs
    )
    rng = random.Random(args.seed)
    calls = []
    for spec in args.sources.split(","):
        src, n = spec.split(":")
        # sample from a pool up to 6x the request; small corpora (ACI-Bench: 207) supply what they have.
        # Where an id stream exists, sample ids by position in the raw stream and build only those, so the
        # sample is stable even when the builder later skips an episode (e.g. no identifiable host).
        if src in ID_STREAMS:
            ids = list(itertools.islice(ID_STREAMS[src](), int(n) * 6))
            chosen = set(rng.sample(ids, min(int(n), len(ids))))
            calls += list(BUILDERS[src](only=chosen))  # type: ignore[call-arg]  # only sporc has `only`
        else:
            pool = list(itertools.islice(BUILDERS[src](), int(n) * 6))
            calls += rng.sample(pool, min(int(n), len(pool)))

    def probe(case: Case) -> tuple[dict, float]:
        lines = "\n".join(f"{n + 1}: {ln}" for n, ln in enumerate(case.transcript.lines("clean")))
        roles = ", ".join(case.transcript.speakers)
        prompt = (f"Transcript, one turn per line, each prefixed by the speaker's role as recorded by the source ({roles}).\n\nTRANSCRIPT\n{lines}\n\n"
                  f"Answer EVERY question below strictly by its option criteria, for this call as it stands.\n\n{qtext}\n\n"
                  'Respond with one JSON object mapping each question id to its answer value, e.g. {"gen-01-greeting": "pass", ...}. Nothing else.')
        return ask(args.model, prompt)

    dist: dict[str, collections.Counter] = {q.id: collections.Counter() for q in qs}
    total_cost = 0.0
    # --workers > 1 runs calls concurrently; only sensible for the claude backend (Ollama serialises on the GPU)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(probe, case): case for case in calls}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            case = futures[fut]
            try:
                ans, cost = fut.result()
                total_cost += cost
            except Exception as e:  # noqa: BLE001
                print(f"[{i}] {case.id}: failed ({str(e)[:80]})", flush=True)
                continue
            for q in qs:
                a = ans.get(q.id)
                dist[q.id][a if a in q.values else "?"] += 1
            print(f"[{i}/{len(calls)}] {case.id}", flush=True)

    print(f"\n{'question':32s} {'n':>3}  pass  part  fail    NA   ?   max-share")
    flagged = []
    for q in qs:
        c = dist[q.id]
        n = sum(c.values())
        share = max(c.values()) / n if n else 0
        row = f"{q.id:32s} {n:>3}  {c['pass']:>4}  {c['partial_pass']:>4}  {c['fail']:>4}  {c['NA']:>4}  {c['?']:>2}   {share:.2f}"
        if share >= 0.8:
            flagged.append(q.id)
            row += "  <-- skewed"
        print(row)
    print(f"\n{len(flagged)} of {len(qs)} questions have one answer on >=80% of calls: {flagged}")
    print(f"cost: ${total_cost:.2f} ({args.model}, {len(calls)} calls x {len(qs)} questions)")
    if args.save:
        with open(args.save, "w") as f:
            json.dump({q.id: dict(dist[q.id]) for q in qs}, f, indent=1)


if __name__ == "__main__":
    main()
