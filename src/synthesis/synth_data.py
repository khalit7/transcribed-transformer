"""LLM labelling: pick a transcript at random, ask it every bank question that applies.

Terminology (see CLAUDE.md): answering a question about a transcript is **LLM labelling** and the
model doing it is the labeller; assessing how good a label is, is **LLM-as-a-judge** and lives in
a separate step (grade_labels.py, planned). Nothing in this module judges.

    uv run python -m src.synthesis.synth_data <model> <num_transcripts> <families> [options]

<model> names the backend and model: `claude:sonnet`, `claude:opus`, `ollama:qwen3.8`, ...
<num_transcripts> is the number of transcripts to process. <families> is a comma-separated
list of question families to label: any of vulnerability, complaint, eod, general_qa. The
labeller-selection study (experiments/2026-09-03-labeller-selection) settled the split: a local
model labels general_qa, Sonnet labels the three rare-event families, both over the same
transcripts (same seed, same pool), e.g.

    uv run python -m src.synthesis.synth_data claude:sonnet 200 vulnerability,complaint,eod
    uv run python -m src.synthesis.synth_data ollama:qwen3.8 200 general_qa

Each iteration draws a call uniformly at random without replacement, then runs ALL
bank questions of the requested families whose `dataset_allow_list` contains the call's
dataset (a question is not sampled; if it makes sense for that dataset it is asked). For each (call, question)
pair the labeller answers with evidence first, then answer, summary, confidence and any
tags, and one self-contained record is written. Pairs already on disk are skipped and the
transcript selection does not depend on what is on disk, so a re-run after a bank change or an
interruption fills in only the missing pairs of the same calls. Options add the
research-backed checks:
  --verify-model M    a second, preferably different-family labeller answers blind; agreement recorded
  --ablate            re-label with cited lines removed / kept only; necessity and sufficiency recorded
  --labeller-variant  clean (default) or messy: which transcript variant the labeller reads
  --workers N         concurrent labeller calls; defaults to 16 for claude: (independent API calls) and, for
                      ollama:, to one per GPU: an Ollama server pinned to each GPU is started on demand and
                      reused (ollama_servers.py). Verification/ablation calls run inside the same worker.

Output: data/labelled_data/labelled_data.jsonl (+ questions.jsonl regenerated). Resumable.
The records serve both training and benchmarking; splitting them is decided downstream.
Release copy: `python -m src.synthesis.export --track p`.
"""

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import random
import time
from pathlib import Path

from src.synthesis.cases import BUILDERS
from src.synthesis.label import ablate, dumps, label, verify
from src.synthesis.llm import OLLAMA_URLS, LLMError, set_ollama_urls, split_model
from src.synthesis.ollama_servers import ensure_servers, gpu_count
from src.synthesis.question_bank import OUT_DIR, QUESTIONS, write_questions
from src.synthesis.schema import Case, Generation, LabelledRecord, Question

FAMILIES = ["vulnerability", "complaint", "eod", "general_qa"]

OUT = OUT_DIR / "labelled_data.jsonl"
DEFAULT_WORKERS = 16  # concurrent claude -p calls; the audit probe ran 16 without a single failure


def existing_pairs(out: Path) -> set[str]:
    return {json.loads(l)["id"] for l in out.open()} if out.exists() else set()


def load_pool(sources: list[str], per_source: int) -> list[Case]:
    """A fixed pool of calls to draw from (builders stream corpora in a fixed order)."""
    pool: list[Case] = []
    for s in sources:
        it = BUILDERS[s]()
        for _ in range(per_source):
            try:
                pool.append(next(it))
            except StopIteration:
                break
    return pool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("generation_size", metavar="num_transcripts", type=int,
                   help="number of transcripts to label (each gets every applicable question of the requested families)")
    p.add_argument("families", help="comma-separated question families to label: vulnerability,complaint,eod,general_qa")
    p.add_argument("--sources", default="apptek,taskmaster,aci_bench,sporc",
                   help="sporc episodes need their host identified (identify_speakers.py, cached; Sonnet on demand otherwise)")
    p.add_argument("--pool-per-source", type=int, default=200, help="calls loaded per source to draw from")
    p.add_argument("--labeller-variant", default="clean", choices=["clean", "messy"])
    p.add_argument("--verify-model", default=None)
    p.add_argument("--ablate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help=f"concurrent labeller calls (default {DEFAULT_WORKERS} for claude:, 1 for ollama:)")
    p.add_argument("--out", type=Path, default=OUT,
                   help="output file (default data/labelled_data/labelled_data.jsonl); one file per labeller when comparing labellers")
    args = p.parse_args()
    out: Path = args.out
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = set(families) - set(FAMILIES)
    if not families or unknown:
        p.error(f"families must be a comma-separated subset of {','.join(FAMILIES)}; got {args.families!r}")
    questions = [q for q in QUESTIONS if q.family in families]

    if split_model(args.model)[0] == "ollama":
        if args.dry_run:
            set_ollama_urls([f"http://127.0.0.1:{11435 + i}" for i in range(max(1, gpu_count()))])  # report, start nothing
        else:
            set_ollama_urls(ensure_servers())  # one server per GPU, started on demand
    if args.workers is None:
        # claude -p calls are independent API requests. Ollama serves one request at a time per server, so one
        # worker per server, each call routed to a fixed server so its questions share the prefix cache.
        args.workers = DEFAULT_WORKERS if split_model(args.model)[0] == "claude" else len(OLLAMA_URLS)

    rng = random.Random(args.seed)
    pool = load_pool(args.sources.split(","), args.pool_per_source)
    done_pairs = existing_pairs(out)
    counts: collections.Counter = collections.Counter()
    total_cost = 0.0
    done = failures = 0
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # choose the transcripts first (seeded, without replacement, round-robin over datasets so a run of N draws
    # ~N/len(sources) calls from each), then every applicable unlabelled question of each
    by_ds: dict[str, list[Case]] = {}
    for c in pool:
        by_ds.setdefault(c.dataset, []).append(c)
    for cs in by_ds.values():
        rng.shuffle(cs)
    order = [ds for ds in by_ds]
    pool = [c for i in range(max(len(cs) for cs in by_ds.values())) for ds in order if i < len(by_ds[ds]) for c in [by_ds[ds][i]]]
    jobs: list[tuple[Case, Question]] = []
    call_index: dict[str, int] = {}  # routing key: consecutive calls go to consecutive Ollama servers
    calls_done = 0
    for case in pool:
        if calls_done >= args.generation_size:
            break
        call_index[case.id] = calls_done
        # generation_size counts transcripts in seeded order whether or not they still need work, so a
        # re-run (resume, or a second labeller into another file) selects exactly the same calls
        calls_done += 1
        jobs += [(case, q) for q in questions if case.dataset in q.dataset_allow_list and f"{case.id}::{q.id}" not in done_pairs]

    # order jobs so that `workers` calls are in flight at once (their questions interleaved), then the next
    # chunk of calls: with one worker per Ollama server, each server works on one call at a time
    if args.workers > 1:
        by_call: dict[str, list[tuple[Case, Question]]] = {}
        for case, q in jobs:
            by_call.setdefault(case.id, []).append((case, q))
        calls = list(by_call.values())
        jobs = []
        for i in range(0, len(calls), args.workers):
            chunk = calls[i:i + args.workers]
            for k in range(max(len(c) for c in chunk)):
                jobs += [c[k] for c in chunk if k < len(c)]

    if args.dry_run:
        for case, q in jobs:
            print(f"[dry] {case.id}::{q.id}")
        print(f"dry run: {calls_done} transcripts, {len(jobs)} labels ({', '.join(families)}), workers={args.workers}")
        return

    def work(case: Case, q: Question) -> tuple[LabelledRecord, float, str]:
        """Label one pair (plus optional verification and ablation); returns record, cost, log suffix."""
        lab, cost = label(case, q, args.model, args.labeller_variant, route=call_index[case.id])
        # record the variant actually read: a corpus with only a messy variant (SPoRC) serves it for any request
        used_variant = next((v.kind for v in case.transcript.variants if v.kind == args.labeller_variant),
                            case.transcript.variants[0].kind)
        ver = abl = None
        extra = ""
        if args.verify_model:
            try:
                ver, c = verify(case, q, lab, args.verify_model, args.labeller_variant); cost += c
                extra += f" verify={'agree' if ver.agrees else 'DISAGREE'}"
            except (LLMError, ValueError, KeyError) as e:
                extra += f" verify failed: {str(e)[:80]}"
        if args.ablate:
            try:
                abl, c = ablate(case, q, lab, args.model, args.labeller_variant); cost += c
                extra += f" ablate=nec:{abl.necessary} suf:{abl.sufficient}"
            except (LLMError, ValueError, KeyError) as e:
                extra += f" ablation failed: {str(e)[:80]}"
        rec = LabelledRecord(
            id=f"{case.id}::{q.id}", dataset=case.dataset, source_id=case.source_id, track=case.track, question=q,
            transcript=case.transcript, label=lab, verification=ver, ablation=abl, meta=case.meta,
            generation_info=Generation(name=args.model, labelled_variant=used_variant, cost_usd=round(cost, 5),
                                       timestamp=dt.datetime.now(dt.UTC).isoformat(timespec="seconds")),
        )
        return rec, cost, extra

    # records are appended as they complete (order differs from submission; ids make that irrelevant)
    out.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, out.open("a") as f:
        futures = {ex.submit(work, case, q): (case, q) for case, q in jobs}
        for fut in concurrent.futures.as_completed(futures):
            case, q = futures[fut]
            rid = f"{case.id}::{q.id}"
            try:
                rec, cost, extra = fut.result()
            except (LLMError, ValueError, KeyError) as e:
                failures += 1
                print(f"  {rid}: label failed: {str(e)[:120]}", flush=True)
                continue
            f.write(dumps(rec) + "\n")
            f.flush()
            total_cost += cost
            counts[(q.family, rec.label.answer)] += 1
            done += 1
            print(f"[{done}/{len(jobs)}] {rid} -> {rec.label.answer} ev={rec.label.evidence} "
                  f"conf={rec.label.confidence}{extra} cost=${total_cost:.3f}", flush=True)

    n_q = write_questions()
    print(f"done: {calls_done} transcripts, {done} labels, {failures} failures, ${total_cost:.3f}, "
          f"{time.time() - t0:.0f}s, workers={args.workers} -> {out}; questions.jsonl regenerated ({n_q})")
    if counts:
        print("answers by family:")
        for fam in sorted({f for f, _ in counts}):
            print(f"  {fam:14s} " + "  ".join(f"{a}={counts[(fam, a)]}" for a in ("pass", "partial_pass", "fail", "NA")))


if __name__ == "__main__":
    main()
