"""Generate benchmark labels.

    uv run python -m src.synthesis.synth_data <model_name> <generation_size> [options]

<model_name> goes to `claude -p --model`; <generation_size> is how many calls to label.

Per call the plan is:
  1. one INJECTED label — a bank question (its target answer made true by
     writing turns in) or, with probability --written-fraction, a question the
     generator invents for this call together with the turns;
  2. --as-is-per-call bank questions judged on the final transcript;
  3. with probability --open-fraction, a question written about the call and judged.

Calls rotate over --sources so both licence tracks fill. Output: one
self-contained record per label in data/benchmark/labelled_data.jsonl;
data/benchmark/questions.jsonl is regenerated from it afterwards. Resumable:
calls already labelled are skipped. Release copy: `python -m src.synthesis.export --track p`.
"""

import argparse
import datetime as dt
import json
import random
import time

from src.synthesis.cases import BUILDERS
from src.synthesis.generate import dumps, generate
from src.synthesis.llm import LLMError
from src.synthesis.question_bank import BENCH, QUESTIONS, write_questions
from src.synthesis.schema import Case, Generation, LabelledRecord, Question

FAMILIES = ["vulnerability", "complaint_and_eod", "general_qa"]
OUT = BENCH / "labelled_data.jsonl"


def labelled_calls() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(l)["id"].split("::")[0] for l in OUT.open()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("generation_size", type=int)
    p.add_argument("--sources", default="apptek,taskmaster,aci_bench,sporc")
    p.add_argument("--tag-policy", default="random", choices=["random", "agent_first"])
    p.add_argument("--as-is-per-call", type=int, default=2)
    p.add_argument("--open-fraction", type=float, default=0.25)
    p.add_argument("--written-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="build calls, call no model")
    args = p.parse_args()

    rng = random.Random(args.seed)
    sources = args.sources.split(",")
    iters = {s: BUILDERS[s](args.tag_policy, args.seed) for s in sources}
    done_calls = labelled_calls()
    use_count: dict[str, int] = {}
    total_cost = 0.0
    done = failures = 0
    t0 = time.time()
    BENCH.mkdir(parents=True, exist_ok=True)

    i = 0
    while done < args.generation_size and sources:
        src = sources[i % len(sources)]
        i += 1
        try:
            case: Case = next(iters[src])
        except StopIteration:
            sources.remove(src)
            continue
        if case.id in done_calls:
            continue
        if args.dry_run:
            print(f"[dry] {case.id} ({src}) {case.transcript.n_lines} lines, "
                  f"variants={[v.kind for v in case.transcript.variants]}")
            done += 1
            continue

        # the plan: (mode, question or None, family for written questions, preferred target)
        plan: list[tuple[str, Question | None, str | None, str | None]] = []
        if rng.random() < args.written_fraction:
            plan.append(("injected", None, rng.choice(FAMILIES), rng.choice(["pass", "fail", "partial_pass"])))
            injected_q = None
        else:
            injected_q = min(QUESTIONS, key=lambda x: (use_count.get(x.id, 0), rng.random()))
            use_count[injected_q.id] = use_count.get(injected_q.id, 0) + 1
            prefer = rng.choice([v for v in injected_q.values if v != "NA"])
            plan.append(("injected", injected_q, None, prefer))
        others = [x for x in QUESTIONS if x is not injected_q]
        rng.shuffle(others)
        plan += [("as_is", x, None, None) for x in others[: args.as_is_per_call]]
        if rng.random() < args.open_fraction:
            plan.append(("as_is", None, rng.choice(FAMILIES), None))

        records: list[LabelledRecord] = []
        for mode, question, family, prefer in plan:
            try:
                case, q, label, cost = generate(case, mode, args.model, question=question, family=family, prefer=prefer)
            except (LLMError, ValueError, KeyError) as e:
                failures += 1
                print(f"  {mode} {(question.id if question else family)} failed: {str(e)[:120]}", flush=True)
                if mode == "injected":
                    records = []
                    break  # a call without its injected label is not worth keeping
                continue
            total_cost += cost
            gen = Generation(name=args.model, mode=mode, cost_usd=round(cost, 5),
                             timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            records.append(LabelledRecord.build(case, q, label, gen))
        if not records:
            continue

        with OUT.open("a") as f:
            for r in records:
                f.write(dumps(r) + "\n")
        done_calls.add(case.id)
        done += 1
        first = records[0]
        print(f"[{done}/{args.generation_size}] {case.id} ({src}) injected={first.question.id}:{first.label.answer} "
              f"labels={len(records)} cost=${total_cost:.3f}", flush=True)

    n_q = write_questions()
    print(f"done: {done} calls, {failures} generator failures, ${total_cost:.3f}, {time.time() - t0:.0f}s; "
          f"questions.jsonl regenerated ({n_q} questions) -> {BENCH}")


if __name__ == "__main__":
    main()
