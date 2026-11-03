"""Label v2 episodes: every bank question asked of every episode, k questions per call, one record per label.

    uv run python -m src.synthesis_v2.label_episodes claude:opus --ids data/labelled_data_v2/pilot_ids.txt \\
        --k 7 --seed 0 --out data/labelled_data_v2/pilot/opus_k7.jsonl --claude-account p

The transcript is sent once per call with k questions (k = 1 reproduces v1's one-question prompt shape; the pilot
compares k = 1 and k = 7 on the same episodes because the `claude -p` route charges per call). Questions are grouped
and ordered by a seed so a second pass under another seed measures order effects. Each question's answer, evidence and
summary come back keyed by question id; one `LabelledRecord` is written per question, the call's cost split equally
over its questions. No question takes a per-episode argument (D78). Resumable: ids already in the output file are
skipped.

Prompt shape: the podcast preamble of `src/train/data.task_prompt` (what the fine-tuned models see), the numbered
transcript, then the questions. Evidence-first order and the "empty list when nothing bears on it" rule are v1's.

`--lean` (D82, the production route): every question of an episode in one call, ordered by the seed; the shared
definitions (`question_bank.VOCAB`) printed once before the questions instead of opening every description
(`label_episodes_lean.txt`); and a lean `claude -p` call (`llm.claude_command`: the one-line system prompt in
`labeller_system.txt`, no tools, prompt caching off). Measured on 60 pilot episodes against the pilot's k = 7 Opus
labels: answer agreement 0.916, the same as Opus against itself across two lean runs (0.914), at 6.4x lower cost.
Records carry the bank's question with its own description; the shared definitions are stored once (`definitions.txt`).

    uv run --no-sync python -m src.synthesis_v2.label_episodes claude:opus --lean --ids data/labelled_data_v2/selected_ids.txt \\
        --seed 0 --out data/labelled_data_v2/labelled_data.jsonl --claude-account p --workers 2
"""

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import random
from pathlib import Path

from src import prompts
from src.synthesis import llm
from src.synthesis.llm import ask_json_full, set_claude_account
from src.synthesis.schema import Case, Generation, Label, LabelledRecord, Question
from src.synthesis_v2 import OUT_DIR, append_jsonl, numbered, read_jsonl
from src.synthesis_v2 import question_bank as bank
from src.synthesis_v2.cases import pool_cases

PROMPT = prompts.V2 / "label_episodes.txt"  # several questions per call; its preamble is the training prompt's
PROMPT_LEAN = prompts.V2 / "label_episodes_lean.txt"  # the same with a slot for the shared definitions, printed once
SYSTEM_LEAN = prompts.V2 / "labeller_system.txt"  # replaces Claude Code's system prompt on the lean route


def question_block(q: Question) -> str:
    opts = "\n".join(f"- {o.value}: {o.criteria}" for o in q.options)
    ctx = f"\nContext: {q.description}" if q.description else ""
    return f"QUESTION {q.id}: {q.text}{ctx}\nAnswer options and grading rules:\n{opts}"


def prompt(case: Case, questions: list[Question]) -> str:
    questions = [bank.with_definitions(q) for q in questions]  # this route repeats the definitions in every question
    lines = case.transcript.lines("messy")
    blocks = "\n\n".join(question_block(q) for q in questions)
    ids = [q.id for q in questions]
    shape = ", ".join(f'"{i}": {{"evidence": [<line numbers>], "answer": "<one of the options>", "summary": "<reasoning>", '
                      f'"confidence": <0-1>}}' for i in ids)
    return prompts.render(PROMPT, speakers=", ".join(case.transcript.speakers), transcript=numbered(lines),
                          n_questions=len(questions), blocks=blocks, shape=shape)


def prompt_lean(case: Case, questions: list[Question]) -> str:
    """Every question in one prompt, the shared definitions once: the template prints them a single time before the
    questions, and each question shows only its own description. Otherwise the same blocks, shape and order as
    `prompt`."""
    stripped = [bank.without_definitions(q) for q in questions]
    lines = case.transcript.lines("messy")
    blocks = "\n\n".join(question_block(q) for q in stripped)
    shape = ", ".join(f'"{q.id}": {{"evidence": [<line numbers>], "answer": "<one of the options>", "summary": "<reasoning>", '
                      f'"confidence": <0-1>}}' for q in stripped)
    return prompts.render(PROMPT_LEAN, speakers=", ".join(case.transcript.speakers), transcript=numbered(lines),
                          n_questions=len(stripped), definitions=bank.VOCAB, blocks=blocks, shape=shape)


def parse(resp: dict, q: Question, n_lines: int) -> tuple[Label, int]:
    """The label for one question from the keyed reply, and how many evidence entries were dropped (out of range or
    not a line number). Evidence must be a list of integers or digit strings; anything else fails the question."""
    r = resp.get(q.id)
    if not isinstance(r, dict):
        raise TypeError(f"no answer object for {q.id}")
    answer = r.get("answer")
    if answer not in q.values:
        raise ValueError(f"{q.id}: answer {answer!r} not in {q.values}")
    raw = r.get("evidence", [])
    if not isinstance(raw, list):
        raise TypeError(f"{q.id}: evidence is not a list")
    kept: set[int] = set()
    dropped = 0
    for e in raw:
        if isinstance(e, bool):
            dropped += 1
        elif isinstance(e, int) or (isinstance(e, float) and e.is_integer()) or (isinstance(e, str) and e.strip().isdigit()):
            n = int(float(e))
            if 1 <= n <= n_lines:
                kept.add(n)
            else:
                dropped += 1
        else:
            dropped += 1
    conf = r.get("confidence")
    conf = min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else None
    summary = r.get("summary")
    return Label(answer=answer, evidence=sorted(kept), summary=str(summary).strip() if summary is not None else "",
                 tags=[], confidence=conf), dropped


def batches(questions: list[Question], k: int, seed: str) -> list[list[Question]]:
    order = list(questions)
    random.Random(seed).shuffle(order)
    return [order[i:i + k] for i in range(0, len(order), k)]


def label_case(case: Case, questions: list[Question], model: str, k: int, seed: int, stratum: str,
               calls_log: Path | None = None, lean: bool = False) -> tuple[list[LabelledRecord], int]:
    """All labels for one episode; returns (records, failed questions). Every paid call is written to `calls_log` with
    its full cost (every attempt), the CLI's token usage and the question ids it carried, whether or not its answers
    parsed, so the measured cost is the real spend."""
    records: list[LabelledRecord] = []
    failed = 0
    n_lines = len(case.transcript.lines("messy"))
    if lean:
        k = len(questions)  # one call per episode
    system = prompts.text(SYSTEM_LEAN) if lean else None
    for b_index, batch in enumerate(batches(questions, k, f"{seed}:{case.id}")):
        stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        try:
            resp, cost_total, info = ask_json_full(prompt_lean(case, batch) if lean else prompt(case, batch), model, None,
                                                   system=system)
        except Exception as e:  # noqa: BLE001
            if calls_log is not None:
                append_jsonl(calls_log, {"case": case.id, "batch_index": b_index, "questions": [q.id for q in batch],
                                         "cost_usd": round(getattr(e, "cost", 0.0), 6), "failed": str(e)[:200], "timestamp": stamp})
            print(f"  {case.id}: call failed ({str(e)[:100]})", flush=True)
            failed += len(batch)
            continue
        if calls_log is not None:
            append_jsonl(calls_log, {"case": case.id, "batch_index": b_index, "questions": [q.id for q in batch],
                                     "cost_usd": round(cost_total, 6), "attempts": info["attempts"], "usage": info["usage"],
                                     "timestamp": stamp})
        if not isinstance(resp, dict):
            print(f"  {case.id}: reply is not an object", flush=True)
            failed += len(batch)
            continue
        per_q = cost_total / len(batch)
        for pos, q in enumerate(batch):
            try:
                lab, dropped = parse(resp, q, n_lines)
            except Exception as e:  # noqa: BLE001  (one bad answer must not lose the batch's other answers)
                print(f"  {case.id}::{q.id}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                failed += 1
                continue
            records.append(LabelledRecord(
                id=f"{case.id}::{q.id}", dataset=case.dataset, source_id=case.source_id, track=case.track, question=q,
                transcript=case.transcript, label=lab,
                generation_info=Generation(name=model, labelled_variant="messy", cost_usd=round(per_q, 6),
                                           timestamp=stamp, claude_account=llm.CLAUDE_ACCOUNT),
                meta={**case.meta, "stratum": stratum, "k": k, "seed": seed, "batch_size": len(batch),
                      "format": "lean" if lean else "standard",
                      "batch_index": b_index, "position_in_batch": pos, "evidence_dropped": dropped,
                      "held_out": q.id in bank.HELD_OUT or q.id in bank.PROVISIONAL_HELD_OUT,
                      "provisional": q.id in {p.id for p in bank.PROVISIONAL}}))
    return records, failed


def existing(out: Path) -> set[str]:
    """The record ids already written, read one line at a time: the production file holds 264,000 records with their
    transcripts (about 8 GB), which `read_jsonl` would hold in memory whole. A final line without its newline (cut short by a
    kill) is not counted: the next append cuts it away, so its label must be asked again."""
    ids: set[str] = set()
    if not out.exists():
        return ids
    with out.open(encoding="utf-8", errors="replace") as fh:  # a kill can cut a multi-byte character
        for line in fh:
            if not line.strip() or not line.endswith("\n"):  # a final line without its newline is unfinished
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("model", help="claude:opus or claude:sonnet")
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--k", type=int, default=7, help="questions per call (ignored with --lean: all in one call)")
    ap.add_argument("--lean", action="store_true", help="the production route (D82): one lean call per episode")
    ap.add_argument("--seed", type=int, default=0, help="grouping and order of the questions")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--questions", default=None, help="comma-separated question ids (default: the whole bank)")
    ap.add_argument("--claude-account", "--claude_account", dest="claude_account", choices=["p", "w"], default=None,
                    help="required for claude: models; recorded on every record")
    ap.add_argument("--dry-run", action="store_true", help="write the first prompt to <out>.dryrun.txt and the job count, call nothing")
    a = ap.parse_args()
    if a.model.startswith("claude:") and a.claude_account is None:
        ap.error("--claude-account p|w is required for claude: models")
    set_claude_account(a.claude_account)
    ids = [x.strip() for x in a.ids.read_text().split() if x.strip()]
    rows = {r["doc_id"]: r for r in (json.loads(line) for line in (OUT_DIR / "episodes.jsonl").open())}
    from src.synthesis_v2.agent_label import (
        question_ids,  # imported here: agent_label imports this module
    )

    questions = question_ids(a.questions) or bank.BANK
    done = existing(a.out)
    skipped: dict[str, int] = {}
    cases = {c.id.removeprefix("sporc-"): c for c in pool_cases(ids, skipped)}
    jobs: list[tuple[str, Case, list[Question]]] = []
    for d in ids:
        if d not in cases:
            continue
        qs = [q for q in questions if f"{cases[d].id}::{q.id}" not in done]
        if qs:
            jobs.append((d, cases[d], qs))
    n_labels = sum(len(j[2]) for j in jobs)
    print(f"{len(ids)} episodes, {len(jobs)} with work, {n_labels} labels to produce with {a.model}, "
          f"{'lean, one call per episode' if a.lean else f'k={a.k}'}; skipped {skipped}",
          flush=True)
    if a.dry_run:
        if jobs:
            d, case, qs = jobs[0]
            dry = a.out.with_suffix(".dryrun.txt")
            dry.parent.mkdir(parents=True, exist_ok=True)
            k = len(qs) if a.lean else a.k
            first = batches(qs, k, f"{a.seed}:{case.id}")[0]
            dry.write_text(prompt_lean(case, first) if a.lean else prompt(case, first))
            print(f"first prompt written to {dry}")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    calls_log = a.out.with_suffix(".calls.jsonl")
    spent = 0.0
    counts: collections.Counter = collections.Counter()
    failed_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(label_case, case, qs, a.model, a.k, a.seed, rows[d]["stratum"], calls_log, a.lean): d
                for d, case, qs in jobs}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            d = futs[fut]
            try:
                recs, failed = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(jobs)}] {d}: episode failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                failed_total += len(next(qs for dd, _c, qs in jobs if dd == d))
                continue
            failed_total += failed
            for r in recs:
                append_jsonl(a.out, json.loads(r.model_dump_json()))
                spent += r.generation_info.cost_usd
                counts[r.question.id] += 1
            print(f"[{i}/{len(jobs)}] {d}: {len(recs)} labels, {failed} failed, ${spent:.2f} so far", flush=True)
    calls = read_jsonl(calls_log)
    print(json.dumps({"labels": sum(counts.values()), "failed": failed_total, "episodes": len(jobs),
                      "cost_usd_recorded_on_labels": round(spent, 2),
                      "cost_usd_all_calls_in_log": round(sum(c.get("cost_usd", 0.0) for c in calls), 2),
                      "calls_in_log": len(calls), "per_question": dict(counts)}))


if __name__ == "__main__":
    main()
