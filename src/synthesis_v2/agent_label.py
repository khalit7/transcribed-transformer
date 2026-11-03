"""Labelling through the Claude Code harness: work packages for labelling subagents, and ingestion of their output.

    uv run python -m src.synthesis_v2.agent_label prepare --ids <ids file> --out data/labelled_data_v2/agent/<run>
    uv run python -m src.synthesis_v2.agent_label ingest --run data/labelled_data_v2/agent/<run> --labeller agent:opus \\
        --out data/labelled_data_v2/agent/<run>/labels.jsonl

The owner's decision (2026-10-31, D69): v2 gold is produced by Claude Opus running as a Claude Code subagent, not by a
headless `claude -p` prompt. A subagent reads the episode with its tools (Read, Grep), can check every mention of a
name or every line a speaker says instead of trusting one pass of attention, and writes its labels to a file; the orchestrating
session spawns one subagent per batch of episodes and ingests the files. Nothing in the labelling path calls a model from
this code: `prepare` writes packages, `ingest` validates and converts.

A package is one directory per episode under `<run>/episodes/<doc_id>/`:
- `transcript.txt`: the rendering the fine-tuned models will see (`n. host_1:text`, `n. guest_1:text`, `n. SPEAKER_NN:text`
  for any other voice; every corpus turn verbatim, D75/D76);
- `questions.json`: the questions, each with id, text, description, options;
- `labels.json` is what the subagent writes: {"<question id>": {"evidence": [...], "answer": "...", "summary": "...",
  "confidence": 0-1}, ...}; `notes.md` is optional, for anything the subagent wants to record about the episode.
`<run>/TASK.md` is the subagent's brief, written once per run so every subagent reads the identical instructions.

`ingest` rebuilds every question from the package's own `questions.json` (the text the labeller saw, never the current
bank), checks that `transcript.txt` still equals the rendering, refuses an answer outside the question's
options (after trimming whitespace and case, which it reports), clips evidence to the line count (counting what it
dropped), reports a `none` answer that carries evidence, a non-`none` answer without evidence, a blank summary and any
coerced confidence, and writes one `LabelledRecord` per label with `generation_info.name = "agent:<model>"`, `cost_usd =
0.0` (the harness bills the session, not the call; token counts from the agent reports go in `meta.agent_tokens` when
given) and `meta.agent` / `meta.batch` naming the subagent and batch that produced it (`meta.batch_tokens` is the batch's
token count, the same on every record of the batch, not a per-label figure), so every label traces to its agent. A revised `labels.json` is not re-ingested over an existing record: delete the record first. No question takes a
per-episode argument (D78).
"""

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

from src import prompts
from src.synthesis.schema import Case, Generation, Label, LabelledRecord, Question
from src.synthesis_v2 import OUT_DIR, append_jsonl, numbered, read_jsonl
from src.synthesis_v2 import question_bank as bank
from src.synthesis_v2.cases import pool_cases

TASK_FILE = prompts.V2 / "labelling_task.md"  # the subagents' brief, copied verbatim into every run as TASK.md
TASK = prompts.text(TASK_FILE) + "\n"


def write_task(run: Path) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "TASK.md").write_text(TASK)


def package(case: Case, questions: list[Question], run: Path) -> Path:
    doc_id = case.id.removeprefix("sporc-")
    d = run / "episodes" / doc_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.txt").write_text(numbered(case.transcript.lines("messy")) + "\n")
    (d / "questions.json").write_text(json.dumps(
        [{"id": q.id, "text": q.text, "description": bank.with_definitions(q).description,
          "options": [{"value": o.value, "criteria": o.criteria} for o in q.options]} for q in questions],
        indent=1, ensure_ascii=False))
    return d


def prepare(ids: list[str], run: Path, questions: list[Question] | None = None) -> dict:
    """Write TASK.md and one package per buildable episode; returns the manifest (episode -> questions, skips)."""
    write_task(run)
    rows = {r["doc_id"]: r for r in read_jsonl(OUT_DIR / "episodes.jsonl")}
    questions = questions if questions is not None else bank.BANK
    manifest: dict = {"episodes": {}, "skipped": {}}
    skipped: dict[str, int] = {}
    for case in pool_cases(ids, skipped):
        d = case.id.removeprefix("sporc-")
        package(case, questions, run)
        manifest["episodes"][d] = {"stratum": rows[d]["stratum"], "n_lines": len(case.transcript.lines("messy")),
                                   "questions": [q.id for q in questions]}
    manifest["skipped"].update(skipped)
    manifest["not_built"] = [d for d in ids if d not in manifest["episodes"]]
    (run / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def batches(doc_ids: list[str], manifest: dict, per_agent: int) -> list[list[str]]:
    """Group episodes for the subagents: by stratum, `per_agent` at a time, so one agent's context holds its episodes."""
    out: list[list[str]] = []
    strata = sorted({manifest["episodes"][d]["stratum"] for d in doc_ids if d in manifest["episodes"]})
    for stratum in strata:
        ids = [d for d in doc_ids if manifest["episodes"].get(d, {}).get("stratum") == stratum]
        out += [ids[i:i + per_agent] for i in range(0, len(ids), per_agent)]
    assert sorted(d for g in out for d in g) == sorted(d for d in doc_ids if d in manifest["episodes"])
    return out


def package_questions(run: Path, doc_id: str) -> list[Question]:
    """The questions exactly as the package showed them to the labeller (the bank entry with the package's text,
    description and options), so a later edit to the bank can never relabel a record with different wording."""
    by_id = bank.by_id()
    out = []
    for entry in json.loads((run / "episodes" / doc_id / "questions.json").read_text()):
        base = by_id.get(entry["id"])
        if base is None:  # a question removed from the bank after the package was written: its labels are not ingested
            continue
        out.append(bank.without_definitions(base.model_copy(update={  # stored like every route: definitions once, apart
            "text": entry["text"], "description": entry["description"],
            "options": [type(base.options[0])(**x) for x in entry["options"]]})))
    return out


def question_ids(spec: str | None) -> list[Question] | None:
    """The bank or provisional questions named in a comma-separated `--questions` value, in bank order; an id that is
    in neither is an error (a removed id is named as such), so a run can never silently come out short."""
    if spec is None:
        return None
    wanted = [x.strip() for x in spec.split(",") if x.strip()]
    if not wanted:
        raise SystemExit("--questions names no question id")
    by_id = bank.by_id()
    unknown = [x for x in wanted if x not in by_id]
    if unknown:
        removed = [x for x in unknown if x in bank.REMOVED]
        raise SystemExit(f"unknown question ids {unknown}" + (f" (removed from the bank after the pilot: {removed})" if removed else ""))
    return [q for q in by_id.values() if q.id in set(wanted)]


_DIGITS = re.compile(r"\s*\d+\s*", re.ASCII)


def ingest_episode(run: Path, doc_id: str, case: Case, questions: list[Question], labeller: str, agent: str, batch: str,
                   agent_tokens: int | None) -> tuple[list[LabelledRecord], list[str]]:
    """Validate one episode's labels.json and build its records; returns (records, problems). Problems that drop a
    label say so; problems that only flag a label (coercions, a `none` with evidence, a blank summary) are reported and
    the label is kept with the flag in its meta."""
    d = run / "episodes" / doc_id
    problems: list[str] = []
    try:
        labels = json.loads((d / "labels.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return [], [f"{doc_id}: labels.json unreadable ({type(e).__name__})"]
    if not isinstance(labels, dict):
        return [], [f"{doc_id}: labels.json is not an object"]
    expected = numbered(case.transcript.lines("messy")) + "\n"
    if (d / "transcript.txt").read_text() != expected:
        return [], [f"{doc_id}: transcript.txt differs from the current rendering; not ingested"]
    n_lines = len(case.transcript.lines("messy"))
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    records: list[LabelledRecord] = []
    for q in questions:
        r = labels.get(q.id)
        if not isinstance(r, dict):
            problems.append(f"{doc_id}::{q.id}: missing")
            continue
        flags: list[str] = []
        answer = r.get("answer")
        if isinstance(answer, str) and answer not in q.values and answer.strip().lower() in q.values:
            flags.append(f"answer coerced from {answer!r}")
            answer = answer.strip().lower()
        if answer not in q.values:
            problems.append(f"{doc_id}::{q.id}: answer {answer!r} not in {q.values}; dropped")
            continue
        raw = r.get("evidence", None)
        if raw is None:
            raw = []
            flags.append("evidence key missing")
        if not isinstance(raw, list):
            problems.append(f"{doc_id}::{q.id}: evidence is not a list; dropped")
            continue
        kept: set[int] = set()
        dropped = 0
        for e in raw:
            if isinstance(e, bool) or not (isinstance(e, int) or (isinstance(e, str) and _DIGITS.fullmatch(e))):
                dropped += 1
            elif 1 <= int(e) <= n_lines:
                kept.add(int(e))
            else:
                dropped += 1
        if dropped:
            flags.append(f"{dropped} evidence entries dropped")
        if answer == "none" and kept:
            flags.append("none with evidence")
        if answer != "none" and not kept:
            flags.append("non-none without evidence")
        conf = r.get("confidence")
        if isinstance(conf, str):
            try:
                conf = float(conf)
                flags.append("confidence coerced from string")
            except ValueError:
                conf = None
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or (isinstance(conf, float) and math.isnan(conf)):
            if conf is not None:
                flags.append("confidence unusable")
            conf = None
        elif not 0.0 <= conf <= 1.0:
            flags.append(f"confidence clipped from {conf}")
            conf = min(1.0, max(0.0, float(conf)))
        summary = r.get("summary")
        if not (isinstance(summary, str) and summary.strip()):
            flags.append("blank summary")
        for f in flags:
            problems.append(f"{doc_id}::{q.id}: {f}")
        records.append(LabelledRecord(
            id=f"{case.id}::{q.id}", dataset=case.dataset, source_id=case.source_id, track=case.track, question=q,
            transcript=case.transcript,
            label=Label(answer=answer, evidence=sorted(kept), summary=str(summary).strip() if isinstance(summary, str) else "",
                        tags=[], confidence=None if conf is None else float(conf)),
            generation_info=Generation(name=labeller, labelled_variant="messy", cost_usd=0.0, timestamp=stamp,
                                       claude_account=None),
            meta={**case.meta, "agent": agent, "batch": batch, "batch_tokens": agent_tokens, "evidence_dropped": dropped,
                  "flags": flags, "held_out": q.id in bank.HELD_OUT or q.id in bank.PROVISIONAL_HELD_OUT,
                  "provisional": q.id in {p.id for p in bank.PROVISIONAL}, "method": "claude-code-subagent"}))
    extra = set(labels) - {q.id for q in questions}
    if extra:
        problems.append(f"{doc_id}: {len(extra)} unknown question ids ignored")
    return records, problems


def ingest(run: Path, labeller: str, out: Path, agents: dict[str, dict] | None = None) -> dict:
    """Ingest every episode of a run that has labels.json. `agents` maps doc_id -> {"agent": ..., "batch": ...,
    "tokens": ...} (from the orchestrator's batch log); unknown episodes get agent "unknown"."""
    manifest = json.loads((run / "manifest.json").read_text())
    done = {r["id"] for r in read_jsonl(out)}
    ids = [d for d in manifest["episodes"] if (run / "episodes" / d / "labels.json").exists()]
    summary: dict = {"episodes_with_labels": len(ids), "missing_labels": [d for d in manifest["episodes"] if d not in ids],
                     "records": 0, "problems": [], "skipped_done": 0, "not_built": []}
    built: set[str] = set()
    for case in pool_cases(ids):
        d = case.id.removeprefix("sporc-")
        built.add(d)
        info = (agents or {}).get(d, {})
        qs = package_questions(run, d)
        recs, problems = ingest_episode(run, d, case, qs, labeller, info.get("agent", "unknown"), info.get("batch", "unknown"),
                                        info.get("tokens"))
        summary["problems"] += problems
        for r in recs:
            if r.id in done:
                summary["skipped_done"] += 1
                continue
            append_jsonl(out, json.loads(r.model_dump_json()))
            summary["records"] += 1
    summary["not_built"] = [d for d in ids if d not in built]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="the run directory")
    p.add_argument("--questions", default=None, help="comma-separated question ids (default: the whole bank)")
    p.add_argument("--per-agent", type=int, default=4)
    i = sub.add_parser("ingest")
    i.add_argument("--run", type=Path, required=True)
    i.add_argument("--labeller", required=True, help="recorded as generation_info.name, e.g. agent:opus")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--agents", type=Path, default=None, help="JSON {doc_id: {agent, batch, tokens}} from the orchestrator")
    a = ap.parse_args()
    if a.cmd == "prepare":
        ids = [x.strip() for x in a.ids.read_text().split() if x.strip()]
        qs = question_ids(a.questions)
        m = prepare(ids, a.out, qs)
        groups = batches(list(m["episodes"]), m, a.per_agent)
        (a.out / "batches.json").write_text(json.dumps(groups, indent=1))
        print(json.dumps({"episodes": len(m["episodes"]), "skipped": m["skipped"], "batches": len(groups),
                          "labels_expected": sum(len(e["questions"]) for e in m["episodes"].values())}))
    else:
        agents = json.loads(a.agents.read_text()) if a.agents else None
        s = ingest(a.run, a.labeller, a.out, agents)
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
