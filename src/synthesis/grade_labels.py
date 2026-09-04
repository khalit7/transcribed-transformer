"""LLM-as-a-judge: grade labellers' outputs on the same (call, question) pairs.

Terminology (CLAUDE.md): the models that produced the labels are labellers; the model run
here is the judge. It never labels for the dataset; it grades.

Input: one labelled_data-format file per labeller (synth_data --out), all over the same calls.
For a stratified sample of pairs present in every file, the judge sees the transcript, the
question with its grading rules, and every labeller's output (anonymised as A, B, C ... in a
per-pair shuffled order so position and identity carry no information). It first answers the
question itself, evidence first, so its grades are anchored on its own reading, then grades
each label on four criteria:

  answer   correct | wrong                       (against the judge's own answer, by the rules)
  evidence 0 irrelevant or contradicting lines / 1 partly supports or misses key lines /
           2 the cited lines support the answer and cover the material that decides it
  summary  0 unfaithful (claims not in the transcript, or contradicts the answer) /
           1 faithful but thin or partly off / 2 faithful and justifies the answer
  tags     for tagged questions only: 0 wrong or missing characteristics / 1 partly right /
           2 matches what is disclosed; null otherwise

Output: one record per pair in data/labelled_data/judgements/<name>.jsonl with the judge's
own answer and evidence, the grades per labeller, the anonymisation order and the cost, plus
a review file for human calibration. Judge grades are LLM judgements too; the human review
of a subset is what says how far to trust them.

    uv run python -m src.synthesis.grade_labels data/labelled_data/labellers/*.jsonl --sample 400 --judge claude:opus
"""

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import random
import string
from pathlib import Path

from src.synthesis.label import numbered
from src.synthesis.llm import LLMError, ask_json
from src.synthesis.question_bank import OUT_DIR
from src.synthesis.schema import LabelledRecord

JUDGE_DIR = OUT_DIR / "judgements"
GRADE = {"type": "object",
         "properties": {"answer_correct": {"type": "boolean"},
                        "evidence": {"type": "integer", "minimum": 0, "maximum": 2},
                        "summary": {"type": "integer", "minimum": 0, "maximum": 2},
                        "tags": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
                        "reason": {"type": "string"}},
         "required": ["answer_correct", "evidence", "summary", "tags", "reason"]}


def schema(letters: list[str], answers: list[str]) -> dict:
    return {"type": "object",
            "properties": {"judge_evidence": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                           "judge_answer": {"type": "string", "enum": answers},
                           "judge_tags": {"type": "array", "items": {"type": "string"}},
                           "grades": {"type": "object", "properties": {x: GRADE for x in letters}, "required": letters}},
            "required": ["judge_evidence", "judge_answer", "judge_tags", "grades"]}


def prompt(rec: LabelledRecord, labels: list[tuple[str, LabelledRecord]]) -> str:
    q = rec.question
    lines = rec.transcript.lines("clean")
    opts = "\n".join(f"- {o.value}: {o.criteria}" for o in q.options)
    ctx = f"\nContext: {q.description}" if q.description else ""
    tag_txt = ("\nThe answer must carry tags from this list (empty if none applies): " + "; ".join(q.tags)) if q.tags else ""
    shown = []
    for letter, r in labels:
        tags = f", tags={r.label.tags}" if q.tags else ""
        shown.append(f"LABEL {letter}: answer={r.label.answer}, evidence lines={r.label.evidence}{tags}\n  summary: {r.label.summary}")
    labels_txt = "\n".join(shown)
    letters = [x for x, _ in labels]
    grade_fields = ", ".join(f'"{x}": {{"answer_correct": <bool>, "evidence": <0-2>, "summary": <0-2>, "tags": <0-2 or null>, "reason": "<one sentence>"}}' for x in letters)
    return f"""You are the judge of automatic labels for a conversation-transcript quality-assurance task. One turn per line, numbered; each line starts with the speaker's role as recorded by the source ({', '.join(rec.transcript.speakers)}).

TRANSCRIPT
{numbered(lines)}

QUESTION: {q.text}{ctx}
Answer options and grading rules:
{opts}{tag_txt}

Several labellers answered this question. Each gave an answer, the line numbers it treats as evidence, and a summary of its reasoning{', and tags' if q.tags else ''}:

{labels_txt}

Work in this order. First answer the question yourself: list every line that bears on it, then decide the answer strictly by the rules{', then list the tags that apply' if q.tags else ''}. Then grade every label independently against the transcript and the rules, not against the other labels:
- answer_correct: true only if the label's answer is the one the rules give for this transcript.
- evidence: 2 if the cited lines support the label's answer and cover the material that decides it; 1 if they partly support it or miss lines that matter; 0 if they are irrelevant, contradict the answer, or the list is empty when evidence exists.
- summary: 2 if the summary is faithful to the transcript and justifies the answer; 1 if faithful but thin or partly beside the point; 0 if it asserts things the transcript does not say or contradicts the label's own answer.
- tags: {'2 if the tags match the characteristics actually disclosed, 1 if partly, 0 if wrong or missing' if q.tags else 'null (this question takes no tags)'}.
Judge a label on what it actually cites and says; a right answer with wrong evidence gets answer_correct true and a low evidence score.

Respond with a single JSON object and nothing else:
{{"judge_evidence": [<line numbers>], "judge_answer": "<one of {q.values}>", "judge_tags": [<tags or empty>], "grades": {{{grade_fields}}}}}"""


def load(files: list[Path]) -> dict[str, dict[str, LabelledRecord]]:
    """labeller name -> {pair id -> record}."""
    out = {}
    for f in files:
        recs = {}
        for line in f.open():
            r = LabelledRecord.model_validate_json(line)
            recs[r.id] = r
        out[f.stem] = recs
    return out


def sample_pairs(by_labeller: dict[str, dict[str, LabelledRecord]], n: int, seed: int) -> list[str]:
    """Pairs every labeller completed, stratified by (dataset, family) with rare answers over-represented:
    a pair where any labeller answered fail or partial_pass on a family question, or where labellers
    disagree, is kept with priority; the rest fill the quota at random."""
    common = set.intersection(*(set(r) for r in by_labeller.values()))
    rng = random.Random(seed)
    any_rec = next(iter(by_labeller.values()))
    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    for pid in sorted(common):
        r = any_rec[pid]
        answers = {by_labeller[m][pid].label.answer for m in by_labeller}
        interesting = len(answers) > 1 or (r.question.family != "general_qa" and bool(answers & {"fail", "partial_pass"}))
        strata[(r.dataset, r.question.family, interesting)].append(pid)
    for v in strata.values():
        rng.shuffle(v)
    # up to half the quota from the interesting strata (round-robin so every dataset x family is
    # represented), then the remainder from all strata round-robin
    def take(keys: list[tuple], quota: int) -> None:
        idx = {k: 0 for k in keys}
        while len(chosen) < quota:
            progressed = False
            for k in keys:
                if idx[k] < len(strata[k]) and len(chosen) < quota and strata[k][idx[k]] not in chosen_set:
                    chosen.append(strata[k][idx[k]])
                    chosen_set.add(strata[k][idx[k]])
                    progressed = True
                idx[k] = idx[k] + 1 if idx[k] < len(strata[k]) else idx[k]
            if not progressed:
                break

    chosen: list[str] = []
    chosen_set: set[str] = set()
    n = min(n, len(common))
    take(sorted(k for k in strata if k[2]), n // 2)
    take(sorted(strata), n)
    return chosen


def grade_pair(pid: str, by_labeller: dict[str, dict[str, LabelledRecord]], judge: str, rng: random.Random) -> dict:
    names = list(by_labeller)
    rng.shuffle(names)
    letters = list(string.ascii_uppercase[: len(names)])
    labels = [(letter, by_labeller[m][pid]) for letter, m in zip(letters, names)]
    rec = labels[0][1]
    resp, cost = ask_json(prompt(rec, labels), judge, schema(letters, rec.question.values))
    grades = {}
    for letter, m in zip(letters, names):
        g = resp["grades"][letter]
        grades[m] = {"answer_correct": bool(g["answer_correct"]), "evidence": int(g["evidence"]), "summary": int(g["summary"]),
                     "tags": (int(g["tags"]) if g.get("tags") is not None and rec.question.tags else None),
                     "reason": str(g.get("reason", ""))[:300], "answer": by_labeller[m][pid].label.answer}
    return {"id": pid, "dataset": rec.dataset, "question_id": rec.question.id, "family": rec.question.family, "judge": judge,
            "judge_answer": resp["judge_answer"], "judge_evidence": sorted({int(x) for x in resp.get("judge_evidence", []) if 1 <= int(x) <= rec.transcript.n_lines}),
            "judge_tags": [t for t in resp.get("judge_tags", []) if t in rec.question.tags],
            "order": dict(zip(letters, names)), "grades": grades, "cost_usd": round(cost, 5),
            "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="one labelled_data file per labeller")
    p.add_argument("--judge", default="claude:opus")
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="output name (default <judge>_<sample>)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    by_labeller = load(args.files)
    pids = sample_pairs(by_labeller, args.sample, args.seed)
    name = args.name or f"{args.judge.split(':')[1]}_{len(pids)}"
    out = JUDGE_DIR / f"{name}.jsonl"
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [x for x in pids if x not in done]
    print(f"{len(by_labeller)} labellers, {len(pids)} sampled pairs, {len(todo)} to grade with {args.judge} -> {out}", flush=True)

    orders = {pid: random.Random(f"{args.seed}:{pid}") for pid in todo}  # per-pair shuffle, reproducible
    total = 0.0
    n = fails = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex, out.open("a") as f:
        futs = {ex.submit(grade_pair, pid, by_labeller, args.judge, orders[pid]): pid for pid in todo}
        for fut in concurrent.futures.as_completed(futs):
            pid = futs[fut]
            try:
                rec = fut.result()
            except (LLMError, ValueError, KeyError, TypeError) as e:
                fails += 1
                print(f"  {pid}: judge failed: {str(e)[:120]}", flush=True)
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            total += rec["cost_usd"]
            n += 1
            ok = " ".join(f"{m[:6]}={'✓' if g['answer_correct'] else '✗'}{g['evidence']}{g['summary']}" for m, g in rec["grades"].items())
            print(f"[{n}/{len(todo)}] {pid} judge={rec['judge_answer']} {ok} cost=${total:.2f}", flush=True)
    print(f"done: {n} graded, {fails} failed, ${total:.2f} -> {out}")


if __name__ == "__main__":
    main()
