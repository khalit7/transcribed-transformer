"""Analyse the labeller comparison: judge grades (and, when present, human grades) per labeller.

Reads a judgements file written by grade_labels.py and the labeller files it graded, and reports
per labeller, overall and per question family:

  answer     share of graded pairs the judge marked answer_correct
  rare       recall on rare events: among pairs where the judge's own answer is fail or
             partial_pass on a family question (vulnerability, complaint, eod), the share where
             the labeller also answered fail / partial_pass
  na-fail    how often the labeller said NA where the judge said fail, and vice versa
  evidence   mean judge evidence grade (0-2), plus mean Jaccard overlap between the labeller's
             cited lines and the judge's own evidence lines
  summary    mean judge summary grade (0-2)
  tags       mean judge tag grade (0-2) on tagged questions
  cost/time  mean cost per label and labels per minute, from the labeller files

With --human <file>, a human grading file in the same shape as the judgements (id, grades per
labeller with answer_correct / evidence / summary / tags) is compared with the judge: agreement
per criterion, which is what says how far the judge's verdicts can be trusted.

--export-review N writes N judged pairs to data/labelled_data/judgements/review_<name>.md with the
labels anonymised (the letter->labeller map is kept in review_<name>_key.json), for a human to
grade blind; --import-review reads the filled-in file back into a human grading file.

    uv run python -m src.synthesis.analyse_labellers data/labelled_data/judgements/opus_400.jsonl
    uv run python -m src.synthesis.analyse_labellers data/labelled_data/judgements/opus_400.jsonl --export-review 60
    uv run python -m src.synthesis.analyse_labellers data/labelled_data/judgements/opus_400.jsonl --human data/labelled_data/judgements/human_opus_400.jsonl
"""

import argparse
import collections
import datetime as dt
import json
import random
import re
from pathlib import Path

from src.synthesis.grade_labels import JUDGE_DIR, load
from src.synthesis.label import numbered

LABELLER_DIR = JUDGE_DIR.parent / "labellers"
RARE = {"fail", "partial_pass"}


def read_judgements(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def jaccard(a: list[int], b: list[int]) -> float | None:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    return len(sa & sb) / len(sa | sb)


def summarise(judgements: list[dict], by_labeller: dict, labellers: list[str]) -> None:
    fams = sorted({j["family"] for j in judgements})
    print(f"{len(judgements)} judged pairs; families: {', '.join(fams)}\n")
    for fam in ["all", *fams]:
        rows = [j for j in judgements if fam == "all" or j["family"] == fam]
        if not rows:
            continue
        print(f"== {fam} ({len(rows)} pairs)")
        print(f"{'labeller':14s} {'answer':>7s} {'rare':>9s} {'NA→fail':>8s} {'fail→NA':>8s} {'evid':>5s} {'jacc':>5s} {'summ':>5s} {'tags':>5s}")
        for m in labellers:
            g = [(j, j["grades"][m]) for j in rows if m in j["grades"]]
            if not g:
                continue
            acc = sum(x["answer_correct"] for _, x in g) / len(g)
            rare_rows = [(j, x) for j, x in g if j["family"] != "general_qa" and j["judge_answer"] in RARE]
            rare = sum(x["answer"] in RARE for _, x in rare_rows) / len(rare_rows) if rare_rows else None
            na_fail = sum(1 for j, x in g if j["judge_answer"] == "fail" and x["answer"] == "NA")
            fail_na = sum(1 for j, x in g if j["judge_answer"] == "NA" and x["answer"] == "fail")
            ev = sum(x["evidence"] for _, x in g) / len(g)
            jac = [v for v in (jaccard(by_labeller[m][j["id"]].label.evidence, j["judge_evidence"]) for j, _ in g)
                   if v is not None]
            sm = sum(x["summary"] for _, x in g) / len(g)
            tg = [x["tags"] for _, x in g if x["tags"] is not None]
            print(f"{m:14s} {acc:7.2f} {(f'{rare:.2f} ({len(rare_rows)})' if rare is not None else '-'):>9s} {na_fail:8d} {fail_na:8d} "
                  f"{ev:5.2f} {(sum(jac) / len(jac) if jac else 0):5.2f} {sm:5.2f} {(f'{sum(tg) / len(tg):.2f}' if tg else '-'):>5s}")
        print()

    print("cost and speed (from the labeller files, all labels, not only the judged sample):")
    for m in labellers:
        recs = list(by_labeller[m].values())
        cost = sum(r.generation_info.cost_usd for r in recs)
        ts = sorted(dt.datetime.fromisoformat(r.generation_info.timestamp) for r in recs)
        minutes = (ts[-1] - ts[0]).total_seconds() / 60 if len(ts) > 1 else 0
        print(f"  {m:14s} {len(recs):5d} labels  ${cost / len(recs):.4f}/label  {len(recs) / minutes if minutes else 0:6.1f} labels/min  "
              f"empty-evidence {sum(1 for r in recs if not r.label.evidence) / len(recs):.0%}")


def compare_human(judgements: list[dict], human: list[dict], labellers: list[str]) -> None:
    hj = {h["id"]: h for h in human}
    pairs = [(j, hj[j["id"]]) for j in judgements if j["id"] in hj]
    print(f"\n== judge vs human on {len(pairs)} pairs the human graded")
    agree: collections.Counter[str] = collections.Counter()
    n: collections.Counter[str] = collections.Counter()
    for j, h in pairs:
        for m in labellers:
            if m not in j["grades"] or m not in h["grades"]:
                continue
            a, b = j["grades"][m], h["grades"][m]
            for k in ("answer_correct", "evidence", "summary", "tags"):
                if a.get(k) is None or b.get(k) is None:
                    continue
                n[k] += 1
                agree[k] += a[k] == b[k]
                if k != "answer_correct":
                    agree[k + "±1"] += abs(int(a[k]) - int(b[k])) <= 1
                    n[k + "±1"] += 1
    for k in ("answer_correct", "evidence", "evidence±1", "summary", "summary±1", "tags", "tags±1"):
        if n[k]:
            print(f"  {k:14s} exact agreement {agree[k] / n[k]:.2f}  (n={n[k]})")
    # per-labeller answer accuracy under the human, for the same pairs
    print("  answer accuracy by human vs by judge, same pairs:")
    for m in labellers:
        hs = [h["grades"][m]["answer_correct"] for _, h in pairs if m in h["grades"]]
        js = [j["grades"][m]["answer_correct"] for j, _ in pairs if m in j["grades"]]
        if hs:
            print(f"    {m:14s} human {sum(hs) / len(hs):.2f}   judge {sum(js) / len(js):.2f}   (n={len(hs)})")


def export_review(judgements: list[dict], by_labeller: dict, n: int, name: str, seed: int) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(judgements, min(n, len(judgements)))
    intro = ("Grade each label blind (the letters are shuffled per pair; the judge's grades are not shown). "
             "For every label fill `correct` (y/n), `evidence` (0-2), `summary` (0-2) and, on tagged questions, `tags` (0-2), "
             "using the same rubric as the judge: evidence 2 = cited lines support the answer and cover what decides it, 1 = partly, "
             "0 = irrelevant/contradicting/empty when evidence exists; summary 2 = faithful and justifies the answer, 1 = faithful but thin, "
             "0 = unfaithful; tags 2 = matches what is disclosed, 1 = partly, 0 = wrong or missing. Then run --import-review.\n")
    md = [f"# Human review: {name}\n", intro]
    key = {}
    for i, j in enumerate(chosen, 1):
        rec = next(iter(by_labeller.values()))[j["id"]]
        q = rec.question
        letters = sorted(j["order"])
        key[j["id"]] = j["order"]
        md.append(f"\n---\n\n## Pair {i}: `{j['id']}`\n\n**Question ({q.family}):** {q.text}\n")
        if q.description:
            md.append(f"\n*Context:* {q.description}\n")
        md.append("\n**Options:**\n" + "\n".join(f"- `{o.value}`: {o.criteria}" for o in q.options) + "\n")
        if q.tags:
            md.append("\n**Tag vocabulary:** " + "; ".join(q.tags) + "\n")
        md.append("\n<details><summary>Transcript</summary>\n\n```\n" + numbered(rec.transcript.lines("clean")) + "\n```\n\n</details>\n")
        for letter in letters:
            m = j["order"][letter]
            r = by_labeller[m][j["id"]]
            tags = f"; tags {r.label.tags}" if q.tags else ""
            md.append(f"\n**Label {letter}:** answer `{r.label.answer}`; evidence {r.label.evidence}{tags}\n\n> {r.label.summary}\n")
        md.append("\n```yaml\n# your grades for pair " + str(i) + f" ({j['id']})\n")
        for letter in letters:
            md.append(f"{letter}: {{correct: , evidence: , summary: {', tags: ' if q.tags else ''}}}\n")
        md.append("```\n")
    out = JUDGE_DIR / f"review_{name}.md"
    out.write_text("".join(md))
    (JUDGE_DIR / f"review_{name}_key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {len(chosen)} pairs to {out} (key in review_{name}_key.json; do not open it while grading)")


def import_review(name: str) -> None:
    text = (JUDGE_DIR / f"review_{name}.md").read_text()
    key = json.loads((JUDGE_DIR / f"review_{name}_key.json").read_text())
    out = JUDGE_DIR / f"human_{name}.jsonl"
    n = 0
    with out.open("w") as f:
        for block in re.finditer(r"# your grades for pair \d+ \((?P<id>[^)]+)\)\n(?P<body>.*?)```", text, re.DOTALL):
            pid = block.group("id")
            grades = {}
            for line in block.group("body").strip().split("\n"):
                m = re.match(r"([A-Z]): \{(.*)\}", line.strip())
                if not m:
                    continue
                fields = dict(re.findall(r"(\w+):\s*([^,}]*)", m.group(2)))
                if not fields.get("correct", "").strip():
                    continue  # not graded
                g = {"answer_correct": fields["correct"].strip().lower() in ("y", "yes", "true", "1"),
                     "evidence": int(fields["evidence"]) if fields.get("evidence", "").strip() else None,
                     "summary": int(fields["summary"]) if fields.get("summary", "").strip() else None,
                     "tags": int(fields["tags"]) if fields.get("tags", "").strip() else None}
                grades[key[pid][m.group(1)]] = g
            if grades:
                f.write(json.dumps({"id": pid, "grades": grades, "grader": "human"}) + "\n")
                n += 1
    print(f"imported {n} human-graded pairs -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("judgements", type=Path)
    p.add_argument("--labellers", nargs="*", type=Path, default=None, help="labeller files (default: all in data/labelled_data/labellers)")
    p.add_argument("--human", type=Path, default=None)
    p.add_argument("--export-review", type=int, default=None)
    p.add_argument("--import-review", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    name = args.judgements.stem
    if args.import_review:
        import_review(name)
        return
    judgements = read_judgements(args.judgements)
    files = args.labellers or sorted(LABELLER_DIR.glob("*.jsonl"))
    by_labeller = load(files)
    labellers = sorted({m for j in judgements for m in j["grades"]} & set(by_labeller))
    if args.export_review:
        export_review(judgements, by_labeller, args.export_review, name, args.seed)
        return
    summarise(judgements, by_labeller, labellers)
    if args.human:
        compare_human(judgements, read_judgements(args.human), labellers)


if __name__ == "__main__":
    main()
