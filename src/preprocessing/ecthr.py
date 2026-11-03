"""ECtHR (LexGLUE) -> data/labelled_data/ecthr/{train,val,benchmark}.jsonl (Track NC, tier 3; transfer dataset).

Input: data/raw/ecthr/ecthr_{a,b}.{train,validation,test}.jsonl, exported from `coastalcph/lex_glue` (Chalkidis et
al. 2022). Each case is the facts section of a European Court of Human Rights judgment as a list of paragraphs, with
the set of Convention articles the Court found violated (task A) and the set the applicant alleged (task B).

Decisions:
- The set-valued label is decomposed into one binary question per article: ten task-A questions ("did the Court find
  a violation of Article N?", `violated | not_violated`) and ten task-B questions ("was a violation of Article N
  alleged?", `alleged | not_alleged`). This matches the compliance task's shape (one question, one answer from the
  options supplied in the prompt), lets the evaluator's macro-F1 handle the class imbalance per question, and gives a
  natural zero-shot cell: `--held-out` names articles whose questions are dropped from train and val and marked
  `cell = unseen_q` on the benchmark. Task A is the default; `--tasks a,b` adds task B.
- Paragraphs are kept verbatim, one per line, including the dataset's own paragraph numbers at the start of each.
  Cases longer than the training sequence cap are written, not dropped; the training loader applies the cap and the
  datasheet records how many there are.
- Official splits: train -> train, validation -> val, test -> benchmark.

Usage: uv run python -m src.preprocessing.ecthr [--tasks a,b] [--held-out 9,14]
"""

import argparse
import json
from pathlib import Path

from src.preprocessing.common import RAW
from src.preprocessing.transfer import SplitWriter, document, record
from src.synthesis.schema import Option, Question

DATASET = "ecthr"
TRACK = "track-nc"
ORIGIN = "LexGLUE ecthr case facts (HUDOC text), paragraph per line"
# LexGLUE's label order; the titles are the Convention's own section headings.
ARTICLES = [
    ("2", "Article 2 (right to life)"),
    ("3", "Article 3 (prohibition of torture and of inhuman or degrading treatment)"),
    ("5", "Article 5 (right to liberty and security)"),
    ("6", "Article 6 (right to a fair trial)"),
    ("8", "Article 8 (right to respect for private and family life)"),
    ("9", "Article 9 (freedom of thought, conscience and religion)"),
    ("10", "Article 10 (freedom of expression)"),
    ("11", "Article 11 (freedom of assembly and association)"),
    ("14", "Article 14 (prohibition of discrimination)"),
    ("P1-1", "Article 1 of Protocol No. 1 (protection of property)"),
]
DESCRIPTION = ("The text is the facts section of a judgment of the European Court of Human Rights, as the Court set "
               "them out. Decide from the facts alone; the judgment's own legal assessment is not included.")
SPLITS = {"train": "train", "validation": "val", "test": "benchmark"}
TASKS = {
    "a": ("Did the Court find a violation of {title} in this case?",
          [("violated", "The Court found that the respondent State violated {title}."),
           ("not_violated", "The Court found no violation of {title}, or the article was not at issue.")]),
    "b": ("Did the applicant allege a violation of {title} in this case?",
          [("alleged", "The applicant complained of a breach of {title}, whatever the Court then decided."),
           ("not_alleged", "No complaint under {title} was raised.")]),
}


def questions(task: str) -> list[tuple[int, Question]]:
    text, opts = TASKS[task]
    out = []
    for i, (code, title) in enumerate(ARTICLES):
        q = Question(id=f"ecthr-{task}-{code}", source="written", family="transfer", text=text.format(title=title),
                     description=DESCRIPTION, options=[Option(value=v, criteria=c.format(title=title)) for v, c in opts],
                     tags=[], dataset_allow_list=[DATASET])
        out.append((i, q))
    return out


def main(raw: Path = RAW / "ecthr", tasks: tuple[str, ...] = ("a",), held_out: frozenset[str] = frozenset(),
         out_dir: Path | None = None) -> dict:
    writer = SplitWriter(DATASET, out_dir)
    for src_split, target in SPLITS.items():
        for task in tasks:
            positive = TASKS[task][1][0][0]
            negative = TASKS[task][1][1][0]
            with open(raw / f"ecthr_{task}.{src_split}.jsonl") as f:
                for n, line in enumerate(f):
                    case = json.loads(line)
                    doc_id = f"ecthr-{src_split}-{n:05d}"
                    transcript = document(case["text"], ORIGIN)
                    labels = set(case["labels"])
                    for i, q in questions(task):
                        code = ARTICLES[i][0]
                        if code in held_out and target != "benchmark":
                            continue
                        meta = {"task": task, "n_paragraphs": len(case["text"]), "n_labels": len(labels),
                                "label_set": ",".join(ARTICLES[j][0] for j in sorted(labels))}
                        writer.add(record(dataset=DATASET, doc_id=doc_id, source_id=f"lex_glue/ecthr_{task}/{src_split}/{n}",
                                          track=TRACK, question=q, transcript=transcript,
                                          answer=positive if i in labels else negative, meta=meta,
                                          benchmark=target == "benchmark",
                                          cell="unseen_q" if code in held_out else "seen_q"), target)
    return writer.close({"tasks": list(tasks), "held_out": sorted(held_out)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--tasks", default="a", help="comma-separated subset of a,b (default a)")
    p.add_argument("--held-out", default="", help="comma-separated article codes whose questions are held out of training")
    a = p.parse_args()
    main(tasks=tuple(t.strip() for t in a.tasks.split(",") if t.strip()),
         held_out=frozenset(c.strip() for c in a.held_out.split(",") if c.strip()))
