"""Export a releasable subset of the benchmark filtered by licence track.

    uv run python -m src.synthesis.export --track p [--out data/benchmark/release-p]

Writes labelled_data.jsonl filtered to records of that track, and a
questions.jsonl derived from those records (the bank plus the questions
written for calls of that track). Track P is built from CC-BY, CC-BY-SA and
public-domain sources and may be published; Track NC contains SPoRC text whose
gate forbids redistribution and must not be.
"""

import argparse
import json
from pathlib import Path

from src.synthesis.question_bank import BENCH, write_questions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", required=True, choices=["p", "nc"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    track = f"track-{args.track}"
    out = Path(args.out) if args.out else BENCH / f"release-{args.track}"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "labelled_data.jsonl").open("w") as f:
        for line in (BENCH / "labelled_data.jsonl").open():
            if json.loads(line)["track"] == track:
                f.write(line)
                n += 1
    n_q = write_questions(out / "labelled_data.jsonl", out / "questions.jsonl")
    print(f"labelled_data: {n} records; questions: {n_q} -> {out}")


if __name__ == "__main__":
    main()
