"""Audit every (question, dataset) pair: does the question make sense there, and do answers vary?

Reads the per-dataset probe files written by probe_bank (--family all, --save), for one or
several labellers, and prints for every question and dataset the answer distribution with
two flags:

  NA-dominant   NA share >= --na-threshold (default 0.75): the question has no occasion
                to arise on this dataset -> should not be in its allow list
  skewed        one answer >= --skew-threshold (default 0.90) of calls: little variety.
                Skew is NOT by itself a reason to drop (Khalid, 2026-09-03): a question
                whose rare answer is important to catch (profanity, a vulnerability
                disclosure, a complaint, uncaveated advice, undisclosed promotion) is kept
                as a rare-event detection test; only trivially skewed questions go
                (the common answer is uninformative and the rare one does not matter).
                Rare-positive calls then need oversampling when a benchmark is assembled.

With several labellers a cell is NA-dominant / skewed when a MAJORITY of labellers say so, and
cells where labellers disagree are marked `?` for a human read. It proposes an allow list per
question = datasets that are not NA-dominant, and diffs against the current bank so
changes are explicit and evidence-backed rather than applied blindly. Since synth_data
asks every allowed question of every call, the allow list is a hard gate on which labels
exist, which is why the decision is taken on agreement rather than on one model. (Labellers answer; no LLM-as-a-judge step is involved here.)

    uv run python -m src.synthesis.audit_bank data/labelled_data/probes/audit_*_seed3.json
    uv run python -m src.synthesis.audit_bank data/labelled_data/probes/audit_*_sonnet_seed3.json   # one labeller
"""

import argparse
import collections
import json
import re
from pathlib import Path

from src.synthesis.cases import BUILDERS
from src.synthesis.question_bank import QUESTIONS

VALUES = ["pass", "partial_pass", "fail", "NA"]
DATASETS = list(BUILDERS)
NAME = re.compile(rf"audit_({'|'.join(DATASETS)})_(.+?)_seed\d+\.json$")


def flags(dist: dict, na_thr: float, skew_thr: float) -> tuple[bool, bool, int]:
    """(NA-dominant, skewed, n) for one labeller's distribution on one cell."""
    n = sum(dist.get(v, 0) for v in VALUES)
    if not n:
        return False, False, 0
    na = dist.get("NA", 0) / n
    top = max(dist.get(v, 0) for v in VALUES) / n
    return na >= na_thr, (na < na_thr and top >= skew_thr), n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--na-threshold", type=float, default=0.75)
    p.add_argument("--skew-threshold", type=float, default=0.90)
    args = p.parse_args()

    # probes[dataset][labeller] = {qid: {answer: count}}
    probes: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for f in args.files:
        m = NAME.search(Path(f).name)
        if not m:
            raise SystemExit(f"cannot parse dataset/labeller from {f}")
        probes[m.group(1)][m.group(2)] = json.load(open(f))  # noqa: SIM115
    datasets = [d for d in DATASETS if d in probes]
    labellers = sorted({j for d in datasets for j in probes[d]})
    print(f"labellers: {', '.join(labellers)}   (majority of {len(labellers)} decides; ? = labellers disagree)\n")

    width = 9 * len(labellers) + 6
    print(f"{'question':36s} " + "  ".join(f"{d:>{width}s}" for d in datasets))
    proposal: dict[str, list[str]] = {}
    disputed: list[str] = []
    for q in QUESTIONS:
        cells = []
        allowed = []
        for d in datasets:
            per_labeller = {j: flags(probes[d][j].get(q.id, {}), args.na_threshold, args.skew_threshold) for j in probes[d]}
            if not any(n for _, _, n in per_labeller.values()):
                cells.append(f"{'-':>{width}s}")
                continue
            votes_na = sum(na for na, _, _ in per_labeller.values())
            votes_skew = sum(sk for _, sk, _ in per_labeller.values())
            k = len(per_labeller)
            na_dom = votes_na * 2 > k
            skewed = votes_skew * 2 > k
            unanimous = votes_na in (0, k) and votes_skew in (0, k)
            # one summary per labeller: pass/partial/fail/NA compressed to NA share and top answer
            parts = []
            for j in labellers:
                dist = probes[d].get(j, {}).get(q.id)
                if not dist:
                    parts.append(f"{'-':>8s}")
                    continue
                n = sum(dist.get(v, 0) for v in VALUES)
                top = max(VALUES, key=lambda v: dist.get(v, 0))
                parts.append(f"{top[0].upper()}{round(100 * dist.get(top, 0) / n):>3d}n{round(100 * dist.get('NA', 0) / n):>3d}")
            flag = "NA!" if na_dom else ("skew" if skewed else "")
            if not unanimous:
                flag += "?"
            cells.append((" ".join(parts) + f" {flag:>5s}").rjust(width))
            if not unanimous:
                disputed.append(f"{q.id}@{d}")
            if not na_dom:
                allowed.append(d)  # skewed cells stay allowed; they are flagged for the human call
        proposal[q.id] = allowed
        cur = set(q.dataset_allow_list) & set(datasets)  # only diff over datasets that were probed
        add = sorted(set(allowed) - cur)
        drop = sorted(cur - set(allowed))
        change = (f"  +{','.join(add)}" if add else "") + (f"  -{','.join(drop)}" if drop else "")
        print(f"{q.id:36s} " + "  ".join(cells) + change)

    print("\nper labeller: <top answer initial><its share %>n<NA share %>; NA! = NA-dominant by majority, "
          "skew = one answer >= threshold by majority, ? = labellers disagree on a flag; "
          "+/- = proposed additions/removals vs the current allow list")
    print(f"disputed cells ({len(disputed)}): {', '.join(disputed)}")


if __name__ == "__main__":
    main()
