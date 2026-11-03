"""Draw pilot episodes: a fixed number per stratum, spread over categories, from the v2 pool.

    uv run python -m src.synthesis_v2.pilot [--per-stratum 40] [--seed 0]
    uv run python -m src.synthesis_v2.pilot --within data/labelled_data_v2/selected_ids.txt --per-stratum 30 \
        --seed 4 --name pilot4      # round four (D79): from the selected episodes, the four strata

Writes data/labelled_data_v2/<name>_ids.txt (one doc_id per line) and <name>_summary.json (default name `pilot`, the
first two pilots' draw). The draw is seeded and depends only on the pool file (and the `--within` list); the file is
frozen once written (--force to redraw, which makes a different pilot).
Category spread: within a stratum, episodes are taken round-robin over categories in descending pool frequency, one
per podcast, so no category or show dominates a stratum.
"""

import argparse
import collections
import json
import random
from pathlib import Path

from src.synthesis_v2 import OUT_DIR
from src.synthesis_v2.episodes import POOL, STRATA

IN_CAP = list(STRATA)


def draw(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen: list[dict] = []
    pods: set[str] = set()
    for stratum in IN_CAP:
        pool = [r for r in rows if r["stratum"] == stratum]
        by_cat: dict[str, list[dict]] = collections.defaultdict(list)
        for r in pool:
            by_cat[r["category"] or "uncategorised"].append(r)
        for cat in by_cat.values():
            rng.shuffle(cat)
        order = sorted(by_cat, key=lambda c: -len(by_cat[c]))
        taken = 0
        while taken < per_stratum and any(by_cat.values()):
            for cat in order:
                while by_cat[cat]:
                    r = by_cat[cat].pop()
                    if r["podcast_id"] in pods:
                        continue
                    pods.add(r["podcast_id"])
                    chosen.append(r)
                    taken += 1
                    break
                if taken >= per_stratum:
                    break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--per-stratum", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--within", type=Path, default=None, help="draw only from these doc_ids (one per line)")
    ap.add_argument("--name", default="pilot", help="output stem: <name>_ids.txt and <name>_summary.json")
    a = ap.parse_args()
    ids_out, summary_out = OUT_DIR / f"{a.name}_ids.txt", OUT_DIR / f"{a.name}_summary.json"
    if ids_out.exists() and not a.force:
        raise SystemExit(f"{ids_out} exists; the pilot draw is frozen (--force to redraw)")
    rows = [json.loads(line) for line in POOL.open()]
    if a.within is not None:
        keep = {x.strip() for x in a.within.read_text().split() if x.strip()}
        rows = [r for r in rows if r["doc_id"] in keep]
    chosen = draw(rows, a.per_stratum, a.seed)
    ids_out.write_text("".join(r["doc_id"] + "\n" for r in chosen))
    summary = {"seed": a.seed, "per_stratum": a.per_stratum, "within": str(a.within) if a.within else None, "episodes": len(chosen),
               "podcasts": len({r["podcast_id"] for r in chosen}),
               "by_stratum": dict(collections.Counter(r["stratum"] for r in chosen)),
               "by_category": dict(collections.Counter(r["category"] or "uncategorised" for r in chosen).most_common()),
               "tokens_p50_by_stratum": {s: sorted(r["tokens"] for r in chosen if r["stratum"] == s)[len([r for r in chosen if r["stratum"] == s]) // 2]
                                         for s in IN_CAP}}
    summary_out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
