"""Re-stratify the v2 pool on the decided rendering and choose about 1,500 conversations per stratum (D75, D76, D77).

    uv run python -m src.synthesis_v2.strata measure [--seed 11] [--sample 3000]   # size the pool and a population sample
    uv run python -m src.synthesis_v2.strata draw [--per-stratum 1500]             # cell targets; new episodes for the gaps
    uv run python -m src.synthesis_v2.sporc_meta                                   # metadata for the appended episodes
    uv run python -m src.synthesis_v2.identify_hosts --ids data/labelled_data_v2/strata/draw_ids.txt --claude-account w
    uv run python -m src.synthesis_v2.strata select [--per-stratum 1500]           # the final selection

Strata (owner's decision 2026-11-01): `0-4k`, `4-8k`, `8-12k`, `12-16k`, by the Gemma 3 token count of the transcript
rendered `<n>. <SPEAKER_NN>:<utterance>`, every corpus turn verbatim (`episodes.render`). The training rendering puts
`host_N` / `guest_N` in place of the tags, which is never longer (checked per selected episode by `select`), so a size
measured here bounds the transcript the arms see.

Ceiling. Training drops an example whose prompt, target and end token exceed the sequence length (16,384), so the top
stratum ends at `ceiling()`: the sequence length less the end token, one token of tokenisation slack (a full prompt
measures frame + transcript + 1), the longest prompt frame around the transcript (the podcast task prompt for every
bank question, approved or provisional, with eight speaker labels) and a target allowance of TARGET_ALLOWANCE tokens. The allowance is a bound, not a measurement: the
longest pilot target was 278 tokens (Sonnet; Opus 190) on the merged rendering, with at most 34 cited lines numbered
under 200; on the unmerged rendering a top-stratum episode has 600 to 1,600 lines and Gemma spends a token per digit, so
70 cited four-digit lines cost about 420 tokens before the answer and summary. How many labels exceed it is measured on
the production labels. The widest frame (874 tokens) is a question that takes no argument, so removing the argument
templates (D78) left it unchanged; a provisional question with a longer frame would lower the ceiling. The bound is for the Gemma tokenizer, the arms' own.

Representing a stratum (owner's request: a stratum should cover its band, not sit at its lower edge as the old `4-8k`
did, every one of whose episodes came from the `2-4k` candidates and so had at most 3,333 words). `measure` draws a
seeded random sample of the eligible SPoRC episodes (the pool's filters: 2 to 4 speakers, at least 40 turns, no v1
episode or podcast, a known duration at 90 words per minute or more, no monologue, repetition loop or collapsed
transcript) whose word-count estimate puts them near each stratum, and measures them exactly. Each stratum's sample is
taken from its own shuffled window only (an episode measured for another window is reused when this window reaches it in
its own order, never counted before), so the in-band sample is a uniform random sample of the stratum's population; the
windows hold over 99.9% of each band (review round 15). Each stratum is cut into cells: 1,000-token sub-bins (a last
bin under 500 tokens wide joins the one before) crossed with quintiles of tokens per word (a dense, fragment-heavy
transcript has more tokens per word; the old `4-8k` was 1.70 at the median against the population's 1.46). A stratum of
N episodes gets N times each cell's population share, by largest remainder.

Reuse. A cell is filled first from pool episodes the identifier has already accepted (a conversation with a host),
chosen at random within the cell; only the remainder is drawn new, from the population sample (never a pool episode or a
podcast already holding PER_PODCAST pool episodes; only episodes sampled from the stratum's own window), with a
DRAW_MARGIN for the episodes the identifier will reject. Drawn episodes still awaiting identification count as filling
their cells, so a second `draw` does not draw them again. The drawn episodes are appended to the pool (`draw` and `seed`
fields) and `strata/draw_ids.txt` lists every drawn episode not yet identified. `select`
then fills every cell, reused episodes first and new ones in their drawn order, and writes `strata/selection.jsonl` and
`selected_ids.txt`; a cell still short is reported (run `draw` again with a new seed). A cell no draw can fill (its
population sample exhausted or podcast-capped) then takes spare accepted episodes from the nearest cell of the same
stratum, length sub-bin weighted twice the density quintile; such rows keep their own `cell` and carry `fills`.

The pool's previous token counts and strata (the D66 rendering `n: SPEAKER_NN: text`, strata 2-4k / 4-8k / 8-16k /
16-32k) stay on each row as `tokens_d66` and `stratum_d66`, and the pool as it stood is kept as
`episodes.three_strata.2026-11-01.jsonl`.
"""

import argparse
import bisect
import collections
import json
import math
import random
import shutil
from collections.abc import Callable
from pathlib import Path

from src.synthesis_v2 import OUT_DIR, identify_hosts
from src.synthesis_v2.episodes import (
    INTERIM,
    PER_PODCAST,
    POOL,
    STRATA,
    WINDOW,
    gemma_counter,
    read_text,
    render,
    scan,
    stratum_of,
    text_checks,
    turns_of,
    v1_footprint,
)

DIR = OUT_DIR / "strata"
POPULATION = DIR / "population.jsonl"
PLAN = DIR / "plan.json"
DRAW_IDS = DIR / "draw_ids.txt"
SELECTION = DIR / "selection.jsonl"
SELECTED_IDS = OUT_DIR / "selected_ids.txt"
BACKUP = OUT_DIR / "episodes.three_strata.2026-11-01.jsonl"
MAX_SEQ_LEN = 16_384  # the Gemma-family configs' max_seq_len
TARGET_ALLOWANCE = 700  # tokens kept for the answer; a bound, see the module docstring
SPEAKER_SLOTS = 8  # speaker labels assumed in the prompt's speaker list when sizing the frame
SUB_BIN = 1_000
RATIO_BINS = 5  # tokens-per-word quantile bins, cut once per stratum and crossed with its sub-bins
DRAW_MARGIN = 0.06  # extra new episodes drawn per cell for the identifier's rejections (1.8% measured on the pool)

Cell = tuple[int, int]


def ceiling() -> dict:
    """The largest transcript size (in the sizing rendering) whose longest prompt still fits with the target allowance."""
    from transformers import AutoTokenizer

    from src.synthesis_v2 import question_bank as bank
    from src.train.data import SEP, task_prompt

    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    speakers = [f"host_{i}" for i in range(1, 3)] + [f"guest_{i}" for i in range(1, 3)] + \
               [f"SPEAKER_{i:02d}" for i in range(SPEAKER_SLOTS - 4)]
    frame, worst = 0, ""
    for q in bank.BANK + bank.PROVISIONAL:
        n = len(tok(task_prompt([], speakers, q) + SEP)["input_ids"])  # with the start token, as training tokenises
        if n > frame:
            frame, worst = n, q.text
    top = MAX_SEQ_LEN - 1 - 1 - frame - TARGET_ALLOWANCE  # end token, tokenisation slack
    return {"ceiling": top, "frame_tokens": frame, "frame_question": worst,
            "target_allowance": TARGET_ALLOWANCE, "max_seq_len": MAX_SEQ_LEN}


def size(row: dict, count_tokens: Callable[[str], int]) -> dict:
    """A pool or candidate row measured on the sizing rendering, with the text checks' verdict (`reject`)."""
    turns = turns_of(read_text(INTERIM, row["offset"]))
    reason, stats = text_checks(turns)
    tokens = count_tokens(render(turns))
    return {**row, "tokens": tokens, "stratum": stratum_of(tokens), "reject": reason,
            "tags_in_text": sorted({r for r, _ in turns}), **stats}


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write through a temporary file, so a crash never leaves a short file in place."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def resize_pool(count_tokens: Callable[[str], int], top: int) -> list[dict]:
    """Every pool row re-measured on the sizing rendering; the D66 values kept as `tokens_d66` / `stratum_d66`. A row
    above the ceiling gets `stratum: None` (old 16-32k rows shrink to about 15.7k on this rendering, inside the nominal
    12-16k band but out of reach of training)."""
    rows = [json.loads(line) for line in POOL.open()]
    if not BACKUP.exists():
        tmp = BACKUP.with_suffix(".jsonl.tmp")
        shutil.copyfile(POOL, tmp)
        tmp.replace(BACKUP)
    out = []
    for r in rows:
        old = {"tokens_d66": r.get("tokens_d66", r["tokens"]), "stratum_d66": r.get("stratum_d66", r["stratum"])}
        new = size(r, count_tokens)
        if new["tokens"] > top:
            new["stratum"] = None
        out.append({**new, **old})
    write_jsonl(POOL, out)
    return out


def sample_population(rows: list[dict], seed: int, per_stratum: int, measure: Callable[[dict], dict],
                      top: int) -> list[dict]:
    """Walk each stratum's window in its own seeded shuffle, measuring (or reusing a measurement of) each episode, until
    `per_stratum` episodes in the band pass the text checks or the window runs out. Each measurement records the
    windows that reached it (`windows`), and only those count for a stratum, so its in-band sample stays uniform."""
    rng = random.Random(seed)
    measured: dict[str, dict] = {}
    for s, (lo, hi) in STRATA.items():
        hi = min(hi, top + 1)
        window = [r for r in rows if WINDOW[0] * lo <= r["est_tokens"] < WINDOW[1] * hi]
        rng.shuffle(window)
        have = 0
        for r in window:
            if have >= per_stratum:
                break
            m = measured.get(r["doc_id"])
            if m is None:
                m = measured[r["doc_id"]] = {**measure(r), "windows": []}
            m["windows"].append(s)
            if m["reject"] is None and lo <= m["tokens"] < hi:
                have += 1
        print(f"{s}: {have} in-band episodes in the population sample", flush=True)
    return list(measured.values())


def measure_population(seed: int, per_stratum: int, count_tokens: Callable[[str], int], top: int) -> list[dict]:
    docs, pods = v1_footprint()
    rows, _funnel = scan(exclude_docs=docs, exclude_pods=pods)
    return sample_population(rows, seed, per_stratum, lambda r: size(r, count_tokens), top)


def cell_edges(stratum: str, top: int) -> list[int]:
    lo, hi = STRATA[stratum]
    hi = min(hi, top + 1)
    edges = list(range(lo, hi, SUB_BIN)) + [hi]
    if len(edges) > 2 and edges[-1] - edges[-2] < SUB_BIN // 2:  # a sliver at the ceiling joins the bin before
        del edges[-2]
    return edges


def cell_of(row: dict, edges: list[int], cuts: list[float]) -> Cell | None:
    t = row["tokens"]
    if not edges[0] <= t < edges[-1]:
        return None
    b = bisect.bisect_right(edges, t) - 1
    return b, bisect.bisect_right(cuts, t / max(1, row["n_words"]))


def in_sample(m: dict, stratum: str, edges: list[int]) -> bool:
    """A population measurement that counts for this stratum: reached from its own window, passing, in the band."""
    return stratum in m["windows"] and m["reject"] is None and edges[0] <= m["tokens"] < edges[-1]


def allocate(n: int, shares: dict[Cell, int]) -> dict[Cell, int]:
    """n split over cells in proportion to their counts, by largest remainder (ties by cell order)."""
    total = sum(shares.values())
    raw = {c: n * v / total for c, v in shares.items()}
    out = {c: math.floor(x) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - out[c]), c))[: n - sum(out.values())]:
        out[c] += 1
    return out


def design(population: list[dict], top: int) -> dict:
    """Per stratum: the cell edges, the tokens-per-word quantile cuts and the population count per cell."""
    out = {}
    for s in STRATA:
        edges = cell_edges(s, top)
        inband = [m for m in population if in_sample(m, s, edges)]
        ratios = sorted(m["tokens"] / max(1, m["n_words"]) for m in inband)
        cuts = [ratios[len(ratios) * i // RATIO_BINS] for i in range(1, RATIO_BINS)] if ratios else []
        counts = collections.Counter(cell_of(m, edges, cuts) for m in inband)
        out[s] = {"edges": edges, "ratio_cuts": cuts, "population": len(inband),
                  "cells": {f"{b},{k}": counts.get((b, k), 0) for b in range(len(edges) - 1) for k in range(RATIO_BINS)}}
    return out


def buildable(cache: dict[str, dict], doc_id: str) -> bool | None:
    """True if the identifier accepted the episode, False if it rejected it, None if it has not identified it."""
    rec = cache.get(doc_id)
    if rec is None or not identify_hosts.current(rec):
        return None
    return rec["kind"] == "conversation" and bool(rec["hosts"])


def usable(r: dict, s: str, top: int) -> bool:
    return r["stratum"] == s and r["tokens"] <= top and r.get("reject") is None


def pending(pool: list[dict], cache: dict[str, dict]) -> list[dict]:
    """Drawn episodes the identifier has not yet seen."""
    return [r for r in pool if "draw" in r and buildable(cache, r["doc_id"]) is None]


def _cells(d: dict) -> dict[Cell, int]:
    out: dict[Cell, int] = {}
    for c, v in d["cells"].items():
        b, k = c.split(",")
        out[(int(b), int(k))] = v
    return out


def targets(plan: dict, n: int) -> dict[str, dict[Cell, int]]:
    return {s: allocate(n, {c: v for c, v in _cells(plan["design"][s]).items() if v > 0}) for s in STRATA}


def draw(plan: dict, pool: list[dict], population: list[dict], cache: dict[str, dict], n: int, seed: int) -> list[dict]:
    """New episodes for every cell the pool's accepted (or pending) episodes cannot fill; returns rows to append."""
    if seed in {d["seed"] for d in plan.get("draws", [])}:
        raise SystemExit(f"seed {seed} was already used for a draw; a further draw needs a new seed")
    rng = random.Random(seed)
    top = plan["ceiling"]["ceiling"]
    in_pool = {r["doc_id"] for r in pool}
    # the podcast cap counts every pool row, usable or not: stricter than `select` needs, never looser
    per_pod = collections.Counter(r["podcast_id"] for r in pool)
    waiting = {r["doc_id"] for r in pending(pool, cache)}
    new: list[dict] = []
    report = {}
    for s, tgt in targets(plan, n).items():
        d = plan["design"][s]
        edges, cuts = d["edges"], d["ratio_cuts"]
        accepted = collections.Counter(cell_of(r, edges, cuts) for r in pool
                                       if usable(r, s, top) and buildable(cache, r["doc_id"]))
        reusable = collections.Counter(cell_of(r, edges, cuts) for r in pool  # identified before any draw
                                       if usable(r, s, top) and buildable(cache, r["doc_id"]) and "draw" not in r)
        waits = collections.Counter(cell_of(r, edges, cuts) for r in pool if usable(r, s, top) and r["doc_id"] in waiting)
        cands = [m for m in population if in_sample(m, s, edges) and m["doc_id"] not in in_pool]
        rng.shuffle(cands)
        short = {}
        for c, t in sorted(tgt.items()):
            need = max(0, t - accepted.get(c, 0) - waits.get(c, 0))
            want = math.ceil(need * (1 + DRAW_MARGIN)) if need else 0
            got = 0
            for m in cands:
                if got >= want:
                    break
                if cell_of(m, edges, cuts) != c or m["doc_id"] in in_pool or per_pod[m["podcast_id"]] >= PER_PODCAST:
                    continue
                in_pool.add(m["doc_id"])
                per_pod[m["podcast_id"]] += 1
                new.append({k: v for k, v in m.items() if k not in ("reject", "windows")}
                           | {"reject": None, "draw": f"strata-{seed}", "seed": seed})
                got += 1
            if got < want:
                short[f"{c[0]},{c[1]}"] = want - got
        report[s] = {"target": n, "reused_accepted": sum(min(t, reusable.get(c, 0)) for c, t in tgt.items()),
                     "pending_before": sum(waits.values()), "drawn": sum(1 for r in new if r["stratum"] == s),
                     "short": short}
    plan.setdefault("draws", []).append({"seed": seed, "per_stratum": n, "report": report})
    return new


def select(plan: dict, pool: list[dict], cache: dict[str, dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Fill every cell: accepted pool episodes from before the draws first (random order), then drawn ones in draw order;
    then fill each cell still short from the nearest cell's spares (sub-bin distance counts double)."""
    rng = random.Random(seed)
    top = plan["ceiling"]["ceiling"]
    chosen: list[dict] = []
    report = {}
    per_pod: collections.Counter = collections.Counter()
    order: dict[str, list[dict]] = {}
    shorts: dict[str, dict[Cell, int]] = {}
    for s, tgt in targets(plan, n).items():
        d = plan["design"][s]
        edges, cuts = d["edges"], d["ratio_cuts"]
        ok = [r for r in pool if usable(r, s, top) and buildable(cache, r["doc_id"])]
        old = [r for r in ok if "draw" not in r]
        rng.shuffle(old)
        drawn = [r for r in ok if "draw" in r]  # pool order is draw order
        order[s] = old + drawn
        shorts[s] = {}
        for c, t in sorted(tgt.items()):
            got = 0
            for r in order[s]:
                if got >= t:
                    break
                if cell_of(r, edges, cuts) != c or per_pod[r["podcast_id"]] >= PER_PODCAST:
                    continue
                per_pod[r["podcast_id"]] += 1
                chosen.append({**r, "cell": f"{c[0]},{c[1]}", "reused": "draw" not in r})
                got += 1
            if got < t:
                shorts[s][c] = t - got
    # A cell the pool cannot fill (its population is exhausted or podcast-capped) takes a spare episode from the
    # nearest cell of the same stratum; run after every stratum's own fill so no other selection changes.
    for s, missing in shorts.items():
        d = plan["design"][s]
        edges, cuts = d["edges"], d["ratio_cuts"]
        taken = {r["doc_id"] for r in chosen}
        spare = [(i, r, cell_of(r, edges, cuts)) for i, r in enumerate(order[s]) if r["doc_id"] not in taken]
        for c, k in sorted(missing.items()):
            spare.sort(key=lambda x: (2 * abs(x[2][0] - c[0]) + abs(x[2][1] - c[1]), x[0]))  # keep length first
            for i, r, rc in list(spare):
                if k == 0:
                    break
                if per_pod[r["podcast_id"]] >= PER_PODCAST:
                    continue
                per_pod[r["podcast_id"]] += 1
                chosen.append({**r, "cell": f"{rc[0]},{rc[1]}", "reused": "draw" not in r, "fills": f"{c[0]},{c[1]}"})
                spare.remove((i, r, rc))
                k -= 1
            missing[c] = k
    for s, missing in shorts.items():
        mine = [r for r in chosen if r["stratum"] == s]
        report[s] = {"selected": len(mine), "reused": sum(r["reused"] for r in mine),
                     "filled_from_nearest_cell": sum("fills" in r for r in mine),
                     "short": {f"{c[0]},{c[1]}": k for c, k in missing.items() if k}}
    return chosen, report


def train_sizes(chosen: list[dict], cache: dict[str, dict], count_tokens: Callable[[str], int]) -> list[str]:
    """Measure each chosen episode as the arms will see its transcript (`cases.build`, `numbered`) into `train_tokens`;
    return the ids where that exceeds the sizing count, which would void the ceiling."""
    from src.synthesis_v2 import numbered
    from src.synthesis_v2.cases import build

    over = []
    for r in chosen:
        rec = cache[r["doc_id"]]
        case = build(r["doc_id"], read_text(INTERIM, r["offset"]), rec["hosts"], {}, rec.get("guests", []))
        r["train_tokens"] = count_tokens(numbered(case.transcript.lines("messy")))
        if r["train_tokens"] > r["tokens"]:
            over.append(r["doc_id"])
    return over


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cmd", choices=["measure", "draw", "select"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sample", type=int, default=3000, help="in-band population episodes to measure per stratum")
    ap.add_argument("--per-stratum", type=int, default=1500)
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    count = gemma_counter()
    if a.cmd == "measure":
        if POPULATION.exists():
            raise SystemExit(f"{POPULATION} exists; the population sample is measured once")
        ceil = ceiling()
        print(json.dumps(ceil), flush=True)
        pool = resize_pool(count, ceil["ceiling"])
        print(f"pool re-sized: {dict(collections.Counter(r['stratum'] for r in pool))}", flush=True)
        population = measure_population(a.seed, a.sample, count, ceil["ceiling"])
        plan = {"seed": a.seed, "ceiling": ceil, "design": design(population, ceil["ceiling"])}
        write_text(PLAN, json.dumps(plan, indent=1))
        write_jsonl(POPULATION, population)  # last: its presence marks a finished measure
        print(json.dumps({s: {k: v for k, v in d.items() if k != "cells"} for s, d in plan["design"].items()}))
        return
    plan = json.loads(PLAN.read_text())
    pool = [json.loads(line) for line in POOL.open()]
    cache = identify_hosts.load_cache(identify_hosts.CACHE)
    if a.cmd == "draw":
        population = [json.loads(line) for line in POPULATION.open()]
        new = draw(plan, pool, population, cache, a.per_stratum, a.seed)
        write_jsonl(POOL, pool + new)
        DRAW_IDS.write_text("".join(r["doc_id"] + "\n" for r in pending(pool + new, cache)))
        write_text(PLAN, json.dumps(plan, indent=1))
        print(json.dumps(plan["draws"][-1]))
        return
    chosen, report = select(plan, pool, cache, a.per_stratum, a.seed)
    over = train_sizes(chosen, cache, count)
    if over:
        raise SystemExit(f"{len(over)} selected episodes are longer in the training rendering than in the sizing one: "
                         f"{over[:5]}")
    write_jsonl(SELECTION, chosen)
    SELECTED_IDS.write_text("".join(r["doc_id"] + "\n" for r in chosen))
    plan["selection"] = {"seed": a.seed, "per_stratum": a.per_stratum, "report": report}
    write_text(PLAN, json.dumps(plan, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
