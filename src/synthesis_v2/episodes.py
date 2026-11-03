"""The v2 episode pool: SPoRC episodes filtered and stratified by rendered length.

    uv run python -m src.synthesis_v2.episodes [--per-stratum 2000 2000 2000 1000] [--seed 0]
    uv run python -m src.synthesis_v2.episodes --top-up 60 43 17 --seed 1    # add episodes to the in-cap strata

Reads the conversational SPoRC subset (`data/interim/sporc/train.jsonl`, one diarised episode per line) and writes
`data/labelled_data_v2/episodes.jsonl`: one line per chosen episode with its metadata, its byte offset in the interim
file (the text is not copied; the case builder seeks to it), its exact rendered length in Gemma 3 tokens and the
length stratum it falls in, plus `episodes_summary.json` with the eligibility funnel and per-stratum statistics.

Filters (decided 2026-10-30, D66):
- 2 to 4 speakers by the corpus's own `n_speakers`, and at least two distinct diarisation tags in the text;
- at least 40 turns, so a monologue with a jingle does not pass;
- not an episode, and not from a podcast, that the v1 compliance labelling touched (`data/labelled_data/
  labelled_data.jsonl`), so the two iterations share no transcript and no show;
- at most two episodes per podcast in the pool, so the show-level split has many shows to work with;
- (added after the analysts' reading of the first pool, 2026-10-30) no partial transcript: at least 90 words per minute
  of audio where the duration is known, and a known duration; no monologue that passed the speaker count: the
  second-largest speaker holds at least 10% of the words; no Whisper repetition loop: at most 20% of lines repeat the
  previous line; no collapsed transcript: the longest line holds at most 25% of the words.

Strata (owner's decision 2026-11-01, D75/D76, replacing the D66 strata 2-4k / 4-8k / 8-16k / 16-32k): `0-4k`, `4-8k`,
`8-12k`, `12-16k`, by the exact length in Gemma 3 tokens (the arms' own tokenizer) of the transcript rendered
`<n>. <SPEAKER_NN>:<utterance>`, one corpus turn per line, verbatim (`render`). The arms see the same lines with
`host_N` / `guest_N` in place of the identified tags, which is never longer. The top stratum ends below 16k at the
ceiling `strata.ceiling()` computes from the real prompt. A word count times 1.5 pre-selects candidates (measured on v1
SPoRC prompts); the exact count decides. Sampling is seeded and depends only on the interim file and the seed.

The pool was first built by `choose` under the D66 strata (seed 0, 7,000 episodes) and topped up once (seed 1, 120
episodes); it was then re-stratified under the strata above, and the per-stratum selection made, by `strata.py`, which
reuses every identified episode it can and draws new ones only where a stratum's length cells need them. `choose` now
keeps a measured candidate only in the stratum whose candidate list it was drawn from (until 2026-11-01 it kept it in
any stratum with room, so the `2-4k` draw filled the old `4-8k` stratum with short, fragment-dense episodes).

Top-up: `--top-up N N N N` draws more episodes for the strata under the same filters and the same podcast cap, counted
across the existing pool, from episodes and podcasts not already in it, with a new seed, and appends them to the pool
marked `topup: <seed>`; `episodes_summary.json` gains a `topups` list and the new ids are written to
`topup_<seed>_ids.txt`. Order of the steps after a top-up: `sporc_meta` (rebuilds episode_meta.jsonl for every pool id,
so the metadata-first identification rule applies to the new episodes too), then `identify_hosts --ids` on the new ids.
"""

import argparse
import collections
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path

from src.synthesis_v2 import OUT_DIR, REPO_ROOT

INTERIM = REPO_ROOT / "data" / "interim" / "sporc" / "train.jsonl"
V1_LEDGER = REPO_ROOT / "data" / "labelled_data" / "labelled_data.jsonl"
POOL = OUT_DIR / "episodes.jsonl"
SUMMARY = OUT_DIR / "episodes_summary.json"
TOKENIZER = "google/gemma-3-270m"

STRATA: dict[str, tuple[int, int]] = {"0-4k": (0, 4_000), "4-8k": (4_000, 8_000), "8-12k": (8_000, 12_000),
                                     "12-16k": (12_000, 16_000)}
WINDOW = (0.4, 1.4)  # a stratum's candidates: word estimate in [0.4 lo, 1.4 hi), which holds over 99% of its episodes
TOKENS_PER_WORD = 1.5  # the pre-selection estimate; the exact count decides the stratum
MIN_TURNS = 40
SPEAKERS = (2, 3, 4)
PER_PODCAST = 2
MIN_WPM = 90.0  # below this the transcript covers only part of the audio
MIN_SECOND_SHARE = 0.10  # words of the second-largest speaker, as a share of all words
MAX_DUPLICATE_LINES = 0.20
MAX_LONGEST_LINE_SHARE = 0.25


def render(turns: list[tuple[str, str]]) -> str:
    """The transcript the strata are sized on: one corpus turn per line, `<n>. <tag>:<utterance>`, 1-based (D76)."""
    return "\n".join(f"{i}. {role}:{text}" for i, (role, text) in enumerate(turns, 1))


def turns_of(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.split("\n"):
        role, _, said = line.partition(": ")
        out.append((role, said))
    return out


def stratum_of(tokens: int) -> str | None:
    for name, (lo, hi) in STRATA.items():
        if lo <= tokens < hi:
            return name
    return None


def v1_footprint(ledger: Path = V1_LEDGER) -> tuple[set[str], set[str]]:
    """Episode doc_ids and podcast titles the v1 labelling touched (podcast ids were not stored in v1 records)."""
    docs: set[str] = set()
    pods: set[str] = set()
    if not ledger.exists():
        return docs, pods
    with ledger.open() as f:
        for line in f:
            if '"dataset":"sporc"' not in line and '"dataset": "sporc"' not in line:
                continue
            r = json.loads(line)
            docs.add(r["id"].split("::")[0].removeprefix("sporc-"))
            pods.add(r.get("meta", {}).get("pod_title", ""))
    return docs, pods


def scan(interim: Path = INTERIM, exclude_docs: frozenset[str] | set[str] = frozenset(),
         exclude_pods: frozenset[str] | set[str] = frozenset()) -> tuple[list[dict], collections.Counter]:
    """Pass 1: every episode's metadata and byte offset, with the eligibility funnel. Text is not kept."""
    funnel: collections.Counter = collections.Counter()
    rows: list[dict] = []
    with interim.open("rb") as f:
        offset = 0
        for raw in f:
            d = json.loads(raw)
            m = d.get("meta", {})
            row = {"doc_id": d["doc_id"], "offset": offset, "podcast_id": m.get("podcast_id", ""),
                   "pod_title": m.get("pod_title", ""), "category": m.get("category", ""),
                   "duration_s": m.get("duration_s"), "n_turns": d["n_turns"], "n_words": d["n_words"],
                   "n_speakers": m.get("n_speakers", 0), "est_tokens": int(d["n_words"] * TOKENS_PER_WORD)}
            offset += len(raw)
            funnel["episodes"] += 1
            if row["n_speakers"] not in SPEAKERS:
                funnel["dropped: speakers not 2-4"] += 1
                continue
            if row["n_turns"] < MIN_TURNS:
                funnel["dropped: under 40 turns"] += 1
                continue
            if row["doc_id"] in exclude_docs or row["pod_title"] in exclude_pods:
                funnel["dropped: v1 episode or podcast"] += 1
                continue
            dur = row["duration_s"] or 0
            if dur <= 0 or row["n_words"] / (dur / 60) < MIN_WPM:
                funnel["dropped: partial transcript (duration unknown or under 90 words per minute)"] += 1
                continue
            if stratum_of(row["est_tokens"]) is None and not any(
                    WINDOW[0] * lo <= row["est_tokens"] < WINDOW[1] * hi for lo, hi in STRATA.values()):
                funnel["dropped: length outside every stratum"] += 1
                continue
            funnel["eligible"] += 1
            rows.append(row)
    return rows, funnel


def text_checks(turns: list[tuple[str, str]]) -> tuple[str | None, dict]:
    """Reject a monologue, a repetition loop or a collapsed transcript; return (reason or None, measured stats)."""
    words: collections.Counter = collections.Counter()
    longest = 0
    dups = 0
    prev = None
    for role, said in turns:
        n = len(said.split())
        words[role] += n
        longest = max(longest, n)
        if prev is not None and said == prev:
            dups += 1
        prev = said
    total = sum(words.values()) or 1
    shares = sorted((v / total for v in words.values()), reverse=True)
    stats = {"word_shares": [round(x, 3) for x in shares], "duplicate_line_share": round(dups / max(1, len(turns)), 3),
             "longest_line_share": round(longest / total, 3)}
    if len(shares) < 2:
        return "one speaker tag in the text", stats
    if shares[1] < MIN_SECOND_SHARE:
        return "monologue (second speaker under 10% of words)", stats
    if stats["duplicate_line_share"] > MAX_DUPLICATE_LINES:
        return "repetition loop (over 20% duplicate lines)", stats
    if stats["longest_line_share"] > MAX_LONGEST_LINE_SHARE:
        return "collapsed transcript (one line over 25% of words)", stats
    return None, stats


def read_text(interim: Path, offset: int) -> str:
    with interim.open("rb") as f:
        f.seek(offset)
        return json.loads(f.readline())["text"]


def gemma_counter() -> Callable[[str], int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def choose(rows: list[dict], per_stratum: dict[str, int], seed: int, count_tokens: Callable[[str], int],
           interim: Path = INTERIM, oversample: float = 3.0, funnel: collections.Counter | None = None,
           taken: list[dict] | None = None) -> Iterator[dict]:
    """Pass 2: draw candidates per stratum by the estimate, measure them exactly, keep the ones that land in a
    stratum that still has room, at most PER_PODCAST per podcast across the whole pool. `taken` is the existing pool
    when topping up: its episodes are never drawn again and its podcasts count towards the cap."""
    rng = random.Random(seed)
    need = dict(per_stratum)
    per_pod: collections.Counter = collections.Counter(r["podcast_id"] for r in (taken or []))
    seen: set[str] = {r["doc_id"] for r in (taken or [])}
    # candidates whose estimate sits in a stratum's window, shuffled once so the draw is reproducible
    by_stratum: dict[str, list[dict]] = {s: [] for s in STRATA}
    for r in rows:
        for s, (lo, hi) in STRATA.items():
            if WINDOW[0] * lo <= r["est_tokens"] < WINDOW[1] * hi:
                by_stratum[s].append(r)
    for cands in by_stratum.values():
        rng.shuffle(cands)
    for s, want in per_stratum.items():
        measured = 0
        for r in by_stratum[s]:
            if need[s] <= 0:
                break
            if r["doc_id"] in seen or per_pod[r["podcast_id"]] >= PER_PODCAST:
                continue
            if measured >= int(want * oversample) + 50:
                break
            measured += 1
            text = read_text(interim, r["offset"])
            turns = turns_of(text)
            reason, stats = text_checks(turns)
            if reason is not None:
                seen.add(r["doc_id"])
                if funnel is not None:
                    funnel[f"measured, dropped: {reason}"] += 1
                continue
            tokens = count_tokens(render(turns))
            final = stratum_of(tokens)
            if final != s:  # kept only in the stratum it was drawn for, so no stratum fills from another's candidates
                continue
            seen.add(r["doc_id"])
            per_pod[r["podcast_id"]] += 1
            need[final] -= 1
            yield {**r, "tags_in_text": sorted({role for role, _ in turns}), "tokens": tokens, "stratum": final,
                   "seed": seed, **stats}


def quantiles(xs: list[int]) -> dict:
    if not xs:
        return {}
    ls = sorted(xs)
    pick = lambda p: ls[min(len(ls) - 1, int(p * len(ls)))]
    return {"min": ls[0], "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "max": ls[-1], "n": len(ls)}


def summarise(pool: list[dict], funnel: collections.Counter, per_stratum: dict[str, int], seed: int) -> dict:
    out: dict = {"seed": seed, "requested": per_stratum, "funnel": dict(funnel), "pool": len(pool),
                 "podcasts": len({r["podcast_id"] for r in pool}), "strata": {}}
    for s in STRATA:
        rows = [r for r in pool if r["stratum"] == s]
        out["strata"][s] = {"episodes": len(rows), "tokens": quantiles([r["tokens"] for r in rows]),
                            "turns": quantiles([r["n_turns"] for r in rows]),
                            "words": quantiles([r["n_words"] for r in rows]),
                            "speakers": dict(collections.Counter(r["n_speakers"] for r in rows)),
                            "categories": dict(collections.Counter(r["category"] for r in rows).most_common())}
    out["categories"] = dict(collections.Counter(r["category"] for r in pool).most_common())
    return out


def top_up(want: dict[str, int], seed: int, oversample: float = 3.0) -> list[dict]:
    """Append `want` more episodes per stratum to the frozen pool, from a seed not used before, under the pool's
    filters and podcast cap; records the draw in the summary's `topups` list. Returns the new rows."""
    if not POOL.exists():
        raise SystemExit(f"{POOL} does not exist; build the pool first")
    pool = [json.loads(line) for line in POOL.open()]
    used = {r.get("topup", r["seed"]) for r in pool}
    if seed in used:
        raise SystemExit(f"seed {seed} was already used for this pool ({sorted(used)}); a top-up needs a new seed")
    docs, pods = v1_footprint()
    rows, funnel = scan(exclude_docs=docs, exclude_pods=pods)
    per_stratum = {s: want.get(s, 0) for s in STRATA}
    new = [{**r, "topup": seed} for r in choose(rows, per_stratum, seed, gemma_counter(), oversample=oversample,
                                                 funnel=funnel, taken=pool)]
    with POOL.open("a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    (OUT_DIR / f"topup_{seed}_ids.txt").write_text("".join(r["doc_id"] + "\n" for r in new))
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    got = dict(collections.Counter(r["stratum"] for r in new))
    short = {s: per_stratum[s] - got.get(s, 0) for s in per_stratum if got.get(s, 0) < per_stratum[s]}
    summary.setdefault("topups", []).append({"seed": seed, "requested": per_stratum, "added": got, "short": short,
                                             "funnel": dict(funnel), "v1_ledger_lines": len(docs),
                                             "podcasts_new": len({r["podcast_id"] for r in new} - {r["podcast_id"] for r in pool}),
                                             "pool_after": len(pool) + len(new)})
    summary["current"] = summarise(pool + new, collections.Counter(), per_stratum, seed)["strata"]  # the whole pool now
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps({"top_up_seed": seed, "requested": per_stratum, "added": got, "short": short,
                      "pool_after": len(pool) + len(new)}))
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--per-stratum", type=int, nargs=4, default=[1500, 1500, 1500, 1500],
                    metavar=("N_0_4K", "N_4_8K", "N_8_12K", "N_12_16K"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pool")
    ap.add_argument("--oversample", type=float, default=3.0,
                    help="candidates measured per stratum, as a multiple of the stratum size (the estimate misplaces some)")
    ap.add_argument("--top-up", type=int, nargs=4, metavar=("N_0_4K", "N_4_8K", "N_8_12K", "N_12_16K"),
                    help="append this many more episodes per stratum to the existing pool (needs a seed not used before)")
    a = ap.parse_args()
    if a.top_up:
        top_up(dict(zip(STRATA, a.top_up, strict=True)), a.seed, a.oversample)
        return
    if POOL.exists() and not a.force:
        raise SystemExit(f"{POOL} exists; the pool is frozen once written (--force to rebuild)")
    want = dict(zip(STRATA, a.per_stratum, strict=True))
    docs, pods = v1_footprint()
    print(f"v1 footprint: {len(docs)} episodes, {len(pods)} podcasts excluded")
    rows, funnel = scan(exclude_docs=docs, exclude_pods=pods)
    print(json.dumps(dict(funnel)))
    pool = list(choose(rows, want, a.seed, gemma_counter(), oversample=a.oversample, funnel=funnel))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with POOL.open("w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    summary = summarise(pool, funnel, want, a.seed)
    SUMMARY.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"}))
    for s, st in summary["strata"].items():
        print(s, st["episodes"], "tokens", st["tokens"])


if __name__ == "__main__":
    main()
