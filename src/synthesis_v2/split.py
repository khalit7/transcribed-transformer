"""Split labelled data v2 into train, val and benchmark, by episode, equal in every length band.

    uv run python -m src.synthesis_v2.split [--seed 0] [--force]

Reads the ledger (`data/labelled_data_v2/labelled_data.jsonl`), the selection (`strata/selection.jsonl`), the episode
pool and metadata (`episodes.jsonl`, `episode_meta.jsonl`) and the bank (`questions.jsonl`), and writes `splits.json`:
every selected episode assigned to train | val | benchmark, plus the question ids and the held-out ones. The file is
frozen once written (`--force` to overwrite); `src.synthesis_v2.benchmark` turns it into the three JSONL files.

Rules (owner's decision 2026-11-05, D84, after three independent reviews of the sizes):

- Per length band (0-4k, 4-8k, 8-12k, 12-16k; 1,500 episodes each): 1,150 train, 50 val, 300 benchmark.
- Split by episode, never by (episode, question) pair, and no show on two sides. SPoRC files some shows under several
  feed ids and some episodes twice, so episodes are grouped (connected components) by any of: the same feed id; the same
  normalised show title; the same normalised episode title with durations within DURATION_TOLERANCE seconds;
  transcripts sharing at least OVERLAP_LINK of their sampled 8-word passages (an interview cross-posted under another
  show's name); a shared full host name in SPoRC's predicted host names (sister shows with the same host); or a shared
  iTunes owner or author name of two or more words (sister shows whose host names SPoRC did not predict). Val and
  benchmark are drawn only from groups of one episode, so the grouping holds by construction; every larger group goes to
  train. The owner link also joins whole networks (one owner name over many feeds: "Locked On Podcast Network", "The
  Athletic", "SB Nation", "New Books Network"); that over-groups on purpose, at a small cost in eligible episodes.
  Shows of one network with different hosts, no shared owner name and no shared text remain unlinked. Consequence,
  measured 2026-11-05: val and benchmark hold no show whose owner name of two or more words owns three or more feeds,
  and few shows with several episodes in the pool, so the benchmark leans to single-feed, smaller shows. One-word owner
  names (station groups such as "iheartradio", "audacy") are not linked, so sister shows of those networks with
  different hosts and no shared text can sit in val or benchmark (eight did on 2026-11-05; none shares text beyond
  adverts).
- Val and benchmark exclude everything an earlier checkpoint was trained on, at the level of the show: each config in
  `configs/adapt/` that samples `data/interim/sporc/` is replayed for its train and val samples (`adaptation_seen`, the
  sampler of `src.train.data.build_corpus_examples`, every document its seeded draw keeps, a superset of what the run
  took), and v1's labelled SPoRC episodes (v1 fine-tunes trained on them) are added; all join the grouping as outside
  nodes with feed id, show title, episode title, host and owner names and transcript passages. Any group touching one
  goes to train, so no val or benchmark episode, nor its show, nor a copy of it under another id, was read by an
  adapted or v1 checkpoint.
- Val and benchmark episodes must carry every question's label and must not be, or be grouped with, a bank-design
  episode (`pilot_ids.txt`, `pilot4_ids.txt`, including the design episodes outside the 6,000): the questions were
  drafted, probed and cut against them, so they and their shows go to train.
- The benchmark of each band is the best of DRAWS seeded draws refined by SWAPS single-episode swaps, scored on how far
  its category and length/density cell shares are from the band's eligible pool, plus a floor on rare answers: every
  (question, answer) class gets at least min(RARE_FLOOR, pool count // 2) benchmark episodes where the pool allows. The
  floor over-represents the rarest classes (measured up to about 2.2 times their pool share where it
  binds; a class of one or two in the pool can show a larger ratio by chance); the report lists every class it
  inflates by RARE_REPORT or more, so benchmark class rates are never quoted as population rates.
- Val is a seeded random draw from the band's remaining eligible episodes; it is for checkpoint selection only and is
  never reported.
- The held-out questions (`held_out` in questions.jsonl) are dropped from train and val and form the unseen-question
  cell of the benchmark.

`assign(record, splits)` returns train | val | benchmark | drop for a ledger record, the one place the rule lives.
"""

import argparse
import collections
import datetime as dt
import json
import random
import re
import zlib
from pathlib import Path

from src.synthesis_v2 import OUT_DIR
from src.synthesis_v2.episodes import STRATA

LEDGER = OUT_DIR / "labelled_data.jsonl"
QUESTIONS = OUT_DIR / "questions.jsonl"
SELECTION = OUT_DIR / "strata" / "selection.jsonl"
POOL = OUT_DIR / "episodes.jsonl"
EPISODE_META = OUT_DIR / "episode_meta.jsonl"
DESIGN_IDS = (OUT_DIR / "pilot_ids.txt", OUT_DIR / "pilot4_ids.txt")  # episodes the bank was written and cut against
SPLITS = OUT_DIR / "splits.json"

SIZES = {"train": 1150, "val": 50, "benchmark": 300}  # episodes per band
RARE_FLOOR = 10  # benchmark episodes per (question, answer) class, where the pool allows
RARE_REPORT = 1.25  # report classes whose benchmark share is at least this multiple of their pool share
DRAWS = 300  # candidate benchmark draws per band
SWAPS = 20000  # single-episode swap attempts that refine the best draw
DURATION_TOLERANCE = 2.0  # seconds: the same episode title within this is the same episode
SHINGLE = 8  # words per passage
SAMPLE_MOD = 8  # keep passages whose hash is 0 mod this (a fixed 1-in-8 sample, the same for every episode)
OVERLAP_LINK = 0.25  # shared sampled passages / the smaller episode's: measured 2026-11-05, every pair at or above
#                      0.25 was the same episode or the same show; below, shared intros and adverts
ADAPT_CONFIGS = Path("configs/adapt")  # relative, as the runs were launched: the sampler seeds on the path string
INTERIM = Path("data/interim/sporc/train.jsonl")


def episode_of(record_id: str) -> str:
    return record_id.split("::")[0].removeprefix("sporc-")


def norm_show(title: str) -> str:
    """Lower case, a closed parenthetical or bracketed suffix dropped ("Master Your Finances(Official 107.7 ...)"),
    punctuation to spaces, a leading "the" and a trailing "podcast" or "the podcast" dropped ("The From Imposter to
    Empowered Podcast" is "From Imposter to Empowered"); never emptied."""
    raw = (title or "").lower()
    t = re.sub(r"\s*[(\[][^()\[\]]*[)\]]\s*$", "", raw) or raw  # a closed trailing group only; never empty a title
    t = re.sub(r"[\W_]+", " ", t).strip()
    core = re.sub(r"^the\s+", "", re.sub(r"(\s+the)?\s+podcast$", "", t)).strip()
    return core or t


def norm_name(name: str) -> str:
    """A full name (two or more words) normalised, or empty: single first names are too common to link shows."""
    t = re.sub(r"[\W_]+", " ", (name or "").lower()).strip()
    return t if len(t.split()) >= 2 else ""


def sporc_rows(doc_ids: set[str]) -> dict[str, dict]:
    """Grouping metadata for SPoRC episodes from the raw parquet: feed, show, episode title, duration, predicted host
    names, and the iTunes owner and author names (the same owner across feeds marks sister shows)."""
    import glob

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from src.synthesis_v2.sporc_meta import PARTS, parse_names
    cols = ["episode_id", "podcast_id", "pod_title", "ep_title", "duration_seconds", "host_predicted_names",
            "itunes_author", "itunes_owner_name"]
    out: dict[str, dict] = {}
    if not doc_ids:  # pyarrow cannot type an empty value set
        return out
    want = pa.array(sorted(doc_ids))
    for f in sorted(glob.glob(str(PARTS / "part-*.parquet"))):
        t = pq.read_table(f, columns=cols)
        for r in t.filter(pc.is_in(t["episode_id"], value_set=want)).to_pylist():
            out[r["episode_id"]] = {"podcast": r["podcast_id"] or "", "show": r["pod_title"] or "",
                                    "title": (r["ep_title"] or "").strip(), "duration": r["duration_seconds"] or 0.0,
                                    "hosts": parse_names(r["host_predicted_names"]),
                                    "owners": sorted({norm_name(r["itunes_author"]), norm_name(r["itunes_owner_name"])} - {""})}
        if len(out) >= len(doc_ids):
            break
    return out


def _with_rows(nodes: dict[str, dict], what: str) -> None:
    """Fill title, hosts and owners from the raw parquet; refuse if more than 1% are missing, since the links that
    use them would silently stop reaching these nodes."""
    if not nodes:
        return
    rows = sporc_rows(set(nodes))
    if len(rows) < 0.99 * len(nodes):
        raise SystemExit(f"SPoRC metadata found for {len(rows)} of {len(nodes)} {what}; check data/raw/sporc/episodes")
    for doc, node in nodes.items():
        r = rows.get(doc) or {}
        node["title"] = node.get("title") or r.get("title", "")
        node["hosts"] = node.get("hosts") or r.get("hosts", [])
        node["owners"] = r.get("owners", [])


def norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", (title or "").lower()).strip()


def passages(lines: list[str]) -> set[int]:
    """A fixed sample of the episode's 8-word passages (speaker labels removed), as hashes."""
    words = re.findall(r"[a-z0-9']+", " ".join(ln.partition(": ")[2] for ln in lines).lower())
    out = set()
    for i in range(len(words) - SHINGLE + 1):
        h = zlib.crc32(" ".join(words[i:i + SHINGLE]).encode())
        if h % SAMPLE_MOD == 0:
            out.add(h)
    return out


def load_questions(path: Path = QUESTIONS) -> tuple[list[str], set[str]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return [r["id"] for r in rows], {r["id"] for r in rows if r.get("held_out")}


def load_design(paths: tuple[Path, ...] = DESIGN_IDS) -> set[str]:
    return {line.strip() for path in paths for line in path.open() if line.strip()}


def _titles(meta: Path, ids: set[str]) -> dict[str, tuple[str, list[str]]]:
    """doc_id -> (episode title, SPoRC's predicted host names)."""
    out = {}
    for line in meta.open():
        r = json.loads(line)
        if r["doc_id"] in ids:
            out[r["doc_id"]] = (r.get("ep_title") or "", list(r.get("host_predicted_names") or []))
    return out


def load_episodes(ledger: Path = LEDGER, selection: Path = SELECTION, meta: Path = EPISODE_META) -> dict[str, dict]:
    """episode id -> band, show, category, cell, title, duration, sampled passages, and its answers by question id."""
    eps: dict[str, dict] = {}
    for line in selection.open():
        if line.strip():
            r = json.loads(line)
            eps[r["doc_id"]] = {"band": r["stratum"], "podcast": r["podcast_id"], "show": r["pod_title"],
                                "category": r["category"], "cell": r["cell"], "duration": r["duration_s"],
                                "answers": {}, "passages": None}
    titles = _titles(meta, set(eps))
    for ep, e in eps.items():
        e["title"], e["hosts"] = titles.get(ep, ("", []))
    if meta == EPISODE_META:  # the real data: owners from the raw parquet (tests pass their own files and no owners)
        _with_rows(eps, "selected episodes")
    for line in ledger.open():
        r = json.loads(line)
        ep = episode_of(r["id"])
        if ep not in eps:
            raise SystemExit(f"{r['id']}: episode not in the selection")
        e = eps[ep]
        if r["meta"].get("stratum") != e["band"]:
            raise SystemExit(f"{r['id']}: ledger stratum {r['meta'].get('stratum')} != selection {e['band']}")
        qid = r["question"]["id"]
        if qid in e["answers"]:
            raise SystemExit(f"{r['id']}: duplicate label")
        e["answers"][qid] = r["label"]["answer"]
        if e["passages"] is None:
            e["passages"] = passages(r["transcript"]["variants"][0]["lines"])
    return eps


def load_outside_design(design: set[str], selected: set[str], pool: Path = POOL, meta: Path = EPISODE_META,
                        interim: Path = INTERIM) -> dict[str, dict]:
    """Bank-design episodes outside the selection (no production labels): show, title, hosts and transcript passages."""
    from src.synthesis_v2.episodes import read_text
    ids = design - selected
    out = {}
    for line in pool.open():
        r = json.loads(line)
        if r["doc_id"] in ids:
            out[r["doc_id"]] = {"podcast": r["podcast_id"], "show": r["pod_title"], "duration": r["duration_s"],
                                "passages": passages(read_text(interim, r["offset"]).splitlines())}
    titles = _titles(meta, set(out))
    for ep, e in out.items():
        e["title"], e["hosts"] = titles.get(ep, ("", []))
    _with_rows(out, "design episodes outside the selection")
    missing = ids - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} design episodes are in neither the selection nor the pool: {sorted(missing)[:3]}")
    return out


def group(eps: dict[str, dict]) -> tuple[dict[str, str], collections.Counter]:
    """episode -> group root (connected components over the six links), and how many unions each link made."""
    parent = {ep: ep for ep in eps}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links: collections.Counter = collections.Counter()

    def union(a: str, b: str, why: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            links[why] += 1

    def by_key(key, why: str) -> None:
        first: dict = {}
        for ep in sorted(eps):
            k = key(eps[ep])
            if k:
                union(first.setdefault(k, ep), ep, why)

    by_key(lambda e: e["podcast"], "feed id")
    by_key(lambda e: norm_show(e["show"]), "show title")
    titled: dict[str, list[str]] = collections.defaultdict(list)
    for ep in sorted(eps):
        if norm_title(eps[ep]["title"]):
            titled[norm_title(eps[ep]["title"])].append(ep)
    for members in titled.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if abs(eps[a]["duration"] - eps[b]["duration"]) <= DURATION_TOLERANCE:
                    union(a, b, "episode title and duration")
    hosted: dict[str, list[str]] = collections.defaultdict(list)
    for ep in sorted(eps):
        for name in {norm_name(n) for n in eps[ep].get("hosts") or []} - {""}:
            hosted[name].append(ep)
    for members in hosted.values():
        for b in members[1:]:
            union(members[0], b, "host name")
    owned: dict[str, list[str]] = collections.defaultdict(list)
    for ep in sorted(eps):
        for name in set(eps[ep].get("owners") or []):
            owned[name].append(ep)
    for members in owned.values():
        for b in members[1:]:
            union(members[0], b, "itunes owner or author")
    index: dict[int, list[str]] = collections.defaultdict(list)
    for ep in sorted(eps):
        for h in eps[ep]["passages"] or ():
            index[h].append(ep)
    shared: collections.Counter = collections.Counter()
    for members in index.values():
        if 1 < len(members) <= 50:  # a passage in more than 50 episodes is boilerplate, not a duplicate
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    shared[(a, b)] += 1
    for (a, b), n in sorted(shared.items()):
        if n / max(1, min(len(eps[a]["passages"]), len(eps[b]["passages"]))) >= OVERLAP_LINK:
            union(a, b, "transcript overlap")
    return {ep: find(ep) for ep in eps}, links


def eligible(eps: dict[str, dict], roots: dict[str, str], qids: list[str], tainting: set[str]) -> set[str]:
    """Selected episodes that may go to val or benchmark: fully labelled, alone among the selected episodes in their
    group, and the group free of every tainting document (design episodes, adaptation-read documents)."""
    size = collections.Counter(roots[ep] for ep in eps)
    tainted = {roots[d] for d in tainting if d in roots}
    return {ep for ep, e in eps.items() if set(qids) <= e["answers"].keys()
            and size[roots[ep]] == 1 and roots[ep] not in tainted}


_DOC_ID = re.compile(r'"doc_id"\s*:\s*"([^"]+)"')


def _node(d: dict) -> dict:
    m = d.get("meta") or {}
    return {"podcast": m.get("podcast_id", ""), "show": m.get("pod_title", ""), "title": "", "hosts": [],
            "duration": m.get("duration_s") or 0.0, "passages": passages(d["text"].splitlines())}


def replay_sample(path: Path, want: int, total: int, seed: int, also: set[str] = frozenset()  # type: ignore[assignment]
                  ) -> tuple[dict[str, dict], dict[str, dict]]:
    """Every document the seeded draw of `build_corpus_examples` keeps from `path` (`<side>.jsonl`), to the end of the
    file: a superset of what the run took (it also skipped its exclusion set and stopped at `want`), so no run-time
    detail can shrink it; and, from the same pass, the documents in `also`. Both as outside grouping nodes."""
    rng = random.Random(f"{seed}|{path}")
    p_keep = min(1.0, want / total)
    kept, extra = {}, {}
    with path.open() as f:
        for line in f:
            drawn = rng.random() <= p_keep  # one draw per line, as the sampler makes
            if not drawn and not also:
                continue
            d = json.loads(line) if drawn or _DOC_ID.search(line).group(1) in also else None  # type: ignore[union-attr]
            if d is None:
                continue
            if drawn:
                kept[d["doc_id"]] = _node(d)
            if d["doc_id"] in also:
                extra[d["doc_id"]] = _node(d)
    return kept, extra


def adaptation_seen(configs: Path = ADAPT_CONFIGS, v1_docs: set[str] = frozenset()  # type: ignore[assignment]
                    ) -> tuple[dict[str, dict], list[dict]]:
    """Every SPoRC document an adaptation config in `configs` may have read (its train and val samples), plus the v1
    labelled SPoRC episodes `v1_docs` (v1 fine-tunes trained on them), as tainting outside grouping nodes with SPoRC's
    titles, host names and owners, so every grouping link reaches them; and a record of what was replayed. Refuses to
    return no adaptation documents: finding no SPoRC sample would silently drop the exclusion."""
    import yaml
    specs = set()
    for f in sorted(configs.glob("*.yaml")):
        cfg = yaml.safe_load(f.read_text()) or {}
        for c in ((cfg.get("adapt") or {}).get("corpora") or []):
            if "/sporc/" in str(c["path"]):
                base = Path(c["path"]).parent
                specs.add((str(base / "train.jsonl"), "train", int(c["docs"]), cfg.get("seed", 0)))
                specs.add((str(base / "val.jsonl"), "val", int(c.get("val_docs", 50)), cfg.get("seed", 0)))
    seen: dict[str, dict] = {}
    v1: dict[str, dict] = {}
    replayed = []
    for path, side, want, seed in sorted(specs):
        total = json.loads((Path(path).parent / "summary.json").read_text())["docs"][side]
        kept, extra = replay_sample(Path(path), want, total, seed, v1_docs)
        seen.update(kept)
        v1.update(extra)
        replayed.append({"path": path, "docs": want, "seed": seed, "kept_by_draw": len(kept)})
    if not seen:
        raise SystemExit(f"no adaptation config under {configs.resolve()} samples SPoRC; run from the repository root")
    _with_rows(seen, "adaptation documents")
    if v1:
        _with_rows(v1, "v1 labelled episodes")
    if len(v1) != len(v1_docs):  # a v1 episode the scan did not find would silently escape the exclusion
        raise SystemExit(f"found {len(v1)} of {len(v1_docs)} v1 labelled SPoRC episodes in the interim files")
    replayed.append({"v1_labelled_episodes": len(v1), "v1_asked": len(v1_docs)})
    return {**v1, **seen}, replayed


def _features(eps: dict[str, dict], ep: str) -> list[tuple]:
    e = eps[ep]
    return [("category", e["category"]), ("cell", e["cell"])] + [("answer", q, a) for q, a in sorted(e["answers"].items())]


class _Scorer:
    """Score of a candidate benchmark against its pool: stratum share deviation plus missing rare-answer support."""

    def __init__(self, eps: dict[str, dict], pool: list[str], n: int):
        self.n, self.pool_n = n, len(pool)
        self.feats = {ep: _features(eps, ep) for ep in pool}
        self.pool_counts = collections.Counter(f for ep in pool for f in self.feats[ep])
        self.share = {f: c / len(pool) for f, c in self.pool_counts.items() if f[0] != "answer"}
        self.floor = {f: min(RARE_FLOOR, c // 2) for f, c in self.pool_counts.items() if f[0] == "answer"}

    def counts(self, chosen: list[str]) -> collections.Counter:
        return collections.Counter(f for ep in chosen for f in self.feats[ep])

    def score(self, c: collections.Counter) -> float:
        # each share term is an L1 distance in [0, 2]; each missing rare answer costs 1, so the floor dominates in
        # practice (measured share deviations are about 0.01-0.02) without being absolute
        dev = sum(abs(c[f] / self.n - s) for f, s in self.share.items())
        missing = sum(max(0, fl - c[f]) for f, fl in self.floor.items())
        return dev + 1.0 * missing

    def inflated(self, c: collections.Counter) -> dict[str, list]:
        """Answer classes whose benchmark share is at least RARE_REPORT times their pool share: [bench, pool, ratio]."""
        out = {}
        for f, pc in sorted(self.pool_counts.items()):
            if f[0] == "answer" and c[f]:
                ratio = (c[f] / self.n) / (pc / self.pool_n)
                if ratio >= RARE_REPORT:
                    out[f"{f[1]}={f[2]}"] = [c[f], pc, round(ratio, 2)]
        return out


def sample_benchmark(eps: dict[str, dict], pool: list[str], n: int, rng: random.Random,
                     draws: int = DRAWS, swaps: int = SWAPS) -> tuple[list[str], dict]:
    if len(pool) < n:
        raise SystemExit(f"cannot draw {n} benchmark episodes from an eligible pool of {len(pool)}")
    sc = _Scorer(eps, pool, n)
    best: tuple[float, list[str]] | None = None
    for _ in range(draws):
        chosen = rng.sample(pool, n)
        s = sc.score(sc.counts(chosen))
        if best is None or s < best[0]:
            best = (s, chosen)
    assert best is not None
    score, chosen = best
    c = sc.counts(chosen)
    outside = sorted(set(pool) - set(chosen))
    for _ in range(swaps):
        if not outside:
            break
        i, j = rng.randrange(n), rng.randrange(len(outside))
        out_ep, in_ep = chosen[i], outside[j]
        c.subtract(sc.feats[out_ep])
        c.update(sc.feats[in_ep])
        s2 = sc.score(c)
        if s2 < score:
            score, chosen[i], outside[j] = s2, in_ep, out_ep
        else:
            c.subtract(sc.feats[in_ep])
            c.update(sc.feats[out_ep])
    chosen = sorted(chosen)
    final = sc.counts(chosen)
    short = {f"{f[1]}={f[2]}": [final[f], fl] for f, fl in sorted(sc.floor.items()) if final[f] < fl}
    report = {"score": round(score, 4), "pool": len(pool), "rare_floors_short": short,
              "classes_inflated": sc.inflated(final),
              "category_deviation": round(sum(abs(final[f] / n - s) for f, s in sc.share.items() if f[0] == "category"), 4),
              "cell_deviation": round(sum(abs(final[f] / n - s) for f, s in sc.share.items() if f[0] == "cell"), 4)}
    return chosen, report


def build(eps: dict[str, dict], qids: list[str], held: set[str], design: set[str], seed: int,
          outside: dict[str, dict] | None = None, sizes: dict[str, int] = SIZES,
          draws: int = DRAWS, swaps: int = SWAPS, seen: dict[str, dict] | None = None,
          replayed: list[dict] | None = None) -> dict:
    """`eps` are the selected episodes (with labels); `outside` the design episodes outside the selection and `seen`
    the adaptation-read documents (grouping nodes only; a selected episode among them keeps its selected entry)."""
    rng = random.Random(seed)
    seen = seen or {}
    every = {**seen, **(outside or {}), **eps}
    roots, links = group(every)
    seen_roots = {roots[d] for d in seen}
    ok = eligible(eps, roots, qids, design | set(seen))
    assignment: dict[str, str] = {}
    reports: dict[str, dict] = {}
    for band in STRATA:
        members = sorted(ep for ep, e in eps.items() if e["band"] == band)
        if len(members) != sum(sizes.values()):
            raise SystemExit(f"{band}: {len(members)} episodes, the sizes need {sum(sizes.values())}")
        pool = [ep for ep in members if ep in ok]
        bench, rep = sample_benchmark(eps, pool, sizes["benchmark"], rng, draws, swaps)
        rest = sorted(set(pool) - set(bench))
        if len(rest) < sizes["val"]:
            raise SystemExit(f"{band}: {len(rest)} eligible episodes left for val, need {sizes['val']}")
        val = set(rng.sample(rest, sizes["val"]))
        for ep in members:
            assignment[ep] = "benchmark" if ep in bench else "val" if ep in val else "train"
        rep["episodes"] = dict(collections.Counter(assignment[ep] for ep in members))
        rep["design_episodes_in_train"] = sum(1 for ep in members if ep in design)
        rep["trained_on_in_train"] = sum(1 for ep in members if ep in seen)  # read by adaptation or labelled in v1
        rep["trained_on_group_in_train"] = sum(1 for ep in members if roots[ep] in seen_roots)
        reports[band] = rep
    sides = collections.defaultdict(set)
    for ep, side in assignment.items():
        sides[roots[ep]].add(side)
    for ep in {**seen, **(outside or {})}:
        if ep not in eps:
            sides[roots[ep]].add("outside")
    for ep in design | set(seen):
        if ep in eps:
            sides[roots[ep]].add("outside")
    straddling = sorted(r for r, s in sides.items() if len(s - {"outside"}) > 1 or ("outside" in s and s & {"val", "benchmark"}))
    if straddling:
        raise SystemExit(f"groups on two sides: {straddling[:5]}")
    size = collections.Counter(roots[ep] for ep in eps)
    return {
        "version": "v2", "seed": seed, "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "policy": {"episodes_per_band": sizes, "rare_floor": RARE_FLOOR, "draws": draws, "swaps": swaps,
                   "grouping": {"duration_tolerance_s": DURATION_TOLERANCE, "shingle_words": SHINGLE,
                                "sample_mod": SAMPLE_MOD, "overlap_link": OVERLAP_LINK},
                   "val_benchmark_from": "fully labelled episodes alone among the selected episodes in their group, the group "
                                         "free of bank-design episodes, of every SPoRC document an adaptation run may have read, "
                                         "and of v1's labelled SPoRC episodes",
                   "design_ids": sorted(design)},
        "trained_on": {"replayed": replayed or [],  # adaptation samples and v1's labelled SPoRC episodes
                       "selected_episodes_trained_on": sum(1 for ep in eps if ep in seen),
                       "selected_episodes_in_a_trained_on_group": sum(1 for ep in eps if roots[ep] in seen_roots)},
        "grouping": {"unions_by_link": dict(links), "groups_of_two_or_more": sum(1 for n in size.values() if n > 1),
                     "episodes_in_them": sum(n for n in size.values() if n > 1)},
        "questions": qids, "held_out_questions": sorted(held), "report": reports,
        "episodes": dict(sorted(assignment.items())),
    }


def load_splits(path: Path = SPLITS) -> dict:
    s = json.loads(path.read_text())
    s["_held"] = set(s["held_out_questions"])
    return s


def assign(record: dict, splits: dict) -> str:
    """train | val | benchmark | drop for one ledger record (drop = a held-out question on a train or val episode)."""
    ep = episode_of(record["id"])
    side = splits["episodes"].get(ep)
    if side is None:  # v2 is a closed set: an unknown episode means the ledger and the split disagree
        raise KeyError(f"{record['id']}: episode {ep} is not in the split")
    held = splits.get("_held", set(splits["held_out_questions"]))
    if side != "benchmark" and record["question"]["id"] in held:
        return "drop"
    return side


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=SPLITS)
    p.add_argument("--force", action="store_true", help="overwrite an existing (frozen) split")
    args = p.parse_args(argv)
    if args.out.exists() and not args.force:
        p.error(f"{args.out} exists and is frozen; --force to rebuild (this changes the benchmark)")
    qids, held = load_questions()
    from src.synthesis_v2.benchmark import (
        check_bank,  # a stale questions.jsonl must not be frozen into a split
    )
    check_bank({"questions": qids, "held_out_questions": sorted(held)}, qids, held)
    design = load_design()
    eps = load_episodes()
    outside = load_outside_design(design, set(eps))
    from src.synthesis_v2.episodes import V1_LEDGER, v1_footprint
    if not V1_LEDGER.exists():  # v1_footprint returns nothing for a missing ledger; the exclusion must not vanish
        raise SystemExit(f"{V1_LEDGER} is missing; the v1 exclusion needs it")
    v1_docs, _ = v1_footprint()
    seen, replayed = adaptation_seen(v1_docs=v1_docs)
    s = build(eps, qids, held, design, args.seed, outside, seen=seen, replayed=replayed)
    args.out.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {dict(collections.Counter(s['episodes'].values()))}; held-out questions {len(held)}")
    print(f"grouping: {json.dumps(s['grouping'])}")
    print(f"trained on: {json.dumps(s['trained_on'])}")
    for band, rep in s["report"].items():
        print(f"{band}: {json.dumps(rep, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
