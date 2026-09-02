"""Fit ASR channel-model statistics from paired reference/hypothesis text (channel v2).

Reference: AppTek verbatim transcripts (`data/raw/apptek_callcenter/test/<locale>/
metadata.jsonl`, one row per split-channel WAV, known speaker).
Hypothesis: self-ASR output (`data/interim/apptek_callcenter_selfasr/
test-<variant>/<locale>/<stem>.json` from scripts/transcribe.py).

Alignment is word-level Levenshtein (rapidfuzz editops) over normalised tokens,
whole file at a time: split-channel files pair one reference blob with one
hypothesis stream, so no time-windowing is needed (unlike turn-interleaved
corpora, where ASR segmentation is finer than the transcriber's).

Normalisation (recorded in the artifact): lowercase; AppTek hesitation markup
`(um)` becomes the token `um` (Whisper emits fillers as words, and fillers are
signal here, not noise); partial-word `~` suffixes are stripped from the token;
leading/trailing punctuation is stripped per token; empty tokens dropped.
Casing and punctuation behaviour are surface-layer concerns fitted separately
(SYNTHSHEET section 4), not part of the word-edit channel.

Counts are stored, not rates, so fits can be summed across corpora and systems.
Per-word reference counts and the hypothesis vocabulary are stored too, because
the apply side (src/channel/apply.py) needs conditional edit probabilities and a
backoff vocabulary for words never seen in the reference.

`--holdout F` deterministically holds out a fraction of files (by stem hash) from
the fit and lists them in the artifact, so the QC gate (src/channel/gate.py) can
compare synthetic against real hypotheses on files the channel never saw.

Output: data/channel/channel_v2_<variant>.json with per-accent and pooled
substitution/deletion/insertion counters, reference counts, and WER.

Usage: uv run python -m src.channel.fit --variant degraded --holdout 0.1
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUNCT_EDGE = re.compile(r"^[^\w']+|[^\w']+$")
_HESITATION = re.compile(r"\(([a-z]+)\)")


def is_holdout(stem: str, fraction: float) -> bool:
    """Deterministic per-file holdout: both channels of a call share the
    stem prefix before `_channel`, so a call is held out whole."""
    call = stem.split("_channel")[0]
    h = int(hashlib.sha1(call.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def normalise(text: str) -> list[str]:
    text = text.lower()
    text = _HESITATION.sub(r"\1", text)
    out = []
    for tok in text.split():
        tok = tok.rstrip("~")
        tok = _PUNCT_EDGE.sub("", tok)
        if tok:
            out.append(tok)
    return out


MAX_REF_SPAN = 4
MAX_HYP_SPAN = 6


def clusters(ops, ref: list[str], hyp: list[str]):
    """Group adjacent edit ops into contiguous (ref_span, hyp_span) clusters.

    `gonna` -> `going to` arrives as replace(gonna->going) + insert(to) at
    adjacent positions; a hallucinated `thank you` arrives as two adjacent
    inserts. Treating those as units is what the word-independent model missed.
    Yields (ref_tokens, hyp_tokens); an empty ref span is a pure insertion run.
    """
    cur = None  # [r0, r1, h0, h1]
    for op, s, d in ops:
        r_len, h_len = {"replace": (1, 1), "delete": (1, 0), "insert": (0, 1)}[op]
        if cur is not None and s == cur[1] and d == cur[3]:
            cur[1] += r_len
            cur[3] += h_len
        else:
            if cur is not None:
                yield ref[cur[0]:cur[1]], hyp[cur[2]:cur[3]]
            cur = [s, s + r_len, d, d + h_len]
    if cur is not None:
        yield ref[cur[0]:cur[1]], hyp[cur[2]:cur[3]]


def align(ref: list[str], hyp: list[str], stats: dict) -> None:
    """Count edit operations into stats; rapidfuzz editops is C++-fast."""
    ops = Levenshtein.editops(ref, hyp)
    n_edits = 0
    for op, ri, hi in ops:
        n_edits += 1
        if op == "replace":
            stats["substitutions"][(ref[ri], hyp[hi])] += 1
        elif op == "delete":
            stats["deletions"][ref[ri]] += 1
        else:
            stats["insertions"][hyp[hi]] += 1
    for r_span, h_span in clusters(ops, ref, hyp):
        stats["word_involvement"].update(r_span)
        if not r_span:
            if len(h_span) <= MAX_HYP_SPAN:
                stats["insertion_phrases"][" ".join(h_span)] += 1
                stats["n_insertion_clusters"] += 1
        elif len(r_span) <= MAX_REF_SPAN and len(h_span) <= MAX_HYP_SPAN:
            stats["span_edits"][(" ".join(r_span), " ".join(h_span))] += 1
        # Longer garbled stretches count towards each word's involvement only:
        # splitting them into per-token fragments produced orphaned halves of
        # phrases (`thank` without `you`) that a discriminator picked out.
    for n in range(1, MAX_REF_SPAN + 1):
        stats["span_counts"].update(" ".join(ref[i:i + n]) for i in range(len(ref) - n + 1))
    if ref:
        stats["file_wers"].append(round(n_edits / len(ref), 4))
    stats["n_ref"] += len(ref)
    stats["n_hyp"] += len(hyp)
    stats["reference_counts"].update(ref)
    stats["hypothesis_counts"].update(hyp)


def new_stats() -> dict:
    return {
        "substitutions": Counter(),
        "deletions": Counter(),
        "insertions": Counter(),
        "reference_counts": Counter(),
        "hypothesis_counts": Counter(),
        "span_edits": Counter(),
        "span_counts": Counter(),
        "word_involvement": Counter(),
        "insertion_phrases": Counter(),
        "n_insertion_clusters": 0,
        "file_wers": [],
        "n_ref": 0,
        "n_hyp": 0,
        "n_files": 0,
    }


def wer(s: dict) -> float:
    edits = sum(s["substitutions"].values()) + sum(s["deletions"].values()) + sum(
        s["insertions"].values()
    )
    return edits / s["n_ref"] if s["n_ref"] else 0.0


def serialise(s: dict) -> dict:
    return {
        "n_files": s["n_files"],
        "n_ref": s["n_ref"],
        "n_hyp": s["n_hyp"],
        "wer": round(wer(s), 4),
        "sub_rate": round(sum(s["substitutions"].values()) / s["n_ref"], 4),
        "del_rate": round(sum(s["deletions"].values()) / s["n_ref"], 4),
        "ins_rate": round(sum(s["insertions"].values()) / s["n_ref"], 4),
        "n_distinct_subs": len(s["substitutions"]),
        "substitutions": {f"{r}\t{h}": c for (r, h), c in s["substitutions"].most_common()},
        "deletions": dict(s["deletions"].most_common()),
        "insertions": dict(s["insertions"].most_common()),
        "reference_counts": dict(s["reference_counts"].most_common()),
        "hypothesis_counts": dict(s["hypothesis_counts"].most_common()),
        "span_edits": {f"{r}\t{h}": c for (r, h), c in s["span_edits"].most_common()},
        "span_counts": {k: c for k, c in s["span_counts"].items() if c >= 2},
        "word_involvement": dict(s["word_involvement"].most_common()),
        "insertion_phrases": dict(s["insertion_phrases"].most_common()),
        "n_insertion_clusters": s["n_insertion_clusters"],
        "file_wers": s["file_wers"],
    }


def calibrate(artifact: dict, refs: list[list[str]], target_wer: float, rounds: int = 3) -> float:
    """Global scalar on edit probabilities so the applied channel reproduces the
    fitted WER on the training references. Span probabilities overlap (a word
    can be edited via its own 1-gram or via a containing n-gram), so the raw
    model over- or under-shoots; this pins the total. Uses a fixed seed and a
    sample of files, burst off."""
    import numpy as np

    from src.channel.apply import Channel  # lazy: apply imports normalise from here

    rng = np.random.default_rng(0)
    sample = [refs[i] for i in rng.choice(len(refs), size=min(300, len(refs)), replace=False)]
    factor, achieved = 1.0, 0.0
    for _ in range(rounds):
        ch = Channel(artifact, severity=factor, burst=False, calibrated=False)
        s = new_stats()
        for ref in sample:
            align(ref, ch.noise(ref, rng), s)
        achieved = wer(s)
        factor *= target_wer / achieved if achieved else 1.0
    print(f"calibration factor {factor:.4f} (target WER {target_wer:.4f}, "
          f"achieved before final adjustment {achieved:.4f})")
    return round(factor, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="degraded", choices=["degraded", "clean"])
    parser.add_argument("--corpus", default="apptek_callcenter")
    parser.add_argument(
        "--holdout", type=float, default=0.0,
        help="fraction of calls held out of the fit (deterministic by stem hash)",
    )
    args = parser.parse_args()

    ref_root = REPO_ROOT / "data" / "raw" / args.corpus / "test"
    hyp_root = (
        REPO_ROOT / "data" / "interim" / f"{args.corpus}_selfasr" / f"test-{args.variant}"
    )
    suffix = f"_holdout{int(args.holdout * 100)}" if args.holdout else ""
    out_path = REPO_ROOT / "data" / "channel" / f"channel_v2_{args.variant}{suffix}.json"

    pooled = new_stats()
    per_accent: dict[str, dict] = {}
    missing = 0
    holdout: list[dict] = []
    train_refs: list[list[str]] = []

    for meta in sorted(ref_root.glob("*/metadata.jsonl")):
        locale = meta.parent.name
        acc = per_accent.setdefault(locale, new_stats())
        for line in meta.open():
            row = json.loads(line)
            stem = Path(row["file_name"]).stem
            hyp_file = hyp_root / locale / f"{stem}.json"
            if not hyp_file.exists():
                missing += 1
                continue
            if args.holdout and is_holdout(stem, args.holdout):
                holdout.append({"locale": locale, "stem": stem})
                continue
            hyp_rec = json.loads(hyp_file.read_text())
            ref_toks = normalise(row["text"])
            hyp_toks = normalise(" ".join(s["text"] for s in hyp_rec["segments"]))
            for stats in (acc, pooled):
                align(ref_toks, hyp_toks, stats)
                stats["n_files"] += 1
            train_refs.append(ref_toks)

    artifact = {
        "version": f"v2.2-{args.variant}",  # v2.x: span edits, insertion phrases, involvement, calibration
        "reference": f"{args.corpus} verbatim transcripts (split-channel)",
        "asr_system": "faster-whisper 1.2.1 large-v3 float16",
        "degradation": None
        if args.variant == "clean"
        else "resample 8kHz; bandpass 300-3400Hz; mu-law 8-bit; resample 16kHz",
        "normalisation": "lowercase; (um)->um; strip ~; strip edge punctuation",
        "missing_hypotheses": missing,
        "holdout_fraction": args.holdout,
        "holdout_files": holdout,
        "pooled": serialise(pooled),
        "per_accent": {k: serialise(v) for k, v in sorted(per_accent.items())},
    }
    artifact["calibration"] = calibrate(artifact, train_refs, wer(pooled))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    print(f"files={pooled['n_files']} holdout={len(holdout)} missing={missing} "
          f"ref_words={pooled['n_ref']:,}")
    print(f"pooled WER={wer(pooled):.4f} "
          f"(sub={artifact['pooled']['sub_rate']} del={artifact['pooled']['del_rate']} "
          f"ins={artifact['pooled']['ins_rate']})")
    for loc, s in sorted(per_accent.items(), key=lambda kv: -wer(kv[1])):
        print(f"  {loc:16s} WER={wer(s):.4f} files={s['n_files']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
