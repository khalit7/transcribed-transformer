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

Output: data/channel/channel_v2_<variant>.json with per-accent and pooled
substitution/deletion/insertion counters, reference counts, and WER.

Usage: uv run python -m src.channel.fit --variant degraded
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUNCT_EDGE = re.compile(r"^[^\w']+|[^\w']+$")
_HESITATION = re.compile(r"\(([a-z]+)\)")


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


def align(ref: list[str], hyp: list[str], stats: dict) -> None:
    """Count edit operations into stats; rapidfuzz editops is C++-fast."""
    for op, ri, hi in Levenshtein.editops(ref, hyp):
        if op == "replace":
            stats["substitutions"][(ref[ri], hyp[hi])] += 1
        elif op == "delete":
            stats["deletions"][ref[ri]] += 1
        else:
            stats["insertions"][hyp[hi]] += 1
    stats["n_ref"] += len(ref)
    stats["n_hyp"] += len(hyp)


def new_stats() -> dict:
    return {
        "substitutions": Counter(),
        "deletions": Counter(),
        "insertions": Counter(),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="degraded", choices=["degraded", "clean"])
    parser.add_argument("--corpus", default="apptek_callcenter")
    args = parser.parse_args()

    ref_root = REPO_ROOT / "data" / "raw" / args.corpus / "test"
    hyp_root = (
        REPO_ROOT / "data" / "interim" / f"{args.corpus}_selfasr" / f"test-{args.variant}"
    )
    out_path = REPO_ROOT / "data" / "channel" / f"channel_v2_{args.variant}.json"

    pooled = new_stats()
    per_accent: dict[str, dict] = {}
    missing = 0

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
            hyp_rec = json.loads(hyp_file.read_text())
            ref_toks = normalise(row["text"])
            hyp_toks = normalise(" ".join(s["text"] for s in hyp_rec["segments"]))
            for stats in (acc, pooled):
                align(ref_toks, hyp_toks, stats)
                stats["n_files"] += 1

    artifact = {
        "version": f"v2-{args.variant}",
        "reference": f"{args.corpus} verbatim transcripts (split-channel)",
        "asr_system": "faster-whisper 1.2.1 large-v3 float16",
        "degradation": None
        if args.variant == "clean"
        else "resample 8kHz; bandpass 300-3400Hz; mu-law 8-bit; resample 16kHz",
        "normalisation": "lowercase; (um)->um; strip ~; strip edge punctuation",
        "missing_hypotheses": missing,
        "pooled": serialise(pooled),
        "per_accent": {k: serialise(v) for k, v in sorted(per_accent.items())},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    print(f"files={pooled['n_files']} missing={missing} ref_words={pooled['n_ref']:,}")
    print(f"pooled WER={wer(pooled):.4f} "
          f"(sub={artifact['pooled']['sub_rate']} del={artifact['pooled']['del_rate']} "
          f"ins={artifact['pooled']['ins_rate']})")
    for loc, s in sorted(per_accent.items(), key=lambda kv: -wer(kv[1])):
        print(f"  {loc:16s} WER={wer(s):.4f} files={s['n_files']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
