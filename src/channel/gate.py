"""QC gate for a fitted channel (SYNTHSHEET section 6): can a discriminator tell
channel-noised text from real ASR output?

Controlled design: uses the calls held out of the channel fit. For each held-out
split-channel file, REAL = the recogniser's actual (normalised) hypothesis and
SYNTHETIC = the channel applied to that file's verbatim reference. Same speech,
same speaker, same content, so the only difference is synthetic vs real errors.

Two measurements:

1. Distributional table: WER against the verbatim reference for real vs
   synthetic, length ratio, filler rate. Synthetic should match real.
2. Adversarial discriminator: TF-IDF word (1-2 gram) + char (3-5 gram) features,
   logistic regression, grouped 5-fold CV by call so no call leaks across folds,
   AUC from out-of-fold scores. Pass threshold: AUC <= 0.65.
   Power control: the same discriminator on UNNOISED verbatim vs real must reach
   a high AUC, otherwise a low gate AUC only means the discriminator is weak.

Usage: uv run python -m src.channel.gate --artifact data/channel/channel_v2_degraded_holdout10.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.channel.apply import Channel
from src.channel.fit import align, new_stats, normalise, wer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WINDOW = 60
FILLERS = {"um", "uh", "ah", "er", "erm", "mm", "hmm"}


def chunks(tokens: list[str], n: int = WINDOW) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n // 2, n)]


def discriminate(a_docs: list[str], b_docs: list[str], groups: list[str]) -> float:
    """Out-of-fold AUC for separating class A (label 1) from class B (label 0)."""
    docs = a_docs + b_docs
    y = np.array([1] * len(a_docs) + [0] * len(b_docs))
    g = np.array(groups)
    scores = np.zeros(len(docs))
    for tr, te in GroupKFold(n_splits=5).split(docs, y, g):
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
        tr_docs = [docs[i] for i in tr]
        te_docs = [docs[i] for i in te]
        x_tr = hstack([wv.fit_transform(tr_docs), cv.fit_transform(tr_docs)]).tocsr()
        x_te = hstack([wv.transform(te_docs), cv.transform(te_docs)]).tocsr()
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_tr, y[tr])
        scores[te] = clf.decision_function(x_te)
    return float(roc_auc_score(y, scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--corpus", default="apptek_callcenter")
    parser.add_argument("--variant", default="degraded")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact = json.loads(Path(args.artifact).read_text())
    holdout = artifact["holdout_files"]
    if not holdout:
        raise SystemExit("artifact has no held-out files; refit with --holdout")
    channel = Channel(artifact, severity=args.severity)

    ref_root = REPO_ROOT / "data" / "raw" / args.corpus / "test"
    hyp_root = REPO_ROOT / "data" / "interim" / f"{args.corpus}_selfasr" / f"test-{args.variant}"
    refs: dict[str, str] = {}
    for meta in ref_root.glob("*/metadata.jsonl"):
        for line in meta.open():
            row = json.loads(line)
            refs[Path(row["file_name"]).stem] = row["text"]

    real_docs, synth_docs, clean_docs = [], [], []
    real_groups, synth_groups, clean_groups = [], [], []
    st_real, st_synth = new_stats(), new_stats()
    ref_all: Counter = Counter()
    real_all: Counter = Counter()
    synth_all: Counter = Counter()
    n_ref = n_real = n_synth = 0

    for i, item in enumerate(holdout):
        stem, locale = item["stem"], item["locale"]
        call = stem.split("_channel")[0]
        ref = normalise(refs[stem])
        hyp = json.loads((hyp_root / locale / f"{stem}.json").read_text())
        real = normalise(" ".join(s["text"] for s in hyp["segments"]))
        synth = channel.noise(ref, np.random.default_rng(args.seed * 100_003 + i))

        align(ref, real, st_real)
        align(ref, synth, st_synth)
        n_ref += len(ref); n_real += len(real); n_synth += len(synth)
        ref_all.update(ref); real_all.update(real); synth_all.update(synth)

        for doc in chunks(real):
            real_docs.append(doc); real_groups.append(call)
        for doc in chunks(synth):
            synth_docs.append(doc); synth_groups.append(call)
        for doc in chunks(ref):
            clean_docs.append(doc); clean_groups.append(call)

    table = {
        "holdout_files": len(holdout),
        "reference_words": n_ref,
        "length_ratio_real": round(n_real / n_ref, 4),
        "length_ratio_synth": round(n_synth / n_ref, 4),
        "wer_real_vs_ref": round(wer(st_real), 4),
        "wer_synth_vs_ref": round(wer(st_synth), 4),
        "filler_per_1k_ref": round(1000 * sum(ref_all[f] for f in FILLERS) / n_ref, 2),
        "filler_per_1k_real": round(1000 * sum(real_all[f] for f in FILLERS) / n_real, 2),
        "filler_per_1k_synth": round(1000 * sum(synth_all[f] for f in FILLERS) / n_synth, 2),
        "vocab_real": len(real_all),
        "vocab_synth": len(synth_all),
    }

    auc_gate = discriminate(real_docs, synth_docs, real_groups + synth_groups)
    auc_power = discriminate(real_docs, clean_docs, real_groups + clean_groups)

    result = {
        "channel_version": channel.version,
        "severity": args.severity,
        "windows_per_class": len(real_docs),
        "distribution": table,
        "auc_real_vs_synthetic": round(auc_gate, 4),
        "auc_real_vs_clean_control": round(auc_power, 4),
        "threshold": 0.65,
        "passed": auc_gate <= 0.65,
    }
    out = Path(args.artifact).with_name(Path(args.artifact).stem.replace("channel_", "gate_") + ".json")
    out.write_text(json.dumps(result, indent=1))
    for k, v in table.items():
        print(f"{k:24s} {v}")
    print(f"AUC real vs synthetic      {auc_gate:.4f}   (gate: <= 0.65 -> {'PASS' if result['passed'] else 'FAIL'})")
    print(f"AUC real vs clean control  {auc_power:.4f}   (separately trained classifier, real vs "
          f"un-noised verbatim: the gap with no channel at all; should be high)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
