"""Shortcut probes and the evidence-position audit for labelled data v2, run before any arm trains.

    uv run python -m src.synthesis_v2.probes            # writes data/labelled_data_v2/probes/report.{json,md}

Pre-registered in experiments/2026-10-31-v2-long-podcast/README.md ("Checks run before any arm trains"):

- Shortcut probes (rule revised before any arm trained, 2026-11-05; the record says why). For each question: (a) its
  majority answer over the train episodes; (b) a classifier on the speaker structure alone (how many host, guest and
  other speaker labels the transcript shows, and the band), which reads no transcript content; (c) a bag-of-words
  classifier (TF-IDF of word unigrams and bigrams) on the first OPENING_WORDS words, a partial-reading baseline. Each
  classifier is logistic regression with balanced class weights, fitted on the train episodes and scored on the
  benchmark episodes per band by the evaluator's macro-F1 (src.train.evaluate.macro_f1, over the question's options),
  against a null: the same classifier fitted on train labels shuffled across episodes, averaged over NULL_SHUFFLES
  shuffles. Comparing with the null, not with the constant majority answer, removes what balanced weights alone buy
  under macro-F1. A question is shortcut-prone in a band if the speaker-structure probe beats its null by at least
  SHORTCUT_MARGIN macro-F1 and is at least as accurate as the majority answer there. The opening-words probe is
  reported against its own null but never excludes a question: the opening is part of the episode, and a short
  episode's first 500 words are a large share of it. Held-out questions are probed too: their train-episode labels exist
  in the ledger, and a probe is not a model in the grid.
- Evidence-position audit. Where the gold's cited lines fall in the episode (first, middle, last third of its lines),
  per band and per question, over every label of every selected episode. The rule: if at 12-16k the middle third holds
  under half the first third's share while at 0-4k it does not, the labeller is read as under-reading the middle of
  long episodes.
"""

import collections
import json
import random
from pathlib import Path

from src.synthesis_v2 import OUT_DIR
from src.synthesis_v2.split import LEDGER, SPLITS, episode_of

OPENING_WORDS = 500
SHORTCUT_MARGIN = 0.15
NULL_SHUFFLES = 5
OUT = OUT_DIR / "probes"


def _structure(lines: list[str], band: str) -> dict[str, float | str]:
    labels = [ln.partition(": ")[0] for ln in lines]
    distinct = set(labels)
    return {"hosts": sum(1 for x in distinct if x.startswith("host_")),
            "guests": sum(1 for x in distinct if x.startswith("guest_")),
            "others": sum(1 for x in distinct if not x.startswith(("host_", "guest_"))),
            "band": band}


def _opening(lines: list[str]) -> str:
    words = " ".join(ln.partition(": ")[2] for ln in lines).split()
    return " ".join(words[:OPENING_WORDS])


def _position(line: int, n_lines: int) -> str:
    third = (line - 1) * 3 // max(1, n_lines)
    return ("first", "middle", "last")[min(2, third)]


def load(ledger: Path = LEDGER, splits: Path = SPLITS) -> tuple[dict, dict]:
    """(episodes: id -> side, band, opening text, structure, n_lines; labels: (episode, qid) -> answer, evidence, options)."""
    sides = json.loads(splits.read_text())["episodes"]
    eps: dict[str, dict] = {}
    labels: dict[tuple[str, str], dict] = {}
    for line in ledger.open():
        r = json.loads(line)
        ep = episode_of(r["id"])
        if ep not in eps:
            lines = r["transcript"]["variants"][0]["lines"]
            band = r["meta"]["stratum"]
            eps[ep] = {"side": sides[ep], "band": band, "opening": _opening(lines),
                       "structure": _structure(lines, band), "n_lines": len(lines)}
        q = r["question"]
        labels[(ep, q["id"])] = {"answer": r["label"]["answer"], "evidence": r["label"]["evidence"],
                                 "options": [o["value"] for o in q["options"]]}
    return eps, labels


def probe(eps: dict, labels: dict, held: set[str]) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from src.train.evaluate import macro_f1

    train = sorted(ep for ep, e in eps.items() if e["side"] == "train")
    bench = sorted(ep for ep, e in eps.items() if e["side"] == "benchmark")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)
    x_text_tr = tfidf.fit_transform([eps[e]["opening"] for e in train])
    x_text_be = tfidf.transform([eps[e]["opening"] for e in bench])
    dv = DictVectorizer()
    x_str_tr = dv.fit_transform([eps[e]["structure"] for e in train])
    x_str_be = dv.transform([eps[e]["structure"] for e in bench])
    bands = sorted({eps[e]["band"] for e in bench})
    out: dict = {}
    for qid in sorted({q for _, q in labels}):
        y_tr = [labels[(e, qid)]["answer"] for e in train]
        y_be = [labels[(e, qid)]["answer"] for e in bench]
        options = labels[(bench[0], qid)]["options"]
        counts = collections.Counter(y_tr)
        majority = min(counts, key=lambda a: (-counts[a], a))
        preds: dict[str, list] = {"majority": [majority] * len(bench)}
        nulls: dict[str, list[list]] = {}
        for name, (xtr, xbe) in {"opening_words": (x_text_tr, x_text_be), "speaker_structure": (x_str_tr, x_str_be)}.items():
            if len(counts) < 2:
                preds[name] = [majority] * len(bench)
                nulls[name] = [[majority] * len(bench)]
                continue

            def fit(y, xtr=xtr, xbe=xbe):
                clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
                clf.fit(xtr, y)
                return list(clf.predict(xbe))
            preds[name] = fit(y_tr)
            nulls[name] = []
            for k in range(NULL_SHUFFLES):
                y_sh = list(y_tr)
                random.Random(f"{qid}|{name}|{k}").shuffle(y_sh)
                nulls[name].append(fit(y_sh))
        per_band: dict = {}
        for band in bands:
            idx = [i for i, e in enumerate(bench) if eps[e]["band"] == band]

            def score(p, idx=idx, y_be=y_be, options=options):
                rows = [{"gold": y_be[i], "pred": p[i], "options": options} for i in idx]
                return macro_f1(rows), sum(r["gold"] == r["pred"] for r in rows) / len(rows)
            scores: dict = {}
            for name, p in preds.items():
                f1, acc = score(p)
                scores[name] = {"macro_f1": round(f1, 4), "accuracy": round(acc, 4)}
                if name in nulls:
                    null_f1 = sum(score(q)[0] for q in nulls[name]) / len(nulls[name])
                    scores[name]["null_macro_f1"] = round(null_f1, 4)
                    scores[name]["margin_over_null"] = round(f1 - null_f1, 4)
            s_ = scores["speaker_structure"]
            prone = s_["margin_over_null"] >= SHORTCUT_MARGIN and s_["accuracy"] >= scores["majority"]["accuracy"]
            per_band[band] = {**scores, "shortcut_prone": prone}
        out[qid] = {"held_out": qid in held, "bands": per_band}
    return out


def positions(eps: dict, labels: dict) -> dict:
    by_band: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_q_band: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for (ep, qid), lab in labels.items():
        n = eps[ep]["n_lines"]
        for line in lab["evidence"]:
            pos = _position(line, n)
            by_band[eps[ep]["band"]][pos] += 1
            by_q_band[(qid, eps[ep]["band"])][pos] += 1

    def shares(c: collections.Counter) -> dict:
        t = sum(c.values()) or 1
        return {k: round(c[k] / t, 4) for k in ("first", "middle", "last")} | {"cited_lines": sum(c.values())}

    bands = {b: shares(c) for b, c in sorted(by_band.items())}
    under = {b: s["middle"] < 0.5 * s["first"] for b, s in bands.items()}
    return {"by_band": bands, "middle_under_half_first": under,
            "rule_fires": bool(under.get("12-16k")) and not under.get("0-4k"),
            "by_question": {f"{q}|{b}": shares(c) for (q, b), c in sorted(by_q_band.items())}}


def markdown(report: dict) -> str:
    lines = ["# v2 shortcut probes and evidence-position audit", "",
             (f"Probes fitted on the train episodes, scored on the benchmark episodes; macro-F1 over each question's options; "
              f"each classifier against the same classifier on shuffled labels (mean of {NULL_SHUFFLES}). Shortcut-prone: the "
              f"speaker-structure probe at least {SHORTCUT_MARGIN} above its null and at least as accurate as the majority "
              "answer. The opening-words probe is a partial-reading baseline and never excludes a question."), "",
             "| question | held out | band | majority F1 / acc | structure F1 / acc / over null | opening F1 / over null | shortcut-prone |",
             "|---|---|---|---|---|---|---|"]
    for qid, q in report["probes"].items():
        for band, s in q["bands"].items():
            st, op, mj = s["speaker_structure"], s["opening_words"], s["majority"]
            lines.append(f"| `{qid}` | {'yes' if q['held_out'] else ''} | {band} | {mj['macro_f1']:.3f} / {mj['accuracy']:.3f} | "
                         f"{st['macro_f1']:.3f} / {st['accuracy']:.3f} / {st['margin_over_null']:+.3f} | "
                         f"{op['macro_f1']:.3f} / {op['margin_over_null']:+.3f} | {'**yes**' if s['shortcut_prone'] else ''} |")
    flagged = report["shortcut_prone"]
    lines += ["", f"Shortcut-prone (question, band) pairs: {len(flagged)}: " + (", ".join(f"`{f}`" for f in flagged) or "none"), "",
              "## Evidence position (share of cited lines by third of the episode)", "",
              "| band | first | middle | last | cited lines | middle under half of first |", "|---|---|---|---|---|---|"]
    pos = report["positions"]
    for band, s in pos["by_band"].items():
        lines.append(f"| {band} | {s['first']:.3f} | {s['middle']:.3f} | {s['last']:.3f} | {s['cited_lines']} | "
                     f"{'yes' if pos['middle_under_half_first'][band] else 'no'} |")
    lines += ["", f"Pre-registered rule (12-16k under half while 0-4k is not): **{'fires' if pos['rule_fires'] else 'does not fire'}**.", ""]
    return "\n".join(lines)


def main() -> None:
    held = {json.loads(line)["id"] for line in (OUT_DIR / "questions.jsonl").open() if json.loads(line).get("held_out")}
    eps, labels = load()
    report = {"probes": probe(eps, labels, held), "positions": positions(eps, labels),
              "opening_words": OPENING_WORDS, "shortcut_margin": SHORTCUT_MARGIN, "null_shuffles": NULL_SHUFFLES}
    report["shortcut_prone"] = sorted(f"{q}|{b}" for q, v in report["probes"].items()
                                      for b, s in v["bands"].items() if s["shortcut_prone"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    (OUT / "report.md").write_text(markdown(report))
    print(markdown(report).split("## Evidence")[0][-600:])
    print(json.dumps(report["positions"]["by_band"], indent=1), "rule fires:", report["positions"]["rule_fires"])


if __name__ == "__main__":
    main()
