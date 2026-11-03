"""The pilot's analysis: per question and per stratum, agreement between two labellers, answer distribution, evidence
position and cost, with the pre-registered cut and stop rules applied.

    uv run python -m src.synthesis_v2.pilot_report data/labelled_data_v2/pilot/opus_k7.jsonl \\
        data/labelled_data_v2/pilot/sonnet_k7.jsonl --out data/labelled_data_v2/pilot/report.md

The first file is the gold labeller's pass (Opus, D67); the second the comparison pass (Sonnet). Rules (D67):
- a question is measurable only when every in-cap stratum holds at least 20 pairs whose question text matches
  between the passes, compared on text, description and options (a question edited between passes is not the same
  question); an unmeasurable
  question is reported as such and is neither a survivor nor cut;
- a question is cut if Opus-against-Sonnet Cohen's kappa is under 0.40 in any in-cap stratum, or if the gold pass
  answers `none` on over 75% of episodes, or one answer on over 90% of episodes. A declared rare-event detector
  (`question_bank.RARE_EVENT`) is exempt from the two skew rules, and its agreement rule is raw agreement under 0.80
  rather than kappa, because kappa is undefined or near zero when the gold answer is almost always the same; it is
  measurable only with at least three gold positives, and its agreement on those positives is reported;
- the project stops if the bank-mean kappa in the longest stratum (over every measurable non-rare question) is
  under 0.40, or more than 0.20 below the shortest stratum's (8-16k against 2-4k in the first two pilots, 12-16k
  against 0-4k from round four, D79): labeller consistency would then be a function of length, the
  confound the dataset exists to measure.
Everything reported is measured on the two files; nothing is estimated.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from src.synthesis_v2 import question_bank as bank
from src.synthesis_v2 import read_jsonl
from src.synthesis_v2.episodes import STRATA as STRATA_NOW

STRATA = tuple(STRATA_NOW)  # the strata of the labels being analysed; the first two pilots used ("2-4k", "4-8k", "8-16k")
MIN_PAIRS = 20
KAPPA_CUT = 0.40
RARE_AGREEMENT_CUT = 0.80
MIN_RARE_POSITIVES = 3  # a rare-event question with fewer gold positives than this is unmeasurable, not a survivor
NONE_CUT = 0.75
TOP_CUT = 0.90
STOP_KAPPA = 0.40
STOP_GAP = 0.20


def load(path: Path) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(path):
        qd = r["question"]  # the whole question the labeller saw: a pair only counts when both passes saw the same one
        desc = qd.get("description") or ""  # records before 2026-11-02 store the shared definitions inside it
        desc = desc[len(bank.VOCAB):].strip() if desc.startswith(bank.VOCAB) else desc
        text = json.dumps([qd["text"], desc, qd.get("options")], sort_keys=True)
        out[r["id"]] = {"qid": qd["id"], "text": text, "answer": r["label"]["answer"],
                        "evidence": r["label"]["evidence"], "n_lines": len(r["transcript"]["variants"][0]["lines"]),
                        "stratum": r["meta"].get("stratum"), "cost": r["generation_info"]["cost_usd"],
                        "episode": r["id"].split("::")[0]}
    return out


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over (gold, other) answer pairs; None when undefined (no pairs, or no variation in either)."""
    if not pairs:
        return None
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    pa = collections.Counter(a for a, _ in pairs)
    pb = collections.Counter(b for _, b in pairs)
    expected = sum(pa[k] / n * pb.get(k, 0) / n for k in pa)
    if expected >= 1.0:
        return 1.0 if agree == 1.0 else None
    return (agree - expected) / (1 - expected)


def analyse(gold: dict[str, dict], other: dict[str, dict]) -> dict:
    by_q: dict[str, dict] = {}
    text_mismatch = 0
    for qid in sorted({r["qid"] for r in gold.values()}):
        rows = [r for r in gold.values() if r["qid"] == qid and r["stratum"] in STRATA]  # out-of-strata labels never count
        q: dict = {"n": len(rows), "answers": dict(collections.Counter(r["answer"] for r in rows).most_common()),
                   "kappa": {}, "agreement": {}, "pairs": {}, "evidence_last_third": None, "evidence_empty": None,
                   "cost_per_label": round(statistics.mean(r["cost"] for r in rows), 5) if rows else None}
        pairs_all = []
        for s in STRATA:
            pairs = []
            for k, r in gold.items():
                if r["qid"] != qid or r["stratum"] != s or k not in other:
                    continue
                if other[k].get("text", r["text"]) != r["text"]:
                    text_mismatch += 1
                    continue
                pairs.append((r["answer"], other[k]["answer"]))
            pairs_all += pairs
            q["pairs"][s] = len(pairs)
            ks = kappa(pairs)
            q["kappa"][s] = None if ks is None else round(ks, 3)
            q["agreement"][s] = round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None
        ka = kappa(pairs_all)
        q["kappa"]["all"] = None if ka is None else round(ka, 3)
        q["agreement"]["all"] = round(sum(a == b for a, b in pairs_all) / len(pairs_all), 3) if pairs_all else None
        with_ev = [r for r in rows if r["evidence"]]
        if with_ev:
            last = [sum(1 for e in r["evidence"] if e > 2 * r["n_lines"] / 3) / len(r["evidence"]) for r in with_ev]
            q["evidence_last_third"] = round(statistics.mean(last), 3)
        q["evidence_empty"] = round(sum(1 for r in rows if not r["evidence"]) / len(rows), 3) if rows else None
        # cut rules
        rare = qid in bank.RARE_EVENT
        q["rare_event"] = rare
        gold_positives = sum(c for a, c in q["answers"].items() if a != "none")
        q["gold_positives"] = gold_positives
        if rare:  # agreement on the episodes where the gold says the event happened
            pos_pairs = [(r["answer"], other[k]["answer"]) for k, r in gold.items()
                         if r["qid"] == qid and r["answer"] != "none" and k in other and other[k].get("text", r["text"]) == r["text"]]
            q["positive_agreement"] = round(sum(a == b for a, b in pos_pairs) / len(pos_pairs), 3) if pos_pairs else None
        q["measurable"] = all(q["pairs"][s] >= MIN_PAIRS for s in STRATA) and (not rare or gold_positives >= MIN_RARE_POSITIVES)
        reasons = []
        if q["measurable"]:
            for s in STRATA:
                if rare:
                    ag = q["agreement"][s]
                    if ag is not None and ag < RARE_AGREEMENT_CUT:
                        reasons.append(f"agreement {ag} in {s}")
                else:
                    kv = q["kappa"][s]
                    if kv is not None and kv < KAPPA_CUT:
                        reasons.append(f"kappa {kv} in {s}")
        share_none = q["answers"].get("none", 0) / q["n"] if q["n"] else 0
        top_share = max(q["answers"].values()) / q["n"] if q["n"] else 0
        if share_none > NONE_CUT and not rare:
            reasons.append(f"none on {share_none:.0%}")
        if top_share > TOP_CUT and not rare:
            reasons.append(f"one answer on {top_share:.0%}")
        q["cut"] = reasons
        q["held_out"] = qid in bank.HELD_OUT or qid in bank.PROVISIONAL_HELD_OUT
        by_q[qid] = q
    # stop rule on the bank mean per stratum, over every measurable non-rare question (cut ones included: a bank whose
    # questions all degrade on the long stratum must trigger the stop, not escape it by being cut one by one)
    means = {}
    for s in STRATA:
        ks = [q["kappa"][s] for q in by_q.values()
              if q["measurable"] and not q["rare_event"] and q["kappa"][s] is not None]
        means[s] = round(statistics.mean(ks), 3) if ks else None
    stop: list[str] = []
    short, long = STRATA[0], STRATA[-1]
    if means[long] is None or means[short] is None:
        stop.append(f"not evaluable: no measurable non-rare question in {short} or {long}")
    if means[long] is not None:
        if means[long] < STOP_KAPPA:
            stop.append(f"bank-mean kappa in {long} is {means[long]}")
        if means[short] is not None and means[short] - means[long] > STOP_GAP:
            stop.append(f"{long} is {means[short] - means[long]:.2f} below {short}")
    episodes = {r["episode"] for r in gold.values()}
    unknown_stratum = sum(1 for r in gold.values() if r["stratum"] not in STRATA)
    cost = {"gold_on_labels": round(sum(r["cost"] for r in gold.values()), 2),
            "other_on_labels": round(sum(r["cost"] for r in other.values()), 2),
            "gold_per_episode_on_labels": round(sum(r["cost"] for r in gold.values()) / len(episodes), 4) if episodes else None}
    return {"episodes": len(episodes), "labels_gold": len(gold), "labels_other": len(other), "bank_mean_kappa": means,
            "stop": stop, "cost": cost, "text_mismatches_excluded": text_mismatch,
            "gold_labels_outside_strata": unknown_stratum, "questions": by_q,
            "survivors": [q for q, v in by_q.items() if v["measurable"] and not v["cut"]],
            "unmeasurable": [q for q, v in by_q.items() if not v["measurable"]],
            "cut": {q: v["cut"] for q, v in by_q.items() if v["cut"]}}


def markdown(rep: dict) -> str:
    lines = [f"# Pilot report: {rep['episodes']} episodes, {rep['labels_gold']} gold labels, {rep['labels_other']} comparison labels", "",
             f"Bank-mean kappa by stratum: {rep['bank_mean_kappa']}. Stop rule: {('NOT EVALUABLE: ' if any(x.startswith('not evaluable') for x in rep['stop']) else 'TRIGGERED: ') + '; '.join(rep['stop']) if rep['stop'] else 'not triggered'}.",
             f"Cost on the labels: {rep['cost']}. Every call, from the call logs: {rep.get('calls', 'not read')}.", "",
             f"| question | HO | n | answers | kappa {' / '.join(STRATA)} / all | agreement all | evidence in last third | empty evidence | cut |",
             "|---|---|---|---|---|---|---|---|---|"]
    for qid, q in rep["questions"].items():
        k = " / ".join("-" if q["kappa"][s] is None else f"{q['kappa'][s]:.2f}" for s in (*STRATA, "all"))
        lines.append(f"| `{qid}` | {'HO' if q['held_out'] else ''} | {q['n']} | {q['answers']} | {k} | {q['agreement']['all']} | "
                     f"{q['evidence_last_third']} | {q['evidence_empty']} | {'; '.join(q['cut']) or ''} |")
    lines += ["", f"Survivors ({len(rep['survivors'])}): {', '.join(rep['survivors'])}", "",
              f"Cut ({len(rep['cut'])}): " + "; ".join(f"{q} ({', '.join(r)})" for q, r in rep["cut"].items()), "",
              f"Unmeasurable, fewer than {MIN_PAIRS} matched pairs in some stratum ({len(rep['unmeasurable'])}): "
              + ", ".join(rep["unmeasurable"]), (f"Pairs excluded for a question mismatch (text, description or options): {rep['text_mismatches_excluded']}. "
               f"Gold labels whose stratum is not one of {list(STRATA)} (never paired): {rep['gold_labels_outside_strata']}.")]
    return "\n".join(lines) + "\n"


def calls_summary(path: Path) -> dict:
    """Total spend and cache reads from a `.calls.jsonl` log (every call, failed ones included)."""
    calls = read_jsonl(path)
    usage = [c.get("usage") or {} for c in calls]
    return {"calls": len(calls), "failed_calls": sum(1 for c in calls if "failed" in c),
            "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in calls), 2),
            "cache_read_tokens": sum(int(u.get("cache_read_input_tokens") or 0) for u in usage),
            "input_tokens": sum(int(u.get("input_tokens") or 0) for u in usage),
            "calls_with_cache_reads": sum(1 for u in usage if int(u.get("cache_read_input_tokens") or 0) > 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("gold", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strata", default=None, help="comma-separated strata of the labels, shortest first (default: the "
                    "current ones; the first two pilots used 2-4k,4-8k,8-16k)")
    a = ap.parse_args()
    if a.strata:
        global STRATA
        STRATA = tuple(s.strip() for s in a.strata.split(","))
    rep = analyse(load(a.gold), load(a.other))
    rep["calls"] = {"gold": calls_summary(a.gold.with_suffix(".calls.jsonl")),
                    "other": calls_summary(a.other.with_suffix(".calls.jsonl"))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(markdown(rep))
    a.out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(markdown(rep))


if __name__ == "__main__":
    main()
