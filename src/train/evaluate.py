"""Score generated answers against a split file and write the results.

    uv run python -m src.train.evaluate <split.jsonl> <outputs.jsonl> <out_dir> [--wandb-run ID --wandb-project P]

Outputs: one JSON line per (record id, variant) with the model's text and prompt length. The split file's
records carry the gold label: for the benchmark two of them (D61, 2026-09-29): `label`, the teacher's (the
training labeller's own label, the headline gold, scored over every pair) and `opus_label`, Opus's as a second
labeller, scored over the pairs that have one (a selected subset: the general_qa pairs the teacher and a second local
labeller disagreed on, four that had no second opinion, and the family pairs the teacher called `fail`). Every variant a
record carries is scored, so a generation that skipped a variant scores it as empty; generate both. Scoring is strict: the whole output
must be one JSON object with the answer among the question's options, a valid sorted evidence list of line
numbers and a summary string; a case-variant of an option earns partial credit but never counts as correct.
Answer accuracy and macro-F1 over the question's options, evidence precision / recall / F1 / exact match,
format validity, per dataset, family, cell (seen / unseen question, and unseen without the twinned held-out
questions) and length bucket, plus the transcript-blind floors (each question's majority gold answer with an
empty evidence list; the Opus section carries both the Opus-majority and the teacher-majority floor) that every number
is read against.
Writes metrics.json and results.md; --wandb-run logs the headline numbers to that training run.
"""

import argparse
import collections
import json
import re
from pathlib import Path

INVALID = "<invalid>"
PRIORITY = ("vulnerability", "complaint", "eod")
# held-out questions with a near-verbatim or strong semantic twin among the training questions (the benchmark audits of
# 2026-09-29: the split ties `tm-` rewrites to their `gen-` originals but not the `aci-` clinical rewrites, and eight more pairs
# share a concept). `cell=unseen_q_no_twin` scores the held-out questions without them: the number to read as zero-shot.
TWINNED_HELD_OUT = {
    "aci-15-timescales", "aci-21-next-steps", "aci-26-full-name", "aci-34-appointment-quoted", "aci-35-precautions-volunteered",
    "aci-39-plan-recap", "aci-63-tests-explained", "gen-16-read-back", "tm-16-read-back", "gen-18-empathy", "gen-23-explicit-consent",
    "tm-23-explicit-consent", "aci-75-empathy-statement", "spk-36-question-dodged", "spk-29-personal-disclosure",
    "gen-02-account-holder", "gen-55-date-of-birth", "gen-07-needs-established", "spk-35-audience-assumed", "tm-29-preference-honoured",
}
# Labelled data v2 (podcast family). Its one held-out twin (per-03, answerable from trained per-05) moved to training
# before the split was frozen (twin audit, 2026-11-05), so `unseen_q_no_twin` equals `unseen_q` on v2 by design. Two
# held-out questions share a gate with trained questions: their `none` answer is settled by a trained question's answer
# or by the speaker structure. `cell=unseen_q_ungated` scores the held-out cell without those gated pairs: per-06 only on
# episodes with a story (gold per-05 is not `none`), nam-02 only on episodes with a rendered speaker who is not a host.
V2_STORY_GATE = ("v2-per-06-story-questioned", "v2-per-05-storytellers", "none")
V2_NON_HOST_GATE = "v2-nam-02-name-taken-up"
BUCKETS = [(0, 2048, "<=2k"), (2048, 4096, "2-4k"), (4096, 8192, "4-8k"), (8192, 16384, "8-16k"), (16384, 10**9, ">16k")]


def parse(text: str) -> dict | None:
    """Strict: the whole output, less surrounding whitespace, must be one JSON object. No repair of surrounding text,
    fences or truncation (the benchmark scores format validity strictly, and a lenient parser hides the raw error rate)."""
    try:
        o = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return o if isinstance(o, dict) else None


def _norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def score_pair(r: dict, out: dict, gold_key: str = "label") -> dict:
    """One scored row: the output against the record's `gold_key` label (`label`, the teacher's, or `opus_label`)."""
    q = r["question"]
    values = [o["value"] for o in q["options"]]
    n_lines = len(r["transcript"]["variants"][0]["lines"])
    gold_label = r[gold_key]
    gold_ev = set(gold_label["evidence"])
    gold_tags = set(gold_label.get("tags", []))
    o = parse(out["text"])
    ans = o.get("answer") if o else None
    if isinstance(ans, str) and ans in values:
        status, pred = "exact", ans
    elif isinstance(ans, str) and _norm(ans) in {_norm(v): v for v in values}:
        status, pred = "recovered", {_norm(v): v for v in values}[_norm(ans)]
    else:
        status, pred = "invalid", INVALID
    ev = o.get("evidence") if o else None
    ev_valid = (isinstance(ev, list) and all(isinstance(e, int) and not isinstance(e, bool) for e in ev)
                and all(1 <= e <= n_lines for e in ev) and ev == sorted(set(ev)))
    pev = set(ev) if ev_valid and isinstance(ev, list) else None
    if pev is None:
        prec = rec = f1 = exact = 0.0
    else:
        inter = len(pev & gold_ev)
        prec = (inter / len(pev)) if pev else (1.0 if not gold_ev else 0.0)
        rec = (inter / len(gold_ev)) if gold_ev else (1.0 if not pev else 0.0)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        exact = float(pev == gold_ev)
    tag_j = None
    if q.get("tags"):
        pt = o.get("tags") if o else None
        if isinstance(pt, list) and all(isinstance(t, str) for t in pt):
            ps = set(pt)  # a tag outside the question's vocabulary is a wrong tag, not an ignored one
            tag_j = 1.0 if not ps and not gold_tags else len(ps & gold_tags) / len(ps | gold_tags)
        else:
            tag_j = 0.0
    return {"id": r["id"], "variant": out["variant"], "dataset": r["dataset"], "track": r["track"], "family": q["family"],
            "qid": q["id"], "cell": r.get("cell", "seen_q"), "prompt_tokens": out.get("prompt_tokens"), "gold_key": gold_key,
            "band": (r.get("meta") or {}).get("stratum"),  # v2's length band (the selection's), absent elsewhere
            "difficult": (r.get("meta") or {}).get("difficult"), "over_cap": (r.get("_over_cap") or {}).get(out["variant"]),
            # a recovered answer is a failure with partial credit (`credit`), never a correct prediction: the answer
            # metrics see it as invalid, `pred_raw` keeps what it would have been
            "gold": gold_label["answer"], "pred": pred if status == "exact" else INVALID, "pred_raw": pred, "status": status,
            "parsed": o is not None, "options": values,
            "credit": {"exact": 1.0, "recovered": 0.3, "invalid": 0.0}[status],
            "ev_valid": bool(ev_valid), "ev_prec": prec, "ev_rec": rec, "ev_f1": f1, "ev_exact": exact,
            "gold_empty": not gold_ev, "tag_jaccard": tag_j,
            "format_valid": status == "exact" and bool(ev_valid) and isinstance((o or {}).get("summary"), str)}


def macro_f1(rows: list[dict], key: str = "pred") -> float:
    """Macro-F1 over the answer values the questions supplied (every option of every question in the rows, plus any
    gold value), so a wrong prediction of an option that never occurs as gold is a false positive for that class. A
    class with neither gold nor predictions has no F1 and is left out of the mean."""
    classes = sorted({r["gold"] for r in rows} | {v for r in rows for v in r.get("options", [])})
    f1s = []
    for c in classes:
        tp = sum(1 for r in rows if r["gold"] == c and r[key] == c)
        fp = sum(1 for r in rows if r["gold"] != c and r[key] == c)
        fn = sum(1 for r in rows if r["gold"] == c and r[key] != c)
        if 2 * tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else 0.0


def mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(rows: list[dict]) -> dict:
    ne = [r for r in rows if not r["gold_empty"]]
    ge = [r for r in rows if r["gold_empty"]]
    return {"n": len(rows), "parsed": mean(rows, "parsed"), "format_valid": mean(rows, "format_valid"),
            "answer_exact": mean(rows, "credit") and sum(r["status"] == "exact" for r in rows) / len(rows),
            "answer_recovered": sum(r["status"] == "recovered" for r in rows) / len(rows),
            "answer_invalid": sum(r["status"] == "invalid" for r in rows) / len(rows),
            "answer_credit": mean(rows, "credit"),
            "accuracy": sum(r["status"] == "exact" and r["pred"] == r["gold"] for r in rows) / len(rows),
            "macro_f1": macro_f1(rows), "majority_macro_f1": macro_f1(rows, "majority"),
            "ev_valid": mean(rows, "ev_valid"), "ev_precision": mean(rows, "ev_prec"), "ev_recall": mean(rows, "ev_rec"),
            "ev_f1": mean(rows, "ev_f1"), "ev_exact": mean(rows, "ev_exact"),
            "gold_empty_share": len(ge) / len(rows), "gold_empty_matched": mean(ge, "ev_exact"),
            "ev_precision_nonempty": mean(ne, "ev_prec"), "ev_recall_nonempty": mean(ne, "ev_rec"),
            "tag_jaccard": mean(rows, "tag_jaccard"),
            "pred_distribution": dict(collections.Counter(r["pred"] for r in rows)),
            "gold_distribution": dict(collections.Counter(r["gold"] for r in rows))}


def bucket(tokens: int | None) -> str:
    if tokens is None or tokens <= 0:
        return "unknown"
    return next(name for lo, hi, name in BUCKETS if lo < tokens <= hi)


def confusion(rows: list[dict]) -> dict:
    m: dict = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        m[r["gold"]][r["pred"]] += 1
    return {g: dict(c) for g, c in m.items()}


def _kinds(r: dict) -> set[str]:
    """The transcript variants a record has: the only variants an output may name (`labelled` is a training-side alias)."""
    return {x["kind"] for x in r["transcript"]["variants"]}


def _load_outputs(gold: dict, outputs: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """(raw outputs, one per (id, variant), with every missing pair of every variant the records carry filled by an empty
    output; the pairs that had one). A pair with no output is scored as empty, so a partial, interrupted or entirely absent
    generation of a variant cannot inflate a slice; an output for a variant the record does not have is ignored."""
    seen: set[tuple[str, str]] = set()
    outs = []
    for line in outputs.open():
        o = json.loads(line)
        if o["id"] in gold and o["variant"] in _kinds(gold[o["id"]]) and (o["id"], o["variant"]) not in seen:
            seen.add((o["id"], o["variant"]))  # one output per pair: a duplicate never counts twice
            outs.append(o)
    expected = sorted({k for r in gold.values() for k in _kinds(r)})
    for variant in expected:
        for r in gold.values():
            if (r["id"], variant) not in seen and variant in _kinds(r):
                outs.append({"id": r["id"], "variant": variant, "text": "", "prompt_tokens": None})
    return outs, seen


def _floor(gold: dict, rows: list[dict], variant: str, gold_key: str, answer_of) -> dict:
    """The transcript-blind floor over `rows`: `answer_of(row)` as the answer, an empty evidence list and no tags, scored
    exactly as a model output would be."""
    f_rows = [score_pair(gold[r["id"]], {"variant": variant, "prompt_tokens": r["prompt_tokens"],
                                         "text": json.dumps({"evidence": [], "answer": answer_of(r), "summary": "", "tags": []})},
                         gold_key) for r in rows]
    for f_, r in zip(f_rows, rows):
        f_["majority"] = answer_of(r)
    return summarise(f_rows)


def _slices(gold: dict, rows: list[dict], variant: str, gold_key: str, majority: dict, teacher_majority: dict | None = None,
            band_majority: dict | None = None) -> dict:
    """Every slice of one variant's rows under one gold: overall, dataset, family, cell (plus the held-out cell without the
    twinned questions), the transcript-blind floor, length buckets and the rare-event families per dataset."""
    slices: dict = {"overall": summarise(rows)}
    for key in ("dataset", "family", "cell"):
        for val in sorted({r[key] for r in rows}):
            slices[f"{key}={val}"] = summarise([r for r in rows if r[key] == val])
    # per-question slices where the question set is small (ECtHR: one binary question per article); the compliance
    # benchmark's hundreds of questions are not sliced this way
    qids = sorted({r["qid"] for r in rows})
    if len(qids) <= MAX_QID_SLICES:
        for qid in qids:
            slices[f"qid={qid}"] = summarise([r for r in rows if r["qid"] == qid])
    for key in ("difficult", "over_cap"):  # record-level flags: a dataset's own difficulty flag; a prompt cut to the cap (--cap-tokens)
        vals = sorted({r[key] for r in rows if r.get(key) is not None})
        for val in vals:
            slices[f"{key}={val}"] = summarise([r for r in rows if r.get(key) == val])
    no_twin = [r for r in rows if r["cell"] == "unseen_q" and r["qid"] not in TWINNED_HELD_OUT]
    if no_twin:
        slices["cell=unseen_q_no_twin"] = summarise(no_twin)
    if any(r.get("gated") for r in rows):  # v2: the held-out cell without the pairs a trained gate settles
        slices["cell=unseen_q_ungated"] = summarise([r for r in no_twin if not r.get("gated")])
    if band_majority and len(qids) > MAX_QID_SLICES:  # v2: too many questions for every one, but its 12 held-out ones are each a claim
        for qid in sorted({r["qid"] for r in rows if r["cell"] == "unseen_q"}):
            sub = [r for r in rows if r["qid"] == qid]
            slices[f"qid={qid}"] = summarise(sub)
            if any(r.get("gated") for r in sub):
                slices[f"qid={qid}/ungated"] = summarise([r for r in sub if not r.get("gated")])
    if band_majority:  # v2's length bands: every band, every band by cell, and the floor of each from its own majorities
        for band in sorted({r["band"] for r in rows if r.get("band")}):
            in_band = [r for r in rows if r.get("band") == band]
            slices[f"band={band}"] = summarise(in_band)
            slices[f"band={band}/floor=majority_answer_in_band"] = _floor(
                gold, in_band, variant, gold_key, lambda r, b=band: band_majority[(r["qid"], b)])
            for cell in sorted({r["cell"] for r in in_band}):
                sub = [r for r in in_band if r["cell"] == cell]
                slices[f"band={band}/cell={cell}"] = summarise(sub)
                slices[f"band={band}/cell={cell}/floor=majority_answer_in_band"] = _floor(
                    gold, sub, variant, gold_key, lambda r, b=band: band_majority[(r["qid"], b)])
    # the floor: each question's majority gold answer, no transcript read, an empty evidence list and no tags, scored
    # exactly as a model output would be (so an empty key still matches empty gold); every number above is read against it
    slices["floor=majority_answer"] = _floor(gold, rows, variant, gold_key, lambda r: majority[r["qid"]])
    if teacher_majority is not None:  # on the Opus subset, the teacher's own majority answers: the floor an imitator of the teacher gets
        t_rows = [score_pair(gold[r["id"]], {"variant": variant, "prompt_tokens": r["prompt_tokens"],
                                             "text": json.dumps({"evidence": [], "answer": teacher_majority[r["qid"]], "summary": "", "tags": []})},
                             gold_key) for r in rows]
        for f_ in t_rows:
            f_["majority"] = teacher_majority[f_["qid"]]
        slices["floor=teacher_majority_answer"] = summarise(t_rows)
    for name in [b[2] for b in BUCKETS] + ["unknown"]:  # `unknown`: pairs with no output, so they stay visible by length too
        sub = [r for r in rows if bucket(r["prompt_tokens"]) == name]
        if sub:
            slices[f"length={name}"] = summarise(sub)
    for fam in PRIORITY:
        for ds in sorted({r["dataset"] for r in rows if r["family"] == fam}):
            sub = [r for r in rows if r["family"] == fam and r["dataset"] == ds]
            slices[f"priority={fam}/{ds}"] = {**summarise(sub), "confusion": confusion(sub)}
    return slices


def _majority(answers: list[str]) -> str:
    """The most frequent answer; a tie goes to the alphabetically first, so the floor is the same on every run."""
    counts = collections.Counter(answers)
    return min(counts, key=lambda a: (-counts[a], a))


def v2_gated(gold: dict, gold_key: str = "label") -> set[str]:
    """Record ids of v2 held-out pairs whose answer a trained question or the speaker structure already settles (see
    V2_STORY_GATE, V2_NON_HOST_GATE): per-06 on an episode whose gold per-05 is `none`, nam-02 on an episode where every
    rendered speaker label is `host_*`. Empty on other datasets."""
    held_q, gate_q, gate_none = V2_STORY_GATE
    out = set()
    for i, r in gold.items():
        qid = r["question"]["id"]
        ep = i.split("::")[0]
        if qid == held_q:
            g = gold.get(f"{ep}::{gate_q}")
            if g is None:
                raise SystemExit(f"{i}: the story gate needs {gate_q} on the same episode, which the split does not hold")
            if (g.get(gold_key) or {}).get("answer") == gate_none:
                out.add(i)
        elif qid == V2_NON_HOST_GATE:  # the speakers the model sees: every line's label, as rendered
            labels = {ln.partition(": ")[0] for ln in r["transcript"]["variants"][0]["lines"]}
            if labels and all(lab.startswith("host_") for lab in labels):
                out.add(i)
    return out


def _by_gold(gold: dict, outs: list[dict], gold_key: str, teacher_majority: dict | None = None) -> tuple[dict, list[dict]]:
    """Per-variant slices under one gold, over the records that carry that label."""
    have = {i for i, r in gold.items() if r.get(gold_key)}
    rows = [score_pair(gold[o["id"]], o, gold_key) for o in outs if o["id"] in have]
    majority = {qid: _majority([gold[i][gold_key]["answer"] for i in have if gold[i]["question"]["id"] == qid])
                for qid in {gold[i]["question"]["id"] for i in have}}
    # a question asked of a single pair in the whole split (a dataset whose every question belongs to one document) would make
    # the floor its own gold; such questions take the majority gold over every pair that offers the same options instead
    # (the option-position prior), so the floor stays transcript-blind. Judged over the whole split, not over `have`:
    # a compliance question with one Opus-labelled pair among many teacher pairs keeps its per-question floor
    per_q = collections.Counter(r["question"]["id"] for r in gold.values())
    singles = {qid for qid in majority if per_q[qid] == 1}
    if singles:
        by_opts: dict[tuple, list[str]] = {}
        for i in have:
            by_opts.setdefault(tuple(o["value"] for o in gold[i]["question"]["options"]), []).append(gold[i][gold_key]["answer"])
        for i in have:
            qid = gold[i]["question"]["id"]
            if qid in singles:
                majority[qid] = _majority(by_opts[tuple(o["value"] for o in gold[i]["question"]["options"])])
    for r in rows:
        r["majority"] = majority[r["qid"]]
    # v2: the answer mix shifts with length, so each band also gets a floor from each question's majority within the band
    band_majority = {(qid, band): _majority([gold[i][gold_key]["answer"] for i in have if gold[i]["question"]["id"] == qid
                                             and (gold[i].get("meta") or {}).get("stratum") == band])
                     for qid, band in {(gold[i]["question"]["id"], (gold[i].get("meta") or {}).get("stratum")) for i in have}
                     if band is not None}
    # gating reads the headline gold (a second gold on a subset need not hold the gating question's label)
    gated = (v2_gated({i: gold[i] for i in have}, gold_key)
             if gold_key == "label" and any(r["family"] == "podcast" for r in rows) else set())
    for r in rows:
        r["gated"] = r["id"] in gated
    out: dict = {"n_pairs": len(have), "variants": {}}
    for variant in sorted({r["variant"] for r in rows}):
        vr = [r for r in rows if r["variant"] == variant]
        out["variants"][variant] = _slices(gold, vr, variant, gold_key, majority, teacher_majority, band_majority)
    return out, rows


def flag_over_cap(gold: dict, cap_tokens: int, tokenizer_dir: str) -> int:
    """Mark every record whose full rendered prompt (as training renders it, plus the separator) exceeds `cap_tokens`,
    so a benchmark read under head truncation can also be read on the records that were not cut. Counts the marked."""
    from transformers import AutoTokenizer

    from src.train.data import SEP, record_prompt
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    n = 0
    for r in gold.values():
        r["_over_cap"] = {}
        for v in r["transcript"]["variants"]:
            r["_over_cap"][v["kind"]] = len(tok(record_prompt(r, v["kind"]) + SEP)["input_ids"]) > cap_tokens
        n += any(r["_over_cap"].values())
    return n


def evaluate(split: Path, outputs: Path, cap_tokens: int | None = None, tokenizer_dir: str | None = None) -> dict:
    from src.train.data import check_questions
    check_questions(split.parent)  # a v2 benchmark rebuilt from another bank or split is refused, as in training
    gold = {json.loads(l)["id"]: json.loads(l) for l in split.open()}
    if cap_tokens:
        if not tokenizer_dir:
            raise SystemExit("--cap-tokens needs --tokenizer (the model directory whose tokenizer counted the prompts)")
        flag_over_cap(gold, cap_tokens, tokenizer_dir)
    with_prov = [r for r in gold.values() if r.get("provenance")]
    stale = [r["id"] for r in with_prov if r["provenance"].get("method") not in ("teacher", "dataset")]
    if with_prov and len(with_prov) != len(gold):  # a benchmark file: every record must say which gold its label is
        raise SystemExit(f"{split}: {len(gold) - len(with_prov)} records carry no provenance; a benchmark file must mark every "
                         "label as the teacher's or the dataset's own (rebuild it with src.synthesis.benchmark)")
    if stale:  # a benchmark file whose `label` is a mixed gold (built before 2026-09-29) would be reported as the teacher's
        raise SystemExit(f"{split}: {len(stale)} records carry a pre-D61 gold (provenance.method != teacher); rebuild the "
                         "benchmark with src.synthesis.benchmark before scoring")
    outs, seen = _load_outputs(gold, outputs)
    teacher, rows = _by_gold(gold, outs, "label")
    result: dict = {"split": str(split), "outputs": str(outputs), "pairs_in_split": len(gold), "scored": len(rows),
                    "missing_outputs": len(rows) - len(seen),  # pairs scored as empty because no output had them
                    "gold": "teacher", "variants": teacher["variants"]}
    # what `label` is: the training labeller's own label (D61) or, on a transfer dataset, the task's own gold
    if with_prov and all(r["provenance"].get("method") == "dataset" for r in with_prov):
        result["gold"], result["gold_version"] = "dataset", "dataset gold: the task's own labels"
    else:
        result["gold_version"] = "D61: teacher label headline, opus_label on its pairs"
    if any(r.get("opus_label") for r in gold.values()):
        teacher_majority = {qid: _majority([r["label"]["answer"] for r in gold.values() if r["question"]["id"] == qid])
                            for qid in {r["question"]["id"] for r in gold.values()}}
        opus, _ = _by_gold(gold, outs, "opus_label", teacher_majority)
        result["opus"] = opus  # Opus's gold over the pairs Opus labelled: {"n_pairs", "variants": {...}}
    # the variants do not cover the same pairs (SPoRC exists only as messy), so a clean-against-messy reading is only fair
    # on the pairs that have both: the shared set, overall and per dataset, under the teacher's gold
    by_id: dict[str, set[str]] = {}
    for r in rows:
        by_id.setdefault(r["id"], set()).add(r["variant"])
    variants = sorted({r["variant"] for r in rows})
    shared_ids = {i for i, vs in by_id.items() if vs >= set(variants)} if len(variants) > 1 else set()
    result["shared"] = {"n_pairs": len(shared_ids), "variants": {}}
    if shared_ids:
        for variant in variants:
            vr = [r for r in rows if r["variant"] == variant and r["id"] in shared_ids]
            slices = {"overall": summarise(vr)}
            for val in sorted({r["dataset"] for r in vr}):
                slices[f"dataset={val}"] = summarise([r for r in vr if r["dataset"] == val])
            result["shared"]["variants"][variant] = slices
    return result


MAX_QID_SLICES = 32
HEADLINE = ["n", "format_valid", "answer_exact", "accuracy", "macro_f1", "majority_macro_f1", "ev_precision", "ev_recall",
            "ev_f1", "ev_exact", "gold_empty_matched", "tag_jaccard"]


def fmt(x) -> str:
    return "" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _table(title: str, variants: dict) -> list[str]:
    lines = []
    for variant, slices in variants.items():
        header = "| slice | " + " | ".join(HEADLINE) + " |\n|---|" + "---|" * len(HEADLINE)
        lines.append(f"\n## {title}, variant `{variant}`\n\n{header}")
        for name, s in slices.items():
            if name.startswith("priority="):
                continue
            lines.append(f"| {name} | " + " | ".join(fmt(s.get(k)) for k in HEADLINE) + " |")
        for name, s in slices.items():
            if name.startswith("priority="):
                lines.append(f"\n**{name}** (n={s['n']}, macro-F1 {s['macro_f1']:.3f}, majority {s['majority_macro_f1']:.3f}); "
                             f"confusion gold→pred: `{json.dumps(s['confusion'])}`")
    return lines


def markdown(result: dict) -> str:
    head = (f"# Benchmark results\n\nsplit `{result['split']}`, outputs `{result['outputs']}`, "
            f"{result['scored']} outputs scored over {result['pairs_in_split']} pairs; {result['missing_outputs']} pairs had no output "
            "and were scored as empty.\n")
    lines = [head]
    lines += _table("Gold = dataset (the task's own labels)" if result.get("gold") == "dataset"
                    else "Gold = teacher (the training labeller's own label, every pair)", result["variants"])
    if result.get("opus"):
        lines += _table(f"Gold = opus (Opus's label on the {result['opus']['n_pairs']} pairs it labelled, a selected subset not comparable "
                        "with the teacher numbers; each variant's n is the pairs of that subset it has)", result["opus"]["variants"])
    shared = result.get("shared", {})
    if shared.get("variants"):
        lines.append(f"\n## Clean against messy on the {shared['n_pairs']} pairs that have both variants (teacher gold)\n")
        header = "| variant / slice | " + " | ".join(HEADLINE) + " |\n|---|" + "---|" * len(HEADLINE)
        lines.append(header)
        for variant, slices in shared["variants"].items():
            for name, m in slices.items():
                lines.append(f"| {variant} {name} | " + " | ".join(fmt(m.get(k)) for k in HEADLINE) + " |")
    lines.append("\nSummary faithfulness (LLM-as-a-judge): TBD.\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path)
    ap.add_argument("outputs", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--wandb-project", default="tt-decoder")
    ap.add_argument("--cap-tokens", type=int, default=None,
                    help="also slice by whether the full prompt exceeded this many tokens (the head-truncation cap the run "
                         "generated under), so the uncut records can be read on their own")
    ap.add_argument("--tokenizer", default=None, help="model directory whose tokenizer counts the prompts for --cap-tokens")
    ap.add_argument("--wandb-prefix", default="bench/",
                    help="key prefix for the logged numbers; a checkpoint scored on a benchmark other than its own training "
                         "task (the zero-shot transfer scores) logs under its own prefix so the run's bench/ keys stay its own")
    args = ap.parse_args()
    result = evaluate(args.split, args.outputs, args.cap_tokens, args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    md = markdown(result)
    (args.out_dir / "results.md").write_text(md)
    print(md)
    if args.wandb_run:
        import wandb
        run = wandb.init(project=args.wandb_project, id=args.wandb_run, resume="must")
        pre = args.wandb_prefix
        flat = {f"{pre}gold_version": result["gold_version"]}
        groups = [("", result["variants"]), ("shared/", result.get("shared", {}).get("variants", {}))]
        if result.get("opus"):
            groups.append(("opus/", result["opus"]["variants"]))
        for prefix, variants in groups:
            for variant, slices in variants.items():
                for name, s in slices.items():
                    for k in HEADLINE:
                        if s.get(k) is not None:
                            flat[f"{pre}{prefix}{variant}/{name}/{k}"] = s[k]
        run.log(flat)
        path, online = run.path, run.settings.mode == "online"
        run.finish()
        if online:
            # a re-scored run keeps only the current keys: summary entries of an earlier evaluator (other slices, other golds) go,
            # and the fresh values are written in the same update so a stale API copy can never put old numbers back
            api_run = wandb.Api().run(path)
            for k in [k for k in list(api_run.summary.keys()) if k.startswith(pre) and k not in flat]:  # wandb Summary, not a dict
                del api_run.summary[k]
            for k, v in flat.items():
                api_run.summary[k] = v
            api_run.summary.update()
        # written last: a queue that skips "already scored" runs keys on this file, so a wandb failure after metrics.json
        # was written still re-scores (and re-logs) on the next invocation
        (args.out_dir / "wandb.json").write_text(json.dumps({"run": path, "prefix": pre, "keys": len(flat), "online": online}))


if __name__ == "__main__":
    main()
