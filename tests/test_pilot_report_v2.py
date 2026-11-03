from src.synthesis_v2 import pilot_report as pr


def _rec(ep: str, qid: str, answer: str, stratum: str, evidence=(), n_lines: int = 90, cost: float = 0.01, text: str = "t") -> dict:
    return {"qid": qid, "text": text, "answer": answer, "evidence": list(evidence), "n_lines": n_lines, "stratum": stratum,
            "cost": cost, "episode": ep}


def test_kappa_edges():
    assert pr.kappa([]) is None
    assert pr.kappa([("a", "a")] * 10) == 1.0
    assert pr.kappa([("a", "a"), ("b", "b"), ("a", "a"), ("b", "b")]) == 1.0
    assert abs(pr.kappa([("a", "a"), ("b", "a"), ("a", "b"), ("b", "b")]) - 0.0) < 1e-9


def test_analyse_applies_cut_and_stop_rules(monkeypatch):
    monkeypatch.setattr(pr, "STRATA", ("2-4k", "4-8k", "8-16k"))  # the first two pilots' strata
    gold, other = {}, {}
    strata = ["2-4k", "4-8k", "8-16k"]
    for i in range(60):
        s = strata[i % 3]
        ep = f"e{i}"
        # q_good: balanced, perfectly agreed, evidence late
        gold[f"{ep}::q_good"] = _rec(ep, "q_good", ["x", "y", "none"][i % 3], s, evidence=[80, 85])
        other[f"{ep}::q_good"] = _rec(ep, "q_good", ["x", "y", "none"][i % 3], s)
        # q_none: 'none' on 90% of episodes
        gold[f"{ep}::q_none"] = _rec(ep, "q_none", "none" if i % 10 else "x", s)
        other[f"{ep}::q_none"] = _rec(ep, "q_none", "none" if i % 10 else "x", s)
        # q_noisy: the other labeller disagrees whenever i is odd
        gold[f"{ep}::q_noisy"] = _rec(ep, "q_noisy", ["x", "y"][i % 2], s)
        other[f"{ep}::q_noisy"] = _rec(ep, "q_noisy", ["x", "y"][(i + (i % 2)) % 2], s)
    rep = pr.analyse(gold, other)
    assert rep["episodes"] == 60 and rep["questions"]["q_good"]["cut"] == [] and rep["questions"]["q_good"]["kappa"]["all"] == 1.0
    assert rep["questions"]["q_good"]["evidence_last_third"] == 1.0 and rep["questions"]["q_good"]["evidence_empty"] == 0.0
    assert any(c.startswith("none on") for c in rep["questions"]["q_none"]["cut"])
    assert any(c.startswith("kappa") for c in rep["questions"]["q_noisy"]["cut"])
    assert set(rep["survivors"]) == {"q_good"} and rep["stop"] == []
    md = pr.markdown(rep)
    assert "| `q_good` |" in md and "Survivors (1)" in md


def test_stop_rule_fires_when_long_stratum_disagrees(monkeypatch):
    monkeypatch.setattr(pr, "STRATA", ("2-4k", "4-8k", "8-16k"))  # the first two pilots' strata
    gold, other = {}, {}
    for i in range(90):
        s = ["2-4k", "4-8k", "8-16k"][i % 3]
        ep = f"e{i}"
        gold[f"{ep}::q"] = _rec(ep, "q", ["x", "y", "z"][i % 3], s)
        # agree everywhere except the 8-16k stratum, where the other labeller answers at random-looking offsets
        other[f"{ep}::q"] = _rec(ep, "q", ["x", "y", "z"][(i + (1 if s == "8-16k" else 0)) % 3], s)
    rep = pr.analyse(gold, other)
    assert rep["bank_mean_kappa"]["2-4k"] == 1.0 and rep["bank_mean_kappa"]["8-16k"] < 0.4 and rep["stop"]


def test_unmeasurable_rare_and_text_mismatch_handling(monkeypatch):
    monkeypatch.setattr(pr, "STRATA", ("2-4k", "4-8k", "8-16k"))  # the first two pilots' strata
    monkeypatch.setattr(pr.bank, "RARE_EVENT", ("q_rare",))
    gold, other = {}, {}
    for i in range(90):
        s = ["2-4k", "4-8k", "8-16k"][i % 3]
        ep = f"e{i}"
        gold[f"{ep}::q_rare"] = _rec(ep, "q_rare", "none" if i % 15 else "hit", s)   # 94% none: exempt from skew
        other[f"{ep}::q_rare"] = _rec(ep, "q_rare", "none" if i % 15 else "hit", s)  # perfect agreement
        gold[f"{ep}::q_thin"] = _rec(ep, "q_thin", ["x", "y"][i % 2], s)
        if s != "8-16k":  # the second pass lacks the long stratum: unmeasurable
            other[f"{ep}::q_thin"] = _rec(ep, "q_thin", ["x", "y"][i % 2], s)
        gold[f"{ep}::q_tmpl"] = _rec(ep, "q_tmpl", ["x", "y", "none"][i % 3], s, text=f"Topic: {i}")
        other[f"{ep}::q_tmpl"] = _rec(ep, "q_tmpl", ["x", "y", "none"][i % 3], s, text=f"Topic: {i if i % 2 else -1}")
    rep = pr.analyse(gold, other)
    assert rep["questions"]["q_rare"]["cut"] == [] and rep["questions"]["q_rare"]["rare_event"] and "q_rare" in rep["survivors"]
    assert "q_thin" in rep["unmeasurable"] and "q_thin" not in rep["survivors"] and "q_thin" not in rep["cut"]
    assert rep["text_mismatches_excluded"] == 45 and rep["questions"]["q_tmpl"]["pairs"]["2-4k"] == 15
    # the bank mean excludes the rare-event question, so it is None here (q_tmpl is unmeasurable with 15 pairs)
    assert rep["bank_mean_kappa"]["2-4k"] is None and rep["stop"][0].startswith("not evaluable")  # nothing measurable: the rule cannot clear the bank


def test_stop_rule_compares_the_longest_stratum_with_the_shortest():
    assert pr.STRATA == ("0-4k", "4-8k", "8-12k", "12-16k")
    gold, other = {}, {}
    for i in range(120):
        s = pr.STRATA[i % 4]
        ep = f"e{i}"
        gold[f"{ep}::q"] = _rec(ep, "q", ["x", "y", "z"][i % 3], s)
        # the middle strata disagree; only 12-16k against 0-4k decides the stop
        bad = s in ("4-8k", "8-12k")
        other[f"{ep}::q"] = _rec(ep, "q", ["x", "y", "z"][(i + bad) % 3], s)
    rep = pr.analyse(gold, other)
    assert rep["stop"] == [] and rep["bank_mean_kappa"]["4-8k"] < 0.4 and rep["questions"]["q"]["cut"]
    gold["e0::q"] = {**gold["e0::q"], "stratum": "2-4k"}  # an old-strata label is counted, not silently lost
    rep = pr.analyse(gold, other)
    assert rep["gold_labels_outside_strata"] == 1
    md = pr.markdown(rep).splitlines()
    head = next(line for line in md if line.startswith("| question"))
    row = next(line for line in md if line.startswith("| `q`"))
    assert head.count("|") == row.count("|") and "0-4k / 4-8k / 8-12k / 12-16k / all" in head
