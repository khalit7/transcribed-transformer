import collections

from src.synthesis_v2 import identify_hosts, strata

TOP = 15_000


def _ident(kind: str = "conversation", hosts: list[str] | None = None) -> dict:
    rec = {k: None for k in identify_hosts.FIELDS}
    return {**rec, "kind": kind, "hosts": ["SPEAKER_00"] if hosts is None else hosts, "guests": []}


def _row(doc_id: str, tokens: int, words: int, podcast: str | None = None, **kw) -> dict:
    from src.synthesis_v2.episodes import stratum_of
    return {"doc_id": doc_id, "podcast_id": podcast or doc_id, "tokens": tokens, "n_words": words,
            "stratum": stratum_of(tokens), **kw}


def _plan(population: list[dict]) -> dict:
    return {"ceiling": {"ceiling": TOP}, "design": strata.design(population, TOP)}


def test_allocate_is_proportional_and_exact():
    out = strata.allocate(10, {(0, 0): 1, (0, 1): 1, (0, 2): 1})
    assert sum(out.values()) == 10 and sorted(out.values()) == [3, 3, 4]
    assert strata.allocate(7, {(1, 0): 6, (2, 0): 1}) == {(1, 0): 6, (2, 0): 1}


def test_cells_cut_by_sub_bin_and_tokens_per_word_quantile_and_stop_at_the_ceiling():
    assert strata.cell_edges("4-8k", TOP) == [4_000, 5_000, 6_000, 7_000, 8_000]
    assert strata.cell_edges("12-16k", TOP) == [12_000, 13_000, 14_000, 15_001]  # 15,000-15,001 joins the bin before
    cuts = [1.4, 1.5, 1.6, 1.8]
    edges = strata.cell_edges("4-8k", TOP)
    assert strata.cell_of(_row("a", 4_500, 3_500), edges, cuts) == (0, 0)   # 1.29 tokens per word
    assert strata.cell_of(_row("b", 7_900, 5_200), edges, cuts) == (3, 2)   # 1.52
    assert strata.cell_of(_row("c", 6_100, 2_900), edges, cuts) == (2, 4)   # 2.10
    assert strata.cell_of(_row("d", 8_000, 5_000), edges, cuts) is None
    assert strata.cell_of(_row("e", 15_200, 10_000), strata.cell_edges("12-16k", TOP), cuts) is None  # over the ceiling


RATIOS = (1.3, 1.45, 1.6, 1.8, 2.1)


def _population() -> list[dict]:
    # 4-8k population spread evenly over its four sub-bins and five densities; rejected and out-of-band rows ignored
    pop = []
    for b in range(4):
        for k, ratio in enumerate(RATIOS):
            for i in range(10):
                t = 4_000 + 1_000 * b + 50 * i
                pop.append({**_row(f"pop{b}{k}{i}", t, int(t / ratio)), "reject": None, "windows": ["4-8k"]})
    pop.append({**_row("mono", 5_000, 3_000), "reject": "monologue", "windows": ["4-8k"]})
    pop.append({**_row("other", 5_000, 3_000), "reject": None, "windows": ["0-4k"]})  # reached from another window only
    for s, t in (("0-4k", 3_000), ("8-12k", 9_000), ("12-16k", 13_000)):
        pop += [{**_row(f"{s}-{i}", t + i, int((t + i) / 1.5)), "reject": None, "windows": [s]} for i in range(30)]
    return pop


def test_design_counts_the_population_per_cell_from_the_strata_own_window():
    d = strata.design(_population(), TOP)["4-8k"]
    assert d["population"] == 200 and set(d["cells"].values()) == {10} and len(d["cells"]) == 20


def test_draw_reuses_accepted_pool_episodes_first_and_draws_only_the_gaps():
    population = _population()
    plan = _plan(population)
    # the pool: 4-8k episodes all in the lowest sub-bin and the densest quintile, as the old stratum was
    pool = [_row(f"old{i}", 4_100 + i, 1_900) for i in range(40)]
    cache = {r["doc_id"]: _ident() for r in pool}
    cache["old0"] = _ident(kind="scripted")  # rejected by the identifier: not reusable
    new = strata.draw(plan, pool, population, cache, 40, seed=3)
    rep = plan["draws"][-1]["report"]["4-8k"]
    # 40 over 20 equal cells = 2 per cell; the pool can fill only cell (0, 4)
    assert rep["reused_accepted"] == 2 and rep["pending_before"] == 0
    drawn = [r for r in new if r["stratum"] == "4-8k"]
    want = 19 * strata.math.ceil(2 * (1 + strata.DRAW_MARGIN))
    assert len(drawn) == want and rep["drawn"] == want
    assert all(r["draw"] == "strata-3" and r["seed"] == 3 and r["reject"] is None and "windows" not in r for r in new)
    assert not {r["doc_id"] for r in new} & {r["doc_id"] for r in pool} and "other" not in {r["doc_id"] for r in new}


def test_draw_respects_the_podcast_cap_across_the_pool():
    population = [{**r, "podcast_id": "busy"} for r in _population()]
    plan = _plan(population)
    pool = [_row("p1", 4_100, 2_000, podcast="busy"), _row("p2", 4_200, 2_000, podcast="busy")]
    new = strata.draw(plan, pool, population, {}, 40, seed=0)
    assert new == [] and plan["draws"][-1]["report"]["4-8k"]["short"]


def test_select_fills_cells_with_reused_episodes_before_drawn_ones_and_caps_podcasts():
    population = _population()
    plan = _plan(population)
    old = [_row(f"old{i}", 4_100 + i, 1_900, podcast="show" if i < 3 else None) for i in range(5)]  # cell (0, 4)
    old.append(_row("bad", 4_150, 1_900, reject="monologue"))  # fails the text checks: never selected
    drawn = [_row(f"new{b}{k}{i}", 4_000 + 1_000 * b + 10 * i, int((4_000 + 1_000 * b + 10 * i) / r), draw="strata-3")
             for b in range(4) for k, r in enumerate(RATIOS) for i in range(3)]
    pool = old + drawn
    cache = {r["doc_id"]: _ident() for r in pool}
    cache["new000"] = _ident(hosts=[])  # no host: never selected
    chosen, report = strata.select(plan, pool, cache, 40, seed=0)
    mine = [r for r in chosen if r["stratum"] == "4-8k"]
    cell04 = [r for r in mine if r["cell"] == "0,4"]
    assert len(cell04) == 2 and all(r["reused"] for r in cell04)  # reused first
    assert collections.Counter(r["podcast_id"] for r in chosen)["show"] <= 2
    assert not {"new000", "bad"} & {r["doc_id"] for r in chosen}
    assert report["4-8k"]["selected"] == 40 and report["4-8k"]["reused"] == 2 and not report["4-8k"]["short"]


def test_select_fills_an_unfillable_cell_from_the_nearest_cell_without_changing_other_choices():
    population = _population()
    plan = _plan(population)
    # every cell has three drawn episodes except (2, 1), which has none; (2, 0) and (2, 2) hold a fourth spare
    rows = [(b, k, i) for b in range(4) for k in range(5) for i in range(3) if (b, k) != (2, 1)]
    rows += [(2, 0, 3), (2, 2, 3)]
    drawn = [_row(f"new{b}{k}{i}", 4_000 + 1_000 * b + 10 * i, int((4_000 + 1_000 * b + 10 * i) / RATIOS[k]),
                  draw="strata-3") for b, k, i in rows]
    cache = {r["doc_id"]: _ident() for r in drawn}
    chosen, report = strata.select(plan, drawn, cache, 40, seed=0)
    rep = report["4-8k"]
    assert rep["selected"] == 40 and rep["filled_from_nearest_cell"] == 2 and not rep["short"]
    fills = [r for r in chosen if "fills" in r]
    assert {r["fills"] for r in fills} == {"2,1"} and {r["cell"] for r in fills} <= {"2,0", "2,2"}  # same sub-bin
    assert len({r["doc_id"] for r in chosen}) == len(chosen)
    # the first pass is what a pool without the fallback's spares would give: every other choice unchanged
    plain, _ = strata.select(plan, [r for r in drawn if r["doc_id"] not in {"new203", "new223"}], cache, 40, seed=0)
    first = [r for r in chosen if "fills" not in r]
    assert [r["doc_id"] for r in first] == [r["doc_id"] for r in plain if "fills" not in r]


def test_sample_population_counts_only_the_strata_whose_own_window_reached_an_episode():
    # Overlapping windows: every row's estimate lies in both the 0-4k and the 4-8k window. A measurement made while
    # walking 0-4k must not count towards 4-8k until 4-8k's own walk reaches it (review round 15: the 4-8k sample
    # otherwise came wholly from the 0-4k window and missed its upper half).
    rows = [{"doc_id": f"r{i}", "est_tokens": 3_000, "tokens": 4_000 + 10 * i, "n_words": 3_000} for i in range(200)]
    calls = []

    def measure(r: dict) -> dict:
        calls.append(r["doc_id"])
        from src.synthesis_v2.episodes import stratum_of
        return {**r, "stratum": stratum_of(r["tokens"]), "reject": None}

    pop = strata.sample_population(rows, seed=0, per_stratum=20, measure=measure, top=TOP)
    assert len(calls) == len(set(calls))  # each episode measured once
    in_48 = [m for m in pop if strata.in_sample(m, "4-8k", strata.cell_edges("4-8k", TOP))]
    assert len(in_48) == 20  # exactly the 4-8k walk's own 20, not the 200 the 0-4k walk measured
    assert all(m["windows"] == ["0-4k", "4-8k"] for m in in_48)
    assert not [m for m in pop if strata.in_sample(m, "0-4k", strata.cell_edges("0-4k", TOP))]


def test_draw_counts_pending_rows_and_does_not_draw_them_again():
    population = [{**r, "windows": ["4-8k"]} for r in _population()]
    plan = _plan(population)
    first = strata.draw(plan, [], population, {}, 24, seed=1)
    assert first and all(r["seed"] == 1 and r["draw"] == "strata-1" for r in first)
    second = strata.draw(plan, first, population, {}, 24, seed=2)
    assert [r for r in second if r["stratum"] == "4-8k"] == []  # the first draw's rows still await identification
    assert {r["doc_id"] for r in strata.pending(first, {})} == {r["doc_id"] for r in first}


def test_a_sliver_bin_at_the_ceiling_joins_the_bin_before():
    assert strata.cell_edges("12-16k", 15_109) == [12_000, 13_000, 14_000, 15_110]
    assert strata.cell_edges("12-16k", 14_800) == [12_000, 13_000, 14_000, 14_801]


def test_draw_refuses_a_seed_already_used():
    import pytest
    population = _population()
    plan = _plan(population)
    strata.draw(plan, [], population, {}, 40, seed=5)
    with pytest.raises(SystemExit):
        strata.draw(plan, [], population, {}, 40, seed=5)
