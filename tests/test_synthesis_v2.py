import collections
import json

from src.synthesis_v2 import episodes


def _episode(doc_id: str, podcast: str, n_turns: int, words_per_turn: int, n_speakers: int = 2, title: str = "",
             duration_s: float = 60.0, text: str | None = None) -> dict:
    turns = [f"SPEAKER_0{i % n_speakers}: " + " ".join(f"w{i}_{j}" for j in range(words_per_turn)) for i in range(n_turns)]
    return {"source": "sporc", "doc_id": doc_id, "track": "track-nc", "tier": 1, "asr_system": "t", "has_speakers": True,
            "n_turns": n_turns, "n_words": n_turns * words_per_turn, "text": text if text is not None else "\n".join(turns),
            "meta": {"podcast_id": podcast, "pod_title": title or podcast, "category": "arts", "duration_s": duration_s,
                     "n_speakers": n_speakers}}


def _words(s: str) -> int:
    return len(s.split())


def test_render_is_numbered_one_turn_per_line():
    assert episodes.render([("SPEAKER_00", "hi there"), ("SPEAKER_01", "hello")]) == "1. SPEAKER_00:hi there\n2. SPEAKER_01:hello"
    assert episodes.turns_of("SPEAKER_00: a: b\nSPEAKER_01: c") == [("SPEAKER_00", "a: b"), ("SPEAKER_01", "c")]


def test_stratum_edges():
    assert episodes.stratum_of(0) == "0-4k" and episodes.stratum_of(3_999) == "0-4k" and episodes.stratum_of(4_000) == "4-8k"
    assert episodes.stratum_of(7_999) == "4-8k" and episodes.stratum_of(8_000) == "8-12k"
    assert episodes.stratum_of(11_999) == "8-12k" and episodes.stratum_of(12_000) == "12-16k"
    assert episodes.stratum_of(15_999) == "12-16k" and episodes.stratum_of(16_000) is None


def test_scan_filters_and_choose_stratifies_with_a_podcast_cap(tmp_path):
    interim = tmp_path / "train.jsonl"
    rows = [_episode("mono", "p0", 60, 40, n_speakers=1),          # one speaker: dropped
            _episode("short", "p1", 20, 100),                        # under 40 turns: dropped
            _episode("v1", "p2", 60, 40),                            # v1 episode: dropped
            _episode("v1pod", "p3", 60, 40, title="V1 Show"),        # v1 podcast: dropped
            _episode("huge", "p4", 60, 250),                         # 15,000 words -> 22,500 est tokens: outside every window
            _episode("partial", "p5", 60, 40, duration_s=3600.0),    # 2,400 words in an hour: 40 wpm, a partial transcript
            _episode("nodur", "p6", 60, 40, duration_s=0.0),         # duration unknown
            _episode("monotext", "p7", 60, 40, text="\n".join(["SPEAKER_00: " + " ".join(["w"] * 40)] * 59 + ["SPEAKER_01: w"])),
            ] + [_episode(f"a{i}", "pa", 60, 40) for i in range(4)] + [_episode(f"b{i}", f"pb{i}", 60, 40) for i in range(3)]
    #  60 turns x 40 words = 2,400 words; the word counter below gives 2,400 + 60 line-number tokens -> 0-4k stratum
    interim.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scanned, funnel = episodes.scan(interim, exclude_docs={"v1"}, exclude_pods={"V1 Show"})
    assert funnel["dropped: speakers not 2-4"] == 1 and funnel["dropped: under 40 turns"] == 1
    assert funnel["dropped: v1 episode or podcast"] == 2 and funnel["dropped: length outside every stratum"] == 1
    assert funnel["dropped: partial transcript (duration unknown or under 90 words per minute)"] == 2
    assert funnel["eligible"] == 8 and {r["doc_id"] for r in scanned} == {f"a{i}" for i in range(4)} | {f"b{i}" for i in range(3)} | {"monotext"}
    pool = list(episodes.choose(scanned, {"0-4k": 10, "4-8k": 0, "8-12k": 0, "12-16k": 0}, 0, _words, interim, funnel=funnel))
    assert funnel["measured, dropped: monologue (second speaker under 10% of words)"] == 1
    # podcast "pa" has four eligible episodes and the cap is two
    assert collections.Counter(r["podcast_id"] for r in pool)["pa"] == 2 and len(pool) == 5
    assert {r["stratum"] for r in pool} == {"0-4k"} and all(r["tokens"] == 2_400 + 60 for r in pool)
    assert all(r["word_shares"] == [0.5, 0.5] and r["duplicate_line_share"] == 0.0 for r in pool)
    # offsets point back at the episode
    for r in pool:
        assert json.loads(interim.read_bytes()[r["offset"]:].split(b"\n")[0])["doc_id"] == r["doc_id"]
    summary = episodes.summarise(pool, funnel, {"0-4k": 10, "4-8k": 0, "8-12k": 0, "12-16k": 0}, 0)
    assert summary["pool"] == 5 and summary["podcasts"] == 4 and summary["strata"]["0-4k"]["episodes"] == 5


def test_choose_is_deterministic_for_a_seed(tmp_path):
    interim = tmp_path / "train.jsonl"
    rows = [_episode(f"e{i}", f"p{i}", 60, 40) for i in range(30)]
    interim.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scanned, _ = episodes.scan(interim)
    want = {"0-4k": 5, "4-8k": 0, "8-12k": 0, "12-16k": 0}
    first = [r["doc_id"] for r in episodes.choose(scanned, want, 7, _words, interim)]
    second = [r["doc_id"] for r in episodes.choose(scanned, want, 7, _words, interim)]
    assert first == second and len(first) == 5


def test_text_checks_catch_loops_and_collapsed_lines():
    loop = [("SPEAKER_00", "yeah yeah")] * 30 + [("SPEAKER_01", " ".join(["w"] * 30))]
    assert episodes.text_checks(loop)[0] == "repetition loop (over 20% duplicate lines)"
    collapsed = [("SPEAKER_00", " ".join(["w"] * 100)), ("SPEAKER_01", " ".join(["v"] * 60))] + [("SPEAKER_00", "a"), ("SPEAKER_01", "b")] * 20
    assert episodes.text_checks(collapsed)[0] == "collapsed transcript (one line over 25% of words)"
    fine = [("SPEAKER_00", "one two three"), ("SPEAKER_01", "four five six")] * 20
    assert episodes.text_checks(fine)[0] is None


def test_choose_with_taken_pool_skips_its_episodes_and_counts_its_podcasts(tmp_path):
    interim = tmp_path / "train.jsonl"
    rows = [_episode(f"d{i}", f"pod{i // 4}", 60, 40) for i in range(16)]  # four podcasts, four eligible episodes each
    interim.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scanned, _f = episodes.scan(interim)
    # the existing pool holds both slots of pod0 and one slot of pod1
    taken = [{"doc_id": "d0", "podcast_id": "pod0"}, {"doc_id": "d1", "podcast_id": "pod0"}, {"doc_id": "d4", "podcast_id": "pod1"}]
    more = list(episodes.choose(scanned, {"0-4k": 10, "4-8k": 0, "8-12k": 0, "12-16k": 0}, 1, _words, interim, taken=taken))
    by_pod = collections.Counter(r["podcast_id"] for r in more)
    assert by_pod["pod0"] == 0  # pod0 is at the cap already: its two untaken episodes d2, d3 are never drawn
    assert by_pod["pod1"] == 1  # one slot left
    assert by_pod["pod2"] == 2 and by_pod["pod3"] == 2 and len(more) == 5
    assert not {r["doc_id"] for r in more} & {"d0", "d1", "d4"} and all(r["seed"] == 1 for r in more)


def test_choose_keeps_a_candidate_only_in_the_stratum_it_was_drawn_for(tmp_path):
    # 900 words -> 1,350 est tokens: inside the 0-4k window only; the inflating counter measures them at 8-12k. Until
    # 2026-11-01 the 0-4k loop kept such episodes for the 8-12k stratum, so a stratum filled from another's candidates.
    interim = tmp_path / "train.jsonl"
    rows = [_episode(f"e{i}", f"p{i}", 60, 15) for i in range(10)]
    interim.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scanned, _ = episodes.scan(interim)
    inflate = lambda s: 10 * _words(s)
    pool = list(episodes.choose(scanned, {"0-4k": 10, "4-8k": 0, "8-12k": 5, "12-16k": 0}, 0, inflate, interim))
    assert pool == []
