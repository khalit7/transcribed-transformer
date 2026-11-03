import json

from src.synthesis_v2 import cases, identify_hosts, sporc_meta


def test_clean_turns_merges_fragments_strips_non_speech_and_collapses_loops():
    turns = [("SPEAKER_00", "Hi there"), ("SPEAKER_00", ", great"), ("SPEAKER_00", "."),
             ("SPEAKER_01", "[Music]"), ("SPEAKER_01", "Welcome back (laughs) everyone"),
             ("SPEAKER_01", "yeah yeah"), ("SPEAKER_01", "yeah yeah"), ("SPEAKER_01", "yeah yeah"),
             ("SPEAKER_00", "So"), ("SPEAKER_00", "[Applause]")]
    assert cases.clean_turns(turns) == [("SPEAKER_00", "Hi there , great ."),
                                        ("SPEAKER_01", "Welcome back everyone yeah yeah yeah yeah"),
                                        ("SPEAKER_00", "So")]


def test_clean_turns_keeps_genuine_short_repeats_and_suffixes():
    # a real "thanks" after "many thanks" is kept; "yeah" twice is kept; a long fragment repeated twice is a loop
    turns = [("SPEAKER_00", "many thanks"), ("SPEAKER_00", "thanks"), ("SPEAKER_00", "yeah"), ("SPEAKER_00", "yeah"),
             ("SPEAKER_00", "we will be right back after this"), ("SPEAKER_00", "we will be right back after this")]
    assert cases.clean_turns(turns) == [("SPEAKER_00", "many thanks thanks yeah yeah we will be right back after this")]


def test_split_long_turns_at_sentence_ends():
    text = " ".join(f"Sentence number {i} is here." for i in range(80))  # 400 words, 80 sentences
    parts = cases.split_long("host", text, max_words=150)
    assert len(parts) == 3 and all(len(t.split()) <= 150 for _r, t in parts) and " ".join(t for _r, t in parts) == text
    assert all(t.endswith(".") for _r, t in parts)
    run = " ".join(["word"] * 320)  # no punctuation: cut at the word count
    parts = cases.split_long("host", run, max_words=150)
    assert [len(t.split()) for _r, t in parts] == [150, 150, 20]


def test_render_numbers_hosts_and_guests_by_first_appearance_and_keeps_other_tags():
    turns = [("SPEAKER_02", "a"), ("SPEAKER_01", "c"), ("SPEAKER_00", "b"), ("SPEAKER_02", "d"), ("SPEAKER_03", "e")]
    assert cases.render(turns, {"SPEAKER_00", "SPEAKER_02"}) == ["host_1: a", "SPEAKER_01: c", "host_2: b", "host_1: d",
                                                                 "SPEAKER_03: e"]
    assert cases.render(turns, {"SPEAKER_02"}, {"SPEAKER_01", "SPEAKER_03"}) == ["host_1: a", "guest_1: c", "SPEAKER_00: b",
                                                                                 "host_1: d", "guest_2: e"]
    labels = cases.speaker_labels(turns, {"SPEAKER_02", "SPEAKER_07"}, {"SPEAKER_02", "SPEAKER_01"})
    assert labels == {"SPEAKER_02": "host_1", "SPEAKER_01": "guest_1", "SPEAKER_07": "host_2"}


def test_build_keeps_every_corpus_turn_verbatim():
    # owner's decision 2026-11-01 (D75): no merging, no non-speech removal, nothing; only the speaker label changes
    filler = "\n".join(f"SPEAKER_0{(i + 1) % 2}: filler turn number {i} with several words in it" for i in range(20))
    text = "SPEAKER_01: Welcome to the show\nSPEAKER_01: everybody\nSPEAKER_00: Thanks for having me\nSPEAKER_01: [Music]\n" + filler
    c = cases.build("abc", text, ["SPEAKER_01"], {"stratum": "0-4k"}, guests=["SPEAKER_00"])
    assert c.id == "sporc-abc" and c.dataset == "sporc" and c.track == "track-nc"
    assert c.transcript.lines("messy")[:4] == ["host_1: Welcome to the show", "host_1: everybody",
                                               "guest_1: Thanks for having me", "host_1: [Music]"]
    assert len(c.transcript.lines("messy")) == 24
    from src.synthesis_v2 import numbered
    assert numbered(c.transcript.lines("messy")[:2]) == "1. host_1:Welcome to the show\n2. host_1:everybody"
    assert c.transcript.speakers == ["host_1", "guest_1"] and c.meta["host_tags"] == ["SPEAKER_01"]
    assert c.meta["guest_tags"] == ["SPEAKER_00"]
    assert c.meta["n_raw_turns"] == 24 and c.meta["stratum"] == "0-4k"


def test_build_rejects_a_collapsed_transcript():
    text = "SPEAKER_01: " + " ".join(["word"] * 100) + "\n" + "\n".join(f"SPEAKER_0{i % 2}: short turn number {i}" for i in range(20))
    try:
        cases.build("abc", text, ["SPEAKER_01"], {})
    except ValueError as e:
        assert "collapsed" in str(e)
    else:
        raise AssertionError("a collapsed transcript was built")


def test_identify_prompt_names_every_tag_and_truncates_the_middle():
    turns = [(f"SPEAKER_0{i % 3}", f"turn {i}") for i in range(300)]
    p = identify_hosts.prompt(turns)
    assert "SPEAKER_00 (100 turns), SPEAKER_01 (100 turns), SPEAKER_02 (100 turns)" in p and "[middle of the episode omitted]" in p
    assert "TAGS THAT DO NOT APPEAR ABOVE" not in p  # every tag is inside the windows
    assert "turn 0" in p and "turn 299" in p and "turn 150" not in p
    assert '"hosts": [' in p and '"guests": [' in p and '"kind": ' in p


def test_identify_filters_unknown_tags_and_records_sporc_agreement(monkeypatch, tmp_path):
    monkeypatch.setattr(identify_hosts, "CALLS", tmp_path / "calls.jsonl")
    monkeypatch.setattr(identify_hosts, "ask_json_full",
                        lambda prompt, model, schema: ({"kind": " Conversation", "hosts": ["speaker_01"],
                                                        "guests": ["SPEAKER_00", "SPEAKER_01"], "same_person": [],
                                                        "confidence": 0.9, "reason": "opens and closes"}, 0.03,
                                                       {"attempts": 2, "cost_last": 0.01, "cost_total": 0.03, "usage": {"x": 1}}))
    turns = [("SPEAKER_01", "welcome"), ("SPEAKER_00", "hi")]
    rec = identify_hosts.identify("d1", turns, ["SPEAKER_01"], model="claude:sonnet", sporc_guest_tags=["SPEAKER_00"])
    assert rec["hosts"] == ["SPEAKER_01"] and rec["raw_hosts"] == ["speaker_01"]
    assert rec["guests"] == ["SPEAKER_00"] and rec["guests_agree_with_sporc"] is True  # a tag is never both
    assert rec["kind"] == "conversation"
    assert rec["agrees_with_sporc"] is True and rec["cost_usd"] == 0.03  # every attempt's cost
    assert json.loads((tmp_path / "calls.jsonl").read_text())["attempts"] == 2
    rec2 = identify_hosts.identify("d2", turns, [], model="claude:sonnet")
    assert rec2["agrees_with_sporc"] is None
    rec3 = identify_hosts.identify("d3", turns, ["SPEAKER_00"], model="claude:sonnet")
    assert rec3["agrees_with_sporc"] is False


def test_identify_refuses_to_cache_a_reply_with_an_unmatched_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(identify_hosts, "CALLS", tmp_path / "calls.jsonl")
    monkeypatch.setattr(identify_hosts, "ask_json_full",
                        lambda prompt, model, schema: ({"kind": "conversation", "hosts": ["SPEAKER_01", "SPEAKER_7"], "guests": [],
                                                        "same_person": [], "confidence": 0.5, "reason": ""}, 0.01,
                                                       {"attempts": 1, "cost_last": 0.01, "cost_total": 0.01, "usage": {}}))
    turns = [("SPEAKER_01", "welcome"), ("SPEAKER_00", "hi")]
    try:
        identify_hosts.identify("d1", turns, [], model="claude:sonnet")
    except Exception as e:  # noqa: BLE001
        assert "match no tag" in str(e)  # one unmatched tag is enough: a typo must not silently drop a speaker
    else:
        raise AssertionError("a reply with an unmatched tag was accepted")
    assert identify_hosts.normalise_hosts("speaker_00", {"SPEAKER_00"}) == (["SPEAKER_00"], ["speaker_00"])
    assert identify_hosts.normalise_hosts([], {"SPEAKER_00"}) == ([], [])


def test_skipped_counts_cache_states(tmp_path, monkeypatch):
    cache = tmp_path / "sporc.jsonl"
    full = {"guests": [], "same_person": [], "kind": "conversation"}
    cache.write_text("\n".join(json.dumps(r) for r in [
        {"doc_id": "a", "hosts": ["SPEAKER_00"], **full},
        {"doc_id": "b", "hosts": [], **full},
        {"doc_id": "d", "hosts": ["SPEAKER_00"], **full, "kind": "scripted"},
        {"doc_id": "e", "hosts": ["SPEAKER_00"]},  # a host-only record from the first version: not identified
        {"doc_id": "f", "hosts": ["SPEAKER_00"], "guests": ["SPEAKER_01"]},  # host+guest without kind: not identified
    ]) + "\n")
    monkeypatch.setattr(identify_hosts, "CACHE", cache)
    assert cases.skipped(["a", "b", "c", "d", "e", "f"]) == {"not_identified": 3, "no_host_found": 1, "not_a_conversation": 1,
                                                             "buildable": 1}
    assert identify_hosts.current({"doc_id": "x", "hosts": [], **full}) and not identify_hosts.current(None)


def test_render_refuses_a_tag_that_reads_as_a_label():
    try:
        cases.render([("host_3", "a"), ("SPEAKER_00", "b")], {"SPEAKER_00"})
    except ValueError as e:
        assert "host_3" in str(e)
    else:
        raise AssertionError("a verbatim tag that reads as a host label was rendered")


def test_cases_render_verbatim_turns_with_numbered_host_and_guest_labels():
    filler = "\n".join(f"SPEAKER_0{(i + 2) % 3}: filler turn number {i} with several words in it" for i in range(30))
    text = "SPEAKER_01: Welcome\nSPEAKER_01: to the show\nSPEAKER_02: Thanks\nSPEAKER_00: Hi all\n" + filler
    hosts, guests = ["SPEAKER_01", "SPEAKER_00"], ["SPEAKER_02", "SPEAKER_01"]
    c = cases.build("abc", text, hosts, {}, guests=guests)
    assert c.transcript.lines("messy")[:4] == ["host_1: Welcome", "host_1: to the show", "guest_1: Thanks", "host_2: Hi all"]


def test_identify_prompt_shows_the_first_turns_of_tags_outside_the_windows():
    turns = [(f"SPEAKER_0{i % 2}", f"turn {i}") for i in range(300)]
    turns[150] = ("SPEAKER_05", "a mid-episode caller")
    turns[151] = ("SPEAKER_05", "still the caller")
    turns[152] = ("SPEAKER_05", "third turn never shown")
    p = identify_hosts.prompt(turns)
    assert "SPEAKER_05 (3 turns)" in p and "TAGS THAT DO NOT APPEAR ABOVE" in p
    assert "[turn 151 of 300] SPEAKER_05: a mid-episode caller" in p and "still the caller" in p and "never shown" not in p
    assert identify_hosts.normalise_groups([["speaker_00", "SPEAKER_01"], ["SPEAKER_01"], "SPEAKER_00, SPEAKER_01", ["x", "y"]],
                                           {"SPEAKER_00", "SPEAKER_01"}) == [["SPEAKER_00", "SPEAKER_01"]]
    assert identify_hosts.normalise_hosts("SPEAKER_00, speaker_01", {"SPEAKER_00", "SPEAKER_01"})[0] == ["SPEAKER_00", "SPEAKER_01"]


def test_identify_refuses_a_reply_without_guests_or_with_unmatched_guests(monkeypatch, tmp_path):
    monkeypatch.setattr(identify_hosts, "CALLS", tmp_path / "calls.jsonl")
    turns = [("SPEAKER_01", "welcome"), ("SPEAKER_00", "hi")]
    for reply, msg in [({"kind": "conversation", "hosts": ["SPEAKER_01"], "confidence": 0.9, "reason": ""}, "no guests"),
                       ({"kind": "conversation", "hosts": ["SPEAKER_01"], "guests": ["SPEAKER_01", "SPEAKER_7"], "confidence": 0.9, "reason": ""}, "match no tag"),
                       ({"hosts": ["SPEAKER_01"], "guests": [], "confidence": 0.9, "reason": ""}, "kind"),
                       ({"kind": "conversation", "guests": [], "same_person": [], "confidence": 0.9, "reason": ""}, "no hosts")]:
        monkeypatch.setattr(identify_hosts, "ask_json_full", lambda prompt, model, schema, reply=reply: (reply, 0.01, {"attempts": 1, "cost_last": 0.01, "cost_total": 0.01, "usage": {}}))
        try:
            identify_hosts.identify("d1", turns, [], model="claude:sonnet")
        except Exception as e:  # noqa: BLE001
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"accepted a reply that should fail with {msg}")
    monkeypatch.setattr(identify_hosts, "ask_json_full", lambda prompt, model, schema: (
        {"kind": "conversation", "hosts": ["SPEAKER_01"], "guests": "none", "same_person": [], "confidence": 0.9, "reason": ""}, 0.01,
        {"attempts": 1, "cost_last": 0.01, "cost_total": 0.01, "usage": {}}))
    assert identify_hosts.identify("d1", turns, [], model="claude:sonnet")["guests"] == []  # "none" is an empty list
    monkeypatch.setattr(identify_hosts, "ask_json_full", lambda prompt, model, schema: (
        {"kind": "scripted", "hosts": ["none"], "guests": "n/a", "same_person": [], "confidence": 0.9, "reason": ""}, 0.01,
        {"attempts": 1, "cost_last": 0.01, "cost_total": 0.01, "usage": {}}))
    rec = identify_hosts.identify("d1", turns, [], model="claude:sonnet")
    assert rec["hosts"] == [] and rec["kind"] == "scripted"  # "none" for hosts is cached as no host, not refused
    monkeypatch.setattr(identify_hosts, "ask_json_full", lambda prompt, model, schema: (
        {"kind": "scripted", "hosts": ["Narrator"], "guests": ["SPEAKER_00 and SPEAKER_01"], "same_person": [], "confidence": 0.9,
         "reason": ""}, 0.01, {"attempts": 1, "cost_last": 0.01, "cost_total": 0.01, "usage": {}}))
    rec = identify_hosts.identify("d1", turns, [], model="claude:sonnet")
    assert rec["kind"] == "scripted" and rec["hosts"] == []  # a non-conversation is cached whatever it calls its voices
    assert identify_hosts.normalise_hosts(['"SPEAKER_00" (host); speaker_01 and SPEAKER_00.'], {"SPEAKER_00", "SPEAKER_01"})[0] == ["SPEAKER_00", "SPEAKER_01"]
    assert identify_hosts.normalise_hosts(["SPEAKER_00 (12 turns)", "speaker_01, SPEAKER_00"], {"SPEAKER_00", "SPEAKER_01"})[0] == ["SPEAKER_00", "SPEAKER_01"]


def test_sporc_label_parsing():
    assert sporc_meta.parse_labels('{"Sam": "SPEAKER_01", "Jeff": "SPEAKER_02"}') == {"Sam": "SPEAKER_01", "Jeff": "SPEAKER_02"}
    assert sporc_meta.parse_labels("{}") == {} and sporc_meta.parse_labels(None) == {} and sporc_meta.parse_labels("not a dict") == {}
    assert sporc_meta.parse_names("['A', 'B']") == ["A", "B"] and sporc_meta.parse_names("[]") == []


def test_append_jsonl_drops_a_truncated_fragment_and_read_is_tolerant(tmp_path):
    from src.synthesis_v2 import append_jsonl, read_jsonl
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"a": 1})
    p.write_bytes(p.read_bytes() + b'{"a": 2, "tru')  # a crash mid-write
    assert read_jsonl(p) == [{"a": 1}]
    append_jsonl(p, {"a": 3})
    assert read_jsonl(p) == [{"a": 1}, {"a": 3}] and p.read_bytes().count(b"\n") == 2


def test_from_metadata_covers_only_complete_sporc_labels():
    # shares come from the transcript per tag: the main speaker is SPEAKER_02, SPEAKER_00 is a fragment
    turns = [("SPEAKER_00", "ok"), ("SPEAKER_02", " ".join(["w"] * 60)), ("SPEAKER_01", " ".join(["w"] * 39))]
    shares = identify_hosts.word_shares(turns)
    assert shares["SPEAKER_00"] == 0.01 and shares["SPEAKER_02"] == 0.6
    m = {"sporc_host_tags": ["SPEAKER_02"], "sporc_guest_tags": ["SPEAKER_01", "SPEAKER_02"]}
    rec = identify_hosts.from_metadata("d", shares, m)
    assert rec is not None and rec["hosts"] == ["SPEAKER_02"] and rec["guests"] == ["SPEAKER_01"]  # the fragment need not be covered
    assert rec["kind"] == "conversation" and rec["source"] == identify_hosts.METADATA and rec["cost_usd"] == 0.0
    assert identify_hosts.current(rec) and rec["sporc_guest_tags"] == ["SPEAKER_01"] and rec["agrees_with_sporc"] is None
    # the main speaker missing from the metadata: the episode goes to the model (the pool row's sorted shares would have hidden this)
    assert identify_hosts.from_metadata("d", shares, {"sporc_host_tags": ["SPEAKER_00"], "sporc_guest_tags": ["SPEAKER_01"]}) is None
    # no host labels, no tags, or a host tag that never speaks: no record
    assert identify_hosts.from_metadata("d", shares, {"sporc_host_tags": [], "sporc_guest_tags": ["SPEAKER_01"]}) is None
    assert identify_hosts.from_metadata("d", {}, m) is None
    assert identify_hosts.from_metadata("d", shares, {"sporc_host_tags": ["SPEAKER_09"], "sporc_guest_tags": ["SPEAKER_01", "SPEAKER_02"]}) is None
    assert not identify_hosts.current({"doc_id": "d", "source": identify_hosts.METADATA, "revoked": True})  # a revocation marker
