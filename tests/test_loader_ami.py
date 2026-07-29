"""AMI loader round trip.

The fixture is synthetic but shaped exactly like the real NXT archive, so the
parsing logic is exercised without committing corpus data and without touching
the network. What it deliberately reproduces are the properties that are easy to
get wrong and silent when wrong: turns split across per-speaker files, word
ranges addressed by id, punctuation as its own token, and annotation markup
interleaved with words.
"""

import zipfile
from pathlib import Path

import pytest

from tt.data.loaders import ami
from tt.data.schema import Role, Track

HEADER = '<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>'
NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def _words_xml(meeting: str, speaker: str, items: list[str]) -> str:
    body = "\n".join(items)
    return f'{HEADER}\n<nite:root nite:id="{meeting}.{speaker}.words" {NS}>\n{body}\n</nite:root>'


def _w(
    meeting: str,
    speaker: str,
    n: int,
    text: str,
    *,
    punc: bool = False,
    trunc: bool = False,
) -> str:
    extra = ' punc="true"' if punc else ""
    extra += ' trunc="true"' if trunc else ""
    return (
        f'  <w nite:id="{meeting}.{speaker}.words{n}" '
        f'starttime="{n}.0" endtime="{n}.5"{extra}>{text}</w>'
    )


def _markup(meeting: str, speaker: str, n: int, tag: str) -> str:
    return f'  <{tag} nite:id="{meeting}.{speaker}.words{n}" starttime="{n}.0" endtime="{n}.5"/>'


def _segments_xml(meeting: str, speaker: str, segs: list[tuple[str, float, int, int]]) -> str:
    rows = []
    for seg_id, start, first, last in segs:
        href = f"{meeting}.{speaker}.words.xml#id({meeting}.{speaker}.words{first})"
        if last != first:
            href += f"..id({meeting}.{speaker}.words{last})"
        rows.append(
            f'  <segment nite:id="{seg_id}" channel="0" transcriber_start="{start}" '
            f'transcriber_end="{start + 1}">\n    <nite:child href="{href}"/>\n  </segment>'
        )
    body = "\n".join(rows)
    return f'{HEADER}\n<nite:root nite:id="{meeting}.{speaker}.segs" {NS}>\n{body}\n</nite:root>'


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A two-speaker meeting whose turns interleave in time."""
    meeting = "TS0001a"
    path = tmp_path / "ami_fixture.zip"

    # Speaker A: a filled pause, a repetition, a truncation, and punctuation as
    # its own token; plus markup that must not reach the text.
    a_words = [
        _w(meeting, "A", 0, "Um"),
        _w(meeting, "A", 1, "so"),
        _w(meeting, "A", 2, "the"),
        _w(meeting, "A", 3, "the"),
        _w(meeting, "A", 4, "th-", trunc=True),
        _w(meeting, "A", 5, "thing"),
        _w(meeting, "A", 6, ".", punc=True),
        _markup(meeting, "A", 7, "vocalsound"),
        _w(meeting, "A", 8, "Right"),
        _w(meeting, "A", 9, ".", punc=True),
        _markup(meeting, "A", 10, "disfmarker"),
        _markup(meeting, "A", 11, "gap"),
    ]
    a_segs = [
        ("TS0001a.sync.1", 10.0, 0, 6),  # first turn
        ("TS0001a.sync.5", 30.0, 8, 9),  # third turn, after B speaks
        ("TS0001a.sync.7", 40.0, 10, 11),  # markup only: must be dropped entirely
    ]

    b_words = [
        _w(meeting, "B", 0, "Mm-hmm"),
        _w(meeting, "B", 1, ",", punc=True),
        _w(meeting, "B", 2, "yeah"),
        _w(meeting, "B", 3, ".", punc=True),
    ]
    b_segs = [("TS0001a.sync.3", 20.0, 0, 3)]  # second turn, between A's two

    # A speaker with turn boundaries but no word stream. Must be skipped without
    # disturbing anyone else's indices.
    c_segs = [("TS0001a.sync.9", 15.0, 0, 1)]

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"words/{meeting}.A.words.xml", _words_xml(meeting, "A", a_words))
        z.writestr(f"words/{meeting}.B.words.xml", _words_xml(meeting, "B", b_words))
        z.writestr(f"segments/{meeting}.A.segments.xml", _segments_xml(meeting, "A", a_segs))
        z.writestr(f"segments/{meeting}.B.segments.xml", _segments_xml(meeting, "B", b_segs))
        z.writestr(f"segments/{meeting}.C.segments.xml", _segments_xml(meeting, "C", c_segs))
    return path


def test_round_trip_through_canonical_schema(archive: Path) -> None:
    (t,) = ami.load_archive(archive)

    assert t.id == "ami/TS0001a"
    assert t.source == ami.SOURCE
    assert t.track is Track.P
    assert t.is_asr is False
    assert t.asr_system is None

    # Contiguous from 0 is enforced by the schema, but assert it here too: it is
    # what evidence line numbers depend on.
    assert [turn.index for turn in t.turns] == list(range(t.n_turns))
    assert all(turn.speaker for turn in t.turns)
    assert all(turn.role is Role.UNKNOWN for turn in t.turns)


def test_turns_merge_across_speakers_in_time_order(archive: Path) -> None:
    """The load-bearing behaviour: AMI stores turns per speaker, one file each.

    A meeting's line order only exists after merging them by start time. Getting
    this wrong produces a transcript where one speaker talks for the whole
    meeting and then the next replies, which is fluent-looking and completely wrong.
    """
    (t,) = ami.load_archive(archive)
    assert [turn.speaker for turn in t.turns] == ["A", "B", "A"]


def test_disfluencies_and_repetitions_are_preserved(archive: Path) -> None:
    (t,) = ami.load_archive(archive)
    first = t.turns[0].text
    assert first.startswith("Um so")
    assert "the the" in first, "repetition must survive; it is signal, not noise"
    assert "th-" in first, "truncated words must survive"
    assert "Mm-hmm" in t.turns[1].text


def test_punctuation_attaches_to_the_preceding_token(archive: Path) -> None:
    (t,) = ami.load_archive(archive)
    assert t.turns[0].text.endswith("thing.")
    assert " ." not in t.turns[0].text
    assert t.turns[1].text == "Mm-hmm, yeah."


def test_markup_is_dropped_from_text_but_counted(archive: Path) -> None:
    """Markup is not spoken words, so it must not reach the text, and must not vanish silently."""
    (t,) = ami.load_archive(archive)
    joined = " ".join(turn.text for turn in t.turns)
    for tag in ("vocalsound", "disfmarker", "gap"):
        assert tag not in joined
    assert t.meta["dropped_vocalsound"] == "1"
    assert t.meta["dropped_disfmarker"] == "1"
    assert t.meta["dropped_gap"] == "1"


def test_markup_only_segments_produce_no_empty_line(archive: Path) -> None:
    """A turn holding only a laugh has no words.

    An empty line that evidence could cite is worse than no line at all.
    """
    (t,) = ami.load_archive(archive)
    assert all(turn.text.strip() for turn in t.turns)
    assert t.n_turns == 3


def test_speaker_without_a_word_stream_is_skipped(archive: Path) -> None:
    """Speaker C has segments but no words file, and must not shift anyone's indices."""
    (t,) = ami.load_archive(archive)
    assert "C" not in {turn.speaker for turn in t.turns}
    assert t.meta["n_speakers"] == "2"


def test_load_is_deterministic(archive: Path) -> None:
    """Line indices are label anchors, so two loads must agree exactly."""
    first = [(x.index, x.speaker, x.text) for x in next(iter(ami.load_archive(archive))).turns]
    second = [(x.index, x.speaker, x.text) for x in next(iter(ami.load_archive(archive))).turns]
    assert first == second


def test_render_is_one_based_and_resolves_back(archive: Path) -> None:
    (t,) = ami.load_archive(archive)
    lines = t.render().split("\n")
    assert lines[0].startswith("1: A: ")
    assert len(lines) == t.n_turns
    # A model citing line 2 must resolve to speaker B's turn.
    assert t.turns[t.line_to_index(2)].speaker == "B"
