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
    (t,) = ami.load_archive(archive, variant="manual")

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
    (t,) = ami.load_archive(archive, variant="manual")
    assert [turn.speaker for turn in t.turns] == ["A", "B", "A"]


def test_disfluencies_and_repetitions_are_preserved(archive: Path) -> None:
    (t,) = ami.load_archive(archive, variant="manual")
    first = t.turns[0].text
    assert first.startswith("Um so")
    assert "the the" in first, "repetition must survive; it is signal, not noise"
    assert "th-" in first, "truncated words must survive"
    assert "Mm-hmm" in t.turns[1].text


def test_punctuation_attaches_to_the_preceding_token(archive: Path) -> None:
    (t,) = ami.load_archive(archive, variant="manual")
    assert t.turns[0].text.endswith("thing.")
    assert " ." not in t.turns[0].text
    assert t.turns[1].text == "Mm-hmm, yeah."


def test_markup_is_dropped_from_text_but_counted(archive: Path) -> None:
    """Markup is not spoken words, so it must not reach the text, and must not vanish silently."""
    (t,) = ami.load_archive(archive, variant="manual")
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
    (t,) = ami.load_archive(archive, variant="manual")
    assert all(turn.text.strip() for turn in t.turns)
    assert t.n_turns == 3


def test_speaker_without_a_word_stream_is_skipped(archive: Path) -> None:
    """Speaker C has segments but no words file, and must not shift anyone's indices."""
    (t,) = ami.load_archive(archive, variant="manual")
    assert "C" not in {turn.speaker for turn in t.turns}
    assert t.meta["n_speakers"] == "2"


def test_load_is_deterministic(archive: Path) -> None:
    """Line indices are label anchors, so two loads must agree exactly."""

    def snapshot() -> list[tuple[int, str, str]]:
        (t,) = ami.load_archive(archive, variant="manual")
        return [(x.index, x.speaker, x.text) for x in t.turns]

    assert snapshot() == snapshot()


def test_render_is_one_based_and_resolves_back(archive: Path) -> None:
    (t,) = ami.load_archive(archive, variant="manual")
    lines = t.render().split("\n")
    assert lines[0].startswith("1: A: ")
    assert len(lines) == t.n_turns
    # A model citing line 2 must resolve to speaker B's turn.
    assert t.turns[t.line_to_index(2)].speaker == "B"


# --- ASR variant -------------------------------------------------------------
#
# The tier 1 layer. Same NXT shape in a different directory, with one structural
# difference that would fail silently: ASR segments carry no timing of their own,
# so the merge order has to come from the words instead.


def _asr_words_xml(meeting: str, speaker: str, items: list[str]) -> str:
    body = "\n".join(items)
    return f'{HEADER}\n<nite:root nite:id="{meeting}.{speaker}.asr" {NS}>\n{body}\n</nite:root>'


def _aw(meeting: str, speaker: str, n: int, text: str, start: float) -> str:
    return (
        f'  <w nite:id="{meeting}.{speaker}.aw{n}" '
        f'starttime="{start}" endtime="{start + 0.2}">{text}</w>'
    )


def _asil(meeting: str, speaker: str, n: int, start: float) -> str:
    return (
        f'  <sil nite:id="{meeting}.{speaker}.sil{n}" starttime="{start}" endtime="{start + 0.1}"/>'
    )


def _asr_segments_xml(meeting: str, speaker: str, segs: list[tuple[str, str, str]]) -> str:
    """ASR segments have a participant attribute and, notably, NO start time."""
    rows = []
    for seg_id, first, last in segs:
        href = f"{meeting}.{speaker}.words.xml#id({first})"
        if last != first:
            href += f"..id({last})"
        rows.append(
            f'  <segment nite:id="{seg_id}" participant="m000">\n'
            f'    <nite:child href="{href}"/>\n  </segment>'
        )
    body = "\n".join(rows)
    return f'{HEADER}\n<nite:root nite:id="{meeting}.{speaker}.aseg" {NS}>\n{body}\n</nite:root>'


@pytest.fixture
def asr_archive(tmp_path: Path) -> Path:
    """Two speakers whose ASR turns interleave in time."""
    m = "TS0002a"
    d = ami.LAYOUTS["asr"].words_dir
    path = tmp_path / "ami_asr_fixture.zip"

    a_words = [
        _asil(m, "A", 0, 5.0),
        _aw(m, "A", 1, "okay", 10.0),
        _aw(m, "A", 2, "so", 10.3),
        _aw(m, "A", 3, "the", 10.6),
        _asil(m, "A", 4, 11.0),
        _aw(m, "A", 5, "right", 30.0),
    ]
    a_segs = [
        (f"{m}.A.asg.1", f"{m}.A.sil0", f"{m}.A.aw3"),
        (f"{m}.A.asg.2", f"{m}.A.aw5", f"{m}.A.aw5"),
        (f"{m}.A.asg.3", f"{m}.A.sil4", f"{m}.A.sil4"),  # silence only: dropped
    ]
    b_words = [_aw(m, "B", 0, "mm", 20.0), _aw(m, "B", 1, "hmm", 20.4)]
    b_segs = [(f"{m}.B.asg.1", f"{m}.B.aw0", f"{m}.B.aw1")]

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{d}/{m}.A.words.xml", _asr_words_xml(m, "A", a_words))
        z.writestr(f"{d}/{m}.B.words.xml", _asr_words_xml(m, "B", b_words))
        z.writestr(f"{d}/{m}.A.segments.xml", _asr_segments_xml(m, "A", a_segs))
        z.writestr(f"{d}/{m}.B.segments.xml", _asr_segments_xml(m, "B", b_segs))
    return path


def test_asr_is_the_default_variant() -> None:
    """The data-tier rule in CLAUDE.md, encoded so a change to it fails a test.

    Defaulting to the human transcript because it is cleaner is the specific
    mistake this project already made once.
    """
    assert ami.DEFAULT_VARIANT == "asr"
    assert ami.LAYOUTS["asr"].is_asr is True
    assert ami.LAYOUTS["manual"].is_asr is False


def test_asr_variant_round_trips(asr_archive: Path) -> None:
    (t,) = ami.load_archive(asr_archive, variant="asr")
    assert t.id == "ami/TS0002a"
    assert t.track is Track.P
    assert t.is_asr is True
    assert t.asr_system is not None and "2007" in t.asr_system
    assert t.meta["variant"] == "asr"
    assert [turn.index for turn in t.turns] == list(range(t.n_turns))


def test_asr_turns_order_by_word_time_not_segment_attribute(asr_archive: Path) -> None:
    """The structural difference between the layers, and the one that fails silently.

    ASR segments have no transcriber_start. Falling back to 0.0 would give every
    segment the same key and collapse the cross-speaker merge, producing a
    transcript where one speaker talks through the whole meeting.
    """
    (t,) = ami.load_archive(asr_archive, variant="asr")
    assert [turn.speaker for turn in t.turns] == ["A", "B", "A"]
    assert t.turns[0].text == "okay so the"
    assert t.turns[1].text == "mm hmm"
    assert t.turns[2].text == "right"


def test_asr_silences_are_dropped_and_counted(asr_archive: Path) -> None:
    (t,) = ami.load_archive(asr_archive, variant="asr")
    assert "sil" not in " ".join(turn.text for turn in t.turns)
    assert t.meta["dropped_sil"] == "2"  # both in speaker A's stream; B has none
    # A segment holding only silence must not become an empty line.
    assert t.n_turns == 3
    assert all(turn.text.strip() for turn in t.turns)


def test_asr_text_has_no_punctuation(asr_archive: Path) -> None:
    """Part of why ASR output is tier 1: a recogniser emits no punctuation at all."""
    (t,) = ami.load_archive(asr_archive, variant="asr")
    joined = " ".join(turn.text for turn in t.turns)
    assert not any(ch in joined for ch in ".,?!;:")


def test_variants_are_separate_archives() -> None:
    """Different downloads, different checksums. Mixing them would be silent."""
    asr, manual = ami.LAYOUTS["asr"], ami.LAYOUTS["manual"]
    assert asr.archive_name != manual.archive_name
    assert asr.sha256 != manual.sha256
    assert asr.words_dir != manual.words_dir


def test_bare_href_format_with_multiple_children(tmp_path: Path) -> None:
    """AMI writes segment references two ways, and both appear in the ASR archive.

    The range form puts one child on a segment. The bare form puts several, each
    naming a single id with no ``id(...)`` wrapper. Handling only the range form
    dropped 43 of 169 ASR meetings entirely, and dropped them *silently*: the
    files parsed, the segments were found, and every one produced no text.
    """
    m = "TS0003a"
    d = ami.LAYOUTS["asr"].words_dir
    path = tmp_path / "bare.zip"

    words = _asr_words_xml(
        m,
        "A",
        [_asil(m, "A", 0, 4.4), _aw(m, "A", 1, "mm", 7.2), _aw(m, "A", 2, "okay", 7.5)],
    )
    # One segment, three children, bare ids, no id(...) wrapper.
    children = "\n".join(
        f'    <nite:child href="{m}.A.words.xml#{m}.A.{sfx}"/>' for sfx in ("sil0", "aw1", "aw2")
    )
    segments = (
        f'{HEADER}\n<nite:root nite:id="{m}.A.aseg" {NS}>\n'
        f'  <segment nite:id="{m}.A.asg.1" participant="m000">\n{children}\n  </segment>\n'
        f"</nite:root>"
    )

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{d}/{m}.A.words.xml", words)
        z.writestr(f"{d}/{m}.A.segments.xml", segments)

    (t,) = ami.load_archive(path, variant="asr")
    assert t.n_turns == 1
    # All three children resolved: both words present, silence dropped and counted.
    assert t.turns[0].text == "mm okay"
    assert t.meta["dropped_sil"] == "1"


def test_every_meeting_with_files_yields_a_transcript(asr_archive: Path) -> None:
    """A count cross-check, because this failure mode produces no error at all."""
    ids = ami.meeting_ids(asr_archive, variant="asr")
    got = {t.id.split("/", 1)[1] for t in ami.load_archive(asr_archive, variant="asr")}
    assert set(ids) == got
