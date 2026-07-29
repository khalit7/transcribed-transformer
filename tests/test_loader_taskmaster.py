"""Taskmaster loader round trip.

Synthetic fixtures in the real record shape, so no corpus data is committed and
no network is touched. The cases reproduce the defects actually present in the
corpus: dialogues with no utterances, and utterances carrying real text under an
empty speaker label.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tt.data.loaders import taskmaster
from tt.data.schema import Role, Track


def _dialogue(
    cid: str, turns: list[tuple[str, str]], instruction: str = "restaurant-2"
) -> dict[str, Any]:
    return {
        "conversation_id": cid,
        "instruction_id": instruction,
        "utterances": [
            {"index": i, "speaker": speaker, "text": text}
            for i, (speaker, text) in enumerate(turns)
        ],
    }


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "tm2-flights.json"
    path.write_text(
        json.dumps(
            [
                _dialogue(
                    "dlg-0001",
                    [
                        ("USER", "Hi, I'm um looking to book a a flight to Dublin."),
                        ("ASSISTANT", "Sure, what date were you thinking?"),
                        ("USER", "Next Tuesday if that's possible."),
                    ],
                ),
                # Real defect: empty speaker, but the text is genuine speech.
                _dialogue(
                    "dlg-0002",
                    [
                        ("USER", "I need a hotel too."),
                        ("", "I'm sorry what was that?"),
                        ("ASSISTANT", "Of course, let me check."),
                    ],
                ),
                # Real defect: dialogue with no utterances at all.
                _dialogue("dlg-0003", []),
            ]
        )
    )
    return path


def test_round_trip_through_canonical_schema(data_file: Path) -> None:
    transcripts = list(taskmaster.load_file(data_file))
    # The empty dialogue is dropped, the other two survive.
    assert [t.id for t in transcripts] == ["taskmaster/dlg-0001", "taskmaster/dlg-0002"]

    t = transcripts[0]
    assert t.source == taskmaster.SOURCE
    assert t.track is Track.P
    assert t.is_asr is False
    assert [turn.index for turn in t.turns] == [0, 1, 2]
    assert t.meta["subset"] == "tm2-flights"
    assert t.meta["release"] == "tm2"


def test_roles_are_populated_not_unknown(data_file: Path) -> None:
    """The reason this corpus is here: it supports the advisor/customer split.

    AMI cannot do this, so a regression to UNKNOWN would silently remove the one
    structural advantage Taskmaster has over it.
    """
    t = next(iter(taskmaster.load_file(data_file)))
    assert [turn.role for turn in t.turns] == [Role.CUSTOMER, Role.ADVISOR, Role.CUSTOMER]
    assert [turn.speaker for turn in t.turns] == ["USER", "ASSISTANT", "USER"]


def test_disfluencies_are_preserved(data_file: Path) -> None:
    t = next(iter(taskmaster.load_file(data_file)))
    assert "um" in t.turns[0].text
    assert "a a flight" in t.turns[0].text, "repetition must survive"


def test_empty_speaker_keeps_its_text_and_does_not_shift_indices(data_file: Path) -> None:
    """Dropping these would discard real speech and renumber every later line."""
    t = list(taskmaster.load_file(data_file))[1]
    assert t.n_turns == 3
    assert t.turns[1].text == "I'm sorry what was that?"
    assert t.turns[1].role is Role.UNKNOWN
    assert t.turns[1].speaker == "UNKNOWN"
    assert t.meta["n_unknown_speaker"] == "1"
    # The turn after it keeps its position.
    assert t.turns[2].role is Role.ADVISOR


def test_dialogue_with_no_turns_is_dropped(data_file: Path) -> None:
    assert "taskmaster/dlg-0003" not in {t.id for t in taskmaster.load_file(data_file)}


def test_indices_are_rebuilt_not_trusted(tmp_path: Path) -> None:
    """Upstream indices are contiguous today. The schema's guarantee cannot depend on that."""
    path = tmp_path / "tm2-movies.json"
    path.write_text(
        json.dumps(
            [
                {
                    "conversation_id": "dlg-gap",
                    "instruction_id": "x",
                    "utterances": [
                        {"index": 5, "speaker": "USER", "text": "one"},
                        {"index": 9, "speaker": "ASSISTANT", "text": "two"},
                    ],
                }
            ]
        )
    )
    t = next(iter(taskmaster.load_file(path)))
    assert [turn.index for turn in t.turns] == [0, 1]


def test_blank_text_utterances_do_not_become_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "tm2-music.json"
    path.write_text(json.dumps([_dialogue("dlg-blank", [("USER", "hello"), ("ASSISTANT", "   ")])]))
    t = next(iter(taskmaster.load_file(path)))
    assert t.n_turns == 1


def test_self_dialogue_file_is_refused(tmp_path: Path) -> None:
    """The documented trap: same record schema, but typed prose rather than speech.

    Loading it would produce entirely valid Transcript objects and nothing
    downstream would notice, so the refusal has to happen here.
    """
    path = tmp_path / "self-dialogs.json"
    path.write_text(json.dumps([_dialogue("dlg-x", [("USER", "typed")])]))
    with pytest.raises(ValueError, match="self-dialogue"):
        list(taskmaster.load_file(path))


def test_render_resolves_back_to_turns(data_file: Path) -> None:
    t = next(iter(taskmaster.load_file(data_file)))
    lines = t.render().split("\n")
    assert lines[0].startswith("1: USER: ")
    assert t.turns[t.line_to_index(2)].speaker == "ASSISTANT"


def test_self_dialogue_file_is_not_in_the_download_list() -> None:
    """Belt and braces: the refusal above only helps if nothing routes around it."""
    assert not any("self-dialog" in name for name in taskmaster.FILES)
    assert set(taskmaster.FILES) == set(taskmaster.CHECKSUMS)
