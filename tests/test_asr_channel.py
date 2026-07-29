"""ASR channel alignment and fitting.

The channel is what makes Arm E possible at all, so an error here does not fail
loudly — it produces a plausible-looking corpus with the wrong noise in it. These
cases pin the decisions that would otherwise be silent.
"""

import pytest

from tt.data.asr_channel import (
    ChannelStats,
    Edit,
    Op,
    align_streams,
    align_words,
    is_punctuation,
    normalise,
)
from tt.data.loaders.ami import TimedWord


def _ops(edits: list[Edit]) -> list[Op]:
    return [e.op for e in edits]


def test_identical_streams_are_all_matches() -> None:
    edits = align_words(["the", "cat", "sat"], ["the", "cat", "sat"])
    assert _ops(edits) == [Op.MATCH] * 3


def test_substitution_is_one_error_not_two() -> None:
    """A mis-recognised word must not be scored as a deletion plus an insertion.

    Reporting it as two errors would roughly double the apparent error rate and
    lose the (reference, hypothesis) pair that the confusion distribution needs.
    """
    edits = align_words(["the", "cat", "sat"], ["the", "hat", "sat"])
    assert _ops(edits) == [Op.MATCH, Op.SUBSTITUTION, Op.MATCH]
    assert edits[1].ref == "cat"
    assert edits[1].hyp == "hat"


def test_deletion_and_insertion() -> None:
    assert _ops(align_words(["a", "b", "c"], ["a", "c"])) == [
        Op.MATCH,
        Op.DELETION,
        Op.MATCH,
    ]
    assert _ops(align_words(["a", "c"], ["a", "b", "c"])) == [
        Op.MATCH,
        Op.INSERTION,
        Op.MATCH,
    ]


def test_empty_sides() -> None:
    assert _ops(align_words([], ["a", "b"])) == [Op.INSERTION, Op.INSERTION]
    assert _ops(align_words(["a", "b"], [])) == [Op.DELETION, Op.DELETION]
    assert align_words([], []) == []


@pytest.mark.parametrize("token", [",", ".", "?", "--", "..."])
def test_punctuation_only_tokens_are_detected(token: str) -> None:
    assert is_punctuation(token)


@pytest.mark.parametrize("token", ["cat", "it's", "th-", "mm-hmm", "3"])
def test_words_are_not_punctuation(token: str) -> None:
    assert not is_punctuation(token)


def test_normalise_folds_case_and_edge_punctuation() -> None:
    """The recogniser removes both deterministically, so leaving them in would
    score every correctly recognised word as a substitution."""
    assert normalise("David.") == "david"
    assert normalise("I'm") == "i'm", "internal apostrophes are part of the word"
    assert normalise("th-") == "th", "truncations still normalise to a comparable form"


def test_reference_punctuation_does_not_count_as_deletions() -> None:
    """The specific inflation this guards against.

    AMI's manual layer emits punctuation as its own word element. Counting those
    as deletions would measure the transcriber's comma usage rather than anything
    the recogniser did.
    """
    ref = [
        TimedWord("Hi", 0.0, 0.2),
        TimedWord(",", 0.2, 0.2),
        TimedWord("David", 0.2, 0.6),
        TimedWord(".", 0.6, 0.6),
    ]
    hyp = [TimedWord("hi", 0.0, 0.2), TimedWord("david", 0.2, 0.6)]
    edits = align_streams(ref, hyp)
    assert _ops(edits) == [Op.MATCH, Op.MATCH]


def test_casing_difference_alone_is_a_match() -> None:
    ref = [TimedWord("David", 0.0, 0.4)]
    hyp = [TimedWord("david", 0.0, 0.4)]
    assert _ops(align_streams(ref, hyp)) == [Op.MATCH]


def test_time_windows_keep_distant_speech_from_aligning() -> None:
    """Chunking is what stops one bad match propagating through a whole meeting.

    Two unrelated words 100 seconds apart must not be paired as a substitution
    just because they happen to be adjacent in the token sequence.
    """
    ref = [TimedWord("hello", 0.0, 0.5), TimedWord("goodbye", 100.0, 100.5)]
    hyp = [TimedWord("hello", 0.0, 0.5), TimedWord("banana", 100.0, 100.5)]
    edits = align_streams(ref, hyp, window_s=20.0)
    assert _ops(edits) == [Op.MATCH, Op.SUBSTITUTION]

    # Same tokens, but the hypothesis word lands in a different window: now it is
    # a deletion and an insertion, not a substitution.
    hyp_far = [TimedWord("hello", 0.0, 0.5), TimedWord("banana", 40.0, 40.5)]
    ops = _ops(align_streams(ref, hyp_far, window_s=20.0))
    assert Op.SUBSTITUTION not in ops
    assert Op.DELETION in ops and Op.INSERTION in ops


def test_stats_accumulate_and_compute_wer() -> None:
    stats = ChannelStats()
    stats.update(
        [
            Edit(Op.MATCH, "the", "the"),
            Edit(Op.SUBSTITUTION, "cat", "hat"),
            Edit(Op.DELETION, "sat", None),
            Edit(Op.INSERTION, None, "um"),
        ]
    )
    # Insertions are hypothesis-side only, so N_reference counts match+sub+del.
    assert stats.n_reference_words == 3
    assert stats.n_hypothesis_words == 3
    assert stats.wer == pytest.approx(3 / 3)
    assert stats.substitutions[("cat", "hat")] == 1
    assert stats.deletions["sat"] == 1
    assert stats.insertions["um"] == 1


def test_stats_are_additive() -> None:
    """Fits over different meetings must be summable, which is why counts are stored."""
    a, b = ChannelStats(), ChannelStats()
    edits = [Edit(Op.MATCH, "x", "x"), Edit(Op.SUBSTITUTION, "y", "z")]
    a.update(edits)
    b.update(edits)
    combined = ChannelStats()
    combined.update(edits + edits)
    assert combined.n_reference_words == a.n_reference_words + b.n_reference_words
    assert combined.substitutions[("y", "z")] == 2


def test_round_trip_through_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Channel parameters are versioned as artifacts, so they must survive serialisation."""
    stats = ChannelStats(source="ami", asr_system="test-asr")
    stats.update([Edit(Op.SUBSTITUTION, "cat", "hat"), Edit(Op.DELETION, "sat", None)])
    path = tmp_path / "channel.json"
    stats.save(path)
    loaded = ChannelStats.load(path)
    assert loaded.summary() == stats.summary()
    assert loaded.substitutions[("cat", "hat")] == 1
    assert loaded.deletions["sat"] == 1
