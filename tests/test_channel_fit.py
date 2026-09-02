import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.channel.fit import align, new_stats, normalise, wer


def test_normalise_hesitation_and_partials():
    assert normalise("Hello, (um) the n~ new plan.") == ["hello", "um", "the", "n", "new", "plan"]


def test_normalise_keeps_apostrophes():
    assert normalise("I'm 'cause it's fine!") == ["i'm", "'cause", "it's", "fine"]


def test_align_counts_each_edit_type():
    # Each case admits exactly one minimal alignment, so op attribution is forced.
    s = new_stats()
    align(["a", "b", "c"], ["a", "x", "c"], s)
    assert s["substitutions"][("b", "x")] == 1
    align(["a", "x", "b"], ["a", "b"], s)
    assert s["deletions"]["x"] == 1
    align(["a", "b"], ["a", "y", "b"], s)
    assert s["insertions"]["y"] == 1
    assert s["n_ref"] == 3 + 3 + 2 and s["n_hyp"] == 3 + 2 + 3
    assert abs(wer(s) - 3 / 8) < 1e-9


def test_perfect_alignment_zero_wer():
    s = new_stats()
    align(["a", "b"], ["a", "b"], s)
    assert wer(s) == 0.0
