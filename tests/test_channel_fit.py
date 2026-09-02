import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz.distance import Levenshtein

from src.channel.fit import align, clusters, new_stats, normalise, wer


def test_clusters_group_adjacent_ops_into_spans():
    ref = ["i'm", "gonna", "call", "you"]
    hyp = ["i'm", "going", "to", "call", "you", "bye", "now"]
    got = list(clusters(Levenshtein.editops(ref, hyp), ref, hyp))
    assert (["gonna"], ["going", "to"]) in got          # sub + adjacent insert = one span
    assert ([], ["bye", "now"]) in got                  # trailing insertion run = one phrase
    assert len(got) == 2


def test_align_records_span_edits_and_file_wer():
    s = new_stats()
    align(["i'm", "gonna", "call"], ["i'm", "going", "to", "call"], s)
    assert s["span_edits"][("gonna", "going to")] == 1
    assert s["span_counts"]["gonna call"] == 1
    assert s["file_wers"] == [round(2 / 3, 4)]


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
