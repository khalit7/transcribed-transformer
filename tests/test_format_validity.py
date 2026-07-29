"""Format-validity gates.

This scorer decides what the evidence-index probe reports, so a bug here would
show up as a finding rather than as a test failure. The cases below are the ones
that would be silently mis-scored by a tolerant parser.
"""

import pytest

from tt.bench.format_validity import (
    AnswerOutcome,
    parse_json_object,
    score_answer,
    score_evidence,
)

PERMITTED = ["pass", "fail"]
GLOSSES = {
    "pass": "The adviser explained the fees clearly.",
    "fail": "The adviser did not explain the fees.",
}


def test_bare_object_parses() -> None:
    value, failures = parse_json_object('{"evidence": [1, 2], "reasoning": "x"}')
    assert value == {"evidence": [1, 2], "reasoning": "x"}
    assert failures == []


def test_code_fence_is_a_failure_not_something_to_strip() -> None:
    """Stripping the fence would hide the failure the production parser would hit."""
    value, failures = parse_json_object('```json\n{"evidence": [1]}\n```')
    assert "no_fence" in failures
    assert value is None


def test_prose_around_json_fails() -> None:
    _, failures = parse_json_object('Here is my answer: {"evidence": [1]}')
    assert "valid_json" in failures


def test_json_array_at_top_level_is_not_an_object() -> None:
    value, failures = parse_json_object("[1, 2, 3]")
    assert value is None
    assert "valid_json" in failures


def test_wellformed_evidence_passes_every_gate() -> None:
    r = score_evidence([2, 5, 9], n_lines=20)
    assert r.parsed == [2, 5, 9]
    assert r.fully_valid
    assert r.in_range_fraction == 1.0


def test_empty_list_is_valid() -> None:
    """`[]` is the correct way to say nothing was found."""
    r = score_evidence([], n_lines=20)
    assert r.fully_valid
    assert r.parsed == []


def test_string_that_looks_like_a_list_is_a_failure() -> None:
    """The silent one: a presence-only check passes this, then it yields nothing."""
    r = score_evidence("12, 15", n_lines=20)
    assert not r.structurally_valid
    assert "evidence_typed" in r.failures


def test_string_elements_are_a_failure() -> None:
    r = score_evidence(["12", "15"], n_lines=20)
    assert not r.structurally_valid
    assert "evidence_typed" in r.failures


def test_floats_are_a_failure() -> None:
    r = score_evidence([1.0, 2.5], n_lines=20)
    assert not r.structurally_valid


def test_booleans_are_not_line_numbers() -> None:
    """bool subclasses int in Python, so a naive isinstance check would let True through."""
    r = score_evidence([True, 2], n_lines=20)
    assert not r.structurally_valid


@pytest.mark.parametrize("sentinel", ["NA", "N/A", "-1", "none", ""])
def test_sentinels_instead_of_empty_list_are_failures(sentinel: str) -> None:
    r = score_evidence(sentinel, n_lines=20)
    assert "evidence_no_sentinel" in r.failures


def test_out_of_range_is_reported_with_the_fraction() -> None:
    """The fraction matters: 'mostly right' and 'entirely invented' are different failures."""
    r = score_evidence([1, 2, 99], n_lines=10)
    assert "evidence_in_range" in r.failures
    assert r.in_range_fraction == pytest.approx(2 / 3)


def test_zero_is_out_of_range_because_indices_are_one_based() -> None:
    r = score_evidence([0, 1], n_lines=10)
    assert "evidence_in_range" in r.failures


def test_duplicates_and_disorder_are_reported() -> None:
    assert "evidence_dedup_sorted" in score_evidence([3, 3, 5], n_lines=10).failures
    assert "evidence_dedup_sorted" in score_evidence([5, 3], n_lines=10).failures


def test_cardinality_cap() -> None:
    assert "evidence_cardinality" in score_evidence([1, 2, 3], n_lines=10, cap=2).failures
    assert "evidence_cardinality" not in score_evidence([1, 2], n_lines=10, cap=2).failures


def test_exact_answer_is_the_only_full_credit() -> None:
    assert score_answer("pass", PERMITTED, GLOSSES) is AnswerOutcome.EXACT


@pytest.mark.parametrize("emitted", ["Pass", "pass.", '"pass"', " pass "])
def test_near_misses_are_recovered_not_exact(emitted: str) -> None:
    """Case and punctuation variants are failures with partial credit, never successes."""
    assert score_answer(emitted, PERMITTED, GLOSSES) is AnswerOutcome.RECOVERED


def test_emitting_the_gloss_instead_of_the_label_is_recovered() -> None:
    """The specific near miss the strict scoring exists to expose."""
    assert score_answer(GLOSSES["fail"], PERMITTED, GLOSSES) is AnswerOutcome.RECOVERED


def test_template_echo_is_recovered() -> None:
    assert score_answer("Answer (fail): the adviser did not", PERMITTED, GLOSSES) is (
        AnswerOutcome.RECOVERED
    )


def test_unrelated_answer_is_invalid() -> None:
    assert score_answer("maybe", PERMITTED, GLOSSES) is AnswerOutcome.INVALID
    assert score_answer(3, PERMITTED, GLOSSES) is AnswerOutcome.INVALID


def test_longest_consecutive_run_detects_enumeration() -> None:
    """Distinguishes selecting lines from counting upward through them.

    A degenerate `[112, 113, ..., 164]` is well-typed, in range and ascending, so
    every other gate passes it. This is the only signal that catches it.
    """
    from tt.bench.index_probe import longest_consecutive_run

    assert longest_consecutive_run([]) == 0
    assert longest_consecutive_run([5]) == 1
    assert longest_consecutive_run([2, 7, 19]) == 1, "genuine selection has no runs"
    assert longest_consecutive_run(list(range(112, 165))) == 53
    assert longest_consecutive_run([1, 2, 3, 40, 60, 61]) == 3
