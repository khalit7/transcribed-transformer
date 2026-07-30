"""Task heads and case encoding.

Two failure classes matter here and both are silent. If line spans are wrong the
evidence head is trained to tag the wrong lines, and the loss falls perfectly
well. If option positions are wrong the answer head scores the wrong text, and
accuracy just looks mediocre rather than broken.
"""

from typing import Any

import pytest
import torch

from tt.data.schema import (
    AnswerOption,
    CaseSemantics,
    ComplianceQuestion,
    Role,
    Track,
    Transcript,
    Turn,
)
from tt.models.encoding import encode_case
from tt.models.heads import (
    NEG_INF,
    CaseAggregator,
    ComplianceHeads,
    answer_loss,
    evidence_loss,
    pooling_matrix,
)


class _Tok:
    """One token per word, with deterministic ids and real special tokens."""

    cls_token, sep_token = "[CLS]", "[SEP]"
    cls_token_id, sep_token_id = 101, 102

    def __len__(self) -> int:
        return 1000

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        if text == "[unused0]":
            return {"input_ids": [1]}
        if text == "[CLS]":
            return {"input_ids": [101]}
        if text == "[SEP]":
            return {"input_ids": [102]}
        return {"input_ids": [(abs(hash(w)) % 800) + 150 for w in text.split()]}


def _question(n_options: int = 2) -> ComplianceQuestion:
    labels = ["pass", "fail", "partial", "NA", "unknown"][:n_options]
    return ComplianceQuestion(
        id="q1",
        text="Did the advisor explain the fees?",
        options=[AnswerOption(value=v, criteria=f"criteria for {v}") for v in labels],
        family="clarity",
        semantics=CaseSemantics.ALL,
    )


def _transcript(n_turns: int = 5) -> Transcript:
    return Transcript(
        id="t1",
        source="fixture",
        track=Track.P,
        turns=[
            Turn(index=i, speaker="spk", role=Role.UNKNOWN, text=f"um line number {i} here")
            for i in range(n_turns)
        ],
        is_asr=True,
    )


# --- encoding ---------------------------------------------------------------


def test_every_option_gets_a_marker_position() -> None:
    for n in (2, 3, 5):
        enc = encode_case(_transcript(), _question(n), _Tok(), max_length=512)
        assert enc.n_options == n
        assert len(set(enc.option_positions)) == n, "positions must be distinct"
        assert all(enc.input_ids[p] == 1 for p in enc.option_positions), "marker token at each"


def test_line_spans_cover_the_right_tokens() -> None:
    """If spans drift, the evidence head tags the wrong lines and nothing errors."""
    tok = _Tok()
    transcript = _transcript(4)
    enc = encode_case(transcript, _question(), tok, max_length=512)
    assert len(enc.line_spans) == 4
    rendered = transcript.render().split("\n")
    for (start, end), line in zip(enc.line_spans, rendered, strict=True):
        assert enc.input_ids[start:end] == tok(line)["input_ids"]


def test_spans_are_contiguous_and_non_overlapping() -> None:
    enc = encode_case(_transcript(6), _question(), _Tok(), max_length=512)
    for (_, end), (next_start, _) in zip(enc.line_spans, enc.line_spans[1:], strict=False):
        assert end == next_start


def test_question_precedes_the_transcript() -> None:
    """Truncation removes transcript from the end, so options must sit before it.

    An input truncated into the option block would silently change the label set
    the model is choosing from.
    """
    enc = encode_case(_transcript(), _question(3), _Tok(), max_length=512)
    assert max(enc.option_positions) < enc.line_spans[0][0]


def test_truncation_drops_whole_lines_and_reports_it() -> None:
    tok = _Tok()
    enc = encode_case(_transcript(50), _question(), tok, max_length=80)
    assert enc.n_truncated_lines > 0
    assert len(enc.input_ids) <= 80
    # Surviving lines are still whole.
    rendered = _transcript(50).render().split("\n")
    for (start, end), number in zip(enc.line_spans, enc.line_numbers, strict=True):
        assert enc.input_ids[start:end] == tok(rendered[number - 1])["input_ids"]


def test_surviving_lines_keep_their_original_numbers() -> None:
    """Evidence labels are line numbers, so renumbering after truncation would
    silently point every label at different text."""
    enc = encode_case(_transcript(50), _question(), _Tok(), max_length=80)
    assert enc.line_numbers == list(range(1, len(enc.line_spans) + 1))
    assert enc.line_numbers[0] == 1


def test_options_that_cannot_fit_raise_rather_than_truncate() -> None:
    with pytest.raises(ValueError, match="drop answer options"):
        encode_case(_transcript(), _question(5), _Tok(), max_length=12)


# --- pooling ----------------------------------------------------------------


def test_pooling_matrix_averages_each_span() -> None:
    pool, mask = pooling_matrix([[(0, 2), (2, 5)]], seq_len=6, device=torch.device("cpu"))
    assert mask.tolist() == [[True, True]]
    assert pool[0, 0].tolist() == pytest.approx([0.5, 0.5, 0, 0, 0, 0])
    assert pool[0, 1].tolist() == pytest.approx([0, 0, 1 / 3, 1 / 3, 1 / 3, 0])


def test_pooling_masks_empty_spans() -> None:
    """An all-zero pooling row would emit a logit from the zero vector, which the
    head would learn to treat as a real prediction."""
    _, mask = pooling_matrix([[(0, 2), (4, 4)]], seq_len=6, device=torch.device("cpu"))
    assert mask.tolist() == [[True, False]]


def test_pooling_pads_ragged_batches() -> None:
    pool, mask = pooling_matrix([[(0, 2)], [(0, 1), (1, 3)]], seq_len=4, device=torch.device("cpu"))
    assert pool.shape == (2, 2, 4)
    assert mask.tolist() == [[True, False], [True, True]]


# --- heads ------------------------------------------------------------------


def test_option_head_handles_variable_arity() -> None:
    """Two rows offering different numbers of options in one batch."""
    heads = ComplianceHeads(hidden_size=16)
    hidden = torch.randn(2, 20, 16)
    positions = torch.tensor([[1, 2, 0], [1, 2, 3]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    logits = heads.answer(hidden, positions, mask)
    assert logits.shape == (2, 3)
    assert logits[0, 2].item() == pytest.approx(NEG_INF), "unsupplied option is suppressed"
    probs = logits.softmax(-1)
    assert probs[0, 2].item() < 1e-3, "and carries no probability mass"
    assert probs[0, :2].sum().item() == pytest.approx(1.0, abs=1e-3)


def test_evidence_head_emits_one_logit_per_line() -> None:
    heads = ComplianceHeads(hidden_size=16)
    hidden = torch.randn(2, 20, 16)
    pool, mask = pooling_matrix(
        [[(0, 3), (3, 6)], [(0, 2)]], seq_len=20, device=torch.device("cpu")
    )
    logits = heads.evidence(hidden, pool, mask)
    assert logits.shape == (2, 2)
    assert logits[1, 1].item() == pytest.approx(NEG_INF), "padded line suppressed"


def test_output_size_is_fixed_by_line_count() -> None:
    """The structural argument: nothing here can overrun a token budget.

    A generative model emits a variable-length list and can enumerate until it
    runs out of output tokens. This emits exactly one decision per line.
    """
    heads = ComplianceHeads(hidden_size=16)
    for n_lines in (1, 7, 40):
        spans = [[(i, i + 1) for i in range(n_lines)]]
        pool, mask = pooling_matrix(spans, seq_len=64, device=torch.device("cpu"))
        logits = heads.evidence(torch.randn(1, 64, 16), pool, mask)
        assert logits.shape == (1, n_lines)


def test_joint_forward_returns_both_heads() -> None:
    heads = ComplianceHeads(hidden_size=16)
    pool, line_mask = pooling_matrix([[(0, 4), (4, 8)]], seq_len=20, device=torch.device("cpu"))
    out = heads(
        torch.randn(1, 20, 16),
        option_positions=torch.tensor([[1, 2]]),
        option_mask=torch.tensor([[True, True]]),
        line_pool=pool,
        line_mask=line_mask,
    )
    assert out["answer_logits"].shape == (1, 2)
    assert out["evidence_logits"].shape == (1, 2)


def test_losses_are_finite_and_backpropagate() -> None:
    heads = ComplianceHeads(hidden_size=16)
    hidden = torch.randn(1, 20, 16, requires_grad=True)
    pool, line_mask = pooling_matrix([[(0, 4), (4, 8)]], seq_len=20, device=torch.device("cpu"))
    out = heads(
        hidden,
        option_positions=torch.tensor([[1, 2]]),
        option_mask=torch.tensor([[True, True]]),
        line_pool=pool,
        line_mask=line_mask,
    )
    loss = answer_loss(out["answer_logits"], torch.tensor([0])) + evidence_loss(
        out["evidence_logits"], torch.tensor([[1, 0]]), line_mask
    )
    assert torch.isfinite(loss)
    loss.backward()  # type: ignore[no-untyped-call]
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_evidence_loss_ignores_padded_lines() -> None:
    """Counting padding as negatives teaches the head to say 'not evidence' everywhere.

    The logits are deliberately non-uniform. With all-zero logits every element
    has identical BCE, so masking could not change the mean and the test would
    pass no matter what the implementation did.
    """
    logits = torch.tensor([[5.0, 5.0, -5.0, -5.0]])
    target = torch.tensor([[1, 1, 1, 1]])
    all_real = evidence_loss(logits, target, torch.tensor([[True] * 4]))
    confident_only = evidence_loss(logits, target, torch.tensor([[True, True, False, False]]))
    assert all_real > 2.0, "the two badly-predicted lines dominate"
    assert confident_only < 0.1, "masking them leaves only the well-predicted ones"


def test_case_aggregator_pools_over_transcripts() -> None:
    agg = CaseAggregator(hidden_size=16, n_heads=2)
    out = agg(
        torch.randn(2, 16),
        torch.randn(2, 3, 16),
        torch.tensor([[True, True, True], [True, False, False]]),
    )
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all(), "a case with one call must not produce NaN"
