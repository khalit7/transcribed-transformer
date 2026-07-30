"""Task heads: option scoring, per-line evidence, case aggregation.

Shared by arms A, B and C, so nothing here knows which trunk it is attached to.
A head takes hidden states and the positions from :mod:`tt.models.encoding` and
returns logits.

## Option scoring, and why there is no classification layer

The obvious head is a softmax over ``pass | fail | partial_pass | NA``. It cannot
work: answer vocabularies vary in arity, wording and polarity between question
authors, and a question written after training may use a label set nobody has
seen. So the head scores **the options it was given**, reading the hidden state
at each option's marker and projecting it to one number. Arity is whatever the
question supplied, and the softmax runs over that set alone.

This is also why the arm cannot emit an invalid answer: it selects among strings
it was handed and copies one back. There is no decoding path to a label nobody
offered.

## Evidence, and why it is per line rather than generated

The [P1a probe](../../../experiments/2026-07-29-evidence-index-probe/) measured
generative models emitting line indices over long transcripts. They never
produced an out-of-range index in 336 generations; what they did instead was stop
selecting lines and start enumerating them, until the output ran past its token
budget and the JSON was cut off mid-integer.

Tagging pools each line's token states and emits one logit per line. The output
is fixed-size, so there is no list to overrun and no counting fallback when
selection fails. That is the argument, stated as a property of the shape rather
than a hope about behaviour.
"""

from __future__ import annotations

import torch
from torch import nn

NEG_INF = -1e4
"""Mask value. Finite rather than -inf so a row that is entirely masked yields a
uniform distribution instead of NaN, which would poison the whole batch."""


class OptionScoringHead(nn.Module):
    """Score each supplied answer option from its marker's hidden state.

    Variable arity is handled by masking rather than padding to a fixed label
    count, so a two-option question and a five-option one cost the same code path.
    """

    def __init__(self, hidden_size: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        option_positions: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(batch, n_options)`` logits over the options each row supplied.

        ``option_positions`` is ``(batch, n_options)`` token indices and
        ``option_mask`` marks which are real, since rows in a batch may offer
        different numbers of options.
        """
        gathered = torch.gather(
            hidden, 1, option_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )
        logits: torch.Tensor = self.score(gathered).squeeze(-1)
        return logits.masked_fill(~option_mask, NEG_INF)


class EvidenceHead(nn.Module):
    """One logit per transcript line, from the mean of that line's token states.

    Mean-pooling rather than taking the first token: a line's first token is its
    line number, which carries no information about whether the line is evidence,
    and would let the head learn position instead of content.
    """

    def __init__(self, hidden_size: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, hidden: torch.Tensor, line_pool: torch.Tensor, line_mask: torch.Tensor
    ) -> torch.Tensor:
        """``(batch, n_lines)`` logits.

        ``line_pool`` is a ``(batch, n_lines, seq_len)`` averaging matrix built by
        :func:`pooling_matrix`; multiplying is far cheaper than looping over spans
        and keeps the whole thing on device.
        """
        pooled = torch.bmm(line_pool, hidden)
        logits: torch.Tensor = self.score(pooled).squeeze(-1)
        return logits.masked_fill(~line_mask, NEG_INF)


def pooling_matrix(
    spans: list[list[tuple[int, int]]], seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(batch, n_lines, seq_len)`` mean-pooling matrix and its mask.

    Row ``i`` holds ``1/len(span)`` across line ``i``'s tokens and zero elsewhere,
    so one batched matmul pools every line at once.
    """
    batch = len(spans)
    n_lines = max((len(s) for s in spans), default=0)
    pool = torch.zeros(batch, max(n_lines, 1), seq_len, device=device)
    mask = torch.zeros(batch, max(n_lines, 1), dtype=torch.bool, device=device)

    for b, rows in enumerate(spans):
        for i, (start, end) in enumerate(rows):
            end = min(end, seq_len)
            if end <= start:
                # A line truncated to nothing must stay masked: an all-zero
                # pooling row would emit a logit from the zero vector, which the
                # head would happily learn to treat as a real prediction.
                continue
            pool[b, i, start:end] = 1.0 / (end - start)
            mask[b, i] = True
    return pool, mask


class CaseAggregator(nn.Module):
    """Combine per-transcript option scores into one case-level answer.

    A case is several calls. Some questions pass if *any* call satisfies them
    ("was the customer's name established"), others only if *every* call does
    ("did the advisor close politely"). Which rule applies is a property of the
    question, and the model has to infer it from the question text rather than
    being told, so this is a learned pooling over transcripts rather than a
    hard-coded min or max.

    Attention over transcripts, conditioned on the question representation: the
    query comes from the question, so the same set of calls can aggregate
    differently for different questions.
    """

    def __init__(self, hidden_size: int, *, n_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attend = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        question_repr: torch.Tensor,
        transcript_reprs: torch.Tensor,
        transcript_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(batch, hidden)`` case representation.

        ``transcript_mask`` marks real transcripts; cases hold different numbers
        of calls and a padded slot must not contribute.
        """
        query = question_repr.unsqueeze(1)
        attended, _ = self.attend(
            query, transcript_reprs, transcript_reprs, key_padding_mask=~transcript_mask
        )
        out: torch.Tensor = self.norm(attended.squeeze(1) + question_repr)
        return out


class ComplianceHeads(nn.Module):
    """Both task heads over a shared trunk output.

    Kept together because they are trained jointly on the same forward pass; the
    whole efficiency argument for the encoder design is that answering and citing
    evidence cost one pass over the transcript, not two.
    """

    def __init__(self, hidden_size: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.answer = OptionScoringHead(hidden_size, dropout=dropout)
        self.evidence = EvidenceHead(hidden_size, dropout=dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        option_positions: torch.Tensor,
        option_mask: torch.Tensor,
        line_pool: torch.Tensor,
        line_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "answer_logits": self.answer(hidden, option_positions, option_mask),
            "evidence_logits": self.evidence(hidden, line_pool, line_mask),
        }


def answer_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy over the options each row actually supplied.

    Takes no mask: the head has already pushed unsupplied options to a large
    negative logit, so they hold no probability mass by the time loss is taken.
    """
    return nn.functional.cross_entropy(logits, target)


def evidence_loss(
    logits: torch.Tensor, target: torch.Tensor, line_mask: torch.Tensor
) -> torch.Tensor:
    """Binary cross entropy per line, averaged over real lines only.

    Padded lines are excluded rather than counted as negatives: a batch whose
    cases have very different line counts would otherwise be dominated by
    padding, and the head would learn to predict "not evidence" everywhere.
    """
    per_line = nn.functional.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none"
    )
    masked = per_line * line_mask.float()
    denominator = line_mask.float().sum().clamp(min=1.0)
    return masked.sum() / denominator
