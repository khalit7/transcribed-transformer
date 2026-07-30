"""Arm C: turning a causal decoder into a bidirectional encoder.

The cheap conversion. If it works, nobody needs to pretrain an encoder — you take
a decoder that already exists and flip its attention. [Ettin](https://arxiv.org/abs/2507.11412)
predicts it loses to a native encoder and that the gap widens with scale;
[T5Gemma](https://arxiv.org/abs/2504.06225) predicts adaptation succeeds. Whichever
way this lands is worth knowing, which is why it is an arm rather than a bet.

## Flipping the mask without patching the model

``Qwen3Model.forward`` builds its causal mask only when ``attention_mask`` is not
already a dict::

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        ...  # build causal masks here

Passing a dict keyed by layer type therefore replaces mask construction outright,
with no subclassing and no monkeypatching. That matters for maintenance: a patched
`create_causal_mask` would silently stop applying when transformers reorganises
its masking, and the arm would quietly revert to causal attention while still
reporting itself as bidirectional.

The mask is additive float, ``0`` where attention is allowed and a large negative
where it is not, shaped ``(batch, 1, q_len, kv_len)``. Padding is blocked in the
key dimension so no position attends to padding.

## MNTP, and why the labels are shifted

LLM2Vec adapts with **masked next token prediction** rather than plain MLM. The
model spent its whole pretraining predicting position ``i`` from the hidden state
at position ``i-1``, and its output head is trained for exactly that offset. Plain
MLM asks it to predict position ``i`` from position ``i``, which the head has never
done, so a large part of early adaptation is spent relearning the offset rather
than learning bidirectionality.

So a token is masked in the input at position ``i``, and the loss is taken from
the logits at position ``i-1``. That reuses the pretrained head as-is and is the
detail that makes this conversion cheap rather than merely possible.

## A causal decoder has no mask token

Qwen3 reports ``mask_token_id`` of ``None``, because nothing in causal
pretraining ever needed one. Some existing token has to be borrowed, and the
choice is not free: reusing a token that appears in real text teaches the model
to treat that token as a corruption marker everywhere, including where it occurs
legitimately.

Pick a reserved or special token that does not occur in the corpus, record which
one in the run config, and keep it fixed across the arm. Changing it midway
through adaptation silently changes the task.
"""

from __future__ import annotations

from typing import Any

import torch

MASK_VALUE = torch.finfo(torch.float32).min
"""Additive mask value for disallowed attention."""


def bidirectional_mask(
    attention_mask: torch.Tensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build an additive attention mask with no causal restriction.

    ``attention_mask`` is the usual ``(batch, seq)`` of 1 for real tokens and 0
    for padding. Every real position may attend to every other real position, in
    both directions; padding is blocked as a key so nothing attends to it.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"expected (batch, seq) attention mask, got {tuple(attention_mask.shape)}")

    keys_allowed = attention_mask[:, None, None, :].to(torch.bool)
    seq = attention_mask.size(1)
    out = torch.zeros(
        attention_mask.size(0), 1, seq, seq, dtype=dtype, device=attention_mask.device
    )
    return out.masked_fill(~keys_allowed, torch.finfo(dtype).min)


def mask_mapping(
    attention_mask: torch.Tensor, config: Any, *, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """The dict form ``Qwen3Model.forward`` accepts in place of building masks.

    Every layer type present in the config gets the same bidirectional mask. A
    model with sliding-window layers would need its window honoured here; those
    are rejected rather than silently converted, because a sliding window quietly
    limits how far "bidirectional" actually reaches.
    """
    layer_types = getattr(config, "layer_types", None) or ["full_attention"]
    kinds = set(layer_types)
    if kinds - {"full_attention"}:
        raise ValueError(
            f"model has non-full attention layers {sorted(kinds)}. A sliding window "
            "limits how far bidirectional attention actually reaches, so converting "
            "it silently would misrepresent what the arm is testing."
        )
    mask = bidirectional_mask(attention_mask, dtype=dtype)
    return dict.fromkeys(kinds, mask)


class BidirectionalDecoder(torch.nn.Module):
    """A causal decoder run with bidirectional attention.

    Wraps rather than subclasses, so the underlying model stays exactly the
    checkpoint it was loaded from and the conversion is one visible object rather
    than a mutation somewhere in the class hierarchy.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        dtype = next(self.model.parameters()).dtype
        mapping = mask_mapping(attention_mask, self.config, dtype=dtype)
        return self.model(input_ids=input_ids, attention_mask=mapping, **kwargs)


def mntp_batch(
    ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    probability: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Masked next token prediction, as LLM2Vec defines it.

    A token at position ``i`` is masked in the input and its label is placed at
    position ``i-1``, so the loss is taken from the logits the model was
    pretrained to produce there. Position 0 can never be a target: nothing
    precedes it to predict from.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    inputs = ids.clone()
    labels = torch.full_like(ids, -100)

    selected = torch.rand(ids.shape, device=ids.device) < probability
    selected[:, 0] = False  # nothing precedes position 0 to predict from

    inputs[selected] = mask_token_id
    # Shift: the label for a token masked at i is scored at i-1.
    labels[:, :-1] = torch.where(selected[:, 1:], ids[:, 1:], torch.full_like(ids[:, 1:], -100))
    del vocab_size  # kept for signature symmetry with the MLM objective
    return {"input_ids": inputs, "labels": labels}
