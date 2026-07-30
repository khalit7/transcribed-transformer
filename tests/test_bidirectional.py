"""Arm C: the causal-to-bidirectional conversion.

Two things must be true and both fail silently if they are not. The mask flip has
to actually let information flow backwards — a conversion that quietly stays
causal still trains and still reports itself as Arm C. And MNTP's labels have to
sit one position earlier than the masked token, or the model is asked to use an
output head at an offset it was never pretrained for.
"""

import pytest
import torch

from tt.models.bidirectional import (
    BidirectionalDecoder,
    bidirectional_mask,
    mask_mapping,
    mntp_batch,
)


class _Config:
    def __init__(self, layer_types: list[str] | None = None) -> None:
        self.layer_types = layer_types or ["full_attention"] * 4


# --- the mask ----------------------------------------------------------------


def test_mask_allows_both_directions() -> None:
    """The whole point. A causal mask blocks the upper triangle; this must not."""
    mask = bidirectional_mask(torch.ones(1, 4, dtype=torch.long))
    assert mask.shape == (1, 1, 4, 4)
    assert (mask == 0).all(), "every real position may attend to every other"


def test_padding_is_blocked_as_a_key() -> None:
    mask = bidirectional_mask(torch.tensor([[1, 1, 0, 0]]))
    blocked = mask[0, 0]
    assert (blocked[:, :2] == 0).all(), "real keys stay visible"
    assert (blocked[:, 2:] < -1e30).all(), "nothing attends to padding"


def test_mask_requires_two_dimensions() -> None:
    with pytest.raises(ValueError, match="batch, seq"):
        bidirectional_mask(torch.ones(1, 1, 4))


def test_mapping_covers_every_layer_type() -> None:
    mapping = mask_mapping(torch.ones(2, 5, dtype=torch.long), _Config())
    assert set(mapping) == {"full_attention"}
    assert mapping["full_attention"].shape == (2, 1, 5, 5)


def test_sliding_window_models_are_refused() -> None:
    """A window caps how far 'bidirectional' actually reaches, so converting one
    silently would misrepresent what the arm is testing."""
    config = _Config(["full_attention", "sliding_attention"])
    with pytest.raises(ValueError, match="non-full attention"):
        mask_mapping(torch.ones(1, 4, dtype=torch.long), config)


def test_mask_dtype_follows_the_model() -> None:
    mask = bidirectional_mask(torch.tensor([[1, 0]]), dtype=torch.bfloat16)
    assert mask.dtype == torch.bfloat16
    assert mask[0, 0, 0, 1] < -1e30 or mask[0, 0, 0, 1] == torch.finfo(torch.bfloat16).min


# --- MNTP --------------------------------------------------------------------


def test_labels_sit_one_position_before_the_masked_token() -> None:
    """The detail that makes the conversion cheap.

    The decoder was pretrained to predict position i from the hidden state at
    i-1, and its output head is trained for that offset. Scoring at i instead
    would spend early adaptation relearning the offset rather than learning
    bidirectionality.
    """
    torch.manual_seed(0)
    ids = torch.arange(1, 11).unsqueeze(0)
    batch = mntp_batch(ids, mask_token_id=999, vocab_size=1000, probability=1.0)

    masked_positions = (batch["input_ids"] == 999)[0].nonzero().flatten().tolist()
    scored_positions = (batch["labels"] != -100)[0].nonzero().flatten().tolist()
    assert masked_positions, "probability 1.0 must mask something"
    assert scored_positions == [p - 1 for p in masked_positions]

    for position in scored_positions:
        assert batch["labels"][0, position] == ids[0, position + 1], "label is the masked token"


def test_position_zero_is_never_a_target() -> None:
    """Nothing precedes it to predict from."""
    for seed in range(5):
        torch.manual_seed(seed)
        batch = mntp_batch(
            torch.arange(1, 9).unsqueeze(0), mask_token_id=99, vocab_size=100, probability=1.0
        )
        assert batch["input_ids"][0, 0] != 99


def test_unselected_positions_carry_no_loss() -> None:
    batch = mntp_batch(
        torch.arange(1, 9).unsqueeze(0), mask_token_id=99, vocab_size=100, probability=0.0
    )
    assert (batch["labels"] == -100).all()
    assert (batch["input_ids"] == torch.arange(1, 9)).all(), "nothing corrupted"


def test_input_is_actually_corrupted() -> None:
    """Same failure mode MLM had: selecting without corrupting trains identity."""
    ids = torch.arange(1, 33).unsqueeze(0)
    batch = mntp_batch(ids, mask_token_id=999, vocab_size=1000, probability=0.5)
    assert not torch.equal(batch["input_ids"], ids)


def test_probability_is_validated() -> None:
    with pytest.raises(ValueError, match="probability"):
        mntp_batch(
            torch.ones(1, 4, dtype=torch.long), mask_token_id=1, vocab_size=10, probability=2.0
        )


# --- wrapper -----------------------------------------------------------------


def test_wrapper_passes_a_dict_mask_through() -> None:
    """Passing a dict is what stops the model building its own causal mask.

    A tensor here would be rebuilt into a causal mask internally and the arm
    would quietly stay causal.
    """
    seen: dict[str, object] = {}

    class _Model(torch.nn.Module):
        config = _Config()

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self, input_ids: torch.Tensor, attention_mask: object = None, **_: object
        ) -> torch.Tensor:
            seen["mask"] = attention_mask
            return input_ids

    wrapped = BidirectionalDecoder(_Model())
    wrapped(torch.ones(1, 4, dtype=torch.long))
    assert isinstance(seen["mask"], dict), "must be a dict, or the model rebuilds a causal mask"
    assert "full_attention" in seen["mask"]
