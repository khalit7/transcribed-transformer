"""Data pipeline: prepare, stream, pack, shard, resume.

The failures worth testing here are all silent. A stream that restarts on resume
trains twice on the same tokens and the loss curve looks *better*. Two ranks
drawing identical data halves the effective batch and nothing errors. A mixed
licence track makes a model research-only with no visible symptom at all.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tt.data.pipeline import (
    MixtureSpec,
    PackedBatches,
    ShardIndex,
    TokenStream,
    open_mixture,
    prepare,
    token_dtype,
)
from tt.data.schema import Track


class _FakeTokenizer:
    """Deterministic word-id tokenizer, so expected token counts are computable."""

    name_or_path = "fake"
    eos_token_id = 1

    def __len__(self) -> int:
        return 1000

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": [(abs(hash(w)) % 900) + 10 for w in text.split()]}


def _write_stream(directory: Path, tokens: list[int], track: Track = Track.P) -> TokenStream:
    directory.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.uint16).tofile(directory / "shard-00000.bin")
    ShardIndex(
        source=directory.name,
        track=track,
        tokenizer="fake",
        dtype="uint16",
        total_tokens=len(tokens),
        shards=["shard-00000.bin"],
        n_documents=1,
    ).save(directory)
    return TokenStream(directory)


# --- dtype and index ---------------------------------------------------------


def test_dtype_narrows_for_small_vocabularies() -> None:
    """A 3B-token corpus is 6GB in uint16 and 12GB in uint32."""
    assert token_dtype(50_368) == np.uint16
    assert token_dtype(151_000) == np.uint32
    assert token_dtype(2**16) == np.uint32, "boundary: 65536 does not fit in uint16"


def test_index_round_trips(tmp_path: Path) -> None:
    idx = ShardIndex(
        source="ami",
        track=Track.P,
        tokenizer="m",
        dtype="uint16",
        total_tokens=10,
        shards=["a.bin"],
        n_documents=2,
        variant="asr",
    )
    idx.save(tmp_path)
    loaded = ShardIndex.load(tmp_path)
    assert loaded == idx
    assert loaded.track is Track.P
    assert loaded.variant == "asr"


# --- prepare -----------------------------------------------------------------


def test_prepare_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown source"):
        prepare("nope", tmp_path, tokenizer=_FakeTokenizer())


def test_prepare_requires_a_separator(tmp_path: Path) -> None:
    """Without one, two transcripts become one continuous conversation."""

    class NoSep(_FakeTokenizer):
        eos_token_id = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="no separator token"):
        prepare("ami", tmp_path, tokenizer=NoSep())


# --- streaming ---------------------------------------------------------------


def test_block_reads_a_contiguous_span(tmp_path: Path) -> None:
    stream = _write_stream(tmp_path / "c", list(range(100)))
    assert stream.block(10, 5).tolist() == [10, 11, 12, 13, 14]


def test_block_wraps_past_the_end(tmp_path: Path) -> None:
    """A corpus smaller than the token budget is reread, which for Arm E is normal."""
    stream = _write_stream(tmp_path / "c", list(range(10)))
    assert stream.block(8, 4).tolist() == [8, 9, 0, 1]
    assert len(stream.block(0, 35)) == 35


def test_block_spans_multiple_shards(tmp_path: Path) -> None:
    directory = tmp_path / "multi"
    directory.mkdir()
    np.asarray(range(10), dtype=np.uint16).tofile(directory / "shard-00000.bin")
    np.asarray(range(100, 110), dtype=np.uint16).tofile(directory / "shard-00001.bin")
    ShardIndex(
        source="multi",
        track=Track.P,
        tokenizer="fake",
        dtype="uint16",
        total_tokens=20,
        shards=["shard-00000.bin", "shard-00001.bin"],
        n_documents=2,
    ).save(directory)
    stream = TokenStream(directory)
    assert stream.n_tokens == 20
    assert stream.block(8, 4).tolist() == [8, 9, 100, 101], "reads across the shard join"


# --- batching, determinism, sharding, resume ---------------------------------


def _batches(tmp_path: Path, **kw: Any) -> PackedBatches:
    stream = _write_stream(tmp_path / "c", list(range(5000)))
    opts: dict[str, Any] = {"seq_len": 16, "micro_batch_size": 2, "seed": 0}
    opts.update(kw)
    return PackedBatches([MixtureSpec("c", stream, 1.0)], **opts)


def test_batch_shape(tmp_path: Path) -> None:
    batch = next(iter(_batches(tmp_path)))
    assert batch.shape == (2, 16)


def test_stream_is_deterministic(tmp_path: Path) -> None:
    a = [b.copy() for _, b in zip(range(5), _batches(tmp_path), strict=False)]
    b = [x.copy() for _, x in zip(range(5), _batches(tmp_path), strict=False)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_different_seeds_give_different_data(tmp_path: Path) -> None:
    a = next(iter(_batches(tmp_path, seed=0)))
    b = next(iter(_batches(tmp_path, seed=1)))
    assert not np.array_equal(a, b)


def test_ranks_receive_different_data(tmp_path: Path) -> None:
    """Two ranks drawing the same rows halves the effective batch, silently."""
    r0 = next(iter(_batches(tmp_path, rank=0, world_size=2)))
    r1 = next(iter(_batches(tmp_path, rank=1, world_size=2)))
    assert not np.array_equal(r0, r1)


def test_resume_continues_rather_than_restarting(tmp_path: Path) -> None:
    """The silent one.

    A resumed run whose stream restarts re-trains on tokens it already saw while
    believing it moved on. Loss keeps falling — faster, because the data is
    familiar — so nothing looks wrong.
    """
    packed = _batches(tmp_path)
    straight = [b.copy() for _, b in zip(range(6), packed.from_step(0), strict=False)]
    resumed = [b.copy() for _, b in zip(range(3), packed.from_step(3), strict=False)]
    assert all(np.array_equal(straight[3 + i], resumed[i]) for i in range(3))


def test_resume_is_independent_of_world_size_at_the_run_level(tmp_path: Path) -> None:
    """Sequences are indexed globally, so the corpus a run sees does not depend
    on how many GPUs it happened to use."""
    one = _batches(tmp_path, world_size=1, rank=0, micro_batch_size=2)
    two_r0 = _batches(tmp_path, world_size=2, rank=0, micro_batch_size=1)
    two_r1 = _batches(tmp_path, world_size=2, rank=1, micro_batch_size=1)
    single = next(iter(one))
    assert np.array_equal(single[0], next(iter(two_r0))[0])
    assert np.array_equal(single[1], next(iter(two_r1))[0])


# --- mixture safety ----------------------------------------------------------


def test_mixture_rejects_crossed_licence_tracks(tmp_path: Path) -> None:
    """A mixed-track run makes a model research-only with no visible symptom."""
    p = _write_stream(tmp_path / "p", list(range(100)), track=Track.P)
    nc = _write_stream(tmp_path / "nc", list(range(100)), track=Track.NC)
    with pytest.raises(ValueError, match="licence tracks"):
        PackedBatches(
            [MixtureSpec("p", p, 1.0), MixtureSpec("nc", nc, 1.0)],
            seq_len=8,
            micro_batch_size=1,
        )


def test_empty_and_zero_weight_mixtures_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty mixture"):
        PackedBatches([], seq_len=8, micro_batch_size=1)
    stream = _write_stream(tmp_path / "c", list(range(100)))
    with pytest.raises(ValueError, match="sum to zero"):
        PackedBatches([MixtureSpec("c", stream, 0.0)], seq_len=8, micro_batch_size=1)


def test_weights_shift_the_sampling_proportions(tmp_path: Path) -> None:
    """Disjoint id ranges make the source of every sequence identifiable."""
    a = _write_stream(tmp_path / "a", [5] * 2000)
    b = _write_stream(tmp_path / "b", [900] * 2000)
    packed = PackedBatches(
        [MixtureSpec("a", a, 9.0), MixtureSpec("b", b, 1.0)],
        seq_len=4,
        micro_batch_size=1,
        seed=3,
    )
    seqs = [s.copy() for _, s in zip(range(300), packed, strict=False)]
    from_a = sum(1 for s in seqs if s[0][0] == 5)
    assert 0.8 < from_a / len(seqs) < 1.0, "roughly the 9:1 weighting"


def test_open_mixture_explains_a_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not prepared"):
        open_mixture({"ami": 1.0}, prepared_root=tmp_path)


def test_open_mixture_reads_prepared_corpora(tmp_path: Path) -> None:
    _write_stream(tmp_path / "ami", list(range(50)))
    (specs,) = open_mixture({"ami": 1.0}, prepared_root=tmp_path)
    assert specs.name == "ami"
    assert specs.stream.n_tokens == 50
    assert json.loads((tmp_path / "ami" / "index.json").read_text())["source"] == "ami"
