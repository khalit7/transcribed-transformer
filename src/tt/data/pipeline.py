"""Corpus text to training batches.

Two stages, on purpose.

**Prepare** tokenises a corpus once into flat binary shards on disk. Arm E reads
3B tokens per epoch-equivalent; tokenising that on every run would waste hours
per run and, worse, make the data order depend on loader iteration details that
are easy to change by accident.

**Stream** memmaps those shards and yields packed, rank-sharded, resumable
batches. Because a shard is a flat array of token ids, position is just an
integer offset, which is what makes the stream seekable — and seekable is what
makes resume correct.

## Why resume needs more than restoring weights

The trainer restores model, optimiser and RNG. If the data stream restarted from
the beginning anyway, a resumed run would re-train on tokens it had already seen
while believing it had moved on. Nothing would look wrong: loss would keep
falling, in fact faster, because the data is familiar. So the stream is a pure
function of ``(seed, rank, position)`` and the trainer's step count is enough to
place it exactly.

## Packing, and the boundary it respects

Documents are concatenated with a separator and cut into fixed ``seq_len``
blocks, so no batch is mostly padding. A block therefore usually spans a document
boundary, which is standard for pretraining and fine for MLM and CLM.

It is **not** fine for the task heads later on, where a training example is one
case and must not be cut in half. That path uses whole documents, so packing is
opt-out rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tt.data.loaders import ami, taskmaster
from tt.data.schema import Track, Transcript

SHARD_TOKENS = 100_000_000
"""Tokens per shard file. Large enough that per-file overhead is irrelevant,
small enough that a partial prepare loses little work."""


TranscriptSource = Callable[..., Iterator[Transcript]]

REGISTRY: dict[str, TranscriptSource] = {
    "ami": ami.load,
    "taskmaster": taskmaster.load,
}
"""Corpus name to loader. A YAML names these; nothing imports a loader directly.

Adding a corpus is a registry entry plus a datasheet row, never a change to the
training loop.
"""


def token_dtype(vocab_size: int) -> np.dtype[Any]:
    """Smallest integer type that can hold this vocabulary.

    Halves the size of every shard on disk for models with a small vocabulary,
    which for a 3B-token corpus is the difference between 6GB and 12GB.
    """
    return np.dtype(np.uint16) if vocab_size < 2**16 else np.dtype(np.uint32)


@dataclass(frozen=True)
class ShardIndex:
    """What a prepared corpus directory contains.

    Written alongside the shards so a stream can be opened without re-reading
    gigabytes to discover its own shape, and so a corpus carries the provenance
    that makes a training run reproducible.
    """

    source: str
    track: Track
    tokenizer: str
    dtype: str
    total_tokens: int
    shards: list[str]
    n_documents: int
    variant: str | None = None
    channel_version: str | None = None

    def save(self, directory: Path) -> None:
        payload = {**self.__dict__, "track": self.track.value}
        (directory / "index.json").write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, directory: Path) -> ShardIndex:
        raw = json.loads((directory / "index.json").read_text())
        raw["track"] = Track(raw["track"])
        return cls(**raw)


def prepare(
    source: str,
    out_dir: Path,
    *,
    tokenizer: Any,
    variant: str | None = None,
    cache_dir: Path = Path("data/raw"),
    limit: int | None = None,
    separator_id: int | None = None,
) -> ShardIndex:
    """Tokenise a corpus into flat binary shards. Idempotent by directory.

    ``separator_id`` is written between documents so a packed block that spans a
    boundary still has the join marked. Defaults to the tokenizer's EOS, or SEP,
    and is required to be *something*: without it, two unrelated transcripts
    become one apparently-continuous conversation.
    """
    if source not in REGISTRY:
        raise KeyError(f"unknown source {source!r}; registered: {sorted(REGISTRY)}")

    if separator_id is None:
        separator_id = getattr(tokenizer, "eos_token_id", None) or getattr(
            tokenizer, "sep_token_id", None
        )
    if separator_id is None:
        raise ValueError(
            f"{source}: no separator token. Documents would be concatenated with no "
            "boundary marker, silently turning unrelated transcripts into one "
            "continuous conversation. Pass separator_id explicitly."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(len(tokenizer))

    loader = REGISTRY[source]
    kwargs: dict[str, Any] = {"cache_dir": cache_dir / source}
    if variant is not None:
        kwargs["variant"] = variant

    buffer: list[int] = []
    shards: list[str] = []
    total = 0
    n_docs = 0
    track: Track | None = None

    def flush(final: bool = False) -> None:
        nonlocal buffer
        while len(buffer) >= SHARD_TOKENS or (final and buffer):
            take = buffer[:SHARD_TOKENS] if len(buffer) >= SHARD_TOKENS else buffer
            name = f"shard-{len(shards):05d}.bin"
            np.asarray(take, dtype=dtype).tofile(out_dir / name)
            shards.append(name)
            buffer = buffer[len(take) :]

    for transcript in loader(**kwargs):
        if limit is not None and n_docs >= limit:
            break
        track = transcript.track
        ids = tokenizer(transcript.render(), add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(separator_id)
        total += len(ids) + 1
        n_docs += 1
        flush()

    flush(final=True)

    index = ShardIndex(
        source=source,
        track=track or Track.NC,
        tokenizer=getattr(tokenizer, "name_or_path", "unknown"),
        dtype=dtype.name,
        total_tokens=total,
        shards=shards,
        n_documents=n_docs,
        variant=variant,
    )
    index.save(out_dir)
    return index


class TokenStream:
    """A prepared corpus as one seekable sequence of token ids.

    Shards are memmapped rather than read, so opening a 6GB corpus costs nothing
    and only the blocks actually touched come off disk.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.index = ShardIndex.load(self.directory)
        self._dtype = np.dtype(self.index.dtype)
        self._arrays = [
            np.memmap(self.directory / name, dtype=self._dtype, mode="r")
            for name in self.index.shards
        ]
        self._lengths = [len(a) for a in self._arrays]
        self._offsets: list[int] = []
        running = 0
        for length in self._lengths:
            self._offsets.append(running)
            running += length
        self.n_tokens = running

    def block(self, start: int, length: int) -> np.ndarray[Any, Any]:
        """``length`` tokens from ``start``, wrapping past the end.

        Wrapping rather than stopping so a stream never runs dry mid-run: a
        corpus smaller than the token budget is reread, which for Arm E is the
        normal case rather than an error.
        """
        out = np.empty(length, dtype=self._dtype)
        written = 0
        position = start % self.n_tokens
        while written < length:
            shard = self._locate(position)
            local = position - self._offsets[shard]
            take = min(length - written, self._lengths[shard] - local)
            out[written : written + take] = self._arrays[shard][local : local + take]
            written += take
            position = (position + take) % self.n_tokens
        return out

    def _locate(self, position: int) -> int:
        for i, offset in enumerate(self._offsets):
            if position < offset + self._lengths[i]:
                return i
        return len(self._arrays) - 1


@dataclass
class MixtureSpec:
    """One corpus in a training mixture."""

    name: str
    stream: TokenStream
    weight: float


class PackedBatches:
    """Deterministic, rank-sharded, resumable batches of packed token ids.

    Determinism is the point. Given the same seed, world size and step, this
    yields exactly the same tokens, so a resumed run continues rather than
    silently re-reading data it has already trained on.
    """

    def __init__(
        self,
        mixture: Sequence[MixtureSpec],
        *,
        seq_len: int,
        micro_batch_size: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if not mixture:
            raise ValueError("empty mixture: nothing to train on")
        total = sum(m.weight for m in mixture)
        if total <= 0:
            raise ValueError("mixture weights sum to zero")

        self.mixture = list(mixture)
        self.probabilities = np.array([m.weight / total for m in mixture], dtype=np.float64)
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # Tracks tracks: a mixed-track batch makes a model research-only, and the
        # contamination is invisible once training has started.
        tracks = {m.stream.index.track for m in mixture}
        if len(tracks) > 1:
            raise ValueError(
                f"mixture spans licence tracks {sorted(t.value for t in tracks)}. "
                "Tracks must never mix in one training run."
            )
        self.track = tracks.pop()

    def sequence(self, global_index: int) -> np.ndarray[Any, Any]:
        """The ``global_index``-th sequence in the run, independent of who asks.

        Indexing globally rather than per-rank means the data a run sees depends
        only on the seed and the step, not on how many GPUs it happened to use.
        """
        rng = np.random.default_rng((self.seed, global_index))
        choice = int(rng.choice(len(self.mixture), p=self.probabilities))
        stream = self.mixture[choice].stream
        start = int(rng.integers(0, max(1, stream.n_tokens)))
        return stream.block(start, self.seq_len)

    def __iter__(self) -> Iterator[np.ndarray[Any, Any]]:
        return self.from_step(0)

    def from_step(self, step: int) -> Iterator[np.ndarray[Any, Any]]:
        """Yield batches as if the run were at ``step``. This is how resume works."""
        per_step = self.micro_batch_size * self.world_size
        index = step * per_step
        while True:
            rows = [
                self.sequence(index + self.rank * self.micro_batch_size + i)
                for i in range(self.micro_batch_size)
            ]
            yield np.stack(rows)
            index += per_step


def torch_batches(packed: PackedBatches, device: Any, *, start_step: int = 0) -> Iterator[Any]:
    """Adapt :class:`PackedBatches` to what the trainer consumes."""
    import torch

    for array in packed.from_step(start_step):
        yield torch.from_numpy(array.astype(np.int64)).to(device, non_blocking=True)


def open_mixture(
    sources: dict[str, float], prepared_root: Path = Path("data/processed/tokens")
) -> list[MixtureSpec]:
    """Open prepared corpora named by a config's ``data.sources``."""
    specs: list[MixtureSpec] = []
    for name, weight in sorted(sources.items()):
        directory = prepared_root / name
        if not (directory / "index.json").exists():
            raise FileNotFoundError(
                f"{name} is not prepared at {directory}. Run tt.data.pipeline prepare first; "
                "tokenising during training would make the data order depend on loader "
                "iteration details."
            )
        specs.append(MixtureSpec(name=name, stream=TokenStream(directory), weight=weight))
    return specs


def main() -> int:
    """Tokenise a corpus into shards.

    Separate from training on purpose: this runs once per corpus, and doing it
    inside a training job would make the data order depend on loader iteration
    details that are easy to change by accident.
    """
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="Prepare a corpus into token shards.")
    ap.add_argument("source", choices=sorted(REGISTRY))
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-large")
    ap.add_argument("--variant", default=None, help="Transcript layer, e.g. asr.")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed/tokens"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--limit", type=int, default=None, help="Documents, for smoke tests.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    name = args.source if args.variant is None else f"{args.source}-{args.variant}"
    out = args.out_root / name

    index = prepare(
        args.source,
        out,
        tokenizer=tokenizer,
        variant=args.variant,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )
    print(
        f"{name}: {index.n_documents:,} documents, {index.total_tokens:,} tokens, "
        f"{len(index.shards)} shard(s), dtype {index.dtype}, track {index.track.value}"
    )
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
