"""Pack interim JSONL sources into training-ready token streams from a mixture manifest.

Manifest (configs/mixtures/<name>.yaml):
    stage: <name>
    tokenizer: <HF tokenizer id>          # each arm packs with its own; recorded in meta
    track: track-p | track-nc             # every source must carry this track (refused otherwise)
    sources:
      - path: data/interim/<name>/train.jsonl
        name: <name>                      # defaults to the interim directory name
        repeat: 1                         # upsampling factor
    val_sources:
      - path: data/interim/<name>/val.jsonl

Output: data/packed/<stage>/{train.bin,val.bin,meta.json}. Token streams are flat
uint32 (vocabularies above 65k need it), documents separated by the tokenizer's
EOS. meta.json is the machine-readable twin of the datasheet row: per-source
documents and tokens, tokenizer id and a fingerprint of its vocabulary, seed,
manifest path. The fingerprint is checked at load time so a pack built with one
vocabulary can never be trained with another.

Documents are shuffled globally (seeded) before writing so a sequential reader
sees sources interleaved. Usage:
    uv run python -m src.preprocessing.pack configs/mixtures/p_v1.yaml
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml
from tokenizers import Tokenizer

from src.preprocessing.common import REPO_ROOT

PACKED = REPO_ROOT / "data" / "packed"


def tokenizer_fingerprint(tok: Tokenizer) -> str:
    vocab = tok.get_vocab()
    h = hashlib.sha256()
    for token, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        h.update(f"{idx}\t{token}\n".encode())
    return h.hexdigest()[:16]


def eos_id(tok: Tokenizer) -> int:
    for cand in ("<|endoftext|>", "<|im_end|>", "</s>", "<eos>", "[SEP]"):
        i = tok.token_to_id(cand)
        if i is not None:
            return i
    raise SystemExit("no EOS token found in tokenizer vocabulary")


def pack_split(entries: list[dict], tok: Tokenizer, eos: int, track: str,
               out: Path, seed: int) -> dict:
    docs: list[tuple[str, str]] = []  # (source name, text)
    per_source: dict[str, dict] = {}
    for e in entries:
        path = REPO_ROOT / e["path"]
        name = e.get("name") or path.parent.name
        n = 0
        for line in path.open():
            d = json.loads(line)
            if d["track"] != track:
                raise SystemExit(f"track mix refused: {path} has {d['track']}, manifest is {track}")
            for _ in range(int(e.get("repeat", 1))):
                docs.append((name, d["text"]))
            n += 1
        per_source[name] = {"path": e["path"], "documents": n, "repeat": int(e.get("repeat", 1)), "tokens": 0}
    random.Random(seed).shuffle(docs)

    total = 0
    with open(out, "wb") as f:
        batch: list[str] = []
        names: list[str] = []
        for name, text in docs:
            batch.append(text)
            names.append(name)
            if len(batch) == 256:
                total += _write_batch(batch, names, tok, eos, f, per_source)
                batch, names = [], []
        if batch:
            total += _write_batch(batch, names, tok, eos, f, per_source)
    return {"tokens": total, "documents": len(docs), "sources": per_source}


def _write_batch(batch, names, tok, eos, f, per_source) -> int:
    encs = tok.encode_batch(batch)
    ids = []
    for name, enc in zip(names, encs):
        ids.extend(enc.ids)
        ids.append(eos)
        per_source[name]["tokens"] += len(enc.ids) + 1
    arr = np.asarray(ids, dtype=np.uint32)
    arr.tofile(f)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mpath = Path(args.manifest)
    m = yaml.safe_load(mpath.read_text())
    out_dir = PACKED / m["stage"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained(m["tokenizer"])
    eos = eos_id(tok)
    t0 = time.time()

    train = pack_split(m["sources"], tok, eos, m["track"], out_dir / "train.bin", args.seed)
    val = pack_split(m.get("val_sources", []), tok, eos, m["track"], out_dir / "val.bin", args.seed)

    meta = {
        "stage": m["stage"],
        "manifest": str(mpath),
        "track": m["track"],
        "tokenizer": m["tokenizer"],
        "tokenizer_fingerprint": tokenizer_fingerprint(tok),
        "eos_id": eos,
        "dtype": "uint32",
        "seed": args.seed,
        "train": train,
        "val": val,
        "built_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"train tokens {train['tokens']:,} over {train['documents']:,} docs; "
          f"val tokens {val['tokens']:,}; {meta['built_seconds']}s -> {out_dir}")
    for name, s in train["sources"].items():
        print(f"  {name:16s} docs={s['documents']:>8,} x{s['repeat']} tokens={s['tokens']:>14,}")


if __name__ == "__main__":
    main()
