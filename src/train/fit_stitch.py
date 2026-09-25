"""Fit the stitch for E4-mix-and-match: an affine map from one T5Gemma 2 encoder's final states onto another's.

    uv run python -m src.train.fit_stitch --source google/t5gemma-2-1b-1b --target google/t5gemma-2-270m-270m \
        --out checkpoints/stitch/1b-to-270m.pt

Both text encoders run over the same training prompts (the encoder's input at fine-tuning time: transcript and
question, tokenised as the loop does; the two models share the Gemma tokenizer), their final states are paired
token by token, and ridge regression from running moments (`AffineMoments`) gives the map; held-out prompts give the explained variance,
the measure of how much of the target encoder's representation is linearly present in the source's. The map is
the decoder's starting point only; it trains with the rest afterwards.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.train.data import build_examples
from src.train.stitched import AffineMoments, explained_variance


def encoder_of(base: str, attn: str, device):
    full = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn)
    enc = full.model.encoder.text_model.to(device).eval()
    del full
    return enc


@torch.no_grad()
def states(enc, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    return h[mask.bool()].float()  # real tokens only, [n, d]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="encoder whose states are mapped (the donor encoder)")
    ap.add_argument("--target", required=True, help="encoder-decoder whose decoder will read the mapped states")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=Path, default=Path("data/labelled_data/train.jsonl"))
    ap.add_argument("--tokens", type=int, default=3_000_000, help="fit tokens (held-out: a tenth as many)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch-tokens", type=int, default=32768)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.target)
    ex, _ = build_examples(args.train, tok, ["labelled"], 16384, Path("data/interim/sft"))
    ex = [e for e in ex if e.n_prompt <= args.max_len]
    random.Random(args.seed).shuffle(ex)
    src, tgt = encoder_of(args.source, args.attn, device), encoder_of(args.target, args.attn, device)
    d_in, d_out = src.config.hidden_size, tgt.config.hidden_size
    pad = tok.pad_token_id
    t0 = time.time()

    def batches(budget: int, start: int):
        """Paired states over prompts from ex[start:] until `budget` tokens, one padded batch at a time."""
        n, i = 0, start
        while n < budget and i < len(ex):
            batch, longest = [], 0
            while i < len(ex) and (not batch or max(longest, ex[i].n_prompt) * (len(batch) + 1) <= args.batch_tokens):
                batch.append(ex[i]); longest = max(longest, ex[i].n_prompt); i += 1
            ids = torch.full((len(batch), longest), pad, dtype=torch.long); mask = torch.zeros_like(ids)
            for r, e in enumerate(batch):
                p = torch.from_numpy(e.input_ids[:e.n_prompt].astype("int64")); ids[r, :len(p)] = p; mask[r, :len(p)] = 1
            ids, mask = ids.to(device), mask.to(device)
            n += int(mask.sum())
            yield states(src, ids, mask), states(tgt, ids, mask), i

    # the fit set enters the moment matrices chunk by chunk (nothing is kept); the held-out set is small enough to keep
    moments = AffineMoments(d_in, d_out, device)
    nxt = 0
    for x, y, nxt in batches(args.tokens, 0):
        moments.add(x, y)
    xh, yh = [], []
    for x, y, _ in batches(args.tokens // 10, nxt):
        xh.append(x.cpu()); yh.append(y.cpu())
    xh, yh = torch.cat(xh), torch.cat(yh)
    print(f"fit tokens {moments.n:,}, held-out {xh.shape[0]:,}, {d_in} -> {d_out}, {time.time() - t0:.0f}s of encoding", flush=True)
    w, b = moments.solve(args.ridge)
    src.to("cpu"); tgt.to("cpu")  # the encoders are done; the held-out check needs the card's memory for a wide target
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ev = explained_variance(xh.double().to(device), yh.double().to(device), w, b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": w.float().cpu(), "bias": b.float().cpu(), "source": args.source, "target": args.target,
                "explained_variance_heldout": ev, "fit_tokens": moments.n, "ridge": args.ridge}, args.out)
    report = {"source": args.source, "target": args.target, "d_in": d_in, "d_out": d_out, "fit_tokens": moments.n,
              "heldout_tokens": int(xh.shape[0]), "explained_variance_heldout": round(ev, 4),
              "ridge": args.ridge, "seconds": round(time.time() - t0)}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
