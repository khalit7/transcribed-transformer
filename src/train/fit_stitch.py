"""Fit the stitch for E4-mix-and-match: an affine map from one T5Gemma 2 encoder's final states onto another's.

    uv run python -m src.train.fit_stitch --source google/t5gemma-2-1b-1b --target google/t5gemma-2-270m-270m \
        --out checkpoints/stitch/1b-to-270m.pt

Or, for E5-soft-tokens (`--target-embeddings`), onto a decoder-only model's scaled token embeddings of the same tokens, so the
decoder reads something like its own input at step 0:

    uv run python -m src.train.fit_stitch --source google/t5gemma-2-270m-270m --target google/gemma-3-1b-pt \
        --target-embeddings --out checkpoints/stitch/270m-to-gemma3-1b-embed.pt

Or, for E5-merged-attention (`--target-layer-mean`), onto a decoder-only model's layer inputs: the normalised hidden
state (the output of each layer's input norm, what its key and value projections read) that the decoder-only model computes
when it reads the prompt as tokens, averaged over its layers. One map serves every layer (as the encoder output serves
every T5Gemma 2 decoder layer), and least squares onto the average is the single map closest to every layer's input at
once; the held-out explained variance is reported on the average and on each layer:

    uv run python -m src.train.fit_stitch --source google/t5gemma-2-270m-270m --target google/gemma-3-270m \
        --target-layer-mean --train data/labelled_data_v2/train.jsonl --one-per-transcript --max-len 16384 \
        --out checkpoints/stitch/270m-to-gemma3-270m-layermean-v2.pt

`--one-per-transcript` takes one prompt per transcript (labelled data v2 asks 32 questions of every episode, so the
prompts would otherwise repeat each transcript many times), which also keeps the held-out prompts on transcripts the fit
never saw. Both text encoders run over the same training prompts (the encoder's input at fine-tuning time: transcript and
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

from src.train.data import build_examples, encoder_pad_len
from src.train.stitched import AffineMoments, explained_variance


def encoder_of(base: str, attn: str, device):
    full = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn)
    enc = full.model.encoder.text_model.to(device).eval()
    del full
    return enc


class LayerTargets(torch.nn.Module):
    """A decoder-only model read as a stack of targets: running it causally over the prompt, the output of every layer's
    input norm (what that layer's attention projects), one [n, d] tensor per layer."""

    def __init__(self, base: str, attn: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn)
        self.text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
        self.config = self.text.config
        self.n_layers = self.config.num_hidden_layers
        self._seen: list[torch.Tensor | None] = [None] * self.n_layers
        for i, layer in enumerate(self.text.layers[: self.n_layers]):
            layer.input_layernorm.register_forward_hook(self._hook(i))
        del lm

    def _hook(self, i: int):
        def keep(_module, _inputs, output):
            self._seen[i] = output
        return keep

    @torch.no_grad()
    def layer_states(self, ids: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        self.text(input_ids=ids, attention_mask=mask, use_cache=False)  # causal, right-padded: padding never reaches a real token
        out = [s[mask.bool()].float() for s in self._seen]  # type: ignore[union-attr]
        self._seen = [None] * self.n_layers
        return out


class StreamedVariance:
    """Held-out explained variance per target without keeping the rows: residual and total sums of squares (the total
    from Σy and Σy², per dimension), accumulated in float64."""

    def __init__(self, n: int):
        self.res = [0.0] * n
        self.sy: list[torch.Tensor | None] = [None] * n
        self.syy: list[torch.Tensor | None] = [None] * n
        self.rows = 0

    def add(self, x: torch.Tensor, ys: list[torch.Tensor], w: torch.Tensor, b: torch.Tensor) -> None:
        x = x.double()
        for i, y in enumerate(ys):
            y = y.double()
            pred = x @ w[i].to(x.device, x.dtype).T + b[i].to(x.device, x.dtype)
            self.res[i] += float((y - pred).pow(2).sum())
            sy, syy = y.sum(0), y.pow(2).sum(0)
            self.sy[i] = sy if self.sy[i] is None else self.sy[i] + sy
            self.syy[i] = syy if self.syy[i] is None else self.syy[i] + syy
        self.rows += x.shape[0]

    def explained(self) -> list[float]:
        out = []
        for res, sy, syy in zip(self.res, self.sy, self.syy):
            tot = float((syy - sy.pow(2) / self.rows).sum())  # type: ignore[union-attr]
            out.append(1.0 - res / tot)
        return out


class EmbeddingTarget(torch.nn.Module):
    """A decoder-only model's token embedding as the target "encoder": its (scaled) embedding of each token, so the
    stitch maps an encoder state onto what the decoder would have read at that position."""

    def __init__(self, base: str):
        super().__init__()
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model  # multimodal checkpoints (4B+)
        self.embed = text.embed_tokens  # Gemma 3 scales inside the module; the rest of the model is dropped
        self.config = text.config
        del lm

    def forward(self, input_ids, attention_mask=None):
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=self.embed(input_ids))


@torch.no_grad()
def states(enc, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    return h[mask.bool()].float()  # real tokens only, [n, d]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="encoder whose states are mapped (the donor encoder)")
    ap.add_argument("--target", required=True, help="encoder-decoder whose decoder will read the mapped states")
    ap.add_argument("--target-embeddings", action="store_true",
                    help="E5-soft-tokens: the target is a decoder-only model and the states map onto its token embeddings (soft tokens)")
    ap.add_argument("--target-layer-mean", action="store_true",
                    help="E5-merged-attention: one map onto a decoder-only target's normalised layer inputs averaged over its layers")
    ap.add_argument("--one-per-transcript", action="store_true",
                    help="one prompt per transcript (and held-out prompts on transcripts the fit never saw)")
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
    if args.target_layer_mean and args.target_embeddings:
        ap.error("--target-layer-mean and --target-embeddings are different targets; give one")
    tok = AutoTokenizer.from_pretrained(args.target)
    # one prompt per transcript: the seeded one-question subset (src.train.data.question_subset), so the prompts are
    # distinct transcripts and the fit and held-out sets never share one
    ex, _ = build_examples(args.train, tok, ["labelled"], 16384, Path("data/interim/sft"),
                           questions_per_episode=1 if args.one_per_transcript else None, question_seed=args.seed)
    ex = [e for e in ex if e.n_prompt <= args.max_len]
    random.Random(args.seed).shuffle(ex)
    src = encoder_of(args.source, args.attn, device)
    if args.target_layer_mean:
        tgt = LayerTargets(args.target, args.attn).to(device).eval()
    elif args.target_embeddings:
        tgt = EmbeddingTarget(args.target).to(device).eval()
    else:
        tgt = encoder_of(args.target, args.attn, device)
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
            ids = torch.full((len(batch), encoder_pad_len(longest)), pad, dtype=torch.long); mask = torch.zeros_like(ids)  # BUG-2: never an unpadded row
            for r, e in enumerate(batch):
                p = torch.from_numpy(e.input_ids[:e.n_prompt].astype("int64")); ids[r, :len(p)] = p; mask[r, :len(p)] = 1
            ids, mask = ids.to(device), mask.to(device)
            n += int(mask.sum())
            y = tgt.layer_states(ids, mask) if args.target_layer_mean else states(tgt, ids, mask)
            yield states(src, ids, mask), y, i

    if args.target_layer_mean:  # one map onto the layer average; held-out variance on the average and on every layer
        moments = AffineMoments(d_in, d_out, device)
        nxt = 0
        for x, ys, nxt in batches(args.tokens, 0):
            moments.add(x, torch.stack(ys, 0).mean(0))
        w, b = moments.solve(args.ridge)
        sv = StreamedVariance(tgt.n_layers + 1)
        for x, ys, _ in batches(args.tokens // 10, nxt):
            sv.add(x, [torch.stack(ys, 0).mean(0), *ys], [w] * (tgt.n_layers + 1), [b] * (tgt.n_layers + 1))
        evs = sv.explained()
        print(f"fit tokens {moments.n:,}, held-out {sv.rows:,}, {d_in} -> {d_out}, {time.time() - t0:.0f}s", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": w.float().cpu(), "bias": b.float().cpu(), "source": args.source, "target": args.target,
                    "target_layer_mean": True, "explained_variance_heldout": evs[0],
                    "explained_variance_heldout_per_layer": evs[1:], "fit_tokens": moments.n, "ridge": args.ridge,
                    "train": str(args.train), "one_per_transcript": args.one_per_transcript}, args.out)
        report = {"source": args.source, "target": args.target, "target_layer_mean": True, "d_in": d_in, "d_out": d_out,
                  "layers": tgt.n_layers, "fit_tokens": moments.n, "heldout_tokens": sv.rows,
                  "explained_variance_heldout": round(evs[0], 4),
                  "explained_variance_heldout_per_layer": [round(e, 4) for e in evs[1:]], "ridge": args.ridge,
                  "train": str(args.train), "one_per_transcript": args.one_per_transcript, "max_len": args.max_len,
                  "seconds": round(time.time() - t0)}
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report), flush=True)
        return

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
                "target_embeddings": args.target_embeddings,
                "explained_variance_heldout": ev, "fit_tokens": moments.n, "ridge": args.ridge}, args.out)
    report = {"source": args.source, "target": args.target, "target_embeddings": args.target_embeddings, "d_in": d_in, "d_out": d_out, "fit_tokens": moments.n,
              "heldout_tokens": int(xh.shape[0]), "explained_variance_heldout": round(ev, 4),
              "ridge": args.ridge, "train": str(args.train), "one_per_transcript": args.one_per_transcript,
              "max_len": args.max_len, "seconds": round(time.time() - t0)}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
