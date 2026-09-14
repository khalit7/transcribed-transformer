"""Generate the model's output for every pair of a split file, one JSON line per pair.

    .venv-vllm/bin/python -m src.train.generate <model_dir> <split.jsonl> <out.jsonl> [--variants clean,messy]
        [--max-tokens 512] [--tp 1] [--limit N] [--backend vllm|hf]

The prompt is src/train/data.task_prompt, the same text the model was trained on and the API
baseline is given. Greedy decoding. The output records carry the pair id, the variant rendered and
the raw text, and src/train/evaluate.py scores them; nothing here parses or repairs the output.
vLLM runs in its own environment (.venv-vllm) because it pins its own torch; --backend hf is the
slow fallback inside the training environment, and --backend prefixlm is the E2 path (bidirectional
prefill with the prefix-LM mask, then ordinary cached decoding), also inside the training environment.
"""

import argparse
import json
import os
import time
from pathlib import Path

from src.train.data import SEP, record_prompt


def load_pairs(split: Path, variants: list[str], limit: int | None) -> list[tuple[str, str, str]]:
    out = []
    for line in split.open():
        r = json.loads(line)
        have = {v["kind"] for v in r["transcript"]["variants"]}
        for v in variants:
            v = r["generation_info"]["labelled_variant"] if v == "labelled" else v
            if v in have:
                out.append((r["id"], v, record_prompt(r, v) + SEP))
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("split", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--variants", default="clean,messy")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["vllm", "hf", "prefixlm"], default="vllm")
    ap.add_argument("--batch", type=int, default=16, help="prefixlm backend: sequences per batch")
    args = ap.parse_args()
    pairs = load_pairs(args.split, args.variants.split(","), args.limit)
    done = set()
    if args.out.exists():
        done = {(json.loads(l)["id"], json.loads(l)["variant"]) for l in args.out.open()}
        pairs = [p for p in pairs if (p[0], p[1]) not in done]
    print(f"{len(pairs)} prompts to generate ({len(done)} already in {args.out})", flush=True)
    if not pairs:
        return
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "vllm":
        # FlashInfer's sampler fails its device-capability check on sm_120 (RTX 5090); vLLM's own sampler is fine
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
        llm = LLM(model=args.model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
                  tensor_parallel_size=args.tp, gpu_memory_utilization=0.9, enable_prefix_caching=True)
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        with args.out.open("a") as f:
            for i in range(0, len(pairs), 2048):
                chunk = pairs[i:i + 2048]
                outs = llm.generate([p for _, _, p in chunk], sp)
                for (id_, v, _), o in zip(chunk, outs):
                    f.write(json.dumps({"id": id_, "variant": v, "text": o.outputs[0].text,
                                        "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(o.outputs[0].token_ids)}) + "\n")
                f.flush()
                print(f"[{i + len(chunk)}/{len(pairs)}] {time.time() - t0:.0f}s", flush=True)
    elif args.backend == "prefixlm":
        with args.out.open("a") as f:
            generate_prefix_lm(args.model_dir, pairs, f, args.max_tokens, args.batch)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).cuda().eval()  # type: ignore[call-arg]
        with args.out.open("a") as f:
            for i in range(0, len(pairs), 4):
                chunk = pairs[i:i + 4]
                enc = tok([p for _, _, p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(  # type: ignore[call-arg,operator]
                        **enc, max_new_tokens=args.max_tokens, do_sample=False)
                for (id_, v, _), row in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                    f.write(json.dumps({"id": id_, "variant": v, "text": tok.decode(row, skip_special_tokens=True),
                                        "prompt_tokens": int(enc["attention_mask"].sum(1).max()), "output_tokens": int((row != tok.pad_token_id).sum())}) + "\n")
                f.flush()
    print(f"done: {len(pairs)} in {time.time() - t0:.0f}s -> {args.out}")



def generate_prefix_lm(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 16,
                       causal: bool = False) -> None:
    """Greedy decoding for a prefix-LM model (E2): vLLM cannot express the mask, so this is a plain HF
    loop. Prompts are left-padded and sorted by length; the prefill uses the prefix-LM mask (every
    prompt token sees every prompt token), then each generated token attends causally to everything
    before it through the KV cache, which is ordinary decoding with a 2D padding mask. causal=True
    replaces the prefill mask with a causal one, which must reproduce a causal model's outputs (a check
    of the loop, not a mode anyone evaluates with)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    from src.train.masks import (
        FLEX_KERNEL_OPTIONS,
        left_padded_block_mask,
        left_padded_prompt_mask,
    )

    tok = AutoTokenizer.from_pretrained(model_dir)
    # prefill runs on FlexAttention (block-sparse: a dense 4D mask through SDPA materialises the full
    # score matrix at 15k tokens); decode switches to SDPA with the ordinary 2D padding mask
    model = AutoModelForCausalLM.from_pretrained(  # type: ignore[call-arg]
        model_dir, dtype=torch.bfloat16, attn_implementation="flex_attention").cuda().eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs], add_special_tokens=False)["input_ids"]
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        # batch size capped by the KV cache (tokens), set by the longest prompt in the batch; prompts are
        # length-sorted so the cap is tight
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 80_000 // longest))  # SDPA decode expands the KV heads, so the cache costs double
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):  # left padding
            ids[r, n - len(s):] = torch.tensor(s)
            attn[r, n - len(s):] = 1
        ids, attn = ids.cuda(), attn.cuda()
        if causal:  # loop check only: dense causal mask through SDPA on short prompts
            model.set_attn_implementation("sdpa")
            mask = left_padded_prompt_mask(attn, causal=True)
            extra: dict = {}
        else:
            model.set_attn_implementation("flex_attention")
            mask = left_padded_block_mask(attn)
            extra = {"kernel_options": FLEX_KERNEL_OPTIONS}
        pos = (attn.cumsum(1) - 1).clamp(min=0)
        cache = DynamicCache()
        with torch.no_grad():
            out_ = model(input_ids=ids, attention_mask={"full_attention": mask}, position_ids=pos, past_key_values=cache,
                         use_cache=True, logits_to_keep=1, **extra)
            model.set_attn_implementation("sdpa")
            next_tok = out_.logits[:, -1].argmax(-1)
            gen = [next_tok]
            done = next_tok == eos
            attn2 = attn
            for _ in range(max_tokens - 1):
                if bool(done.all()):
                    break
                attn2 = torch.cat([attn2, torch.ones((len(seqs), 1), dtype=torch.long, device="cuda")], 1)
                pos = pos[:, -1:] + 1
                out_ = model(input_ids=next_tok[:, None], attention_mask=attn2, position_ids=pos, past_key_values=cache, use_cache=True)
                next_tok = out_.logits[:, -1].argmax(-1)
                next_tok = torch.where(done, torch.full_like(next_tok, pad), next_tok)
                gen.append(next_tok)
                done = done | (next_tok == eos)
        gen_t = torch.stack(gen, 1).tolist()
        for r, i in enumerate(idx):
            toks = []
            for t in gen_t[r]:
                if t == eos:
                    break
                toks.append(t)
            id_, v, _ = pairs[i]
            out.write(json.dumps({"id": id_, "variant": v, "text": tok.decode(toks, skip_special_tokens=True),
                                  "prompt_tokens": len(seqs[r]), "output_tokens": len(toks) + 1}) + "\n")
        out.flush()
        if n_batches % 50 == 1:
            print(f"[{bi}/{len(order)}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
