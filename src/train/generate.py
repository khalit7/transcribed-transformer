"""Generate the model's output for every pair of a split file, one JSON line per pair.

    .venv-vllm/bin/python -m src.train.generate <model_dir> <split.jsonl> <out.jsonl> [--variants clean,messy]
        [--max-tokens 512] [--tp 1] [--limit N] [--backend vllm|hf]

The prompt is src/train/data.task_prompt, the same text the model was trained on and the API
baseline is given. Greedy decoding. The output records carry the pair id, the variant rendered and
the raw text, and src/train/evaluate.py scores them; nothing here parses or repairs the output.
vLLM runs in its own environment (.venv-vllm) because it pins its own torch; --backend hf is the
slow fallback inside the training environment, and --backend prefixlm is the E2 path (bidirectional
prefill with the prefix-LM mask, then ordinary cached decoding), also inside the training environment;
--backend encdec is the E3 path (encoder once, cached decoding with cross-attention); --backend hf_encdec
is a native Hugging Face encoder-decoder (T5Gemma 2) through the library's own generate.
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
    ap.add_argument("--backend", choices=["vllm", "hf", "prefixlm", "encdec", "hf_encdec", "hf_encdec_hybrid", "hybrid", "stitched"], default="vllm")
    ap.add_argument("--batch", type=int, default=16, help="prefixlm backend: sequences per batch")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="vllm backend: no CUDA graphs (a model whose RoPE cache grows on demand, e.g. Hunyuan's dynamic NTK, cannot be captured)")
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
                  tensor_parallel_size=args.tp, gpu_memory_utilization=0.9, enable_prefix_caching=True,
                  enforce_eager=args.enforce_eager)
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
    elif args.backend == "encdec":
        with args.out.open("a") as f:
            generate_encdec(args.model_dir, pairs, f, args.max_tokens, args.batch)
    elif args.backend == "hf_encdec":
        with args.out.open("a") as f:
            generate_hf_encdec(args.model_dir, pairs, f, args.max_tokens, args.batch)
    elif args.backend == "hf_encdec_hybrid":
        with args.out.open("a") as f:
            generate_hf_encdec_hybrid(args.model_dir, pairs, f, args.max_tokens, args.batch)
    elif args.backend == "hybrid":
        with args.out.open("a") as f:
            generate_hybrid(args.model_dir, pairs, f, args.max_tokens, args.batch)
    elif args.backend == "stitched":
        with args.out.open("a") as f:
            generate_stitched(args.model_dir, pairs, f, args.max_tokens, args.batch)
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



def generate_encdec(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 32) -> None:
    """Greedy decoding for the encoder-decoder (E3): encoder once per prompt (right-padded batch, key mask),
    then cached causal decoding over the output with cross-attention to the fixed encoder states. Prompts
    are length-sorted; the batch is capped by encoder tokens."""
    import torch
    from transformers import DynamicCache

    from src.train.encdec import EncDec, load_tokenizer

    tok = load_tokenizer(model_dir)
    model = EncDec.load(model_dir, gradient_checkpointing=False).cuda().eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs], add_special_tokens=False)["input_ids"]
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 160_000 // longest))
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with torch.no_grad():
            enc = model.encode(ids, mask)
            cache = DynamicCache()
            cur = torch.full((len(seqs), 1), eos, dtype=torch.long, device="cuda")  # decoder start token
            done = torch.zeros(len(seqs), dtype=torch.bool, device="cuda")
            gen = []
            for _ in range(max_tokens):
                h = model.decode(cur, None, enc, mask, past=cache)
                nxt = model.lm_head(h[:, -1]).argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
                gen.append(nxt)
                done = done | (nxt == eos)
                if bool(done.all()):
                    break
                cur = nxt[:, None]
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


def generate_hf_encdec(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 16) -> None:
    """Greedy decoding for a native Hugging Face encoder-decoder (T5Gemma 2) through the library's generate.
    Prompts open with the tokenizer's own start token, as in training; length-sorted, batched under an
    encoder-token cap; the decoder starts from <bos>."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    start = tok.bos_token_id if tok.bos_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs])["input_ids"]
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 160_000 // longest))
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        with torch.no_grad():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=max_tokens,  # type: ignore[operator]
                                 do_sample=False, num_beams=1, decoder_start_token_id=start, eos_token_id=eos, pad_token_id=pad)
        for r, i in enumerate(idx):
            toks = []
            for t in gen[r, 1:].tolist():  # the row opens with the decoder start token
                if t == eos:
                    break
                toks.append(t)
            id_, v, _ = pairs[i]
            out.write(json.dumps({"id": id_, "variant": v, "text": tok.decode(toks, skip_special_tokens=True),
                                  "prompt_tokens": len(seqs[r]), "output_tokens": len(toks) + 1}) + "\n")
        out.flush()
        if n_batches % 50 == 1:
            print(f"[{bi}/{len(order)}] {time.time() - t0:.0f}s", flush=True)


def generate_hf_encdec_hybrid(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 16) -> None:
    """Greedy decoding for E3h: the encoder reads the prompt; the decoder is prefilled with start token +
    prompt (left-padded within a length-sorted batch, so padding is a few tokens; explicit position ids start
    at 0 on the first real token) with cross-attention to the encoder states, then decodes the label."""
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DynamicCache,
        EncoderDecoderCache,
    )

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    dec_cfg = model.config.get_text_config(decoder=True)
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    start = tok.bos_token_id if tok.bos_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs])["input_ids"]
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 80_000 // (longest + 1)))
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        enc_ids = torch.full((len(seqs), n), pad, dtype=torch.long); enc_mask = torch.zeros((len(seqs), n), dtype=torch.long)
        m = n + 1  # decoder prefix: start token + prompt, left-padded
        dec_ids = torch.full((len(seqs), m), pad, dtype=torch.long); dec_mask = torch.zeros((len(seqs), m), dtype=torch.long)
        pos = torch.zeros((len(seqs), m), dtype=torch.long)
        for r, s in enumerate(seqs):
            enc_ids[r, :len(s)] = torch.tensor(s); enc_mask[r, :len(s)] = 1
            k = m - (len(s) + 1)
            dec_ids[r, k] = start; dec_ids[r, k + 1:] = torch.tensor(s); dec_mask[r, k:] = 1
            pos[r, k:] = torch.arange(len(s) + 1)
        enc_ids, enc_mask, dec_ids, dec_mask, pos = (x.cuda() for x in (enc_ids, enc_mask, dec_ids, dec_mask, pos))
        with torch.no_grad():
            enc = model.model.encoder(input_ids=enc_ids, attention_mask=enc_mask, return_dict=True).last_hidden_state
            cache = EncoderDecoderCache(DynamicCache(config=dec_cfg), DynamicCache())
            h = model.model.decoder(input_ids=dec_ids, attention_mask=dec_mask, position_ids=pos, past_key_values=cache,
                                    encoder_hidden_states=enc, encoder_attention_mask=enc_mask, use_cache=True, return_dict=True).last_hidden_state
            nxt = model.lm_head(h[:, -1]).argmax(-1)
            done = torch.zeros(len(seqs), dtype=torch.bool, device="cuda")
            gen = []
            next_pos = pos[:, -1] + 1
            for _ in range(max_tokens):
                nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
                gen.append(nxt)
                done = done | (nxt == eos)
                if bool(done.all()):
                    break
                dec_mask = torch.cat([dec_mask, torch.ones((len(seqs), 1), dtype=torch.long, device="cuda")], 1)
                h = model.model.decoder(input_ids=nxt[:, None], attention_mask=dec_mask, position_ids=next_pos[:, None], past_key_values=cache,
                                        encoder_hidden_states=enc, encoder_attention_mask=enc_mask, use_cache=True, return_dict=True).last_hidden_state
                nxt = model.lm_head(h[:, -1]).argmax(-1)
                next_pos = next_pos + 1
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


def generate_hybrid(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 16) -> None:
    """Greedy decoding for E4b (src/train/hybrid.py): the encoder reads the prompt once and its keys and values
    are cached per layer; the decoder is prefilled with the prompt (its own start token in front, as the prompt
    is tokenised; left-padded within a length-sorted batch; explicit position ids from 0 on the first real
    token) and decodes the label with a growing cache, reading the encoder through cross-attention."""
    import torch
    from transformers import AutoTokenizer, DynamicCache

    from src.train.hybrid import Hybrid, flash_varlen_available

    tok = AutoTokenizer.from_pretrained(model_dir)
    attn = "flash_attention_2" if flash_varlen_available(torch.zeros(1, device="cuda", dtype=torch.bfloat16)) else "sdpa"
    model = Hybrid.load(model_dir, attn=attn, gradient_checkpointing=False).cuda().eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs])["input_ids"]  # opens with the tokenizer's start token, as in training
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 40_000 // (longest + 1)))
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        enc_ids = torch.full((len(seqs), n), pad, dtype=torch.long); enc_mask = torch.zeros((len(seqs), n), dtype=torch.long)
        dec_ids = torch.full((len(seqs), n), pad, dtype=torch.long); dec_mask = torch.zeros((len(seqs), n), dtype=torch.long)
        pos = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):
            enc_ids[r, :len(s)] = torch.tensor(s); enc_mask[r, :len(s)] = 1
            k = n - len(s)
            dec_ids[r, k:] = torch.tensor(s); dec_mask[r, k:] = 1
            pos[r, k:] = torch.arange(len(s))
        enc_ids, enc_mask, dec_ids, dec_mask, pos = (x.cuda() for x in (enc_ids, enc_mask, dec_ids, dec_mask, pos))
        with torch.no_grad():
            enc = model.encode(enc_ids, enc_mask)
            # prefill: the prompt's queries over the encoder's keys through the varlen kernel (SDPA with a key mask
            # materialises the 16k × 16k matrix per layer, 2 GiB, which is where the first run died); decoding
            # steps: one query per step over cached keys and values, cheap under SDPA
            model.set_context(enc, enc_mask, cache_kv=False, dec_mask=dec_mask)
            cache = DynamicCache(config=model.config)
            h = model.decode(dec_ids, dec_mask, pos, cache)
            model.set_context(enc, enc_mask, cache_kv=True)
            nxt = model.lm_head(h[:, -1]).argmax(-1)
            done = torch.zeros(len(seqs), dtype=torch.bool, device="cuda")
            gen = []
            next_pos = pos[:, -1] + 1
            for _ in range(max_tokens):
                nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
                gen.append(nxt)
                done = done | (nxt == eos)
                if bool(done.all()):
                    break
                dec_mask = torch.cat([dec_mask, torch.ones((len(seqs), 1), dtype=torch.long, device="cuda")], 1)
                h = model.decode(nxt[:, None], dec_mask, next_pos[:, None], cache)
                nxt = model.lm_head(h[:, -1]).argmax(-1)
                next_pos = next_pos + 1
            model.set_context(None, None)
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


def generate_stitched(model_dir: str, pairs: list[tuple[str, str, str]], out, max_tokens: int, batch: int = 16) -> None:
    """Greedy decoding for E3-mix-and-match (src/train/stitched.py): encoder pass through the stitch once per batch,
    then the T5Gemma 2 decoder from its start token with a growing cache, reading the stitched states."""
    import torch
    from transformers import AutoTokenizer, DynamicCache, EncoderDecoderCache

    from src.train.stitched import Stitched

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Stitched.load(model_dir, attn="sdpa", gradient_checkpointing=False).cuda().eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    start = tok.bos_token_id if tok.bos_token_id is not None else eos
    enc_all = tok([p for _, _, p in pairs])["input_ids"]
    order = sorted(range(len(pairs)), key=lambda i: len(enc_all[i]))
    t0 = time.time()
    bi = 0
    n_batches = 0
    while bi < len(order):
        longest = len(enc_all[order[min(bi + batch, len(order)) - 1]])
        bs = max(1, min(batch, 160_000 // longest))
        idx = order[bi:bi + bs]
        bi += bs
        n_batches += 1
        seqs = [enc_all[i] for i in idx]
        n = max(len(s) for s in seqs)
        enc_ids = torch.full((len(seqs), n), pad, dtype=torch.long); enc_mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, s in enumerate(seqs):
            enc_ids[r, :len(s)] = torch.tensor(s); enc_mask[r, :len(s)] = 1
        enc_ids, enc_mask = enc_ids.cuda(), enc_mask.cuda()
        with torch.no_grad():
            enc = model.encode(enc_ids, enc_mask)
            cache = EncoderDecoderCache(DynamicCache(config=model.config), DynamicCache())
            dec_mask = torch.ones((len(seqs), 1), dtype=torch.long, device="cuda")
            nxt = torch.full((len(seqs),), start, dtype=torch.long, device="cuda")
            pos = torch.zeros((len(seqs), 1), dtype=torch.long, device="cuda")
            done = torch.zeros(len(seqs), dtype=torch.bool, device="cuda")
            gen = []
            for step in range(max_tokens + 1):
                h = model.decode(nxt[:, None], dec_mask, enc, enc_mask, pos, cache)
                nxt = model.lm_head(h[:, -1]).argmax(-1)
                if step == max_tokens:
                    break
                nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
                gen.append(nxt)
                done = done | (nxt == eos)
                if bool(done.all()):
                    break
                dec_mask = torch.cat([dec_mask, torch.ones((len(seqs), 1), dtype=torch.long, device="cuda")], 1)
                pos = pos + 1
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
