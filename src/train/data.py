"""(transcript, question) -> (prompt, target) examples for fine-tuning, and the batching plan.

The prompt is the labelling prompt of src/synthesis/label.py without the confidence request, so
every arm, the API baseline included, sees the same text: the numbered transcript, the question,
its options with their grading rules, and the instruction to cite evidence first. The target is the
label as compact JSON in the same order (evidence, answer, summary, tags when the question has a
tag vocabulary). Loss is taken on the target tokens only.

Examples are rendered from the transcript variant the label was made on (`generation_info.
labelled_variant`; clean for the service-call corpora, real ASR for SPoRC) unless the config asks
for more variants, which are line-aligned so the label transfers unchanged.
"""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.synthesis.label import numbered
from src.synthesis.schema import Label, Question

SEP = "\n\n"


def task_prompt(lines: list[str], speakers: list[str], q: Question) -> str:
    role_txt = f"Each line starts with the speaker's role as recorded by the source ({', '.join(speakers)})."
    opts = "\n".join(f"- {o.value}: {o.criteria}" for o in q.options)
    ctx = f"\nContext: {q.description}" if q.description else ""
    tag_txt = ("\nThe answer must also carry tags chosen only from this list (empty list if none applies): "
               + "; ".join(q.tags)) if q.tags else ""
    tag_field = ', "tags": ["<tag>", ...]' if q.tags else ""
    return f"""You assess conversation transcripts for quality assurance. One turn per line, numbered. {role_txt}

TRANSCRIPT
{numbered(lines)}

QUESTION: {q.text}{ctx}
Answer options and grading rules:
{opts}{tag_txt}

Work in this order: first find and list every line that bears on the question (empty list if nothing in the transcript relates to it); then decide the answer strictly by the rules; then write one or two sentences of reasoning that refer to what was said.

Respond with a single JSON object and nothing else:
{{"evidence": [<line numbers>], "answer": "<one of {q.values}>", "summary": "<reasoning>"{tag_field}}}"""


def target_text(label: Label, q: Question) -> str:
    out: dict = {"evidence": label.evidence, "answer": label.answer, "summary": label.summary}
    if q.tags:
        out["tags"] = label.tags
    return json.dumps(out, ensure_ascii=False, separators=(", ", ": "))


def record_prompt(r: dict, variant: str) -> str:
    """Prompt for a ledger/split record rendered from one transcript variant."""
    t = r["transcript"]
    lines = next((v["lines"] for v in t["variants"] if v["kind"] == variant), t["variants"][0]["lines"])
    return task_prompt(lines, t["speakers"], Question.model_validate(r["question"]))


@dataclass
class Example:
    id: str
    variant: str
    input_ids: np.ndarray  # int32; a Python list of ints would cost ~30x the memory over 200M tokens
    n_prompt: int  # loss starts after this many tokens

    @property
    def n_target(self) -> int:
        return len(self.input_ids) - self.n_prompt


def _variants_of(r: dict, requested: list[str]) -> list[str]:
    have = [v["kind"] for v in r["transcript"]["variants"]]
    labelled = r["generation_info"]["labelled_variant"]
    out = []
    for v in requested:
        v = labelled if v == "labelled" else v
        if v in have and v not in out:
            out.append(v)
    return out


def build_examples(path: Path, tokenizer, variants: list[str], max_seq_len: int,
                   cache_dir: Path | None = None, limit: int | None = None) -> tuple[list[Example], dict]:
    """Tokenised examples for one side of the split, every record whatever its licence track (training is
    research-only; the tracks govern the labelled-data release, not what a model may see). Cached by
    content as one flat int32 tensor plus offsets; prompts are tokenised in chunks so the text never
    sits in memory all at once."""
    key = hashlib.sha1(f"{path}|{path.stat().st_mtime_ns}|{tokenizer.name_or_path}|{variants}|{max_seq_len}|{limit}"
                       .encode()).hexdigest()[:16]
    cache = (cache_dir / f"{path.stem}-{key}.pt") if cache_dir else None
    if cache and cache.exists():
        return _load_cache(cache)
    eos = tokenizer.eos_token_id
    pending: list[tuple[str, str, str, str]] = []  # (id, variant, prompt, target)
    examples: list[Example] = []
    dropped = 0

    def flush() -> None:
        nonlocal dropped
        if not pending:
            return
        # the prompt opens with the tokenizer's own start token when it has one (<bos> for Gemma; Qwen adds nothing)
        enc_p = tokenizer([p for _, _, p, _ in pending])["input_ids"]
        enc_t = tokenizer([t for _, _, _, t in pending], add_special_tokens=False)["input_ids"]
        for (id_, v, _, _), ip, it in zip(pending, enc_p, enc_t):
            if len(ip) + len(it) + 1 > max_seq_len:
                dropped += 1
                continue
            examples.append(Example(id=id_, variant=v, input_ids=np.asarray(ip + it + [eos], dtype=np.int32), n_prompt=len(ip)))
        pending.clear()

    n_seen = 0
    for line in path.open():
        r = json.loads(line)
        q = Question.model_validate(r["question"])
        tgt = target_text(Label.model_validate(r["label"]), q)
        for v in _variants_of(r, variants):
            pending.append((r["id"], v, record_prompt(r, v) + SEP, tgt))
            n_seen += 1
        if len(pending) >= 1000:
            flush()
        if limit and n_seen >= limit:
            break
    flush()
    stats = {"examples": len(examples), "dropped_too_long": dropped,
             "prompt_tokens": sum(e.n_prompt for e in examples), "target_tokens": sum(e.n_target for e in examples),
             "max_len": max((len(e.input_ids) for e in examples), default=0)}
    if cache:
        _save_cache(cache, examples, stats)
    return examples, stats


def _save_cache(cache: Path, examples: list[Example], stats: dict) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(e.input_ids) for e in examples], dtype=np.int64)
    flat = np.concatenate([e.input_ids for e in examples]) if examples else np.zeros(0, dtype=np.int32)
    torch.save({"ids": [e.id for e in examples], "variants": [e.variant for e in examples],
                "n_prompt": np.asarray([e.n_prompt for e in examples], dtype=np.int64),
                "offsets": np.concatenate([[0], np.cumsum(lengths)]), "flat": torch.from_numpy(flat), "stats": stats}, cache)


def _load_cache(cache: Path) -> tuple[list[Example], dict]:
    d = torch.load(cache, weights_only=False)
    flat = d["flat"].numpy()
    off = d["offsets"]
    examples = [Example(id=i, variant=v, input_ids=flat[off[k]:off[k + 1]], n_prompt=int(d["n_prompt"][k]))
                for k, (i, v) in enumerate(zip(d["ids"], d["variants"]))]
    return examples, d["stats"]


def build_transcript_examples(path: Path, tokenizer, variants: list[str], max_seq_len: int,
                              cache_dir: Path | None = None, limit: int | None = None) -> tuple[list[Example], dict]:
    """One example per call (not per pair) for the adaptation stage: the numbered transcript exactly as
    it appears in the prompt's TRANSCRIPT block, plus the end-of-text token, with n_prompt=0 so every
    token is a target. Variants as in build_examples."""
    key = hashlib.sha1(f"transcripts|{path}|{path.stat().st_mtime_ns}|{tokenizer.name_or_path}|{variants}|{max_seq_len}|{limit}"
                       .encode()).hexdigest()[:16]
    cache = (cache_dir / f"{path.stem}-transcripts-{key}.pt") if cache_dir else None
    if cache and cache.exists():
        return _load_cache(cache)
    eos = tokenizer.eos_token_id
    seen: set[tuple[str, str]] = set()
    pending: list[tuple[str, str, str]] = []
    examples: list[Example] = []
    dropped = 0

    def flush() -> None:
        nonlocal dropped
        enc = tokenizer([t for _, _, t in pending], add_special_tokens=False)["input_ids"]
        for (cid, v, _), ids in zip(pending, enc):
            if len(ids) + 1 > max_seq_len:
                dropped += 1
                continue
            examples.append(Example(id=cid, variant=v, input_ids=np.asarray(ids + [eos], dtype=np.int32), n_prompt=0))
        pending.clear()

    for line in path.open():
        r = json.loads(line)
        cid = r["id"].split("::")[0]
        t = r["transcript"]
        for v in _variants_of(r, variants):
            if (cid, v) in seen:
                continue
            seen.add((cid, v))
            lines = next(x["lines"] for x in t["variants"] if x["kind"] == v)
            pending.append((cid, v, numbered(lines)))
        if len(pending) >= 500:
            flush()
        if limit and len(examples) + len(pending) >= limit:
            break
    flush()
    stats = {"examples": len(examples), "dropped_too_long": dropped, "prompt_tokens": 0,
             "target_tokens": sum(e.n_target for e in examples), "max_len": max((len(e.input_ids) for e in examples), default=0)}
    if cache:
        _save_cache(cache, examples, stats)
    return examples, stats


def render_document(doc: dict) -> str:
    """An interim document as numbered lines: speaker turns when the corpus has them (`SPEAKER_NN: text`
    lines, as SPoRC), otherwise one sentence per line. The task's transcripts are always numbered
    lines, and the numbering is what evidence pointing rests on; the speaker label varies by corpus
    anyway (agent/customer, assistant/user, host/SPEAKER_01), so a corpus without one gets none."""
    text = doc["text"].strip()
    if doc.get("has_speakers"):
        lines = [ln for ln in text.split("\n") if ln.strip()]
    else:
        lines = [x for x in re.split(r"(?<=[.?!])\s+", text) if x.strip()]
    return numbered(lines)


def labelled_doc_ids(split_dir: Path) -> set[str]:
    """The tail of every labelled record's source_id, over train/val/benchmark: an interim document whose
    doc_id is among them is a labelled call and stays out of the adaptation text."""
    out: set[str] = set()
    for side in ("train", "val", "benchmark"):
        f = split_dir / f"{side}.jsonl"
        if f.exists():
            for line in f.open():
                out.add(json.loads(line)["source_id"].rsplit("/", 1)[-1])
    return out


def _encode_docs(pending: list[tuple[str, str]], tokenizer, max_seq_len: int, eos: int, out: list[Example]) -> int:
    """Tokenise rendered documents into `out`, truncating to max_seq_len; returns how many were truncated."""
    truncated = 0
    if not pending:
        return 0
    # a 170k-token document is tokenised in full before truncation otherwise; cap the text first
    enc = tokenizer([t[:max_seq_len * 8] for _, t in pending], add_special_tokens=False)["input_ids"]
    for (did, _), ids in zip(pending, enc):
        if len(ids) + 1 > max_seq_len:
            ids = ids[:max_seq_len - 1]
            truncated += 1
        out.append(Example(id=did, variant="corpus", input_ids=np.asarray(ids + [eos], dtype=np.int32), n_prompt=0))
    pending.clear()
    return truncated


def build_corpus_examples(corpora, tokenizer, max_seq_len: int, side: str, seed: int, exclude: set[str],
                          cache_dir: Path | None = None, limit: int | None = None) -> tuple[list[Example], dict]:
    """Adaptation examples from interim corpora: a seeded one-pass sample of each corpus's <side>.jsonl,
    rendered as numbered lines, truncated to max_seq_len (adaptation text, so truncation beats dropping).
    `docs`/`val_docs` per corpus set the expected sample size."""
    spec = [(str(c.path), c.docs if side == "train" else c.val_docs) for c in corpora]
    key = hashlib.sha1(f"corpora|{spec}|{side}|{seed}|{tokenizer.name_or_path}|{max_seq_len}|{limit}|{len(exclude)}".encode()).hexdigest()[:16]
    cache = (cache_dir / f"corpora-{side}-{key}.pt") if cache_dir else None
    if cache and cache.exists():
        return _load_cache(cache)
    eos = tokenizer.eos_token_id
    examples: list[Example] = []
    per_corpus: dict[str, int] = {}
    truncated = 0
    for c in corpora:
        path = c.path.parent / f"{side}.jsonl"
        want = c.docs if side == "train" else c.val_docs
        total = json.loads((c.path.parent / "summary.json").read_text())["docs"][side]
        rng = random.Random(f"{seed}|{path}")
        p_keep = min(1.0, want / total)
        pending: list[tuple[str, str]] = []
        taken = 0

        for line in path.open():
            if rng.random() > p_keep:
                continue
            d = json.loads(line)
            if d["doc_id"] in exclude:
                continue
            pending.append((f"{d['source']}/{d['doc_id']}", render_document(d)))
            taken += 1
            if len(pending) >= 200:
                truncated += _encode_docs(pending, tokenizer, max_seq_len, eos, examples)
            if taken >= want or (limit and len(examples) + len(pending) >= limit):
                break
        truncated += _encode_docs(pending, tokenizer, max_seq_len, eos, examples)
        per_corpus[c.path.parent.name] = taken
        if limit and len(examples) >= limit:
            break
    stats = {"examples": len(examples), "per_corpus": per_corpus, "truncated": truncated, "dropped_too_long": 0, "prompt_tokens": 0,
             "target_tokens": sum(e.n_target for e in examples), "max_len": max((len(e.input_ids) for e in examples), default=0)}
    if cache:
        _save_cache(cache, examples, stats)
    return examples, stats


def mntp_mask(batch: dict[str, torch.Tensor], mask_prob: float, mask_id: int, seed: int) -> dict[str, torch.Tensor]:
    """LLM2Vec masked next-token prediction: replace a share of real tokens (never position 0) with the
    mask token; labels keep only the masked positions. Deterministic in the seed."""
    g = torch.Generator().manual_seed(seed)
    real = batch["attention_mask"].bool()
    real[:, 0] = False
    chosen = (torch.rand(real.shape, generator=g) < mask_prob) & real
    ids = batch["input_ids"].clone()
    ids[chosen] = mask_id
    labels = torch.full_like(batch["labels"], -100)
    labels[chosen] = batch["input_ids"][chosen]
    return {**batch, "input_ids": ids, "labels": labels, "prompt_len": batch["attention_mask"].sum(1)}


def seq2seq_cuts(examples: list[Example], idx: list[int], seed: int, max_target: int, cut: tuple[float, float]) -> list[int]:
    """The cut point per example for a seq2seq micro-batch, deterministic in the seed; collate_encdec and the
    loss normaliser both use it, so every rank can compute every rank's supervised-token count."""
    rng = random.Random(seed)
    out = []
    for i in idx:
        n = len(examples[i].input_ids)
        c = int(n * rng.uniform(*cut))
        out.append(min(max(c, 1), n - 1))
    return out


def seq2seq_target_count(examples: list[Example], idx: list[int], seed: int, max_target: int, cut: tuple[float, float]) -> int:
    return sum(min(len(examples[i].input_ids) - c, max_target) for i, c in zip(idx, seq2seq_cuts(examples, idx, seed, max_target, cut)))


def seq2seq_encoder_count(examples: list[Example], idx: list[int], seed: int, max_target: int, cut: tuple[float, float]) -> int:
    """Encoder tokens eligible for masking (every real token but the first) in a seq2seq micro-batch."""
    return sum(c - 1 for c in seq2seq_cuts(examples, idx, seed, max_target, cut))


def mask_encoder_side(batch: dict[str, torch.Tensor], mask_prob: float, mask_id: int, seed: int) -> dict[str, torch.Tensor]:
    """Mixed objective for the encoder-decoder: replace a share of the encoder's real tokens (never position 0)
    with the mask token and add `enc_labels`, the original tokens at masked positions (-100 elsewhere), for
    an MNTP loss on the encoder's own states (position t predicts enc_labels[t+1]). The decoder reads the
    masked prefix."""
    g = torch.Generator().manual_seed(seed)
    real = batch["enc_mask"].bool().clone()
    real[:, 0] = False
    chosen = (torch.rand(real.shape, generator=g) < mask_prob) & real
    enc_ids = batch["enc_ids"].clone()
    enc_labels = torch.full_like(enc_ids, -100)
    enc_labels[chosen] = batch["enc_ids"][chosen]
    enc_ids[chosen] = mask_id
    return {**batch, "enc_ids": enc_ids, "enc_labels": enc_labels}


def collate_encdec(examples: list[Example], idx: list[int], pad_id: int, start_id: int, seed: int = 0,
                   seq2seq: bool = False, max_target: int = 2048, cut: tuple[float, float] = (0.25, 0.75),
                   dec_prompt: bool = False, dec_start: bool = True) -> dict[str, torch.Tensor]:
    """Encoder/decoder batch for E3. Fine-tuning: the prompt goes to the encoder, the target (with its final
    eos) to the decoder, teacher-forced from `start_id`. seq2seq adaptation: a document is cut at a seeded
    random point in `cut`; the encoder reads the first part, the decoder predicts up to max_target tokens
    of the rest. Both sides right-padded; labels -100 at padding. dec_prompt (E6 native hybrid): the decoder side is
    `start_id`, the prompt, then the target, with labels only on the target, so the decoder reads the raw
    prompt through self-attention as well as the encoder's states through cross-attention. dec_start=False
    (the E6 decoder-only + encoder arm, whose decoder's prompt already opens with its start token): no `start_id` is prepended, so
    the decoder side is the prompt then the target, exactly the decoder's own fine-tuning input."""
    encs, tgts, prefixes = [], [], []
    off = 1 if dec_start else 0
    cuts = seq2seq_cuts(examples, idx, seed, max_target, cut) if seq2seq else [examples[i].n_prompt for i in idx]
    for i, c in zip(idx, cuts):
        ids = examples[i].input_ids
        encs.append(ids[:c]); tgts.append(ids[c:c + max_target] if seq2seq else ids[c:])
        prefixes.append(ids[:c] if dec_prompt and not seq2seq else ids[:0])
    ne = max(len(x) for x in encs); nt = max(off + len(p) + len(x) - 1 for p, x in zip(prefixes, tgts))
    enc_ids = torch.full((len(idx), ne), pad_id, dtype=torch.long); enc_mask = torch.zeros((len(idx), ne), dtype=torch.long)
    dec_ids = torch.full((len(idx), nt), pad_id, dtype=torch.long); dec_mask = torch.zeros((len(idx), nt), dtype=torch.long)
    labels = torch.full((len(idx), nt), -100, dtype=torch.long)
    for r, (en, pre, tg) in enumerate(zip(encs, prefixes, tgts)):
        enc_ids[r, :len(en)] = torch.from_numpy(en.astype(np.int64)); enc_mask[r, :len(en)] = 1
        t = torch.from_numpy(tg.astype(np.int64))
        k = len(pre)
        if off:
            dec_ids[r, 0] = start_id
        if k:
            dec_ids[r, off:off + k] = torch.from_numpy(pre.astype(np.int64))
        dec_ids[r, off + k:off + k + len(tg) - 1] = t[:-1]; dec_mask[r, :off + k + len(tg) - 1] = 1
        labels[r, off + k - 1:off + k - 1 + len(tg)] = t  # the last prompt token (or start_id) predicts the first target token
    return {"enc_ids": enc_ids, "enc_mask": enc_mask, "dec_ids": dec_ids, "dec_mask": dec_mask, "labels": labels}


@dataclass
class Step:
    """One optimizer step: per rank, a list of micro-batches (lists of example indices); the target-token
    total over all ranks normalises the loss so accumulation and data parallelism give the token mean."""
    micro: list[list[list[int]]]  # [rank][micro][example index]
    target_tokens: int


def plan_epoch(examples: list[Example], batch_sequences: int, micro_tokens: int, world_size: int,
               seed: int, epoch: int) -> list[Step]:
    """Deterministic on every rank. Shuffle, cut into groups of batch_sequences, deal each group's
    sequences to ranks alternating by length (balanced tokens), then pack each rank's share into
    micro-batches under the padded-token budget (longest * count). A sequence longer than the budget
    goes alone."""
    order = list(range(len(examples)))
    random.Random(f"{seed}-{epoch}").shuffle(order)
    steps = []
    for g in range(0, len(order) - len(order) % batch_sequences, batch_sequences):
        group = sorted(order[g:g + batch_sequences], key=lambda i: len(examples[i].input_ids), reverse=True)
        shares: list[list[int]] = [[] for _ in range(world_size)]
        for k, i in enumerate(group):
            shares[k % world_size].append(i)
        micro = []
        for share in shares:
            packed: list[list[int]] = []
            cur: list[int] = []
            longest = 0
            for i in share:  # descending length, so `longest` is the first element's
                n = len(examples[i].input_ids)
                if cur and max(longest, n) * (len(cur) + 1) > micro_tokens:
                    packed.append(cur)
                    cur, longest = [], 0
                cur.append(i)
                longest = max(longest, n)
            if cur:
                packed.append(cur)
            micro.append(packed)
        steps.append(Step(micro=micro, target_tokens=sum(examples[i].n_target for i in group)))
    return steps


def collate(examples: list[Example], idx: list[int], pad_id: int, pad_to: int = 1, min_len: int = 1) -> dict[str, torch.Tensor]:
    """Right-pad a micro-batch. pad_to / min_len round the padded length up to a multiple and a floor:
    FlexAttention (E2) wants block-aligned lengths and switches to a decoding kernel below ~128 query
    tokens whose lowering fails under dynamic shapes, so prefix-LM batches use pad_to=128, min_len=256."""
    n = max(len(examples[i].input_ids) for i in idx)
    n = max(n, min_len)
    n = ((n + pad_to - 1) // pad_to) * pad_to
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(idx), n), dtype=torch.long)
    labels = torch.full((len(idx), n), -100, dtype=torch.long)
    prompt_len = torch.zeros(len(idx), dtype=torch.long)
    for row, i in enumerate(idx):
        e = examples[i]
        L = len(e.input_ids)
        ids[row, :L] = torch.from_numpy(e.input_ids.astype(np.int64))
        attn[row, :L] = 1
        labels[row, e.n_prompt:L] = ids[row, e.n_prompt:L]
        prompt_len[row] = e.n_prompt
    return {"input_ids": ids, "attention_mask": attn, "labels": labels, "prompt_len": prompt_len}


def main() -> None:
    """Print decoded examples from a tokenised cache: python -m src.train.data <cache.pt> [--show N] [--first K]"""
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--first", type=int, default=0, help="index of the first example to show")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    examples, stats = _load_cache(args.cache)
    print(f"{stats}\n")
    for e in examples[args.first:args.first + args.show]:
        print(f"===== {e.id} | variant {e.variant} | prompt {e.n_prompt} tokens | target {e.n_target} tokens =====")
        print(tok.decode(e.input_ids[:e.n_prompt].tolist()))
        print("----- target (loss) -----")
        print(tok.decode(e.input_ids[e.n_prompt:].tolist()), "\n")


if __name__ == "__main__":
    main()
