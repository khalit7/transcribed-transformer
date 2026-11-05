"""Score two stitches for the same encoder pair on the held-out tokens of a v2 fit (the selection src.train.fit_stitch
makes with --train data/labelled_data_v2/train.jsonl --one-per-transcript --max-len 16384 and its default --tokens,
--batch-tokens and --seed), so a compliance fit and its v2 refit are compared on the same text.

    uv run python scripts/stitch_cross_ev.py <pair> <donor encoder-decoder> <native encoder-decoder>
    e.g. uv run python scripts/stitch_cross_ev.py 1b-to-270m google/t5gemma-2-1b-1b google/t5gemma-2-270m-270m

Reads checkpoints/stitch/<pair>.pt and <pair>-v2.pt; prints one JSON line. The v2 fit's own number is reproduced exactly,
which checks that the held-out selection is the fit's.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoTokenizer

from src.train.data import build_examples, encoder_pad_len
from src.train.fit_stitch import encoder_of, states
from src.train.stitched import explained_variance

pair, src_b, tgt_b = sys.argv[1], sys.argv[2], sys.argv[3]
TOK, BT, MAXLEN, SEED = 3_000_000, 32768, 16384, 0
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(tgt_b)
ex, _ = build_examples(Path("data/labelled_data_v2/train.jsonl"), tok, ["labelled"], 16384, Path("data/interim/sft"),
                       questions_per_episode=1, question_seed=SEED)
ex = [e for e in ex if e.n_prompt <= MAXLEN]; random.Random(SEED).shuffle(ex)
def spans(budget, start):
    n, i = 0, start
    while n < budget and i < len(ex):
        b, longest = [], 0
        while i < len(ex) and (not b or max(longest, ex[i].n_prompt) * (len(b) + 1) <= BT):
            b.append(ex[i]); longest = max(longest, ex[i].n_prompt); i += 1
        n += sum(e.n_prompt for e in b); yield b, longest, i
nxt = 0
for _, _, nxt in spans(TOK, 0): pass
src, tgt = encoder_of(src_b, "sdpa", dev), encoder_of(tgt_b, "sdpa", dev)
xh, yh = [], []
for b, longest, _ in spans(TOK // 10, nxt):
    ids = torch.full((len(b), encoder_pad_len(longest)), tok.pad_token_id, dtype=torch.long); m = torch.zeros_like(ids)
    for r, e in enumerate(b):
        p = torch.from_numpy(e.input_ids[:e.n_prompt].astype("int64")); ids[r, :len(p)] = p; m[r, :len(p)] = 1
    ids, m = ids.to(dev), m.to(dev)
    with torch.no_grad():
        xh.append(states(src, ids, m).cpu()); yh.append(states(tgt, ids, m).cpu())
xh, yh = torch.cat(xh).double(), torch.cat(yh).double()
src.to("cpu"); tgt.to("cpu"); torch.cuda.empty_cache()
out = {"pair": pair, "heldout_tokens": xh.shape[0]}
for name in (pair, pair + "-v2"):
    d = torch.load(f"checkpoints/stitch/{name}.pt", map_location="cpu")
    out[name] = round(explained_variance(xh.to(dev), yh.to(dev), d["weight"].double().to(dev), d["bias"].double().to(dev)), 4)
print(json.dumps(out), flush=True)
