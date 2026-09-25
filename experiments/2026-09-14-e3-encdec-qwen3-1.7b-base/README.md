# E3: encoder-decoder built from Qwen3-1.7B-Base

The encoder-decoder arm of the README's "Design" table, written before the run on 2026-09-14. My reading of the prefix-LM line (E2, E2+, E2++): the bidirectional arm improved monotonically with adaptation and reached parity on answers with a small lead on evidence, so bidirectional reading has "a little bit of promise" and deserves a dedicated encoder.

## Hypothesis

Prefix-LM makes one set of weights do two jobs, read the transcript bidirectionally and generate causally, and its adaptation curve suggests the two compete. An encoder-decoder separates them: an encoder that only reads (initialised from the same Qwen3-1.7B-Base weights, attention made bidirectional) and a decoder that only writes (the same weights again, with cross-attention layers inserted and initialised from self-attention). With the reading side free of the generation objective, the bidirectional representation should convert into better judgements, not only better evidence.

Against it: RedLLM (arXiv 2510.26622) found the prefix-LM decoder beat their encoder-decoder on quality; T5Gemma needed tens of billions of adaptation tokens for its encoder-decoder to match its decoder-only origin, and this run has 661M.

## Prediction

- Adaptation: the seq2seq val loss on held-out outside documents falls and flattens within the stage (the decoder learns to read the encoder; the encoder's flipped attention repairs as E2++'s did).
- Fine-tuning val loss at or below E1's 0.503 at the end (E2++ ended at 0.506).
- Benchmark unseen-question macro-F1 above E1 by at least 0.02 on both variants, and evidence F1 at or above E2++'s 0.586 / 0.555.
- **Disconfirmed if** E3 is at or below E1 on unseen-question macro-F1 on both variants. Given E2++'s parity, that would say separate read/write weights add nothing at this size, and the encoder-decoder line stops at 1.7B.

## Setup

- Architecture (`src/train/encdec.py`): encoder = Qwen3-1.7B-Base layers with bidirectional attention over the real tokens (FlashAttention 2, non-causal); decoder = the same layers with causal self-attention over the target and a cross-attention sublayer inserted between self-attention and the MLP in every layer (T5 ordering), queries/keys/values copied from that layer's self-attention (with Qwen3's per-head q/k norms, no rotary embedding on cross-attention), output projection zero-initialised so the model starts as the causal decoder and learns to read the encoder; a new pre-cross-attention RMSNorm; output head tied to the decoder embeddings; decoder start token `<|endoftext|>`. **3.79B parameters** (two 1.72B towers, 0.35B of cross-attention; the 0.27B of LoRA belonged to the discarded seq2seq-only attempt), so E3 is size-matched to nothing we have: it is compared with E1 at matched inference cost (the encoder does E1's prompt work once, the decoder runs over ~100 tokens), and the entry says so.
- **Training regime (revised 2026-09-14 evening): every parameter trains, as in E1.** fp32 master weights for 3.8B parameters do not fit one 32 GB card (34 GB before activations), so the first attempt used LoRA on the towers. I switched to full-parameter training instead, so the arm matches E1; a sharded prototype (FSDP2: fp32 master shards, bf16 compute and gradient reduction, 8-bit AdamW on the local shards) measured 20.1 / 21.4 / 25.3 GiB per GPU at 4k / 16k / 32k encoder tokens with 3.0k / 4.1k / 3.4k tokens/s per GPU, about 35% slower than the LoRA run's DDP. `model.sharding: fsdp` in the loop; lr 1e-5 as E1; 12k-token micro-batches and 1k-token decoder targets in adaptation (16k / 2k ran out of memory under the mixed objective). The LoRA config stays in the repo as the fallback.
- Adaptation (`configs/adapt/encdec-mixed-full-qwen3-1.7b-base.yaml`): the E2++ corpus sample (131,426 outside documents, 661.5M tokens; every labelled call excluded). **Objective revised after a first attempt:** the seq2seq-only stage (prefix → continuation, `configs/adapt/encdec-qwen3-1.7b-base.yaml`, LoRA) was stopped at step 1,050 of 4,107 because its held-out val fell only 1.928 → 1.895 over steps 500–1000, a fifth of the rate E2++'s masked stage improved on the same text, with gradient norm ~0.27: the continuation is mostly solvable by the decoder's own language modelling, so the encoder got little signal. The stage now uses a **mixed objective**: the seq2seq loss plus E2++'s masked next-token loss on the encoder's own states (20% of the prefix masked, each masked token predicted from the position before it through the tied head), both losses summed and normalised by the supervised tokens every rank draws. Each document cut at a seeded random point between 25% and 75%; decoder side capped at 1,024 tokens. One pass, 4,107 steps, lr 1e-5. Sanity checks before launch (2026-09-14): the encoder is bidirectional and padding-invariant, the decoder at initialisation reproduces the base model's next-token argmax on 100% of positions, cached decoding matches full recompute; 32k encoder tokens plus 512 decoder tokens take 19.9 GiB forward+backward without an optimizer.
- Fine-tune (`configs/e3/qwen3-1.7b-base-full.yaml`): E1's data, split, rendering, steps, optimiser and lr 1e-5; encoder reads the task prompt, decoder generates the JSON label, loss on the label; sharded as above.
- Generation (`generate.py --backend encdec`): encoder once per prompt, cached causal decoding over the short output with cross-attention to the fixed encoder states; greedy, max 512 tokens; 41 val prompts in 33 s including model load on the untrained model. Scoring as before.
- Control: E1 (= E1+). The adaptation text is extra text the control never saw; E1++ (deferred) is the matched control.

## wandb runs

- Adaptation: [tt-pretrain/1gyti0ha](https://wandb.ai/khalit7-/tt-pretrain/runs/1gyti0ha), 2026-09-15 00:02 → 19:50 (19.8 h; a first full-parameter attempt of 20:30–23:55 on the 14th was discarded so that the two halves of the objective would be logged separately; the LoRA seq2seq-only attempt of 15:17–19:45 is kept under `checkpoints/adapt-encdec-seq2seq-aborted` (deleted in the 2026-09-21 disk clean-up; its wandb run vlmo0aml was deleted on 2026-09-25)). 4,107 steps, 661.5M tokens, all 3.79B parameters, FSDP-sharded, 9.5k tokens/s, peak 29.8 GiB per GPU. Held-out val, seq2seq / masked / combined (token-weighted): step 0 2.160 / 10.015 / 5.613; 250 1.905 / 1.631 / 1.784; 500 1.871 / 1.437 / 1.680; 1000 1.832 / 1.306 / 1.601; 2000 1.798 / 1.219 / 1.543; 3000 1.779 / 1.182 / 1.517; **end 1.770 / 1.168 / 1.505**. The masked half ends where E2++'s masked-only stage ended (1.165), so the encoder is as repaired as E2++'s; the seq2seq half passed the abandoned seq2seq-only run's step-1000 value (1.895) by step 250 and ended 0.12 below it; both halves flat over the last 1,000 steps.
- Fine-tune: [tt-encdec/ul124rc7](https://wandb.ai/khalit7-/tt-encdec/runs/ul124rc7), 2026-09-15 19:52 → 21:25, **stopped at step ~620 of 2,859** under a rule agreed in advance (see Result).

## Result

**Fine-tune val loss (E1 and E2++ at the same steps for reference):**

| step | E3 | E1 | E2++ | E2 (no adaptation) |
|---|---|---|---|---|
| 0 | 3.147 | 1.18 | | |
| 200 | 1.076 | 0.704 | 0.740 | 0.815 |
| 400 | 1.008 | 0.648 | 0.667 | 0.715 |
| 600 | 0.971 | 0.620 | 0.632 | 0.672 |

E3 started the task far behind (3.15 before any update: a decoder trained to emit transcript continuations through cross-attention, asked for JSON judgements) and then improved at the same rate as E1 rather than faster: the gap to E1 was 0.372, 0.360, 0.351 at steps 200, 400, 600, against 0.11 → 0.016 for E2 over a whole run. Training loss on the same batches: 2.28 / 1.43 / 1.26 / 1.01 / 0.92 at steps 10, 50, 100, 200, 600 (E1: 1.07 / 1.28 / 0.79 / 0.70 / 0.58). Throughput 8.3k tokens/s sharded, peak 25 GiB per GPU.

The run was stopped at step ~620 by a stopping rule fixed after the step-200 val, before the outcome was known: continue to step 600, stop if the val gap to E1 is still above 0.20. It was 0.35. At E1's own rate of improvement the run would have ended near 0.85 against E1's 0.503, so the benchmark was not generated; the step-514 checkpoint is kept under `checkpoints/e3-qwen3-1.7b-base/` and the fine-tune can be resumed if a benchmark number is wanted regardless.

## Verdict

**Disconfirmed on the training curve; not benchmarked.** The prediction was a fine-tuning val loss at or below E1's 0.503 and a benchmark gain of 0.02 on unseen questions; at every check the encoder-decoder trailed E1 by 0.35 with no sign of closing, worse than the never-adapted prefix-LM (E2) by 0.30. The adaptation itself succeeded on its own terms (encoder as well repaired as E2++'s, seq2seq converged), so what failed is the transfer: a decoder that reads the transcript only through 661M-token-old cross-attention learns the judgement task far more slowly than a decoder that reads it through pretrained self-attention, and one epoch of 8.5M supervised tokens is not enough to make up the difference. This is consistent with RedLLM (the prefix-LM decoder beat the encoder-decoder) and with T5Gemma's need for tens of billions of adaptation tokens, which we do not have.

Caveats: the run was cut at 22% of the epoch, so the final number is an extrapolation, though the flat gap over 400 steps leaves little room; the adaptation budget (661M tokens) is two orders of magnitude below T5Gemma's; the encoder-decoder is compared with E1 at matched inference cost, not matched parameters; the extra-text confound of the E2++ line applies here too, in E3's favour, and it still lost.

Combined with E2, E2+ and E2++: at 1.7B, on this task and data, no bidirectional-attention design tried beats the causal decoder on answers. Hypothesis 1 is not supported at this size. The one consistent bidirectional gain, evidence precision in the prefix-LM line, stays an observation with its text confound unresolved.

## Follow-ups

- If a measured E3 benchmark is wanted for the record: resume the fine-tune from step 514 (about 6 hours) and evaluate with `--backend encdec`.
- The most plausible rescue, not recommended at this size: a much larger adaptation (billions of tokens) or fine-tuning E3 at a higher learning rate for the cross-attention path than for the rest; either breaks the matched recipe with E1.
- The 4B scale point for E1, and E2 at 4B, are the remaining architecture questions worth GPU time; E3 at 4B is not, on this evidence.
- The frozen-encoder designs (retired from the design table by the 2026-09-25 re-labelling; parked in the notes) stay behind their probe gate; nothing here argues for opening it.
