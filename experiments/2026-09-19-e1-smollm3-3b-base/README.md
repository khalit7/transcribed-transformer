# E1: SmolLM3 3B base, cross-family decoder control

## Hypothesis

The native T5Gemma 2 1b-1b's quality advantage over its Gemma 3 1B decoder origin may not extend to a stronger decoder from another family. A SmolLM3 3B base checkpoint fine-tuned with the E1 recipe tests practical competitiveness, not the causal effect of encoder-decoder architecture. Pretraining, tokenizer, parameter count and architecture all differ.

## Prediction

SmolLM3 will match or exceed T5Gemma 2 1b-1b on unseen-question messy-transcript answer macro-F1. A lower score disconfirms this directional prediction for this seed and recipe. A small difference is inconclusive about a repeatable advantage until repeated seeds and paired, source-group-aware uncertainty estimates are available.

The primary metric is fixed before training: unseen-question messy macro-F1. Report clean unseen and overall macro-F1, the majority predictor, strict format validity, evidence precision, corpus and length slices, and measured inference cost alongside it. Benchmark labels are model-generated, not human-validated gold; summary faithfulness remains TBD.

## Setup

- Config: [`configs/e1/smollm3-3b-base.yaml`](../../configs/e1/smollm3-3b-base.yaml).
- Base: [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base), pretrained, no instruction tuning or chat template.
- Downloaded metadata revision: `d78a42f79198603e614095753484a04c10c2b940`. Config-instantiated parameter count: 3,075,098,624, all text parameters. Attention dropout is zero and there are no active dropout modules.
- Frozen `data/labelled_data/splits.json`; train and validation JSONL materialisations, every licence track, labelled transcript variant. Canonical numbered prompt and JSON targets, zero prompt loss.
- Seed 0, one epoch, 32 sequences per optimiser step, 16,384-token sequence and micro-batch limits. Tokenizer-dependent filtering must be reported; the same token limit does not imply identical retained examples across families.
- E1 precision and optimiser unchanged: bf16 compute, fp32 masters, 8-bit AdamW, learning rate 1e-5, 50 warm-up steps, cosine decay, weight decay 0.1. Full text-parameter training, FlashAttention 2 and activation checkpointing.
- Two RTX 5090s, FSDP required: the unsharded fp32-master probe failed on its first optimiser update, before adding any DDP overhead. Checkpoints every 30 minutes, latest two retained. Validate checkpoint reload before the full run.
- W&B online, as explicitly reaffirmed by the owner. Resolved configuration, tokenizer and split provenance accompany the run.
- Comparison: [native T5Gemma 2/Gemma 3 pairs](../2026-09-15-e3-t5gemma2-1b-1b-native-pair/README.md). This additional baseline does not alter their original hypotheses.

## W&B run

Project: `tt-decoder`. Run: [e82so2fd](https://wandb.ai/khalit7-/tt-decoder/runs/e82so2fd), native online logging.

Launched 2026-09-19 at 23:59 Europe/London in persistent tmux session `tt-e1-smollm3`. Local log: `checkpoints/e1-smollm3-3b-base/train.log`. Resume, if needed, uses the same two ranks and `--resume`:

```bash
WANDB_MODE=online WANDB_ENTITY=khalit7- HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/torchrun --standalone --nproc_per_node=2 -m src.train.train \
  configs/e1/smollm3-3b-base.yaml --resume
```

The initial offline launch was stopped at the owner's request before its first production checkpoint. The owner requested deletion of that aborted run's local log and W&B records; these were removed. The online run starts from the base weights with the same seed and recipe, not from the synthetic preflight checkpoint.

## Result

Training completed successfully at step 2,858; tmux reports exit status 0 and the final HF checkpoint is present. W&B API read on 2026-09-20: final training loss 0.4676753233273272, validation loss 0.46090880036354065 and 239,498,960 tokens seen. The trainer reports 7.09 hours for its training/validation/checkpoint/export section, excluding initial setup.

**Benchmark** (scored 2026-09-20 10:42, `checkpoints/e1-smollm3-3b-base/eval/benchmark/results.md`, 518 `bench/*` keys on run `e82so2fd`), clean / messy, beside the model it was run to compare with and the other large decoders:

| metric | SmolLM3 3B (3.08B) | T5Gemma 2 1b-1b (1.70B) | Gemma 3 4B (3.88B) | E1 Qwen3-1.7B (1.72B) |
|---|---|---|---|---|
| macro-F1 overall | 0.774 / 0.752 | 0.744 / 0.722 | 0.778 / 0.745 | 0.735 / 0.719 |
| macro-F1 unseen questions | 0.698 / 0.688 | 0.677 / 0.642 | 0.702 / 0.684 | 0.654 / 0.652 |
| evidence F1 | 0.638 / 0.608 | 0.619 / 0.590 | 0.635 / 0.604 | 0.603 / 0.570 |
| evidence precision | 0.752 / 0.719 | 0.725 / 0.694 | 0.743 / 0.713 | 0.697 / 0.663 |
| format valid | 0.995 / 0.994 | 0.990 / 0.985 | 0.992 / 0.988 | 0.985 / 0.979 |
| messy SPoRC, macro-F1 / evidence F1 | 0.751 / 0.566 | 0.687 / 0.546 | 0.731 / 0.554 | 0.700 / 0.509 |
| messy 8–16k, macro-F1 / evidence F1 | 0.763 / 0.559 | 0.702 / 0.539 | 0.739 / 0.548 | 0.721 / 0.492 |

Val split: macro-F1 0.806 / 0.775, evidence F1 0.728 / 0.680. Primary metric (unseen-question messy macro-F1): 0.688, against the 1b-1b's 0.642 and the majority predictor's 0.572.

The requested post-training benchmark queue was not fully installed before the session limit interrupted setup. At the owner's next status check on 2026-09-20, training had finished but evaluation had not started. Benchmark launched then in persistent tmux session `tt-eval-smollm3`: GPU 0, vLLM environment, greedy decoding, 16,384-token context, 512-token output cap, both clean/messy variants. The chained job verifies the frozen benchmark hash and final checkpoint, generates raw outputs, requires all 38,220 unique expected keys, then scores and logs online to the same run `e82so2fd`. Failures stop the chain; no repairs are applied. Log: `checkpoints/e1-smollm3-3b-base/benchmark.log`; report: `checkpoints/e1-smollm3-3b-base/eval/benchmark/results.md` once complete. Launcher: `/tmp/tt-smollm3-preflight-fNUMZl/benchmark.sh`.

Online logging verified through the W&B API on 2026-09-20: run state `running`, step-0 validation loss 1.4470646381378174. The tokenizer, frozen split/input-hash manifest and launch-code artefacts are attached as `smollm3-3b-base-tokenizer:v0`, `compliance-qa-frozen-split-provenance:v0` and `smollm3-e1-launch-code:v0`. The metadata-only secondary writer cannot finish the training run. Private notes and raw transcripts are not included.

Data preparation retained 91,485 training examples and excluded 17 over the token cap; validation retained all 5,561. Training contains 231,679,089 prompt tokens and 7,892,771 target tokens, with maximum total length 15,885. The existing native 1b-1b record retains 91,480 training examples under its tokenizer, so the retained sets are not strictly matched.

Existing training tests: 10 passed. A tiny CPU SmolLM3 check with unequal target lengths gives a maximum absolute gradient difference of 1.862645149230957e-08 between accumulated and batched target-token-mean loss. This is a normalisation check, not a training result.

Both weight shards passed SHA256 verification against Hugging Face metadata. The full-length unsharded probe (`tt-decoder/krrthome`, offline) failed with CUDA OOM while initialising AdamW state during the first optimiser update. FSDP is therefore a memory requirement, not an optimisation-recipe change. A DDP/FSDP throughput ratio is unavailable because the unsharded recipe cannot complete an update. The tiny CPU export/reload check preserved all tensors, including the tied embeddings.

The full-size FSDP probe (`tt-decoder/78f10krl`, offline) completed three optimiser updates on fixed synthetic 16,384-token inputs with 256 supervised tokens per GPU. W&B records initial/final losses 11.206123352050781 / 11.199552536010742, peak GPU-0 allocation 17.970627784729004 GiB, and final-step throughput 10,638.672983068156 tokens/s across both GPUs. This is a synthetic memory/mechanics check, not task quality or production throughput. A two-rank checkpoint was written at step 3. A fresh-process reload (`tt-decoder/wuyvvoiq`, offline) restored every optimiser step counter to 3, reproduced the saved next loss within 1e-6 and completed update 4; W&B records `resume_verified: true`. Data position and learning-rate position are deterministic from the step; this zero-dropout model has no stochastic forward operation requiring restored RNG state.

### Validation evaluation

Validation answer generation was also requested to match the previous runs' complete workflow, rather than stopping at validation loss. Launched on GPU 1 in tmux `tt-val-smollm3`, alongside benchmark generation on GPU 0. Same greedy generation and clean/messy variants, all 10,476 validation outputs; exact coverage validation precedes scoring. Outputs: `checkpoints/e1-smollm3-3b-base/eval/gen_val.jsonl`; report: `checkpoints/e1-smollm3-3b-base/eval/val/results.md`; log: `checkpoints/e1-smollm3-3b-base/validation.log`. Validation metrics log to `val_eval/*` on the same W&B run, separate from `bench/*` (the existing scoring CLI hardcodes `bench/*`, so validation uses its local report plus a separate metadata-safe W&B writer). Launcher: `/tmp/tt-smollm3-preflight-fNUMZl/validation.sh`.

## Verdict

**SmolLM3 3B beats the 1b-1b on every slice** (+0.030 / +0.030 macro-F1 overall, +0.021 / +0.046 on unseen questions, +0.019 / +0.018 evidence F1) and ties Gemma 3 4B (within 0.01 on every headline number, ahead on messy answers and on SPoRC and long prompts, behind on clean answers and tags).

What it does and does not verify. It was run "to compare additional E1-recipe models comparable to the 1b-1b, to verify the 1b-1b's performance is architecture". On parameters it is not comparable: 3.08B trainable against the 1b-1b's 1.70B (2.81B non-embedding against 1.40B), so it sits with Gemma 3 4B as a decoder bracketing the encoder-decoder from above, and both say the same thing: a decoder with about twice the encoder-decoder's parameters beats it by 0.02–0.05 macro-F1. The parameter-matched cross-family decoder is still E1 on Qwen3-1.7B (1.41B non-embedding against 1.40B), which the 1b-1b beats on most slices. So the evidence for "architecture" at matched size rests on one cross-family pair (Qwen) and two same-family pairs (Gemma 270M and 1B); the evidence that size overrides it rests on two decoders at 3–4B. On inference compute the 1b-1b reads each prompt token through 0.7B parameters against SmolLM3's 3.08B, about a quarter.

Execution verified on 2026-09-20 against the E1 recipe: same data, split, rendering, target-only loss, lr 1e-5 cosine with 50 warmup, weight decay 0.1, 8-bit AdamW over fp32 masters, one epoch of 2,858 steps at 32 sequences, val every 200 on the fixed 512, 15 wall-clock checkpoints, no resume; FSDP because the unsharded fp32-master state OOM'd (the same recipe, sharded, as the Qwen-built E3 and Gemma 3 4B); FlashAttention 2; the 16k cap of the Gemma pairs (17 records dropped under this tokenizer, against 22 under Gemma's and 1 under Qwen's at 32k). The tokenizer has no start token, so the prompt gets none, as with Qwen. Generation through vLLM, greedy, 512-token cap; scoring by the shared evaluator with the strict parser; benchmark and validation outputs complete (38,220 and 10,476). Tags carry the experiment, base checkpoint and size. Nothing deviates from the recipe.

## Follow-ups

- The parameter-matched control this run was meant to be is SmolLM2-1.7B (1.61B non-embedding, ungated; 8k native context, so it needs RoPE extension or an 8k cap that halves the benchmark) or Llama 3.2 1B/3B (gated). SmolLM2-1.7B under the E1 recipe would give a second cross-family decoder at the 1b-1b's size.
- Repeat E3 and the strongest decoder with a second seed before claiming a robust advantage; select any hyperparameters on validation only with equal tuning budgets.
