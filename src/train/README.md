# src/train — task fine-tuning and evaluation

The training and scoring code for the experiments in the README's "Design" table. E1 (causal SFT, the control) and E2 (prefix-LM: `model.prefix_lm: true` swaps the causal mask for bidirectional-over-the-prompt, on FlexAttention because FlashAttention 2 cannot express it) share this loop; the other arms add their one difference the same way.

## What a run does

1. **Examples** (`data.py`). Every record of `data/labelled_data/train.jsonl` becomes one example per rendered transcript variant: the prompt is the labelling prompt of `src/synthesis/label.py` without the confidence request (numbered transcript, question, options with their grading rules, evidence-first instruction), the target is the label as compact JSON `{"evidence": [...], "answer": "...", "summary": "..."}` plus `"tags"` where the question has a tag vocabulary, then the end-of-text token. The prompt opens with the tokenizer's own start token when it has one (`<bos>` for Gemma; Qwen adds nothing). Every record is used whatever its licence track: training is research-only, and the tracks govern the labelled-data release (`src/synthesis/export.py`), not what a model may see. The default renders the variant each label was made on (clean for the service-call corpora, real ASR for SPoRC). Tokenised examples are cached under `data/interim/sft/` as one flat int32 tensor.
2. **Batching** (`data.plan_epoch`). Deterministic on every rank: shuffle by seed and epoch, cut into groups of `batch_sequences`, deal each group's sequences to the ranks alternating by length, pack each rank's share into micro-batches under a padded-token budget. The loss is scaled by the group's total target tokens, so gradient accumulation and data parallelism give the exact token mean, whatever the sequence lengths.
3. **Loss** (`train.target_loss`). Hidden states at the positions that predict a target token go through the output head; nothing else does. The vocabulary-sized logits never span the prompt.
4. **Precision** (`train.MasterOptimizer`). The model and its gradients are bf16 (FlashAttention 2 needs it, and it halves the saved activations). The optimizer owns an fp32 master copy of every parameter, clips and updates there, and copies back. Pure bf16 would round away updates smaller than a weight's resolution at these learning rates. 8-bit AdamW by default. Under `model.sharding: fsdp` (FSDP2, any arch: transformer blocks and a frozen image tower are the units) the same recipe runs over the local shards, ~5 bytes per parameter per GPU; `optim.weights: bf16_sr` drops the fp32 copy and updates the bf16 shards with stochastic rounding (torchao's 8-bit AdamW), ~3 bytes per parameter per GPU, which is what lets 7.5B parameters train in full on two 32 GB cards. Sharded runs checkpoint their own shards per rank (resume needs the same world size) and export by gathering one parameter at a time into a bf16 copy on rank 0.
5. **Logging.** wandb, project and tags per CLAUDE.md, with the fully resolved config, loss, learning rate, gradient norm, tokens seen, tokens/s, MFU (model FLOPs, PaLM-style estimate, over the spec-sheet peak in the config) and peak GPU memory. Val loss on a fixed subset every `val_every` steps. Checkpoints (model, master weights, optimizer state, step) on a wall-clock interval and at the end, the last two kept; `--resume` continues from the latest. The final bf16 model is written in HF format under `checkpoints/<name>/final/` for generation.

```
torchrun --nproc_per_node 2 -m src.train.train configs/e1/qwen3-1.7b-base-p.yaml            # train
.venv-vllm/bin/python -m src.train.generate checkpoints/<name>/final data/labelled_data/benchmark.jsonl out/gen.jsonl
uv run python -m src.train.evaluate data/labelled_data/benchmark.jsonl out/gen.jsonl out/ --wandb-run <id>
```

`generate.py` runs in its own environment (`.venv-vllm`, vLLM pins its own torch) and writes the raw text per pair and variant; `evaluate.py` scores it per the `eval-bench` skill (format gates, answer exact/recovered/invalid, macro-F1 next to the majority-class predictor, evidence precision/recall/F1 with empty-means-empty, tag Jaccard; slices by track, dataset, family, seen/unseen question, length bucket; confusion matrices for the priority families) and can log the headline numbers to the training run. Summary faithfulness needs an LLM-as-a-judge and is TBD.

## Files

| File | Job |
|---|---|
| `config.py` | Run config (pydantic) loaded from YAML; `wandb_tags()` |
| `data.py` | Prompt and target text, tokenisation and cache, batching plan, collation |
| `train.py` | DDP loop, master-weight optimizer, val loss, checkpoint/resume, wandb, final export |
| `masks.py` | E2's prefix-LM mask (FlexAttention BlockMask or dense SDPA), the 5090 block sizes, the left-padded prefill mask for generation |
| `encdec.py` | E3's encoder-decoder built from one Qwen3 checkpoint: bidirectional encoder, decoder with cross-attention initialised from self-attention (zero output projection), optional LoRA on the towers, save/load. Trained in full with `model.sharding: fsdp` (FSDP2: fp32 master shards, bf16 compute, 8-bit AdamW on the shards; the only way 3.8B parameters fit two 32 GB cards) |
| `generate.py` | Greedy generation for a split file (vLLM; `--backend prefixlm` for E2; `--backend encdec` for E3; `--backend hf_encdec` for a native Hugging Face encoder-decoder such as T5Gemma 2; HF as the slow fallback) |
| `evaluate.py` | Scoring and slicing; `metrics.json` and `results.md` |
| `../../configs/e1/*.yaml`, `e2/`, `e3/`, `adapt/` | E1, E2 and E3 runs (Qwen3-1.7B-Base on every training record; E2 differs by `prefix_lm: true`, E3 by `arch: encdec`), their adapted variants, and the adaptation stages (causal, MNTP, seq2seq). `e3/t5gemma2-1b-1b.yaml` (`arch: hf_encdec`, Google's converged encoder-decoder through the library's own forward, sdpa, DDP) and its control `e1/gemma3-1b-pt.yaml` (the decoder it was adapted from) form the native reference pair, not comparable with the Qwen arms |

## Environment

torch 2.8.0+cu128, transformers 5.x, FlashAttention 2.8.3 from the release wheel (sm_120 kernels included; verified on the RTX 5090s), bitsandbytes 0.50 (8-bit AdamW on sm_120 verified). Memory probe on one GPU, Qwen3-1.7B-Base, batch of one sequence, checkpointing on (2026-09-10): peak 14.0 GiB at 4k tokens, 18.3 GiB at 16k, 21.4 GiB at 32k; forward+backward 7.2k / 8.6k / 6.5k tokens/s at those lengths. The first probe without `model.train()` peaked at 25 GiB at 4k, because transformers applies gradient checkpointing only in train mode.

## Known limitations

- The prompt loss weight is fixed at zero (README "Design"); the loop has no other loss option.
- No sequence packing across examples: micro-batches pad to the longest sequence. FlashAttention unpads internally, so the cost is memory and the collate, not attention compute.
- Gradients accumulate in bf16 across the micro-batches of one step (the fp32 conversion happens at the step). Megatron-style fp32 accumulation would cost another 6.8 GB at 1.7B.
- SPoRC training calls run to 36k tokens; anything over `max_seq_len` (32k by default) is dropped and counted in the run's `train_stats`.
