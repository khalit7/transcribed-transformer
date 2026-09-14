# E2: prefix-LM SFT of Qwen3-1.7B-Base, the mask flip against E1

The first test of hypothesis 1 (bidirectional attention over the transcript should beat causal). Identical to [E1](../2026-09-10-e1-qwen3-1.7b-base/README.md) in checkpoint, data, rendering, loss, optimiser, schedule and steps; the one change is the attention mask: every prompt token attends to every prompt token, answer tokens attend causally. Written before launch on 2026-09-10.

## Hypothesis

Letting the transcript tokens attend to each other in both directions gives the answer positions a representation built from the whole conversation rather than a left-to-right prefix, and that improves the judgements and the evidence selection. RedLLM (arXiv 2510.26622) measured +2.3 to +6.4 points from exactly this flip at fine-tuning time on 1B to 8B causal-pretrained decoders; Bitune's naive-bidirectional ablation +1.5 to +2.2 under LoRA; PaliGemma and Gemma 3 ship the mask. Against it: Wang et al. 2022 found no gain from the flip without an adaptation stage, and LLM2Vec found the flip degrades Llama-family representations until a short masked-next-token stage repairs them.

**Diagnostic run before launch** (LLM2Vec §4.2, per-layer cosine similarity between causal and bidirectional hidden states, 6 val prompts, first 2,048 tokens): Qwen3-1.7B-Base drops to **0.49 at layer 18** (mean 0.67 over layers 1 to 28, 0.95 at the last layer). That is the Llama-shaped reaction, not the Mistral-shaped one, so the flip alone starts from damaged mid-network representations and the 8.5M supervised target tokens of one SFT epoch have to repair them. The pure flip is still run first because it is the one-variable comparison; the adaptation variant is the planned follow-up.

## Prediction

- On the unseen-question cell, macro-F1 above E1's 0.654 (clean) / 0.652 (messy) by at least 0.02, and evidence F1 above E1's 0.573 / 0.550.
- **Disconfirmed if** E2 lands at or below E1 on both unseen-question macro-F1 and evidence F1 on both variants. Given the diagnostic, a plausible outcome is "no gain without adaptation", which would send the follow-up (E2 + masked-next-token stage) rather than kill the hypothesis; the hypothesis is only disconfirmed for this size if the adapted variant also fails.
- Format validity should stay above 0.95; a large drop would indicate the flip broke generation rather than judgement.

## Setup

As E1 (`configs/e2/qwen3-1.7b-base.yaml` differs from `configs/e1/qwen3-1.7b-base.yaml` in `prefix_lm: true` and the name), with the attention implementation forced from FlashAttention 2 to FlexAttention because FA2 cannot express the mask (`src/train/masks.py`, `src/train/train.py`). Generation: `src/train/generate.py --backend prefixlm` (HF, bidirectional prefill then cached causal decoding, greedy, max 512 tokens), not vLLM. Scoring identical to E1. Throughput and memory are measured, not carried over from E1: see Result.

## wandb run

[tt-decoder/huf32gbu](https://wandb.ai/khalit7-/tt-decoder/runs/huf32gbu), launched 2026-09-10 19:36.

## Result

Read from wandb `tt-decoder/huf32gbu` (training summary and the `bench/*` keys logged by `src/train/evaluate.py`; full tables in `checkpoints/e2-qwen3-1.7b-base/eval/benchmark/results.md`). E1's numbers from its entry.

Training: 2,859 steps, 6.12 h on two GPUs (E1: 4.61 h), median 11.5k tokens/s, MFU 0.32 (E1: 15.1k, 0.42), peak 27.7 GiB. Loss after 3 smoke steps 4.50 against E1's 1.11: the mask flip starts from damaged representations, as the diagnostic predicted. Val loss caught up steadily but never fully: 0.815 / 0.715 / 0.672 / 0.638 / 0.613 / 0.595 / 0.580 / 0.564 / 0.553 / 0.543 / 0.535 / 0.527 / 0.523 / 0.520 / **0.519** at the end, against E1's 0.704 → 0.503 at the same steps; the gap shrank from 0.11 at step 200 to 0.016 at the end and was still shrinking. (Logging note: from the checkpoint at step 1879 onwards the run's `tokens_seen`, `tokens_per_s` and `mfu` fields are corrupted in the window after each checkpoint, a cross-GPU reduction of the counters returning garbage; the loss and gradient fields and the weights were unaffected, val loss stayed on trend, and the counters are computed without a reduction in later runs.)

Benchmark, 20,700 pairs, both variants, E2 against E1:

| slice | variant | n | format valid E1 / E2 | macro-F1 E1 / E2 | majority | evidence F1 E1 / E2 |
|---|---|---|---|---|---|---|
| overall | clean | 17,520 | 0.985 / 0.986 | 0.735 / 0.721 | 0.592 | 0.603 / 0.605 |
| overall | messy | 20,700 | 0.979 / 0.981 | 0.719 / 0.703 | 0.594 | 0.570 / 0.570 |
| **unseen questions** | clean | 3,260 | 0.976 / 0.979 | **0.654 / 0.623** | 0.582 | 0.573 / 0.570 |
| **unseen questions** | messy | 3,860 | 0.977 / 0.981 | **0.652 / 0.625** | 0.572 | 0.550 / 0.541 |
| seen questions | clean | 14,260 | 0.987 / 0.988 | 0.752 / 0.742 | 0.591 | 0.609 / 0.613 |
| seen questions | messy | 16,840 | 0.980 / 0.981 | 0.733 / 0.719 | 0.597 | 0.574 / 0.577 |

Where E2 is ahead: evidence precision (+0.01 everywhere), format validity (605 runaway generations against 656), and the small rare-event families: complaint macro-F1 0.789 vs 0.769 (clean), eod 0.836 vs 0.807 (clean) and 0.830 vs 0.782 (messy), vulnerability equal. Where it is behind: general_qa (0.716 vs 0.731 clean, 0.697 vs 0.715 messy), which is 93% of the benchmark; every corpus except Taskmaster; and the longest inputs most of all (8–16k messy: 0.683 vs 0.721; 4–8k clean: 0.641 vs 0.677). Evidence recall is slightly lower everywhere, so evidence F1 is a wash.

## Verdict

**Disconfirmed for the pure mask flip.** E2 does not beat E1 on unseen-question macro-F1; it trails by 0.03 on both variants, outside the 0.02 gain predicted and the wrong sign. The pattern matches the "no gain without adaptation" outcome the entry named in advance: the training curve shows the flipped model spending the epoch repairing what the flip broke (val loss 0.11 behind at step 200, 0.016 behind at the end, still converging), and the deficit is largest on the longest transcripts, where bidirectional encoding was supposed to help most. The gains on complaint and eod are real but on 300 to 640 pairs each and within the noise of a single run.

Per the entry's own rule, this does not yet disconfirm hypothesis 1 at 1.7B: that verdict belongs to E2+ (the flip after a matched masked-next-token adaptation stage, `../2026-09-11-e1plus-e2plus-adapted-qwen3-1.7b-base/`), already queued. What it does establish: at this size and data, flipping a causal-pretrained decoder to a bidirectional prefix at fine-tuning time alone costs about 0.015 macro-F1 overall and 0.03 on unseen questions, for 33% more training time and a much slower decode path (no vLLM), which is the opposite of what RedLLM and Bitune reported on short-prompt benchmarks.

Caveats as for E1: model-produced gold; one epoch at one learning rate; E2 decoded through the HF loop, E1 through vLLM (the causal check of the loop agreed with vLLM on next tokens, so this is not the explanation for a 0.03 gap).

## Follow-ups

- E2+ vs E1+ (queued): the adaptation that the diagnostic and this curve both call for.
- If E2+ also trails: hypothesis 1 is disconfirmed at 1.7B for this task; try the 4B checkpoint before building E3, since the encoder-decoder shares the mechanism.
- The decode path: 3.5 h per half-benchmark against vLLM's 10 minutes. If prefix-LM survives, it needs a real inference path (a bidirectional prefill kernel in vLLM, or a static KV cache and GQA-aware SDPA in the loop).
