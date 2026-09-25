# E6, native hybrid: T5Gemma 2 1b-1b with the decoder reading the prompt too

Written before the run, 2026-09-21. The idea: an encoder-decoder in which the decoder also gets the full input. The encoder encodes the input; the decoder sees the raw input through self-attention and the encoder's states through cross-attention. (Called the hybrid arm while it was being planned, labelled E4 at the time; E6 under the current taxonomy.)

## Hypothesis

The native pairs show the encoder-decoder beating the decoder it was made from at 270M and 1B, and the matched cross-family decoders at 1.4B land below the 1b-1b on evidence and unseen questions. E2 showed that giving a decoder bidirectional attention over the raw prompt does not beat causal attention. Two readings of the encoder-decoder's advantage remain: the encoder's states carry something the decoder cannot recover from the raw tokens, or the encoder-decoder wins because its decoder is spared reading the long prompt (a division of labour). A decoder that reads the raw prompt itself *and* cross-attends to the encoder's states separates them. Prior instances of the mechanism: BERT-fused NMT (+1–2 BLEU), CALM (composition by cross-attention, same input to both models), NVLM-H (multimodal hybrid); no clean text-to-text ablation against both parents was found.

## Prediction

- If the encoder's states add information: the hybrid at or above the 1b-1b on evidence F1 on both variants and on clean unseen-question macro-F1 (1b-1b: 0.619 / 0.590; 0.677).
- If the advantage is division of labour: the hybrid falls back toward the Gemma 3 1B decoder (0.530 / 0.500 evidence F1; 0.496 clean unseen), because the decoder now carries the prompt again and can shortcut past the encoder.
- **The hypothesis "the encoder adds information" is disconfirmed if** the hybrid is below the 1b-1b on evidence F1 on both variants by more than 0.01. A result between the two parents is reported as partial.

## Setup

- Weights: `google/t5gemma-2-1b-1b`, the same starting point as the native E4 run; no new parameters. The decoder's merged attention (one softmax over its own keys and the encoder's) already covers both paths.
- Decoder input: start token, the prompt, then the label; loss on the label only (`model.decoder_sees_prompt: true`, `collate_encdec(dec_prompt=True)`). Causal self-attention over prompt and label, as the decoder was pretrained; the encoder reads the prompt as before. Everything else as `configs/e4/t5gemma2-1b-1b.yaml`: E1's recipe, 16k cap, DDP, SDPA.
- Generation: `generate.py --backend hf_encdec_hybrid`: encoder pass; decoder prefilled with start token + prompt (left-padded within a length-sorted batch, so padding is a few tokens; explicit position ids from 0) with cross-attention to the encoder states; greedy decoding of the label with a growing cache. Verified on the base weights (it regurgitates the prompt, as an untrained model should).
- Probe (one RTX 5090, forward + backward, model and gradients only): 8.3 GiB at 8k and 11.3 GiB at 16k encoder tokens with the same tokens on the decoder side; the loop's masters and optimizer add about 6 GB. Smoke (both GPUs, 3 steps): loss 1.432, val 1.485 at step 3, 24.7 GiB, 8.7k tokens/s across the pair (the decoder now processes the prompt too, so about half the 1b-1b's throughput).
- Cost note: per prompt token this arm spends the prefix-LM's decoder compute plus the encoder pass, more than either parent; it is a diagnostic, not a deployment candidate.
- wandb: tt-encdec/7h3y5uab. Queue 16, 2026-09-21 00:45 → 16:25 (training 9.6 h including a restart from step-271 after the disk filled at step ~410; 7.2k tokens/s over the pair against the 1b-1b's 15.9k, since the decoder processes the prompt too; 26.2 GiB peak; generation 3.4 h with the second benchmark half split over both GPUs).

## Result

Benchmark (38,220 outputs over 20,700 pairs), the E6 native hybrid against its two parents:

| | macro-F1 clean / messy | unseen-question macro-F1 clean / messy | evidence F1 clean / messy | format valid clean / messy |
|---|---|---|---|---|
| E6 native hybrid (1b-1b, decoder sees the prompt) | 0.740 / 0.721 | 0.654 / 0.625 | 0.618 / 0.589 | 0.993 / 0.987 |
| T5Gemma 2 1b-1b (E4, native) | 0.744 / 0.722 | 0.677 / 0.642 | 0.619 / 0.590 | 0.990 / 0.985 |
| Gemma 3 1B (E1, the decoder) | 0.651 / 0.640 | 0.496 / 0.493 | 0.530 / 0.500 | 0.955 / 0.946 |

Per slice (messy), the hybrid and the 1b-1b are within 0.01 of each other on every dataset, length band and question family except aci_bench answers (0.715 vs 0.739) and unseen questions (0.625 vs 0.642; clean 0.654 vs 0.677). Val loss tracked the 1b-1b within 0.005 at every reading and ended at 0.532 vs 0.528. Val-split scores are identical on answers (0.773) and slightly higher on evidence (0.716 vs 0.707).

## Verdict

Neither prediction's disconfirmation fired: the hybrid is not below the 1b-1b on evidence F1 by more than 0.01 (−0.001 on both variants), and it did not fall back toward the decoder. Giving the decoder the raw prompt as well changed nothing on answers or evidence and cost 0.02 on unseen questions, at half the throughput. Reading: the encoder-decoder's advantage over its decoder is not a division of labour that the raw prompt would undo, and the raw tokens add nothing the encoder's states lack; with both paths available the model stays at the encoder-decoder's level. The unseen-question gap is the one signal and is within what a seed replicate is needed to interpret (none exists yet for this pair). Whether the decoder actually read the encoder or shortcut through self-attention is not observable here (merged attention, no cross-attention share logged); the E6 decoder-only + encoder arm logs that share.

## Follow-ups

- The E6 decoder-only + encoder arm and its frozen-decoder variant (D42, D43): the converse arm, a decoder-only model given the encoder, with the cross-attention share logged.
- A seed replicate of the 1b-1b before reading a 0.02 unseen-question difference as anything.
