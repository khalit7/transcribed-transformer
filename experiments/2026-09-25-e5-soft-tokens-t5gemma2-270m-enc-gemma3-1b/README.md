# E5: the T5Gemma 2 270m encoder as soft tokens for the Gemma 3 1B decoder

Written before the run, 2026-09-25 17:30; result and verdict added 2026-09-26 09:50.

## Hypothesis

The same-decoder test of "does an encoder in front help". Every result so far that favours an encoder changes the
decoder too: the native pairs (E4) carry Google's adaptation, mix-and-match swaps decoders, E2 changed the decoder's
own mask and found nothing. Here the decoder is E1's Gemma 3 1B, weights, recipe and input length unchanged, with no new
attention. The only change is what it reads at the prompt positions: the states of a bidirectional encoder (the T5Gemma
2 270m text encoder, from the same family) mapped into its embedding space by an affine stitch, instead of the token
embeddings. The answer tokens enter through the ordinary embeddings and the decoder reads everything through its own
self-attention. If a bidirectional view of the transcript is worth having, this is where a causal decoder can use it.

I replace the prompt embeddings rather than add to them. The E6 decoder-only + encoder arm kept the decoder's easy path
(the raw prompt) and never opened its gate to the encoder; replacing the input leaves the decoder no choice but to read
the encoder.

The 270m encoder first because it is the smallest confound: the arm is the control plus 270M parameters. The 1B encoder
follows if this shows anything, since mix-and-match found the capacity pays in the encoder.

## Related work

The mechanism is not new. [LangBridge (Yoon et al., ACL 2024)](https://aclanthology.org/2024.acl-long.405/) maps an mT5 encoder's hidden states through a trainable linear layer into a frozen decoder-only model's input-embedding space as soft prompts, trained with a language-modelling objective, for cross-lingual transfer. [Dolphin (Chen et al., 2024)](https://arxiv.org/abs/2408.15518) projects a small decoder's encoding of a long context into a 7B decoder's input space, borrowing the vision-language projector recipe; [E2LLM (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.970.pdf), [xRAG (NeurIPS 2024)](https://arxiv.org/pdf/2405.13792) and [ICAE (ICLR 2024)](https://arxiv.org/abs/2307.06945) project a text encoder's output into the decoder's input at various compression ratios with an alignment stage before task tuning. What this record adds is the controlled measurement: the decoder can read the input as tokens anyway, so the question is whether a bidirectional read of the same tokens is worth anything on top of it; the unadapted arm is run as a control; and the adapted arm is read against the same decoder given the identical adaptation data, cuts and budget with the prefix as tokens, so a gain is attributable to the encoder rather than to the extra training. Same-family encoder and decoder, both towers trained, the stitch initialised by regression onto the decoder's own embeddings, and an encoder-size axis at one decoder are the other differences.

## Prediction

- Above the control (`e1-gemma3-1b-pt`: clean macro-F1 0.651, unseen-question 0.496, evidence F1 0.530) on overall and
  unseen-question macro-F1 if the bidirectional read helps.
- The step-0 validation loss says how far the stitched input is from E1's: the control started at 1.732 (measured on the
  E6 arm, whose step 0 is the control exactly). The stitch fit's held-out explained variance is the same measure from the
  other side.
- **Disconfirmed if** the arm is at or below the control on both variants: with the decoder forced to read the encoder,
  a bidirectional read of the transcript then adds nothing a causal decoder of this size can use, and the encoder gains
  elsewhere in this project belong to the decoder side or to Google's adaptation.
- A gap smaller than a seed replicate would move is not a result either way; no replicate exists yet.

## Setup

- Model: `src/train/soft.py`. Decoder `google/gemma-3-1b-pt` (`.model` and `lm_head`, as E1 fine-tuned it); encoder the
  text encoder of `google/t5gemma-2-270m-270m` (640 wide); stitch `nn.Linear(640, 1152)` on the encoder's final states.
  Decoder input: the stitched states at the prompt positions (the two checkpoints share the tokenizer, so the positions
  correspond one to one), the decoder's scaled token embeddings for the answer.
- Stitch initialisation: ridge regression from the encoder's states onto the decoder's scaled embedding of the same
  token, over 3M prompt tokens of the training split (`src/train/fit_stitch.py --target-embeddings`, ridge 1e-2 relative
  to the Gram diagonal), held-out explained variance **0.620** (fit in 52 s; `checkpoints/stitch/270m-to-gemma3-1b-embed.json`).
- Recipe: E1's (`configs/e5/t5gemma2-270m-enc-gemma3-1b.yaml`): lr 1e-5, 50 warm-up steps, weight decay 0.1, 8-bit AdamW
  with fp32 masters, one epoch, 32 sequences per step, 16k cap, loss on the answer only, DDP, flash-attention on the
  decoder, sdpa on the encoder. Every parameter trains; the stitch at 10× the base rate, as in mix-and-match. Micro-batch
  16k tokens (the smoke decides whether it fits).
- Batches: `collate_encdec(dec_prompt=True, dec_start=False)`, the E6 decoder-only + encoder arm's layout.
- Generation: `src/train/generate.py --backend soft` (encoder once, stitched states as the prefill, cached decoding),
  scored by `src/train/evaluate.py` as every arm.
- Control: `e1-gemma3-1b-pt` (`experiments/2026-09-15-e4-t5gemma2-1b-1b-native-pair/README.md` records it).
- wandb: [tt-encdec/iv6h1yif](https://wandb.ai/khalit7-/tt-encdec/runs/iv6h1yif); benchmark keys logged to the same run.

## Result

> **The numbers below come from a run with a bug and are being replaced.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (each rank trained its own replica on its half of every step; rank 0's was saved), so these are for a model trained on half the data at an effective batch of 16. The run is in the re-run queue under the fixed loop ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); its replacement's numbers replace these when scored.

- Smoke (2026-09-26 02:33, both cards): 1.27B trainable (0.7M in the stitch), three steps loss 4.16, val 3.45 at step 3, gradient norm 1,172 before clipping (clip 1.0), 17.5k tokens/s, 18.3 GiB at 16k micro-batches; two steps on the 64 longest records clean. The 16k micro-batch fits, unlike the E6 hybrid's, since there is no second tower of the decoder's size.
- Validation against the control, same steps: **4.44 / 1.73 (step 0)**, 0.975 / 0.921 (200), 0.880 / 0.837 (400), 0.828 / 0.797 (600), 0.794 / 0.764 (800), 0.763 / 0.742 (1000), 0.740 / 0.717 (1200), 0.724 / 0.705 (1400), 0.702 / 0.685 (1600), 0.685 / 0.672 (1800), 0.672 / 0.659 (2000), 0.661 / 0.649 (2200), 0.653 / 0.642 (2400), 0.646 / 0.637 (2600), 0.642 / 0.634 (2800), **0.641 / 0.632 (2858)**. The stitched input starts 2.7 nats above E1's own (the mix-and-match decoders started 0.5 above theirs), closes to 0.05 by step 200 and to 0.009 at the end; training loss level with the control from step 2400. The gradient norm settles into the control's range by step 300. Training 4.35 h (02:35 → 06:56) at about 16.7k tokens/s over the pair, 19.5 GiB peak; E1's own run of this decoder took about the same.
- Generation through the soft backend 06:56 → 09:29 on both cards (160k batch budget, no memory trouble with the 1B decoder); scored 09:29.
- **Benchmark** (38,220 outputs), clean / messy:

| model | macro-F1 | unseen-question macro-F1 | seen-question macro-F1 | evidence F1 | evidence precision | format valid |
|---|---|---|---|---|---|---|
| control: Gemma 3 1B (E1) | 0.651 / 0.640 | 0.496 / 0.493 | 0.683 / 0.670 | 0.530 / 0.500 | 0.589 / 0.555 | 0.955 / 0.946 |
| **E5: 270m encoder into Gemma 3 1B** | 0.636 / 0.620 | 0.461 / 0.450 | 0.672 / 0.655 | 0.531 / 0.494 | 0.594 / 0.550 | 0.959 / 0.944 |
| native 270M encoder + 1B decoder (E4-mix-and-match) | 0.673 / 0.659 | 0.530 / 0.544 | | 0.584 / 0.556 | 0.679 / 0.649 | 0.988 / 0.983 |

  Below the control on answers overall (−0.015 / −0.020) and on unseen questions (−0.035 / −0.043); level on evidence F1 (+0.001 / −0.006), precision and format. Per slice (clean macro-F1, E5 / control): aci_bench 0.556 / 0.582, apptek 0.643 / 0.654, taskmaster 0.654 / 0.666, vulnerability 0.627 / 0.650, complaint 0.725 / 0.700, eod 0.753 / 0.745; messy SPoRC 0.551 / 0.592, messy 8–16k 0.541 / 0.580: the losses are largest on the long, real-ASR slices and on unseen questions, the places a bidirectional read was supposed to help. Val split clean / messy macro-F1 0.678 / 0.647 against the control's 0.687 / 0.670, evidence 0.580 / 0.537 against 0.578 / 0.536.

## Verdict

Disconfirmed for this encoder. With the decoder held exactly at E1's and forced to read the encoder, the 270m encoder's bidirectional view of the transcript is worth nothing to it: answers slip a little, zero-shot transfer slips more, evidence is unchanged. The same encoder behind the T5Gemma 2 1B decoder (a decoder adapted by Google to read an encoder) scores 0.673 / 0.530, so the encoder carries usable information; a causal decoder given that information at its input, and 2.8k steps to learn to use it, does not benefit. Three readings are open and this run cannot separate them: the encoder is too small (the mix-and-match grid says encoder capacity is where the gains live, and the 270M one was the smallest confound by design); the stitched input costs the decoder more than the encoder gives (it starts 2.7 nats from its own input and ends 0.009 above the control, so some of the fine-tune was spent recovering rather than gaining); or a causal decoder needs the encoder's states at every layer, as the native decoder has them, rather than once at the input. What is closed: the cheap version of "put an encoder in front of the decoder you have" does not work at this size. One seed.

## Follow-ups

- The 1B encoder (`google/t5gemma-2-1b-1b`'s text encoder) behind the same decoder: the run that separates "too small" from "does not help". The stitch fit and config are a two-line change; about the same cost as this run plus the encoder's share.
- A seed replicate of the control or of this arm before reading 0.015 as anything more than a direction; the unseen-question gap of 0.035 to 0.043 is larger than the gaps between neighbouring mix-and-match cells and is the finding.
