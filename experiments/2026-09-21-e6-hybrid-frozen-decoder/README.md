# E6, frozen decoder: the decoder-only + encoder arm with the decoder frozen; only the encoder and the cross-attention train

Written before the run, 2026-09-21. The complement of the E6 decoder-only + encoder arm: train only the encoder and the cross-attention layers, and keep the rest of the decoder frozen.

## Hypothesis

The E6 decoder-only + encoder arm lets the optimiser choose between the decoder's own self-attention (pretrained, cheap to adapt) and the new encoder path (zero at initialisation); it may leave the encoder unused, which the logged cross-attention share will show but cannot prevent. The E6 frozen-decoder arm removes the choice: with the decoder frozen, embeddings and head included, the loss can only fall through the encoder and the cross-attention sublayers. It measures how much of the task the encoder path alone can deliver into a fixed decoder, the complement of the E6 decoder-only + encoder arm.

## Prediction

- Comparisons: E1 on the same decoder (`e1-gemma3-1b-pt`, fully fine-tuned, 1.0B trainable) and the E6 decoder-only + encoder arm (everything trained). The E6 frozen-decoder arm trains about 1.08B parameters (the 1.0B encoder plus 77M of cross-attention and norms), none of them in the decoder.
- Expected: below E1 on the benchmark, with a cross-attention share of the residual stream well above the E6 decoder-only + encoder arm's. Format validity is where a frozen decoder should show strain first (the scorer's exact / recovered / invalid split).
- **Surprising, and the result that would matter:** the E6 frozen-decoder arm at or above E1 on evidence F1 and clean unseen-question macro-F1, i.e. the encoder path alone matches full fine-tuning of the decoder.
- Not identifiable from this run alone: whether a gain over the frozen decoder comes from the encoder's content or from any trainable capacity attached to a frozen decoder. The matched control, the frozen decoder with about 77M trainable parameters and no encoder (LoRA), is a follow-up if the E6 frozen-decoder arm looks good.

## Setup

Everything as the E6 decoder-only + encoder arm (`configs/e6/gemma3-1b-t5gemma2-enc.yaml`), the single change being `model.trainable: [encoder, cross]` (`configs/e6/gemma3-1b-t5gemma2-enc-frozen-dec.yaml`): parameter names containing neither substring are frozen (`freeze_except` in `src/train/train.py`). Learning rates unchanged (encoder at 1e-5, cross-attention at 1e-4), zero-initialised output projections, 16k cap, DDP. The backward still runs through the frozen decoder layers to reach the lower cross-attention sublayers and the encoder, so throughput is expected near the E6 decoder-only + encoder arm's; memory should be lower by the decoder's masters and optimizer state.

- wandb: tt-encdec, run id TBD (`checkpoints/e6-gemma3-1b-t5gemma2-enc-frozen-dec/wandb_id`). Queue 18, after queue 17.
- Smoke: TBD.

## Result

TBD.

## Verdict

TBD.

## Follow-ups

- The matched no-encoder control: frozen decoder + LoRA at the cross-attention's parameter count.
