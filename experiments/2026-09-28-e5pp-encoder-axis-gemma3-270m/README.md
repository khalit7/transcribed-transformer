# E5++ into the Gemma 3 270M decoder: our stitching recipe against the native encoder-decoders, cell by cell

Written before the runs, 2026-09-28 16:45. The 270M-decoder column of the encoder-by-decoder grid; companion to the
[1b](../2026-09-26-e5pp-adaptation-t5gemma2-1b-enc-gemma3-1b/README.md),
[270m](../2026-09-27-e5pp-adaptation-t5gemma2-270m-enc-gemma3-1b/README.md) and
[4b (aborted here)](../2026-09-28-e5pp-adaptation-t5gemma2-4b-enc-gemma3-1b/README.md) arms into the 1B decoder.

## Hypothesis

> **Comparator numbers in this record come from runs with a bug.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (half the data at batch 16), so every trained-arm number quoted here as a bar or reference (the native pairs, the mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) is from a buggy run being replaced by a re-run ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); read the predictions against the re-run numbers when they land.

The question is practical: how does our recipe for putting an encoder in front of a decoder, soft tokens through a
regression-fitted stitch plus 594M tokens of continuation adaptation, compare with a native encoder-decoder, whose
decoder was adapted by Google to read the encoder at every layer at far larger cost. The 270M-decoder column is where
every cell has a native counterpart with the same encoder: the native 270m-270m pair (0.660 / 0.509 clean macro-F1 /
unseen-question macro-F1), and the mix-and-match 1b encoder + 270m native decoder (0.724 / 0.646) and 4b encoder + 270m
native decoder (0.774 / 0.694). Three E5++ arms with those encoders into the raw Gemma 3 270M, plus E1++ on that decoder
as the matched control, give two readings per cell: does the encoder help this decoder under our recipe, and how much of
the native advantage does the recipe recover. The gap between the native and the E5++ numbers, as a function of encoder
size, is the answer; it is measured in the fast configuration, a large encoder run once per prompt and a small decoder
run per output token, which is the one that would be deployed.

This is recipe against recipe, not interface against interface: the native decoders also had their own adaptation. The
E1++ control accounts for what our adaptation data does for the decoder alone; nothing here accounts for Google's.

## Prediction

- E1++ 270M within a seed's worth of E1 270M (0.446 / 0.315, evidence F1 0.299, format valid 0.651): the continuation
  stage did nothing for the 1B decoder on the task.
- Each E5++ arm above E1++ 270M on evidence F1 and format validity (the recipe's signature at the 1B decoder), and on
  answers by an amount that grows with the encoder: at 270m level or slightly up, at 1b and 4b clearly up, since the
  270M decoder alone leaves the most room.
- Against the native counterparts: below in every cell, and the gap the measure. If the 4b arm gets near 0.774 / 0.694
  the recipe is competitive with the native pairs at a fraction of the adaptation cost; if it stays well short, the
  native interface earns its cost and the rented 12B run should use a decoder adapted to read the encoder rather than
  soft tokens.
- **Disconfirmed if** the E5++ arms do not order by encoder size on answers, or if the 4b arm is at or below the 1b arm:
  then the soft-token interface caps what a bigger encoder can deliver at this decoder and the encoder axis under the
  recipe is flat, unlike the native one.
- One seed per cell; gaps under about 0.02 are a direction, not a result. The 270M decoder's answer quality and format
  validity are low on their own, so evidence F1 and format validity carry part of the reading.

## Setup

- Control: `adapt-continuation-corpora-gemma3-270m` (`configs/adapt/continuation-corpora-gemma3-270m.yaml`) then
  `e1pp-gemma3-270m` (`configs/e1/gemma3-270m-plusplus.yaml`): the continuation objective on the ++ outside-text sample
  (131,426 documents, one epoch, 4,107 steps, 16k cap), then the E1 fine-tune, vLLM evaluation.
- Arms: `adapt-continuation-corpora-e5-t5gemma2-{270m,1b,4b}-enc-gemma3-270m` then
  `e5pp-t5gemma2-{270m,1b,4b}-enc-gemma3-270m` (`configs/adapt/continuation-corpora-e5-t5gemma2-*-enc-gemma3-270m.yaml`,
  `configs/e5/t5gemma2-*-enc-gemma3-270m-plusplus.yaml`). Stitches `checkpoints/stitch/{270m,1b,4b}-to-gemma3-270m-embed.pt`
  (ridge regression onto the decoder's scaled embeddings over 3M prompt tokens; held-out explained variance 270m 0.675, 1b 0.676, 4b 0.701, the highest of any stitch so far, the 640-wide target being the easiest to explain).
  Everything trains, stitch at 10×; the 270m and 1b arms under DDP at 16k micro-batches, the 4b arm (about 4.2B) under
  FSDP at 8k, every parameter training or not at all (D55). Generation `--backend soft`, 160k batch budget for the small
  encoders and 40k for the 4b; scored by `src/train/evaluate.py`.
- Order: control, 270m, 1b, 4b; every stage smoked on the standard slice and the 64 longest records before the queue
  started (2026-09-28 16:51 to 17:10), every parameter training, all sixteen passing. Peak memory on the 64 longest
  records: stages 6.7 / 10.1 / 18.7 / 26.7 GiB (control, 270m, 1b, 4b arms); fine-tune settings 21.6 / 10.5 / 18.7 / 28.9 GiB.
  The 4b arm (4.15B trainable, FSDP at 8k) sits where the stitched 4B arms sat. Queue 29 launched 17:10.
- Native counterparts: `e4-t5gemma2-270m-270m`, `e4mm-1b-enc-270m-dec`, `e4mm-4b-enc-270m-dec`.
- wandb: stages in tt-pretrain, E1++ in tt-decoder, E5++ in tt-encdec; run ids TBD.

## Result

TBD.

## Verdict

TBD.

## Follow-ups

TBD.
