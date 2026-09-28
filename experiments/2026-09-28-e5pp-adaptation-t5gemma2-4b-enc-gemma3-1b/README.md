# E5++ with the 4b encoder: the third point of the encoder axis at the fixed 1B decoder

Written before the run, 2026-09-28 16:00. Companion to the
[1b-encoder](../2026-09-26-e5pp-adaptation-t5gemma2-1b-enc-gemma3-1b/README.md) and
[270m-encoder](../2026-09-27-e5pp-adaptation-t5gemma2-270m-enc-gemma3-1b/README.md) adaptation pairs.

## Hypothesis

> **Comparator numbers in this record come from runs with a bug.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (half the data at batch 16), so every trained-arm number quoted here as a bar or reference (the native pairs, the mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) is from a buggy run being replaced by a re-run ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); read the predictions against the re-run numbers when they land.

E5++ with the 1b encoder beat its matched control on every metric (0.671 / 0.505 against E1++'s 0.650 / 0.470 on clean
macro-F1 / unseen-question macro-F1, evidence F1 0.591 against 0.532). The mix-and-match grid said that capacity pays in
the encoder and that zero-shot question transfer travels with it. Before spending on a 12B decoder I want to know
whether the recipe scales the same way: the same decoder, the same adaptation stage, the T5Gemma 2 4b text encoder
(about 3.9B) in place of the 1b. With the 270m and 1b arms this gives three points on the encoder axis at one decoder,
one adaptation data set, one budget. The control, E1++ on the 1B decoder, is shared and already scored.

The arm is also the fast configuration for this task's shape, a large encoder run once per prompt and a small decoder
run per output token, so it is read against the Gemma 3 4B decoder (0.778 / 0.702) as well: the mix-and-match
4B encoder + 270M decoder matched that decoder on answers and evidence through the native interface; this is what the
soft-token interface gets from the same encoder.

## Prediction

- Above the 1b arm on unseen-question macro-F1 and evidence F1 by more than 0.02 if the recipe scales with the encoder.
- Against the Gemma 3 4B decoder: I expect it below, because the soft-token interface recovered only part of the
  encoder's value at 1B (E5++ 1b at 0.671 against the native 1B encoder + 270M decoder at 0.724); how far below is the
  measure of the interface's cost at this size, and the number the 12B plan turns on.
- **Disconfirmed if** at or below the 1b arm on both variants: the interface caps what a bigger encoder can deliver,
  and the rented 12B run should not use soft tokens.
- One seed; a gap under about 0.02 is a direction, not a result.

## Setup

- Adaptation stage `adapt-continuation-corpora-e5-t5gemma2-4b-enc-gemma3-1b`
  (`configs/adapt/continuation-corpora-e5-t5gemma2-4b-enc-gemma3-1b.yaml`): the T5Gemma 2 4b text encoder (2560 wide)
  into Gemma 3 1B (1152 wide) through `nn.Linear(2560, 1152)` from `checkpoints/stitch/4b-to-gemma3-1b-embed.pt` (ridge
  regression onto the decoder's scaled token embeddings over 3M prompt tokens, held-out explained variance **0.665**,
  the highest of the three stitches: 270m 0.620, 1b 0.636). Same data, cuts, objective and budget as the other two
  stages (the ++ outside-text sample, 131,426 documents, one epoch, 4,107 steps, 16k cap). Everything trains, stitch at
  10×, FSDP.
- Fine-tune `e5pp-t5gemma2-4b-enc-gemma3-1b` (`configs/e5/t5gemma2-4b-enc-gemma3-1b-plusplus.yaml`) from the adapted
  checkpoint under E1's recipe.
- Memory: about 4.9B parameters, the size that needed frozen embedding tables in the mix-and-match grid. The queue
  smokes three variants in order, 8k micro-batches with every parameter training, 4k micro-batches, then 8k with the
  two embedding tables frozen, on the standard slice and on the 64 longest training records, and runs the first that
  fits. **The frozen-embedding variant ran**: 8k with every parameter training overflowed on the longest records (30.5 GiB
  in use, 1 GiB more requested), 4k with every parameter training overflowed too (the parameter state, not the
  activations, is what does not fit), 8k with the two embedding tables frozen peaked at 27.8 GiB. So 4.88B parameters of
  which the decoder's and encoder's embedding tables (about 1B) are frozen; the control trained its embeddings, a recipe
  difference this record carries until a frozen-embedding E1++ is run. The stage runs at about 5.4k tokens/s, about 30 h
  for the 594M tokens.
- Generation `--backend soft --batch-tokens 40000` (the 4B encoder's budget), scored by `src/train/evaluate.py`.
- Code as the 1b pair, plus the soft model's FSDP path and tied-head handling added for the 4B-decoder configs
  (`experiments/` records under D53); 30 tests + 1 GPU skip.
- wandb: adaptation stage in tt-pretrain, fine-tune in tt-encdec; run ids TBD.

## Result

Not run to completion here. The memory smokes settled the local question: with every parameter training, the 4.88B pair ran out of memory on the 64 longest records at 8k micro-batches (30.5 GiB in use, 1 GiB more requested) and at 4k (the parameter state, not the activations, is what does not fit); only the variant with the two embedding tables frozen fitted (27.8 GiB) and its adaptation stage ran for 29 minutes (step-0 continuation loss 2.42, 6.3k tokens/s) before I stopped it. Freezing the tables is a departure from E1's recipe, and every arm in this project is on that recipe, so the run was aborted rather than continued with a confound (D55); its wandb run was deleted and its logs are under `checkpoints/_archive/`.

## Verdict

No result. The arm needs an 80 GB card under the exact recipe and joins the rented-hardware list with the 1b encoder + 4B decoder pair, which is the same size and would fail the same way.

## Follow-ups

- Run on rented hardware under E1's recipe unchanged (one 80 GB card, DDP): the adaptation stage, the fine-tune, and generation and scoring here.
- The three-point encoder axis at the 1B decoder stays at two points until then.
