# E5++ into the Gemma 3 270M decoder: our stitching recipe against the native encoder-decoders, cell by cell

Written before the runs, 2026-09-28 16:45. The 270M-decoder column of the encoder-by-decoder grid; companion to the
[1b](../2026-09-26-e5pp-adaptation-t5gemma2-1b-enc-gemma3-1b/README.md),
[270m](../2026-09-27-e5pp-adaptation-t5gemma2-270m-enc-gemma3-1b/README.md) and
[4b (aborted here)](../2026-09-28-e5pp-adaptation-t5gemma2-4b-enc-gemma3-1b/README.md) arms into the 1B decoder.

> **Gold and cell (note added 2026-09-29, scoped 2026-10-05).** This note applies to the comparator numbers quoted in the Hypothesis and Prediction as written on 2026-09-28; the thresholds at the top of the Setup were restated under gold = teacher on 2026-09-29 and every number in Result and Verdict is under the two current golds with `cell=unseen_q_no_twin` as the zero-shot slice. The pre-registration's comparator numbers were scored against the pre-D61 mixed gold (the training labeller's answer where a second local labeller agreed, Opus's where they disagreed or where Sonnet said `fail`, with union evidence keys) and their unseen-question figures include the twenty held-out questions that have twins in training. From 2026-09-29 every result is reported under two golds, gold = teacher (the training labeller's own label, the headline) and gold = opus (Opus's label on the 4,012 pairs it labelled), with the no-twin held-out slice as the zero-shot number; the re-run record (`experiments/2026-09-28-rerun-under-the-fixed-loop/`) and the README carry those. The predictions and readings below are kept as written.

## Hypothesis

> **Comparator numbers in this record come from runs with a bug.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (half the data at batch 16), so every trained-arm number quoted here as a bar or reference (the native pairs, the mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) is from a buggy run being replaced by a re-run ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); the predictions are read against the re-run numbers in the re-run record (queue 30, complete 2026-10-05). *The DDP arms among those (the native pairs, the 1B mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) have since been re-run under the fixed loop (queue 30, complete 2026-10-05); the FSDP arms quoted (the 4B mix-and-match arms, Gemma 3 4B, SmolLM3) were sound as run. The sound numbers are in the re-run record.*

The question is practical: how does our recipe for putting an encoder in front of a decoder, soft tokens through a
regression-fitted stitch plus 594M tokens of continuation adaptation, compare with a native encoder-decoder, whose
decoder was adapted by Google to read the encoder at every layer at far larger cost. The 270M-decoder column is where
every cell has a native counterpart with the same encoder: the native 270m-270m pair (0.731 / 0.540 clean macro-F1 / no-twin unseen-question macro-F1 under gold = teacher, the re-run under the fixed loop; thresholds restated from the pre-D61 gold on 2026-09-29), and the mix-and-match 1b encoder + 270m native decoder (0.781 / 0.641, re-run) and 4b encoder + 270m
native decoder (0.803 / 0.683). Three E5++ arms with those encoders into the raw Gemma 3 270M, plus E1++ on that decoder
as the matched control, give two readings per cell: does the encoder help this decoder under our recipe, and how much of
the native advantage does the recipe recover. The gap between the native and the E5++ numbers, as a function of encoder
size, is the answer; it is measured in the fast configuration, a large encoder run once per prompt and a small decoder
run per output token, which is the one that would be deployed.

This is recipe against recipe, not interface against interface: the native decoders also had their own adaptation. The
E1++ control accounts for what our adaptation data does for the decoder alone; nothing here accounts for Google's.

## Prediction

- E1++ 270M within a seed's worth of E1 270M (0.530 / 0.372 under gold = teacher, the re-run under the fixed loop; evidence F1 0.401, format valid 0.790): the continuation
  stage did nothing for the 1B decoder on the task.
- Each E5++ arm above E1++ 270M on evidence F1 and format validity (the recipe's signature at the 1B decoder), and on
  answers by an amount that grows with the encoder: at 270m level or slightly up, at 1b and 4b clearly up, since the
  270M decoder alone leaves the most room.
- Against the native counterparts: below in every cell, and the gap the measure. If the 4b arm gets near 0.803 / 0.683
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
- wandb: E1++ 270M tt-decoder/spn92ttv (stage tt-pretrain/l7zh5f41); E5++ 270m tt-encdec/829j0495 (stage tt-pretrain/b7jyf282); E5++ 1b tt-encdec/17a0vlrr (stage tt-pretrain/4jxxy9xt); E5++ 4b tt-encdec/vhssgknq (stage tt-pretrain/xdgf1kek).

## Result

**E1++ 270M, the control (2026-10-02, queue 30 run 14; tt-decoder `spn92ttv`, stage tt-pretrain `l7zh5f41`).** Gold = teacher clean macro-F1 0.528, no-twin 0.359, evidence F1 0.400, format validity 0.790; messy 0.509 / 0.353 / 0.370; Opus 0.349 / 0.272. Against E1 270M (0.530 / 0.372 / 0.401): −0.002 macro-F1, −0.013 no-twin, −0.002 evidence, −0.001 format validity; validation loss 0.828 against 0.826. The control prediction held: the stage is worth nothing to the 270M decoder, within a seed's worth on every headline number, and both sit below the majority floor (0.569). The three E5++ arms are read against 0.528 / 0.359 / 0.400.

**E5++ with the 270m encoder (2026-10-03, queue 30 run 15; tt-encdec `829j0495`, stage tt-pretrain `b7jyf282`).** Gold = teacher clean macro-F1 0.687, no-twin 0.486, evidence F1 0.650, format validity 0.985; messy 0.670 / 0.504 / 0.607; Opus 0.441 / 0.293. Against E1++ 270M: +0.159 macro-F1, +0.128 no-twin, +0.250 evidence F1, +0.195 format validity, ahead on every clean slice (Opus +0.092 / +0.021 / +0.171). Against the native 270m-270m (0.731 / 0.540 / 0.670): −0.043 / −0.054 / −0.020 (Opus −0.006 / −0.020 / −0.008). Validation loss 0.740 against the control's 0.828. The 270m cell's predictions: the arm beats its control on every headline number and lifts a decoder that sits below the majority floor above it; the gap to the native counterpart is 0.02 to 0.05 under the teacher gold and within 0.01 on answers under Opus's.

**E5++ with the 1b encoder (2026-10-03, queue 30 run 16; tt-encdec `17a0vlrr`, stage tt-pretrain `4jxxy9xt`).** Gold = teacher clean macro-F1 0.747, no-twin 0.530, evidence F1 0.687, format validity 0.985; messy 0.727 / 0.528 / 0.642; Opus 0.466 / 0.276. Against E1++ 270M: +0.219 / +0.171 / +0.288 (Opus +0.118 / +0.003 / +0.192). Against the 270m cell: +0.060 / +0.043 / +0.038. Against the native 1B encoder + 270M decoder (0.781 / 0.641 / 0.713): −0.034 / −0.112 / −0.025 (Opus −0.021 / −0.033 / −0.019). Against the same encoder into the 1B decoder (0.742 / 0.505 / 0.695): +0.005 / +0.024 / −0.008. Validation loss 0.713. The encoder axis at the 270M decoder reads 270m 0.687 / 0.486 / 0.650, 1b 0.747 / 0.530 / 0.687: the larger encoder pays on every number here, where at the 1B decoder it paid mostly on evidence.

**E5++ with the 4b encoder (2026-10-05, queue 30 run 17; tt-encdec `vhssgknq`, stage tt-pretrain `xdgf1kek`; both under FSDP).** Gold = teacher clean macro-F1 0.790, no-twin 0.657, evidence F1 0.710, format validity 0.987; messy 0.752 / 0.651 / 0.665; Opus 0.508 / 0.350. Against E1++ 270M: +0.262 / +0.299 / +0.310 (Opus +0.159 / +0.078 / +0.200). Against the 1b cell: +0.043 / +0.128 / +0.023. Against the native 4B encoder + 270M decoder (0.803 / 0.683 / 0.725): −0.012 / −0.026 / −0.015 (Opus −0.008 / −0.020 / −0.010). Against the Gemma 3 4B decoder (0.804 / 0.690 / 0.729): −0.014 / −0.033 / −0.019. Validation loss 0.690. Stage 24 h 30, fine-tune 15 h 32, generation 2 h 54 at the 40k budget.

**The encoder axis at the 270M decoder, complete** (macro-F1 / no-twin / evidence F1, teacher, clean): control 0.528 / 0.359 / 0.400; 270m 0.687 / 0.486 / 0.650; 1b 0.747 / 0.530 / 0.687; 4b 0.790 / 0.657 / 0.710. Native counterparts: 270m-270m 0.731 / 0.540 / 0.670; 1B encoder + 270M decoder 0.781 / 0.641 / 0.713; 4B encoder + 270M decoder 0.803 / 0.683 / 0.725.

## Verdict

The column answers its question. Encoder capacity pays at the 270M decoder on every number and at every step: 270m to 1b buys +0.060 on answers and +0.043 on zero-shot, 1b to 4b buys +0.043 and +0.128. The gap to the native counterpart narrows with the encoder on answers (0.043, 0.034, 0.012) and not monotonically on zero-shot (0.054, 0.112, 0.026), so the soft-token interface into a raw decoder recovers almost all of a native stitch once the encoder is large. The decoder's own capacity is nearly irrelevant once an encoder is in front: the 1b encoder scores the same into the 270M decoder as into the 1B one, and the 4b encoder into the 270M decoder is level with the native 1b-1b and 0.014 below the Gemma 3 4B decoder with a fifteenth of its trainable decoder. The cost is training time: the 4b-encoder stage and fine-tune ran 40 hours under FSDP on this machine, against about thirteen for the 1b cell (7 h 37 stage, 5 h 02 fine-tune). One seed per cell; the pre-registered thresholds (restated under the teacher gold on 2026-09-29) are met at every encoder size against the control and missed against the native counterparts by the margins above.

## Follow-ups

- A second seed on the column: every cell is one seed, and the 1b to 4b step on zero-shot (+0.128) is the headline.
- The 4b encoder into the 1B decoder (the original third point of the 1B-decoder axis) still waits on rented hardware.
- The column's FSDP cost: 40 hours for the 4b cell against about thirteen for the 1b cell; a bf16 stochastic-rounding recipe is back on the table after the re-read of the 270m-270m check (the native-pair record).
