# Re-running the DDP-trained arms under the fixed loop

Written at the launch, 2026-09-28 18:06 (revised 19:15 after an audit of the cleanup). Queue 30.

## Hypothesis

On 2026-09-28 two independent read-only audits of the training code found
that every DDP-trained run since 2026-09-10 never synchronised gradients: the loss was computed by a forward on the bare
model rather than through `DistributedDataParallel.forward`, so the reducer was never armed and each rank trained its own
replica on its half of every step; rank 0's replica was saved and scored. I reproduced it on CPU with two gloo ranks
(gradients 1.0 and 2.0 on the two ranks where the synchronised value is 1.5). Every DDP arm is therefore a model trained
on half the data at an effective batch of 16 rather than 32. The FSDP runs (Gemma 3 4B, SmolLM3 3B, the two 4B
stitched arms, the E3 stage and fine-tune) are unaffected. Comparisons between DDP arms of one family stayed matched (the same defect on both
sides); absolute numbers, and every comparison of a DDP arm against an FSDP arm, were not.

The re-run answers two questions. First, the ones the affected runs were meant to answer, now on all the data at the
intended batch: the Gemma 3 pairs (native 1b-1b and 270m-270m against their decoders), the 1B mix-and-match arms, the
adaptation pairs at the 1B decoder (E1++ against E5++ with the 1b and 270m encoders), the E5 arms without adaptation, the
cross-family parameter-matched controls, and the 270M-decoder column that queue 29 was about to start. Second, whether the
readings taken from the buggy runs survive: which orderings hold, which gaps change, on twice the data.

Not re-run, and marked in their records as "run had bugs, so don't trust these results fully": the Qwen3 E1, E2, E1+,
E2+ and E2++ arms and their adaptation stages, the two E6 hybrids that ran, and the small (E5+) adaptation pair. The E3 arm and its
stage trained under FSDP and stand (the audit of the cleanup caught them being swept up with the DDP runs; their weights were lost,
their wandb pages were re-synced from local files as far as those allowed, and their results were not lost).
Their within-record comparisons stay matched; their absolute numbers are for half-data models.

## Prediction

- Every re-run lands at or above its buggy predecessor on clean macro-F1: twice the data at the intended batch cannot
  reasonably hurt a one-epoch fine-tune at this learning rate. A re-run more than 0.02 *below* its predecessor would
  mean the fix changed something other than synchronisation, and stops the queue for investigation.
- The orderings that were read as results hold: the native pairs above their decoders; the 1B encoder + 270M decoder
  above the 270M encoder + 1B decoder; E5++ (1b) above E1++ on evidence F1. I do not predict that every small gap
  (under 0.02) keeps its sign.
- The DDP-against-FSDP comparisons tighten: Gemma 3 1B and the 1b-1b move up relative to the 4B arms and SmolLM3.
- **Disconfirmed if** the re-runs land level with the buggy runs across the board: then half the data at batch 16 was
  already enough for this fine-tune, and the bug's cost was only in the absolute numbers' provenance.

## Setup

The fixes, all reviewed by both models before the launch (the `pre-run-review` gate, whose approval hash is logged in
the notes and in `checkpoints/.reviews/approved.jsonl`):

- **Loss routing.** A `LossWrapper` module whose forward computes the task loss; DDP wraps it, `target_loss` calls the
  DDP object so `DDP.forward` runs and the reducer is armed; FSDP roots keep their own loss-forward. A two-process gloo
  test (`tests/test_train.py`) checks that rank 0's gradient is the average of the ranks' own gradients, on the Gemma
  decoder and on the soft model. A GPU equivalence smoke (3 steps, 2 ranks against 1 rank on the same steps) gave loss
  and grad norm agreeing to bf16 precision: E1 fine-tune 2.227 / 24.0 against 2.229 / 24.08; decoder continuation stage
  1.807 / 13.94 against 1.807 / 13.95; soft continuation stage 1.86 / 21.65 against 1.86 / 22.17; stitched 2.494 / 115.2
  against 2.491 / 119.0; native 270m-270m 2.001 / 54.88 against 1.999 / 55.18 (the arms with an SDPA encoder differ a few
  per cent in grad norm between batch layouts; the decoder-only arms under FlashAttention by 0.3%).
- **Continuation cuts** are seeded per document (seed, epoch, document id and variant), not per micro-batch, so the
  E1++ control at 16k micro-batches and the E5++ arms at 8k or 16k see identical cuts; the denominator, the collates and
  the throughput counters draw the same cut.
- **Denominator** of the causal adaptation objective no longer counts the first token of a prompt-less document.
- **Adaptation documents open with the tokenizer's start token** (`<bos>` for the Gemma family, nothing for Qwen and
  Hunyuan), as every fine-tune prompt does. I measured Gemma 3 270M at loss 2.468 on corpus documents without
  it against 1.865 with it, so the continuation stages were adapting to an input format never seen again. Both arms of
  every pair get it. The E1++ and E5++ re-runs therefore differ from their predecessors in two ways, the synchronised
  gradient and the start token; the two are not separated here.
- **Generation keeps `--max-tokens 512`**, as every kept arm (the FSDP arms; the API baseline has since been withdrawn) was generated: 116 of the
  20,700 benchmark gold targets under the teacher gold exceed 512 Gemma tokens (186 under the pre-D61 mixed gold, whose
  union evidence keys were longer; line numbers tokenise digit by digit and the share differs by tokenizer), so a
  perfect model is scored invalid on about 0.6% of pairs. A ceiling on format validity, accuracy and
  evidence F1 that every arm shares, recorded here rather than changed mid-comparison.
- **Throughput** under the continuation objective counts the positions consumed (prefix plus capped continuation), not
  whole documents; both cards' peak memory are logged.
- **Evaluator**: the parser is strict (the whole output is one JSON object); a recovered answer (a case or whitespace
  variant of an option) keeps its 0.3 credit but is no longer a correct prediction for macro-F1 and the confusion table;
  macro-F1 runs over each question's option set; duplicate outputs count once and a missing pair scores as empty;
  format validity requires a summary string; out-of-vocabulary tags count against the tag Jaccard. **Every existing
  run is re-scored with this evaluator** so the numbers that stay (the FSDP arms, the not-re-run arms) are on the same
  scale as the re-runs. Done before the launch: every existing run's clean-variant benchmark macro-F1 is identical to
  three decimals under both evaluators, and format validity moves by 0.001 on one run, so at that point the kept numbers
  stood as published. That comparison was under the pre-D61 mixed gold; on 2026-09-29 every number was re-scored under
  the two golds of D61 (the Result section), which replaced every published number.
- **The queue** (`queue30_rerun.sh`, in the session scratchpad; its text is what the gate hashes) moves each buggy
  predecessor to `checkpoints/_archive/<name>-ddp-bug` before its replacement starts, runs the gate before every
  training stage and every generation, checks every generation's exit status and that every (id, variant) pair is
  present exactly once before scoring, and aborts on any failure. Same configs as the original runs; no recipe change.

The queue, in order (durations are the buggy predecessors' measured wall-clock, for planning only):

| # | Run | Stages | Predecessor's time | Status |
|---|---|---|---|---|
| 1 | `e1-gemma3-1b-pt` | train, vLLM generation, score | 2 h 50 + 10 min | **scored 21:10** (2.94 h + 7 min) |
| 2 | `e1-gemma3-270m` | train, vLLM generation, score | 52 + 6 min | **scored 23:17** (53 + 5 min) |
| 3 | `e4-t5gemma2-1b-1b` | train, `hf_encdec` generation, score | about 3 h + 2.6 h | **scored 06:39** (4.85 h + 2.5 h) |
| 4 | `e4-t5gemma2-270m-270m` | train, `hf_encdec` generation, score | about 1.5 h + 2 h | **scored 10:40** (2 h 23 + 1 h 38) |
| 5 | `e4mm-1b-enc-270m-dec` | train, `stitched` generation, score | about 3 h + 2 h | **scored 17:13** (4 h 39 + 1 h 53) |
| 6 | `e4mm-270m-enc-1b-dec` | train, `stitched` generation, score | about 3 h + 2 h | **scored 22:07** (2 h 38 + 2 h 16), the first run scored under the two golds by the queue |
| 7 | `adapt-continuation-corpora-gemma3-1b-pt` then `e1pp-gemma3-1b-pt` | stage; train, vLLM generation, score | 4 h 22 + 2 h 50 + 10 min | **scored 05:38** (stage 4 h 28 after a relaunch at 22:10, its first launch at 22:07 having segfaulted in NCCL's heartbeat thread 3 s in; fine-tune 2 h 54 + 7 min) |
| 8 | `adapt-continuation-corpora-e5-t5gemma2-1b-enc-gemma3-1b` then `e5pp-t5gemma2-1b-enc-gemma3-1b` | stage; train, `soft` generation, score | 9 h 56 + 6 h 16 + 2 h 30 | **scored 00:26** (stage 10 h 07; fine-tune 6 h 23 + 2 h 17) |
| 9 | `adapt-continuation-corpora-e5-t5gemma2-270m-enc-gemma3-1b` then `e5pp-t5gemma2-270m-enc-gemma3-1b` | stage; train, `soft` generation, score | 7 h 14 + 4 h 22 + 2 h 20 | **scored 14:14** (stage 7 h 15; fine-tune 4 h 26 + 2 h 07) |
| 10 | `e5-t5gemma2-1b-enc-gemma3-1b` | train, `soft` generation, score | 6 h 27 + 2 h 40 | **scored 23:09** (6 h 23 + 2 h 32) |
| 11 | `e5-t5gemma2-270m-enc-gemma3-1b` | train, `soft` generation, score | 4 h 21 + 2 h 30 | **scored 05:51** (4 h 25 + 2 h 17) |
| 12 | `e1-qwen2.5-1.5b` | train, vLLM generation, score | 3 h 38 + 17 min | **scored 09:47** (3 h 40 + 17 min) |
| 13 | `e1-hunyuan-1.8b` | train, vLLM generation (eager), score | 3 h 50 + 30 min | **scored 15:03** (4 h 55 + 20 min) |
| 14 | `adapt-continuation-corpora-gemma3-270m` then `e1pp-gemma3-270m` | stage; train, vLLM generation, score | new (stage about 2.5 h; fine-tune 52 min) | **scored 17:37** (stage 1 h 35; fine-tune 54 + 6 min) |
| 15 | `adapt-continuation-corpora-e5-t5gemma2-270m-enc-gemma3-270m` then `e5pp-t5gemma2-270m-enc-gemma3-270m` | stage; train, `soft` generation, score | new | **scored 02:29** (stage 4 h 39; fine-tune 2 h 46 + 1 h 26) |
| 16 | `adapt-continuation-corpora-e5-t5gemma2-1b-enc-gemma3-270m` then `e5pp-t5gemma2-1b-enc-gemma3-270m` | stage; train, `soft` generation, score | new | **scored 16:50** (stage 7 h 37; fine-tune 5 h 02 + 1 h 42) |
| 17 | `adapt-continuation-corpora-e5-t5gemma2-4b-enc-gemma3-270m` then `e5pp-t5gemma2-4b-enc-gemma3-270m` | stage (FSDP, 8k); train, `soft` generation (40k budget), score | new | **scored 11:46 on 5 Oct** (stage 24 h 30 under FSDP, 27.2 GiB peak per card; fine-tune 15 h 32, 28.3 GiB peak; generation at the 40k budget 2 h 54) |

Every buggy run's weights were deleted on 2026-09-28 at 18:11, before any replacement finished: the 28 DDP run
directories (the ones to be re-run and the ones not; 27 finished runs and one stopped stage) were stripped of their `final/` weight files and step checkpoints
(222.8 GB freed) and moved to `checkpoints/_archive/<name>-ddp-bug`, keeping their training logs, generation outputs and
scoring (lenient and strict) and their wandb ids; their wandb runs were exported (config, summary, step history, as
`wandb_export.json` in each archive folder) and deleted, so a wandb query never returns an abandoned run. An audit of the
cleanup found two archives still holding their weights (deleted) and three FSDP runs swept up by mistake (the E3 stage and
fine-tune, the stopped bf16 stochastic-rounding check): their weights are gone, their wandb exports are kept, their wandb
pages were re-synced from local files where those allowed (the stage in full, the fine-tune to step 590), and their result
folders are back under `checkpoints/`. When a re-run is scored, its predecessor's numbers are replaced in the README and the notes;
the record that reported the predecessor keeps its original tables as history under a banner naming them as the
abandoned run's, with a dated re-read note in its Verdict giving the sound numbers; both sets sit side by side in this
record's result table.

wandb: the run ids are listed at the top of the Result section, one per fine-tune and one per adaptation stage.

## Result

Filled as each run scores. Every number here is under the two golds of 2026-09-29 (D61): the same predictions scored by the
patched evaluator against `benchmark.jsonl` as rebuilt with `label` = the teacher's label and `opus_label` = Opus's. The
predecessors were re-scored the same way, so each pair of rows differs only in the training loop (and, for the two
native pairs, in the generation backend: the predecessors were generated under BUG-2). Clean and messy variants;
macro-F1 over the benchmark, macro-F1 on the held-out questions without a twin in training (`cell=unseen_q_no_twin`, the
zero-shot number), evidence F1, format validity. wandb: `e1-gemma3-1b-pt` tt-decoder `satuk1fl`, `e1-gemma3-270m` tt-decoder
`ictv4tcb`, `e4-t5gemma2-1b-1b` tt-encdec `18lyydir`, `e4-t5gemma2-270m-270m` tt-encdec `eomeqls2`, `e4mm-1b-enc-270m-dec`
tt-encdec `icbx4lyk`, `e4mm-270m-enc-1b-dec` tt-encdec `z6qiy3cg`, `e1pp-gemma3-1b-pt` tt-decoder `kfh8r1g2` (its adaptation stage tt-pretrain `h8efzdam`), `e5pp-t5gemma2-1b-enc-gemma3-1b` tt-encdec `xlbxgsev` (its stage tt-pretrain `us8srg2z`), `e5pp-t5gemma2-270m-enc-gemma3-1b` tt-encdec `4rsknusl` (its stage tt-pretrain `wizcg0a5`), `e5-t5gemma2-1b-enc-gemma3-1b` tt-encdec `p0771u8f`, `e5-t5gemma2-270m-enc-gemma3-1b` tt-encdec `b4qqhyyo`, `e1-qwen2.5-1.5b` tt-decoder `trye2moq`, `e1-hunyuan-1.8b` tt-decoder `k9rknb22`, `e1pp-gemma3-270m` tt-decoder `spn92ttv` (its stage tt-pretrain `l7zh5f41`, a new arm with no predecessor), `e5pp-t5gemma2-270m-enc-gemma3-270m` tt-encdec `829j0495` (its stage tt-pretrain `b7jyf282`, new), `e5pp-t5gemma2-1b-enc-gemma3-270m` tt-encdec `17a0vlrr` (its stage tt-pretrain `4jxxy9xt`, new), `e5pp-t5gemma2-4b-enc-gemma3-270m` tt-encdec `vhssgknq` (its stage tt-pretrain `xdgf1kek`, new); the predecessors' wandb runs are deleted.

**Gold = teacher** (the training labeller's own label: qwen3.8 for general_qa, Sonnet for the rare-event families; 17,520 clean and 20,700 messy pairs; majority-answer floor 0.569 clean, 0.563 messy; no-twin unseen-question floor 0.582).

| Run | Clean macro-F1 | Clean no-twin unseen-q | Clean evidence F1 | Clean format valid | Messy macro-F1 | Messy no-twin unseen-q | Messy evidence F1 | Messy format valid |
|---|---|---|---|---|---|---|---|---|
| e1-gemma3-1b-pt, abandoned predecessor (BUG-1) | 0.669 | 0.482 | 0.578 | 0.955 | 0.654 | 0.486 | 0.539 | 0.946 |
| **e1-gemma3-1b-pt, fixed loop, 2026-09-28** | 0.698 | 0.488 | 0.623 | 0.970 | 0.676 | 0.501 | 0.579 | 0.959 |
| e1-gemma3-270m, abandoned predecessor (BUG-1) | 0.449 | 0.322 | 0.332 | 0.651 | 0.433 | 0.301 | 0.308 | 0.632 |
| **e1-gemma3-270m, fixed loop, 2026-09-28** | 0.530 | 0.372 | 0.401 | 0.790 | 0.513 | 0.346 | 0.372 | 0.766 |
| e4-t5gemma2-1b-1b, abandoned predecessor (BUG-1; generation under BUG-2) | 0.767 | 0.665 | 0.703 | 0.990 | 0.740 | 0.630 | 0.659 | 0.985 |
| **e4-t5gemma2-1b-1b, fixed loop, 2026-09-29** | 0.795 | 0.686 | 0.724 | 0.994 | 0.761 | 0.666 | 0.679 | 0.988 |
| e4-t5gemma2-270m-270m, abandoned predecessor (BUG-1; generation under BUG-2) | 0.683 | 0.482 | 0.646 | 0.982 | 0.660 | 0.490 | 0.603 | 0.973 |
| **e4-t5gemma2-270m-270m, fixed loop, 2026-09-29** | 0.731 | 0.540 | 0.670 | 0.991 | 0.701 | 0.547 | 0.628 | 0.985 |
| e4mm-1b-enc-270m-dec, abandoned predecessor (BUG-1) | 0.750 | 0.622 | 0.692 | 0.988 | 0.725 | 0.610 | 0.648 | 0.981 |
| **e4mm-1b-enc-270m-dec, fixed loop, 2026-09-29** | 0.781 | 0.641 | 0.713 | 0.992 | 0.750 | 0.634 | 0.668 | 0.985 |
| e4mm-270m-enc-1b-dec, abandoned predecessor (BUG-1) | 0.694 | 0.490 | 0.646 | 0.988 | 0.678 | 0.499 | 0.606 | 0.983 |
| **e4mm-270m-enc-1b-dec, fixed loop, 2026-09-29** | 0.742 | 0.565 | 0.677 | 0.986 | 0.713 | 0.565 | 0.637 | 0.985 |
| e1pp-gemma3-1b-pt, abandoned predecessor (BUG-1) | 0.668 | 0.463 | 0.581 | 0.958 | 0.652 | 0.472 | 0.541 | 0.943 |
| **e1pp-gemma3-1b-pt, fixed loop, 2026-09-30** | 0.707 | 0.507 | 0.628 | 0.974 | 0.684 | 0.508 | 0.584 | 0.961 |
| e5pp-t5gemma2-1b-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.694 | 0.478 | 0.665 | 0.984 | 0.677 | 0.493 | 0.623 | 0.979 |
| **e5pp-t5gemma2-1b-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.742 | 0.505 | 0.695 | 0.989 | 0.718 | 0.511 | 0.655 | 0.983 |
| e5pp-t5gemma2-270m-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.666 | 0.448 | 0.629 | 0.979 | 0.648 | 0.462 | 0.590 | 0.970 |
| **e5pp-t5gemma2-270m-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.721 | 0.500 | 0.668 | 0.987 | 0.701 | 0.506 | 0.625 | 0.981 |
| e5-t5gemma2-1b-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.668 | 0.465 | 0.599 | 0.968 | 0.652 | 0.477 | 0.555 | 0.956 |
| **e5-t5gemma2-1b-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.708 | 0.494 | 0.651 | 0.980 | 0.687 | 0.488 | 0.608 | 0.973 |
| e5-t5gemma2-270m-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.653 | 0.452 | 0.580 | 0.959 | 0.635 | 0.455 | 0.533 | 0.944 |
| **e5-t5gemma2-270m-enc-gemma3-1b, fixed loop, 2026-10-02** | 0.694 | 0.452 | 0.646 | 0.982 | 0.675 | 0.464 | 0.604 | 0.974 |
| e1-qwen2.5-1.5b, abandoned predecessor (BUG-1) | 0.725 | 0.558 | 0.627 | 0.971 | 0.702 | 0.570 | 0.588 | 0.960 |
| **e1-qwen2.5-1.5b, fixed loop, 2026-10-02** | 0.751 | 0.613 | 0.650 | 0.971 | 0.723 | 0.627 | 0.604 | 0.961 |
| e1-hunyuan-1.8b, abandoned predecessor (BUG-1) | 0.754 | 0.595 | 0.676 | 0.994 | 0.727 | 0.588 | 0.633 | 0.991 |
| **e1-hunyuan-1.8b, fixed loop, 2026-10-02** | 0.774 | 0.619 | 0.692 | 0.995 | 0.741 | 0.610 | 0.644 | 0.993 |
| **e1pp-gemma3-270m, new arm under the fixed loop, 2026-10-02** | 0.528 | 0.359 | 0.400 | 0.790 | 0.509 | 0.353 | 0.370 | 0.756 |
| **e5pp-t5gemma2-270m-enc-gemma3-270m, new arm under the fixed loop, 2026-10-03** | 0.687 | 0.486 | 0.650 | 0.985 | 0.670 | 0.504 | 0.607 | 0.977 |
| **e5pp-t5gemma2-1b-enc-gemma3-270m, new arm under the fixed loop, 2026-10-03** | 0.747 | 0.530 | 0.687 | 0.985 | 0.727 | 0.528 | 0.642 | 0.977 |
| **e5pp-t5gemma2-4b-enc-gemma3-270m, new arm under the fixed loop, 2026-10-05** | 0.790 | 0.657 | 0.710 | 0.987 | 0.752 | 0.651 | 0.665 | 0.979 |

**Gold = opus** (Opus's label on the 4,012 pairs it labelled, 3,410 of them clean; a selected subset, not comparable with the teacher numbers; Opus-majority floor 0.709 clean, 0.712 messy; teacher-majority floor 0.354 clean, 0.352 messy; no-twin unseen-question floor 0.759).

| Run | Clean macro-F1 | Clean no-twin unseen-q | Clean evidence F1 | Clean format valid | Messy macro-F1 | Messy no-twin unseen-q | Messy evidence F1 | Messy format valid |
|---|---|---|---|---|---|---|---|---|
| e1-gemma3-1b-pt, abandoned predecessor (BUG-1) | 0.406 | 0.287 | 0.389 | 0.945 | 0.421 | 0.295 | 0.376 | 0.936 |
| **e1-gemma3-1b-pt, fixed loop, 2026-09-28** | 0.425 | 0.284 | 0.410 | 0.972 | 0.417 | 0.298 | 0.391 | 0.953 |
| e1-gemma3-270m, abandoned predecessor (BUG-1) | 0.325 | 0.224 | 0.205 | 0.635 | 0.323 | 0.225 | 0.189 | 0.614 |
| **e1-gemma3-270m, fixed loop, 2026-09-28** | 0.345 | 0.255 | 0.260 | 0.782 | 0.350 | 0.265 | 0.238 | 0.751 |
| e4-t5gemma2-1b-1b, abandoned predecessor (BUG-1; generation under BUG-2) | 0.489 | 0.347 | 0.468 | 0.989 | 0.479 | 0.341 | 0.462 | 0.983 |
| **e4-t5gemma2-1b-1b, fixed loop, 2026-09-29** | 0.507 | 0.356 | 0.478 | 0.994 | 0.496 | 0.349 | 0.469 | 0.989 |
| e4-t5gemma2-270m-270m, abandoned predecessor (BUG-1; generation under BUG-2) | 0.419 | 0.270 | 0.431 | 0.981 | 0.409 | 0.285 | 0.411 | 0.968 |
| **e4-t5gemma2-270m-270m, fixed loop, 2026-09-29** | 0.447 | 0.314 | 0.443 | 0.988 | 0.438 | 0.324 | 0.430 | 0.983 |
| e4mm-1b-enc-270m-dec, abandoned predecessor (BUG-1) | 0.464 | 0.310 | 0.468 | 0.986 | 0.462 | 0.330 | 0.456 | 0.975 |
| **e4mm-1b-enc-270m-dec, fixed loop, 2026-09-29** | 0.488 | 0.309 | 0.474 | 0.993 | 0.464 | 0.308 | 0.462 | 0.983 |
| e4mm-270m-enc-1b-dec, abandoned predecessor (BUG-1) | 0.434 | 0.320 | 0.452 | 0.987 | 0.439 | 0.324 | 0.438 | 0.978 |
| **e4mm-270m-enc-1b-dec, fixed loop, 2026-09-29** | 0.471 | 0.317 | 0.458 | 0.989 | 0.459 | 0.313 | 0.438 | 0.980 |
| e1pp-gemma3-1b-pt, abandoned predecessor (BUG-1) | 0.409 | 0.249 | 0.388 | 0.954 | 0.410 | 0.258 | 0.371 | 0.934 |
| **e1pp-gemma3-1b-pt, fixed loop, 2026-09-30** | 0.423 | 0.282 | 0.413 | 0.973 | 0.417 | 0.293 | 0.394 | 0.955 |
| e5pp-t5gemma2-1b-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.435 | 0.280 | 0.445 | 0.984 | 0.434 | 0.277 | 0.430 | 0.977 |
| **e5pp-t5gemma2-1b-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.466 | 0.296 | 0.451 | 0.986 | 0.454 | 0.310 | 0.445 | 0.978 |
| e5pp-t5gemma2-270m-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.412 | 0.283 | 0.422 | 0.978 | 0.414 | 0.319 | 0.409 | 0.963 |
| **e5pp-t5gemma2-270m-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.440 | 0.283 | 0.439 | 0.987 | 0.440 | 0.282 | 0.422 | 0.977 |
| e5-t5gemma2-1b-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.410 | 0.253 | 0.405 | 0.965 | 0.407 | 0.291 | 0.387 | 0.952 |
| **e5-t5gemma2-1b-enc-gemma3-1b, fixed loop, 2026-10-01** | 0.428 | 0.275 | 0.425 | 0.978 | 0.423 | 0.291 | 0.413 | 0.970 |
| e5-t5gemma2-270m-enc-gemma3-1b, abandoned predecessor (BUG-1) | 0.405 | 0.261 | 0.392 | 0.959 | 0.405 | 0.270 | 0.374 | 0.935 |
| **e5-t5gemma2-270m-enc-gemma3-1b, fixed loop, 2026-10-02** | 0.429 | 0.268 | 0.428 | 0.980 | 0.419 | 0.284 | 0.416 | 0.971 |
| e1-qwen2.5-1.5b, abandoned predecessor (BUG-1) | 0.438 | 0.241 | 0.417 | 0.965 | 0.456 | 0.267 | 0.397 | 0.950 |
| **e1-qwen2.5-1.5b, fixed loop, 2026-10-02** | 0.467 | 0.304 | 0.432 | 0.966 | 0.467 | 0.339 | 0.416 | 0.952 |
| e1-hunyuan-1.8b, abandoned predecessor (BUG-1) | 0.470 | 0.295 | 0.455 | 0.992 | 0.483 | 0.291 | 0.437 | 0.987 |
| **e1-hunyuan-1.8b, fixed loop, 2026-10-02** | 0.498 | 0.323 | 0.462 | 0.995 | 0.505 | 0.327 | 0.448 | 0.989 |
| **e1pp-gemma3-270m, new arm under the fixed loop, 2026-10-02** | 0.349 | 0.272 | 0.263 | 0.779 | 0.343 | 0.290 | 0.237 | 0.738 |
| **e5pp-t5gemma2-270m-enc-gemma3-270m, new arm under the fixed loop, 2026-10-03** | 0.441 | 0.293 | 0.434 | 0.983 | 0.432 | 0.318 | 0.415 | 0.971 |
| **e5pp-t5gemma2-1b-enc-gemma3-270m, new arm under the fixed loop, 2026-10-03** | 0.466 | 0.276 | 0.456 | 0.982 | 0.474 | 0.280 | 0.439 | 0.975 |
| **e5pp-t5gemma2-4b-enc-gemma3-270m, new arm under the fixed loop, 2026-10-05** | 0.508 | 0.350 | 0.464 | 0.987 | 0.498 | 0.353 | 0.447 | 0.976 |

Over the first five re-runs, under the teacher gold every re-run beat its predecessor on every clean headline metric (the
270M decoder's tag Jaccard fell, 0.664 to 0.650). Across all thirteen re-runs the exceptions on a headline metric are two:
the 270M encoder + 1B decoder stitch's clean format validity fell 0.988 to 0.986, and the 270m encoder into the 1B
decoder's clean no-twin did not move (0.452 both). The gains are largest where the loss of
half the data bit hardest: the 270M decoder (clean macro-F1 0.449 to 0.530, format validity 0.651 to 0.790), then the
270m-270m pair (0.683 to 0.731), then the three 1B-class arms (about 0.03 each). Under the Opus gold the same ordering
holds for the pairs against the decoders, with smaller gains (0.018 to 0.028 on clean macro-F1, the 270m-270m pair
gaining most and the two 1B-class decoders least). On the no-twin zero-shot slice the 1B-class arms move by
0.006 to 0.021 clean and 0.016 to 0.036 messy under the teacher gold, and by less than 0.01 in either direction under
Opus's: for the 1B decoder the fix bought answers far more than transfer (+0.029 against +0.006), for the two encoder arms
it bought both about equally.

The pairs, both sides under the fixed loop (clean macro-F1 teacher / no-twin, then Opus / no-twin): native 1b-1b 0.795 /
0.686 and 0.507 / 0.356 against Gemma 3 1B 0.698 / 0.488 and 0.425 / 0.284, a native lead of 0.097 on the teacher gold and
0.083 on Opus's (deltas here are of unrounded values); native 270m-270m 0.731 / 0.540 and 0.447 / 0.314 against Gemma 3 270M 0.530 / 0.372 and 0.345 / 0.255,
a lead of 0.201 and 0.102. The 1b encoder + 270m decoder stitch, 0.781 / 0.641 and 0.488 / 0.309, sits between the two
native pairs on the teacher gold and between them on Opus's; the grid under the fixed loop reads 270m-270m 0.731, 1b
encoder + 270m decoder 0.781, 1b-1b 0.795 (teacher, clean). Validation losses: 0.591 against 0.632 (1B decoder), 0.826
against 0.870 (270M decoder), 0.482 against 0.528 (1b-1b), 0.617 at the 270m-270m's close, 0.600 against 0.656 (the
stitch). BUG-2 is gone from the native pairs' outputs: unparseable rows of encoder length 1 mod 32 are 0.7% against 0.9%
elsewhere for the 1b-1b (predecessor 10% against 1%) and 1.6% against 1.2% for the 270m-270m.

The sixth, the 270M encoder + 1B decoder stitch (validation loss 0.528 against the predecessor's 0.581): clean macro-F1
0.694 to 0.742, no-twin 0.490 to 0.565, evidence F1 0.646 to 0.677 under the teacher gold; Opus 0.434 to 0.471. The 1B grid
now has all four cells under the fixed loop (teacher, clean macro-F1 / no-twin / evidence F1): 270m-270m 0.731 / 0.540 /
0.670, 270M encoder + 1B decoder 0.742 / 0.565 / 0.677, 1B encoder + 270M decoder 0.781 / 0.641 / 0.713, 1b-1b 0.795 /
0.686 / 0.724. Of the 270m-270m-to-1b-1b margin (0.064 answers, 0.146 no-twin, 0.053 evidence) the encoder recovers 0.79,
0.69 and 0.80, the decoder 0.17, 0.17 and 0.13. The decoder-side pre-registered threshold (> 0.02 on unseen questions over
the 270m-270m) is met on clean no-twin (+0.025), missed on messy no-twin (+0.018) and on the full held-out cell (+0.013 / −0.003). The
1B → 4B step on the 270M-encoder side adds +0.017 / +0.016 on answers and −0.004 / −0.007 on no-twin; on the
270M-decoder side +0.021 / +0.016 and +0.042 / +0.019: the two steps buy the same on answers and only the encoder step buys
zero-shot transfer.

The seventh, the Gemma 3 1B continuation-adaptation stage then the E1++ fine-tune: the stage's validation loss 1.689
against the predecessor's 1.720, the fine-tune's 0.592 against 0.633. E1++ under the teacher gold: clean macro-F1 0.668
to 0.707, no-twin 0.463 to 0.507, evidence F1 0.581 to 0.628, format validity 0.958 to 0.974; Opus 0.409 to 0.423. **E1++
against E1, both under the fixed loop:** +0.009 / +0.008 macro-F1, +0.019 / +0.007 no-twin, +0.005 / +0.005 evidence F1
under the teacher gold; level under Opus's (−0.002 / +0.000). The adaptation stage that was worth nothing to the buggy
decoder (0.668 against 0.669) is worth a little to the fixed one, most of it on zero-shot. The E5++ comparison is the eighth
paragraph below; its abandoned predecessor's 0.694 / 0.478 was against the abandoned E1++, not this one.

The eighth, the same continuation stage with the 1b encoder in front then the E5++ fine-tune: the stage's validation
loss 1.686 against the predecessor's 1.734, the fine-tune's 0.556 against 0.609. E5++ under the teacher gold: clean
macro-F1 0.694 to 0.742, no-twin 0.478 to 0.505, evidence F1 0.665 to 0.695; Opus 0.435 to 0.466. **E5++ against E1++,
both under the fixed loop** (the same adaptation stage, prefix as the encoder's states against prefix as tokens):
+0.036 / +0.035 macro-F1, +0.023 / +0.025 on the full held-out cell, −0.001 / +0.002 on no-twin, +0.068 / +0.071
evidence F1, +0.076 / +0.081 evidence precision, +0.014 / +0.022 format validity under the teacher gold; +0.043 / +0.037
macro-F1 and +0.014 / +0.016 no-twin under Opus's. Ahead on every clean slice but two: the vulnerability family (0.674
against 0.681) and the no-twin zero-shot slice (level). The predecessor pair's reading (the E5++ 1b record) holds on answers and
evidence and is now level rather than ahead on zero-shot: the encoder interface pays on trained questions and on
locating evidence, not on transfer to new questions.

The ninth, the same stage with the 270m encoder then its E5++ fine-tune: stage validation loss 1.690 against the
predecessor's 1.736, fine-tune 0.576 against 0.627. E5++ 270m under the teacher gold: clean macro-F1 0.666 to 0.721,
no-twin 0.448 to 0.500, evidence F1 0.629 to 0.668; Opus 0.412 to 0.440. **Against E1++, both fixed:** +0.015 / +0.018
macro-F1, −0.007 / −0.002 no-twin, +0.040 / +0.041 evidence F1, +0.046 / +0.049 precision, +0.013 / +0.020 format
validity (teacher); +0.017 / +0.023 macro-F1 and +0.001 / −0.012 no-twin (Opus). The only clean teacher slices where it
trails E1++ are the two held-out cells. **The encoder axis at one decoder, both fixed:** 270m 0.721 / 0.500 / 0.668, 1b
0.742 / 0.505 / 0.695 (macro-F1 / no-twin / evidence F1): the larger encoder adds +0.021 on answers and +0.027 on
evidence and +0.006 on zero-shot. The predecessor pair's reading (the E5++ 270m record, "level on answers, ahead on evidence") becomes
slightly ahead on answers, ahead on evidence, level on zero-shot: the same shape as the 1b arm at about half the size of
the effect.

The tenth, the 1b encoder stitched into the 1B decoder without an adaptation stage: validation loss 0.582 against the
predecessor's 0.624. Under the teacher gold clean macro-F1 0.668 to 0.708, no-twin 0.465 to 0.494, evidence F1 0.599 to
0.651; Opus 0.410 to 0.428. **Against E1, both fixed:** +0.010 / +0.011 macro-F1, +0.006 / −0.014 no-twin, +0.028 / +0.030
evidence F1, +0.036 precision, +0.010 / +0.015 format validity (teacher); +0.004 / +0.006 macro-F1 and +0.016 / +0.021
evidence (Opus); below E1 on one clean slice, the vulnerability family. The predecessor pair read level on answers; the fixed
pair reads a small lift on answers and a clear one on evidence. **What the adaptation stage is worth to each side, all
four fixed:** to the stitched arm +0.034 / +0.032 macro-F1, +0.045 / +0.046 evidence F1, +0.011 / +0.023 no-twin (E5++
over E5); to the decoder alone +0.009 / +0.008, +0.005 / +0.005 and +0.019 / +0.007 (E1++ over E1). The same stage is
worth about four times as much to the encoder interface as to the decoder, so of E5++'s +0.036 over E1++ roughly +0.010 is
the interface as stitched and the rest is the stage teaching the decoder to read the encoder's states.

The eleventh, the 270m encoder stitched into the 1B decoder without a stage: validation loss 0.591 against the
predecessor's 0.641. Under the teacher gold clean macro-F1 0.653 to 0.694, no-twin 0.452 to 0.452, evidence F1 0.580 to
0.646; Opus 0.405 to 0.429. **Against E1, both fixed:** −0.004 / −0.000 macro-F1, −0.036 / −0.037 no-twin, +0.024 /
+0.025 evidence F1, +0.028 precision, +0.012 / +0.015 format validity (teacher); +0.005 / +0.002 macro-F1, −0.016 /
−0.014 no-twin (Opus). Level on answers, behind on zero-shot, ahead on evidence; below E1 on nine clean slices including
the two held-out cells. **The two unadapted encoders at one decoder:** 270m 0.694 / 0.452 / 0.646, 1b 0.708 / 0.494 /
0.651 (macro-F1 / no-twin / evidence F1): the larger encoder adds +0.014 on answers and +0.042 on zero-shot, +0.004 on
evidence. **The stage's worth to the 270m arm:** +0.027 / +0.026 macro-F1, +0.048 / +0.042 no-twin, +0.022 / +0.022
evidence F1 (E5++ 270m over E5 270m), against +0.034 / +0.011 / +0.045 for the 1b arm and +0.009 / +0.019 / +0.005 for
the decoder alone. For the small encoder the stage's largest effect is on zero-shot, where it closes the deficit to E1
from −0.036 to −0.007; for the large one it is on evidence. The E5 picture at the 1B decoder is complete under the fixed loop: the
stitch alone is level with the decoder on answers at either encoder size, the stage opens the interface, and at this
decoder the design never beats the decoder on zero-shot (the 270M-decoder column below is different).

The twelfth, the Qwen2.5-1.5B parameter-matched control: validation loss 0.520 against the predecessor's 0.553. Under the
teacher gold clean macro-F1 0.725 to 0.751, no-twin 0.558 to 0.613, evidence F1 0.627 to 0.650; Opus 0.438 to 0.467. A
zero-shot gain of +0.055 no-twin, second among the thirteen re-runs to the 270M encoder + 1B decoder's +0.076. **The 1b-1b against it, both fixed:** +0.043 / +0.038
macro-F1, +0.073 / +0.040 no-twin, +0.074 / +0.076 evidence F1 (teacher); +0.040 / +0.029 macro-F1, +0.052 / +0.010
no-twin, +0.046 / +0.054 evidence (Opus). The native pair's lead over this cross-family control holds on every headline
number under both golds; the Hunyuan control follows in the next paragraph and the Qwen3-1.7B control stays an abandoned run.

The thirteenth, the Hunyuan-1.8B control: validation loss 0.538 against the predecessor's 0.583. Under the teacher gold
clean macro-F1 0.754 to 0.774, no-twin 0.595 to 0.619, evidence F1 0.676 to 0.692; Opus 0.470 to 0.498. **The 1b-1b
against it, both fixed:** +0.020 / +0.020 macro-F1, +0.067 / +0.057 no-twin, +0.032 / +0.035 evidence F1 (teacher);
+0.009 / −0.009 macro-F1, +0.033 / +0.022 no-twin, +0.016 / +0.021 evidence (Opus). Hunyuan is the strongest sound
cross-family control: +0.023 over Qwen2.5 on answers and +0.042 on evidence, level on zero-shot. Against it the native pair's
lead is clear on zero-shot and evidence under both golds, and on answers it is 0.020 under the teacher gold and within
0.01 either way under Opus's. Both cross-family controls are now sound; the Qwen3-1.7B control is the one matched decoder
left as an abandoned run.

The fourteenth, a new arm: the Gemma 3 270M continuation stage (validation loss 1.865; 1 h 35) then the E1++ 270M
fine-tune (0.828 against E1 270M's 0.826; 54 min). Under the teacher gold clean macro-F1 0.528, no-twin 0.359, evidence
F1 0.400, format validity 0.790; Opus 0.349 / 0.272. **Against E1 270M, both fixed:** −0.002 / −0.004 macro-F1, −0.013 /
+0.007 no-twin, −0.002 / −0.003 evidence F1, −0.001 / −0.010 format validity (teacher); +0.004 / −0.007 macro-F1, +0.017 /
+0.025 no-twin (Opus). The stage is worth nothing to the 270M decoder: the pair is within a seed's worth on every headline
number, which is the control the encoder-axis record pre-registered. Both remain below the majority floor (0.569). The
three E5++ 270M-decoder arms that follow are read against 0.528 / 0.359 / 0.400.

The fifteenth, a new arm: the 270m encoder stitched into the Gemma 3 270M decoder, the continuation stage (validation
loss 1.860; 4 h 39) then the E5++ fine-tune (0.740 against E1++ 270M's 0.828; 2 h 46). Under the teacher gold clean
macro-F1 0.687, no-twin 0.486, evidence F1 0.650, format validity 0.985; Opus 0.441 / 0.293. **Against E1++ 270M:**
+0.159 / +0.161 macro-F1, +0.128 / +0.151 no-twin, +0.250 / +0.237 evidence F1, +0.285 precision, +0.195 / +0.221 format
validity (teacher); +0.092 / +0.089 macro-F1, +0.021 / +0.028 no-twin, +0.171 / +0.178 evidence (Opus); ahead on every
clean slice. The first E5 cell to beat its decoder control decisively, and the first to beat it on zero-shot: a 270M
decoder below the majority floor (0.528 against 0.569) is lifted above it (0.687) by a 270m encoder in front. **Against
the native 270m-270m** (the same encoder, a native decoder): −0.043 / −0.031 macro-F1, −0.054 / −0.042 no-twin, −0.020 /
−0.022 evidence (teacher); −0.006 / −0.006 macro-F1 and −0.020 / −0.006 no-twin (Opus). The stitch into a raw decoder
recovers most of the native pair, all of it under Opus's gold on answers. **Against the same encoder into the 1B decoder**
(0.721 / 0.500 / 0.668): −0.034 macro-F1, −0.014 no-twin, −0.018 evidence; a decoder a quarter the size gives up little
once the encoder is in front.

The sixteenth, a new arm: the 1b encoder into the Gemma 3 270M decoder, the continuation stage (validation loss 1.857;
7 h 37) then the E5++ fine-tune (0.713; 5 h 02). Under the teacher gold clean macro-F1 0.747, no-twin 0.530, evidence F1
0.687, format validity 0.985; Opus 0.466 / 0.276. **Against E1++ 270M:** +0.219 / +0.218 macro-F1, +0.171 / +0.174 no-twin,
+0.288 / +0.273 evidence F1 (teacher); +0.118 / +0.131 macro-F1, +0.003 / −0.010 no-twin, +0.192 / +0.202 evidence (Opus).
**Against the 270m encoder into the same decoder:** +0.060 / +0.057 macro-F1, +0.043 / +0.023 no-twin, +0.038 / +0.035
evidence (teacher): at the 270M decoder the larger encoder pays on every number, where at the 1B decoder it paid +0.021 /
+0.006 / +0.027. **Against the same 1b encoder into the 1B decoder** (0.742 / 0.505 / 0.695): +0.005 / +0.009 macro-F1,
+0.024 / +0.017 no-twin, −0.008 / −0.013 evidence (teacher); under Opus's level on answers (+0.000 / +0.020) and −0.020 /
−0.030 on no-twin: once the encoder is in front, a 270M decoder reads it as well as a 1B one on answers. **Against the native 1B encoder + 270M decoder stitch** (0.781 / 0.641 / 0.713):
−0.034 / −0.023 macro-F1, −0.112 / −0.106 no-twin, −0.025 / −0.026 evidence (teacher); −0.021 / +0.010, −0.033 / −0.029,
−0.019 / −0.023 (Opus). The native stitch keeps a clear zero-shot lead over the soft-token one at the same encoder and
decoder sizes.

The seventeenth and last, a new arm: the 4b encoder into the Gemma 3 270M decoder, the continuation stage under FSDP
(validation loss 1.853; 24 h 30) then the E5++ fine-tune under FSDP (0.690; 15 h 32) and the soft-token generation at the
40k prompt-token budget (2 h 54). Under the teacher gold clean macro-F1 0.790, no-twin 0.657, evidence F1 0.710, format
validity 0.987; Opus 0.508 / 0.350. **Against E1++ 270M:** +0.262 / +0.244 macro-F1, +0.299 / +0.297 no-twin, +0.310 /
+0.296 evidence F1 (teacher); +0.159 / +0.155, +0.078 / +0.062, +0.200 / +0.210 (Opus); ahead on every clean slice. **Against
the 1b encoder into the same decoder:** +0.043 / +0.025 macro-F1, +0.128 / +0.123 no-twin, +0.023 / +0.023 evidence
(teacher); +0.041 / +0.023, +0.074 / +0.073, +0.008 / +0.008 (Opus): the largest encoder step on zero-shot in any E5
column (270m to 1b bought +0.043 here and +0.042 at the unadapted 1B decoder). **Against the native 4B encoder + 270M decoder stitch** (0.803 / 0.683 / 0.725): −0.012 /
−0.013 macro-F1, −0.026 / −0.002 no-twin, −0.015 / −0.015 evidence (teacher); −0.008 / −0.007, −0.020 / +0.006, −0.010 /
−0.015 (Opus). **Against the Gemma 3 4B decoder** (0.804 / 0.690 / 0.729): −0.014 / −0.014, −0.033 / −0.026, −0.019 /
−0.019 (teacher). **Against the native 1b-1b** (0.795 / 0.686 / 0.724): −0.004 / −0.009 macro-F1, −0.029 / −0.015 no-twin,
−0.014 / −0.014 evidence (teacher); +0.000 / +0.002, −0.006 / +0.003, −0.014 / −0.022 (Opus). A 270M decoder that cannot
format its own answers reaches, with a 4b encoder in front and the stage, the level of the 2B native encoder-decoder and
sits 0.012 to 0.014 below the two 4B-class arms.


## Verdict

Queue 30 ran to completion: seventeen runs, twenty-four training jobs (seven adaptation stages and seventeen fine-tunes),
two interruptions (before run 2 a stray empty folder left by a smoke tripped the archive guard; the queue was relaunched
the same minute, re-entered run 1 at its final step and re-exported the same step-2858 weights that had already been
generated from; at run 7's stage launch a transient NCCL segfault, relaunched three minutes later), every launch through
the review gate on one of four approved hashes (aff7cf208d55 for run 1; 530e8c2816ac, the BUG-2 patch, for runs 2 to 5;
5e79aaf7afe7 for run 6; 915c44c93ba0, the two-gold patch, for runs 7 to 17), every scored arm's wandb summary verified
against its metrics file. The fix raised every re-run arm's headline numbers and changed six of the project's earlier
conclusions when re-read under gold = teacher: E2++ no longer ties E1+ on zero-shot (it trails by 0.023 / 0.033 on
no-twin); the 1b-1b's zero-shot lead over the matched Qwen3-1.7B decoder reverses on no-twin; SmolLM3-3B and Gemma 3 4B
are level with the fixed 1b-1b rather than ahead of it; the 4B encoder + 270M decoder is level with the 1b-1b rather than
above it; the 1B grid's decoder-side threshold is missed on the full held-out cell; and the 1B to 4B decoder step buys
nothing on zero-shot where the encoder step buys +0.042. It did not change the shape of the main
result: under both golds the native encoder-decoders lead the decoders of the same size on zero-shot and on evidence,
and the sound 3 to 4B decoders are level with the 2B native pair rather than ahead of it. The 270M-decoder column, all
new, is the queue's clearest finding: an encoder in front of a decoder that cannot read the transcript on its own is worth
+0.16 to +0.26 on answers and +0.13 to +0.30 on zero-shot, scaling with the encoder, where in front of a 1B decoder the same
interface was worth +0.015 to +0.036 on answers and nothing on zero-shot. The soft-token stitch closes on the native stitch as the
encoder grows (−0.043, −0.034, −0.012 on answers; −0.054, −0.112, −0.026 on zero-shot) and is level with it only at the 4b
encoder.

Ranking of the 21 sound arms, gold = teacher clean macro-F1 first:

| # | Run | Teacher clean macro-F1 | Teacher no-twin | Teacher evidence F1 | Teacher messy macro-F1 | Opus clean macro-F1 | Opus no-twin |
|---|---|---|---|---|---|---|---|
| 1 | e1-gemma3-4b-pt | 0.804 | 0.690 | 0.729 | 0.767 | 0.526 | 0.407 |
| 2 | e4mm-4b-enc-270m-dec | 0.803 | 0.683 | 0.725 | 0.766 | 0.516 | 0.370 |
| 3 | e1-smollm3-3b-base | 0.800 | 0.691 | 0.724 | 0.772 | 0.518 | 0.366 |
| 4 | e4-t5gemma2-1b-1b | 0.795 | 0.686 | 0.724 | 0.761 | 0.507 | 0.356 |
| 5 | e5pp-t5gemma2-4b-enc-gemma3-270m | 0.790 | 0.657 | 0.710 | 0.752 | 0.508 | 0.350 |
| 6 | e4mm-1b-enc-270m-dec | 0.781 | 0.641 | 0.713 | 0.750 | 0.488 | 0.309 |
| 7 | e1-hunyuan-1.8b | 0.774 | 0.619 | 0.692 | 0.741 | 0.498 | 0.323 |
| 8 | e4mm-270m-enc-4b-dec | 0.759 | 0.562 | 0.689 | 0.729 | 0.488 | 0.319 |
| 9 | e1-qwen2.5-1.5b | 0.751 | 0.613 | 0.650 | 0.723 | 0.467 | 0.304 |
| 10 | e5pp-t5gemma2-1b-enc-gemma3-270m | 0.747 | 0.530 | 0.687 | 0.727 | 0.466 | 0.276 |
| 11 | e5pp-t5gemma2-1b-enc-gemma3-1b | 0.742 | 0.505 | 0.695 | 0.718 | 0.466 | 0.296 |
| 12 | e4mm-270m-enc-1b-dec | 0.742 | 0.565 | 0.677 | 0.713 | 0.471 | 0.317 |
| 13 | e4-t5gemma2-270m-270m | 0.731 | 0.540 | 0.670 | 0.701 | 0.447 | 0.314 |
| 14 | e5pp-t5gemma2-270m-enc-gemma3-1b | 0.721 | 0.500 | 0.668 | 0.701 | 0.440 | 0.283 |
| 15 | e5-t5gemma2-1b-enc-gemma3-1b | 0.708 | 0.494 | 0.651 | 0.687 | 0.428 | 0.275 |
| 16 | e1pp-gemma3-1b-pt | 0.707 | 0.507 | 0.628 | 0.684 | 0.423 | 0.282 |
| 17 | e1-gemma3-1b-pt | 0.698 | 0.488 | 0.623 | 0.676 | 0.425 | 0.284 |
| 18 | e5-t5gemma2-270m-enc-gemma3-1b | 0.694 | 0.452 | 0.646 | 0.675 | 0.429 | 0.268 |
| 19 | e5pp-t5gemma2-270m-enc-gemma3-270m | 0.687 | 0.486 | 0.650 | 0.670 | 0.441 | 0.293 |
| 20 | e1-gemma3-270m | 0.530 | 0.372 | 0.401 | 0.513 | 0.345 | 0.255 |
| 21 | e1pp-gemma3-270m | 0.528 | 0.359 | 0.400 | 0.509 | 0.349 | 0.272 |

Floors: teacher majority-answer 0.569 clean (no-twin 0.582), Opus-majority 0.709 on the Opus subset.

## Follow-ups

- Every validation loss in this record for an arm with a T5Gemma 2 encoder was computed before the BUG-2 padding reached
  the validation forward (fixed 2026-10-05): about one validation micro-batch in 32 carried an unpadded encoder row at the
  exposed length. The losses were computed the same way for every such arm and predecessor, so the pairwise readings
  stand; the absolute values carry the exposure. No benchmark number is affected.
- The two 4B mix-and-match arms (`e4mm-4b-enc-270m-dec`, `e4mm-270m-enc-4b-dec`) were generated before the BUG-2 fix to
  the T5Gemma generation backends. Measured on their outputs (2026-10-05): on the exposed rows (encoder length 1 mod 32,
  567 clean pairs each) 1.1% and 1.2% are unparseable against 0.8% elsewhere, where the abandoned native runs showed 10%
  against 1%: no large excess of format failures. The effect on their parseable answers and evidence is unmeasured; only a
  paired regeneration on the patched backend (about two GPU-hours each, no retraining) would measure it, so their numbers
  are reported as they stand with this caveat.
- The Qwen3-1.7B control is the one parameter-matched decoder left as an abandoned run, and the one where the zero-shot
  ordering against the 1b-1b reversed under the teacher gold (the re-read of 2026-09-29, summarised in the Verdict). Re-running it is the only way to settle that
  comparison.
- A second seed on the 270M-decoder column would say how much of the +0.043 between the 1b and 4b encoders, and the
  +0.128 on zero-shot, is the encoder and how much is seed noise; every cell here is one seed.
- The 4b-encoder arm into the 1B decoder (the encoder-axis record's original plan) still waits on rented hardware.
