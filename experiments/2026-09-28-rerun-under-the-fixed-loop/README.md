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
stage trained under FSDP and stand (the audit of the cleanup caught them being swept up with the DDP runs; their weights and
wandb pages were lost, their results were not).
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
- **Generation keeps `--max-tokens 512`**, as every kept arm (the API baseline, the FSDP arms) was generated: 186 of the
  20,700 benchmark gold targets exceed 512 Gemma tokens (line numbers tokenise digit by digit; the share differs by
  tokenizer), so a perfect model is scored invalid on about 0.9% of pairs. A ceiling on format validity, accuracy and
  evidence F1 that every arm shares, recorded here rather than changed mid-comparison.
- **Throughput** under the continuation objective counts the positions consumed (prefix plus capped continuation), not
  whole documents; both cards' peak memory are logged.
- **Evaluator**: the parser is strict (the whole output is one JSON object); a recovered answer (a case or whitespace
  variant of an option) keeps its 0.3 credit but is no longer a correct prediction for macro-F1 and the confusion table;
  macro-F1 runs over each question's option set; duplicate outputs count once and a missing pair scores as empty;
  format validity requires a summary string; out-of-vocabulary tags count against the tag Jaccard. **Every existing
  run is re-scored with this evaluator** so the numbers that stay (the FSDP arms, the not-re-run arms) are on the same
  scale as the re-runs. Done before the launch: every existing run's clean-variant benchmark macro-F1 is identical to
  three decimals under both evaluators, and format validity moves by 0.001 on one run, so the kept numbers stand as
  published.
- **The queue** (`queue30_rerun.sh`, in the session scratchpad; its text is what the gate hashes) moves each buggy
  predecessor to `checkpoints/_archive/<name>-ddp-bug` before its replacement starts, runs the gate before every
  training stage and every generation, checks every generation's exit status and that every (id, variant) pair is
  present exactly once before scoring, and aborts on any failure. Same configs as the original runs; no recipe change.

The queue, in order (durations are the buggy predecessors' measured wall-clock, for planning only):

| # | Run | Stages | Predecessor's time | Status |
|---|---|---|---|---|
| 1 | `e1-gemma3-1b-pt` | train, vLLM generation, score | 2 h 50 + 10 min | training since 18:06 |
| 2 | `e1-gemma3-270m` | train, vLLM generation, score | 52 + 6 min | pending |
| 3 | `e4-t5gemma2-1b-1b` | train, `hf_encdec` generation, score | about 3 h + 2.6 h | pending |
| 4 | `e4-t5gemma2-270m-270m` | train, `hf_encdec` generation, score | about 1.5 h + 2 h | pending |
| 5 | `e4mm-1b-enc-270m-dec` | train, `stitched` generation, score | about 3 h + 2 h | pending |
| 6 | `e4mm-270m-enc-1b-dec` | train, `stitched` generation, score | about 3 h + 2 h | pending |
| 7 | `adapt-continuation-corpora-gemma3-1b-pt` then `e1pp-gemma3-1b-pt` | stage; train, vLLM generation, score | 4 h 22 + 2 h 50 + 10 min | pending |
| 8 | `adapt-continuation-corpora-e5-t5gemma2-1b-enc-gemma3-1b` then `e5pp-t5gemma2-1b-enc-gemma3-1b` | stage; train, `soft` generation, score | 9 h 56 + 6 h 16 + 2 h 30 | pending |
| 9 | `adapt-continuation-corpora-e5-t5gemma2-270m-enc-gemma3-1b` then `e5pp-t5gemma2-270m-enc-gemma3-1b` | stage; train, `soft` generation, score | 7 h 14 + 4 h 22 + 2 h 20 | pending |
| 10 | `e5-t5gemma2-1b-enc-gemma3-1b` | train, `soft` generation, score | 6 h 27 + 2 h 40 | pending |
| 11 | `e5-t5gemma2-270m-enc-gemma3-1b` | train, `soft` generation, score | 4 h 21 + 2 h 30 | pending |
| 12 | `e1-qwen2.5-1.5b` | train, vLLM generation, score | 3 h 38 + 17 min | pending |
| 13 | `e1-hunyuan-1.8b` | train, vLLM generation (eager), score | 3 h 50 + 30 min | pending |
| 14 | `adapt-continuation-corpora-gemma3-270m` then `e1pp-gemma3-270m` | stage; train, vLLM generation, score | new (stage about 2.5 h; fine-tune 52 min) | pending |
| 15 | `adapt-continuation-corpora-e5-t5gemma2-270m-enc-gemma3-270m` then `e5pp-t5gemma2-270m-enc-gemma3-270m` | stage; train, `soft` generation, score | new | pending |
| 16 | `adapt-continuation-corpora-e5-t5gemma2-1b-enc-gemma3-270m` then `e5pp-t5gemma2-1b-enc-gemma3-270m` | stage; train, `soft` generation, score | new | pending |
| 17 | `adapt-continuation-corpora-e5-t5gemma2-4b-enc-gemma3-270m` then `e5pp-t5gemma2-4b-enc-gemma3-270m` | stage (FSDP, 8k); train, `soft` generation (40k budget), score | new | pending |

Every buggy run's weights were deleted on 2026-09-28 at 18:11, before any replacement finished: the 28 DDP run
directories (the ones being re-run and the ones not; 27 finished runs and one stopped stage) were stripped of their `final/` weight files and step checkpoints
(222.8 GB freed) and moved to `checkpoints/_archive/<name>-ddp-bug`, keeping their training logs, generation outputs and
scoring (lenient and strict) and their wandb ids; their wandb runs were exported (config, summary, step history, as
`wandb_export.json` in each archive folder) and deleted, so a wandb query never returns an abandoned run. An audit of the
cleanup found two archives still holding their weights (deleted) and three FSDP runs swept up by mistake (the E3 stage and
fine-tune, the stopped bf16 stochastic-rounding check): their weights are gone, their wandb exports are kept, their wandb
pages were re-synced from local files where those allowed (the stage in full, the fine-tune to step 590), and their result
folders are back under `checkpoints/`. When a re-run is scored, its predecessor's numbers are replaced in the README,
the record that reported them and the notes; the old numbers stay only in this record's result table, next to the new
ones.

wandb: run ids TBD (one per run, in the projects the configs name).

## Result

TBD (a table of predecessor against re-run per metric, filled as each run scores).

## Verdict

TBD.

## Follow-ups

TBD.
