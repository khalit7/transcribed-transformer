# E0: Claude Sonnet 5 on the benchmark, through `claude -p`

Written before the run finished, 2026-09-17 01:00. Run on Sonnet 5 through the Claude Code CLI (`claude -p`) rather than the direct API, with 8 workers.

> **Withdrawn, 2026-09-29.** This arm is no longer part of the design. The benchmark-construction audit of 2026-09-29 found that 86% of the family gold labels were Sonnet's own answers and that the general_qa gold was mostly the training labeller's, so a frontier model that wrote none of the training labels could not be compared fairly with the trained arms on this benchmark, and re-labelling the gold for the sake of one baseline was not worth the cost when the promising designs are to be re-tested on other data anyway. The 29,641 requests already generated (of 38,220) are kept under `checkpoints/_archive/e0-sonnet5-cli-withdrawn/`; nothing was scored. The trained arms remain comparable with each other on this benchmark; see the notes' bug register (BUG-3) and D60.

> **Gold and cell as quoted here (note added 2026-09-29).** The comparator numbers in this record were scored against the pre-D61 mixed gold, with the twinned held-out questions in the unseen cell; from 2026-09-29 every result is reported under gold = teacher and gold = opus (the README and `experiments/2026-09-28-rerun-under-the-fixed-loop/`). They are kept as written because this arm was withdrawn and nothing here is a result.

## Hypothesis

> **Comparator numbers in this record come from runs with a bug.** The 2026-09-28 audit found that the DDP training loop never synchronised gradients (half the data at batch 16), so every trained-arm number quoted here as a bar or reference (the native pairs, the mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) is from a buggy run being replaced by a re-run ([the re-run record](../2026-09-28-rerun-under-the-fixed-loop/README.md)); the predictions are read against the re-run numbers in the re-run record (queue 30, complete 2026-10-05). *The DDP arms among those (the native pairs, the 1B mix-and-match arms, the 1B and 270M decoders, E1++ and E5++) have since been re-run under the fixed loop (queue 30, complete 2026-10-05); the FSDP arms quoted (the 4B mix-and-match arms, Gemma 3 4B, SmolLM3) were sound as run. The sound numbers are in the re-run record.*

The README's E0 row: a frontier model behind an API, given the same prompt and transcript rendering as the trained arms, is the baseline every trained arm must beat on quality and cost. After the encoder-decoder pairs, the strongest trained model on the benchmark is T5Gemma 2 1b-1b (macro-F1 0.744 / 0.722 clean / messy; 0.677 / 0.642 on unseen questions). A frontier model has read no training labels but has the general competence the small models lack, and the labels themselves came from a labeller pipeline with Claude models in it (adjudication by Opus), so a Claude model shares a lineage with the gold.

## Prediction

- Sonnet 5 above every trained arm on unseen-question macro-F1 on both variants, by at least 0.05: the zero-shot cell is where the trained models fall furthest and where a frontier model's generality should show.
- On seen questions, at or below the 1b-1b overall: the trained models have seen 91k labelled examples of exactly these questions; the API model has seen none.
- Evidence F1 at or above the 1b-1b's (0.619 / 0.590).
- **Disconfirmed if** Sonnet 5 is at or below the 1b-1b on unseen-question macro-F1 on either variant: a 2B-parameter fine-tuned encoder-decoder would then match the frontier model where the frontier model should be strongest, and the API baseline's advantage would be only on the questions nobody trained for.

## Setup

- Route: `claude -p --model sonnet --output-format json` (resolves to `claude-sonnet-5`), the labelling pipeline's route (`src/synthesis/llm.py`); `src/train/api_baseline.py --backend cli --workers 8` (the module was removed from the tree on 2026-09-29 with the withdrawal). The direct Message Batches path (`--backend batch`, transcript block cached, about $35 at list price) is implemented but was not used for this run. Every CLI call carries Claude Code's own context (16–30k cached tokens per call), so the CLI's reported cost is a notional list-price figure dominated by that overhead: the pilot ran at $0.082 per call cold and about $0.011–0.037 with the cache warm.
- Working directory: an empty one. `claude -p` loads `CLAUDE.md` and memory from its cwd; a first attempt (354 calls, kept under `checkpoints/e0-sonnet5-cli-pilot/`) ran from the repository and carried the project's instructions into every prompt. Discarded and restarted at 00:53.
- Prompt: `src/train/data.task_prompt`, byte-identical to the trained models' input, as the user turn; no system prompt of ours (the harness's own applies); the CLI's default sampling and output cap. Questions of one transcript go to one worker in sequence so the harness's automatic caching can reuse the transcript prefix.
- Output: the same records as `generate.py` (id, variant, raw text, `prompt_tokens` counted with the Qwen3 tokenizer so length buckets match E1's), plus the API's usage and cost per call; scored by `evaluate.py` under the same strict parser as every arm.
- **Known artefact of this route, measured before the run:** Sonnet 5 through the CLI ends its JSON object without the closing brace in a large share of calls (11 of 120 in the pilot from the repository directory; 22 of the first 83 from the empty directory; 4 of 8 on re-runs of failing prompts under the default harness prompt and 5 of 8 under a minimal one). The streamed assistant message itself lacks the brace and the stop reason is `end_turn`, so it is the model's output under this harness, not post-processing. Scoring policy, fixed now: the protocol number is the strict parse, as for every arm; a second, clearly labelled number restores the single missing brace where that alone makes the object parse, so the reader can see the frontier model's judgement separately from the harness artefact. Neither the evaluator nor the recorded outputs are modified.
- wandb: tt-baselines; the run id was to be written to `checkpoints/e0-sonnet5-cli/wandb_id` at the end of the run with the cost totals, and the run was stopped before that point, so no wandb run was ever created for it; the discarded pilot's run (tt-baselines/anupz4wz) was exported to `checkpoints/_archive/e0-sonnet5-cli-pilot-withdrawn/` and deleted on 2026-09-29, and the project is empty.

## Result

None. The run was stopped on 2026-09-26 12:04 at 29,641 of 38,220 requests (8 workers from 00:53 on 2026-09-17, cut to 2 at 10:13 after usage-limit waits, resumed with one worker on 2026-09-26 at 11:00) and never scored; the arm was withdrawn on 2026-09-29 before any number was produced. The generated outputs are archived unscored under `checkpoints/_archive/e0-sonnet5-cli-withdrawn/` and `checkpoints/_archive/e0-sonnet5-cli-pilot-withdrawn/` and will not be scored: the benchmark's gold is its training labellers' own label (and Sonnet's own on the rare-event families), so no number from this route would be a fair comparison.

## Verdict

Withdrawn, not disconfirmed: the hypothesis was never tested. What the attempt did establish is recorded in Setup (the CLI route's truncated-JSON artefact, the need for an empty working directory, the cost and rate of the route). The frontier-baseline question is open for a future dataset whose gold no training labeller wrote (notes D60, D61).

## Follow-ups

- A frontier baseline needs a benchmark gold that no training labeller and no baseline model produced; that is a data-collection decision for the next dataset, not a scoring fix.
- The archived outputs are kept for that day and are not to be scored against the current gold.
