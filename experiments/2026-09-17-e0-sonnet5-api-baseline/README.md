# E0: Claude Sonnet 5 on the benchmark, through `claude -p`

Written before the run finished, 2026-09-17 01:00. Khalid: "Run E0 on sonnet", then "Use claude -p anyway" when offered the direct API, then "Use 8 workers".

## Hypothesis

The README's E0 row: a frontier model behind an API, given the same prompt and transcript rendering as the trained arms, is the baseline every trained arm must beat on quality and cost. After the encoder-decoder pairs, the strongest trained model on the benchmark is T5Gemma 2 1b-1b (macro-F1 0.744 / 0.722 clean / messy; 0.677 / 0.642 on unseen questions). A frontier model has read no training labels but has the general competence the small models lack, and the labels themselves came from a labeller pipeline with Claude models in it (adjudication by Opus), so a Claude model shares a lineage with the gold.

## Prediction

- Sonnet 5 above every trained arm on unseen-question macro-F1 on both variants, by at least 0.05: the zero-shot cell is where the trained models fall furthest and where a frontier model's generality should show.
- On seen questions, at or below the 1b-1b overall: the trained models have seen 91k labelled examples of exactly these questions; the API model has seen none.
- Evidence F1 at or above the 1b-1b's (0.619 / 0.590).
- **Disconfirmed if** Sonnet 5 is at or below the 1b-1b on unseen-question macro-F1 on either variant: a 2B-parameter fine-tuned encoder-decoder would then match the frontier model where the frontier model should be strongest, and the API baseline's advantage would be only on the questions nobody trained for.

## Setup

- Route: `claude -p --model sonnet --output-format json` (resolves to `claude-sonnet-5`), the labelling pipeline's route (`src/synthesis/llm.py`), billed to the personal subscription account; `src/train/api_baseline.py --backend cli --account p --workers 8`. The direct Message Batches path (`--backend batch`, transcript block cached, about $35 at list price) is implemented and was declined for want of an API key. Every CLI call carries Claude Code's own context (16–30k cached tokens per call), so the CLI's reported cost is a notional list-price figure dominated by that overhead: the pilot ran at $0.082 per call cold and about $0.011–0.037 with the cache warm.
- Working directory: an empty one. `claude -p` loads `CLAUDE.md` and memory from its cwd; a first attempt (354 calls, kept under `checkpoints/e0-sonnet5-cli-pilot/`) ran from the repository and carried the project's instructions into every prompt. Discarded and restarted at 00:53.
- Prompt: `src/train/data.task_prompt`, byte-identical to the trained models' input, as the user turn; no system prompt of ours (the harness's own applies); the CLI's default sampling and output cap. Questions of one transcript go to one worker in sequence so the harness's automatic caching can reuse the transcript prefix.
- Output: the same records as `generate.py` (id, variant, raw text, `prompt_tokens` counted with the Qwen3 tokenizer so length buckets match E1's), plus the API's usage and cost per call; scored by `evaluate.py` under the same strict parser as every arm.
- **Known artefact of this route, measured before the run:** Sonnet 5 through the CLI ends its JSON object without the closing brace in a large share of calls (11 of 120 in the pilot from the repository directory; 22 of the first 83 from the empty directory; 4 of 8 on re-runs of failing prompts under the default harness prompt and 5 of 8 under a minimal one). The streamed assistant message itself lacks the brace and the stop reason is `end_turn`, so it is the model's output under this harness, not post-processing. Scoring policy, fixed now: the protocol number is the strict parse, as for every arm; a second, clearly labelled number restores the single missing brace where that alone makes the object parse, so the reader can see the frontier model's judgement separately from the harness artefact. Neither the evaluator nor the recorded outputs are modified.
- wandb: tt-baselines, run id TBD (`checkpoints/e0-sonnet5-cli/wandb_id`), written at the end of the run with the cost totals; the benchmark numbers logged to the same run.

## Result

In progress, paused. Run history: 8 workers from 00:53 on 2026-09-17, cut to 2 workers at 10:13 at Khalid's request after the second subscription-limit window; the machine rebooted on 2026-09-19 with 24,298 of 38,220 requests done; relaunched 2026-09-20 10:47 and stopped at 10:52 at Khalid's request ("stop E0 for now") at 24,330 done. Resumed 2026-09-20 with one worker (10:47, personal account), killed by a full disk at 16:42 (one truncated record dropped), resumed 17:20 and stopped at 17:35 at Khalid's request, then resumed at 17:36 on the **work** subscription account for the remainder (Khalid: "Use claude work for E0. Use one worker"). Account boundary: records 1–26,373 of `gen.jsonl` (in file order) were billed to the personal account, the rest to the work account; the per-record fields do not carry the account. Every completed request is on disk; a relaunch resumes from there. Limit windows on the personal account: about thirty, the longest 202 minutes. Numbers: TBD until the run completes.

## Verdict

TBD.

## Follow-ups

TBD.
