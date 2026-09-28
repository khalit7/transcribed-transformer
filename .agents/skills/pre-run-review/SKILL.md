---
name: pre-run-review
description: Mandatory two-model review gate before any expensive run. Use before launching any training, adaptation or multi-hour generation job (anything beyond a smoke), and whenever code under src/train, a config under configs/, a queue script or the evaluator has changed since the last approved state. Every changed line is reviewed read-only by Codex gpt-6-astra and by a Claude Opus agent, the findings are verified and fixed, and the tree hash is approved before the launcher will run. Exists because the DDP gradient-synchronisation bug of 2026-09-10 to 2026-09-28 ran for two weeks and 27 runs unreviewed.
---

# Pre-run review: two models sign off every change before the expensive part

## The rule

No training stage, adaptation stage or benchmark generation starts on the GPUs until the exact state of the tree that
will run has been reviewed by **both** Codex gpt-6-astra and a Claude Opus agent, independently and read-only, and
every finding either fixed or explicitly rejected in writing. Smokes (a few steps, `--smoke N`) are exempt; they are how
the review's fixes are checked. The gate is mechanical: `scripts/gate.py` computes a hash over everything that can
affect a run (tracked and untracked files under `src/`, `configs/`, `tests/`, `pyproject.toml`, `uv.lock`, and the queue
script being launched) and refuses unless that hash is in `checkpoints/.reviews/approved.jsonl`. Queue scripts call the
gate before every `torchrun` and every `generate` stage; a launch without it is a violation to record in `notes.md`.

Why both models: each catches things the other misses (the DDP bug was found by both; the lenient parser by both; the
cut-seeding flaw by one; the FSDP export filter by the other). Why every change: the bug that costs two weeks is never
the one that looked risky.

## Protocol

1. **Freeze the state.** Commit or stage nothing; just stop editing. Run
   `python3 .claude/skills/pre-run-review/scripts/gate.py hash` and note the hash. Collect the diff since the last
   approved state: `git diff <last-approved-commit>` plus every untracked file the gate hashes, plus the queue script.
   If there is no previous approval, the review covers the whole of `src/train/`, the configs to be run and the script.
2. **Write the brief** to the scratchpad. Self-contained, no repo context assumed: what the code does, what changed and
   why, the exact files or diff, what the run will do (configs, stages, order), and the eight questions below. Ask for
   findings as `file:line`, what is wrong, a concrete failure scenario, a severity (changes results / edge case /
   latent), then a one-line list of what was checked and found correct. Read-only. Terse.
3. **Run both reviewers in parallel.**
   - Codex: `python3 ~/.claude/skills/consult_other_ai/scripts/preflight.py codex` first (quota); then
     `codex exec --sandbox read-only "$(cat brief.md)" > review_astra.md 2> review_astra.err` in the background. The
     default model in `~/.codex/config.toml` is gpt-6-astra; pass `-m gpt-6-astra` if it is not.
   - Opus: the `Agent` tool with `model: opus`, prompt "Read <brief path> and carry out exactly the review it describes,
     read-only", and the same output format.
4. **Verify every finding against the code yourself** before accepting it: reproduce it with a tiny script or a test
   where it is cheap (the DDP check took one minute on CPU with two gloo ranks). Fix what is real, add a test where a
   test is possible, run `pytest` and `ruff`. Reject in writing what is wrong, with the reason.
5. **Re-review if the fixes are non-trivial** (they change the hash; a second, shorter round on the diff of the fixes
   is normal). Trivial fixes (a comment, a config value the reviewers asked for) are re-hashed and approved without a
   second round, and the approval record says so.
6. **Approve**: `python3 .claude/skills/pre-run-review/scripts/gate.py approve --note "<what was reviewed, by whom,
   findings fixed/rejected>"`. This appends the current hash to `checkpoints/.reviews/approved.jsonl` (gitignored with
   the rest of `checkpoints/`). Then log one line in `notes.md` §8: hash prefix, reviewers, findings count, what changed.
7. **Launch.** The queue script's `gate.py check <queue-script>` passes; a hash mismatch (any later edit) blocks the
   launch and says which files differ from the approved state.

## The eight questions every brief asks

1. Loss normalisation under accumulation and data parallelism: does the step's gradient equal the global token mean?
   Is the forward routed through the parallel wrapper (`DDP.forward`, the FSDP root) so gradients synchronise?
2. Objective correctness: label alignment and shifts, the supervised-token count against the denominator, seeding that
   depends only on the example (not on batch layout), truncation and eos.
3. Model wiring: masks under padding and FlashAttention, position ids, tied weights on export and load, sharding units.
4. Resume: optimizer state, schedule position, data order, RNG; checkpoint retention; the end-of-run barrier.
5. Generation: left padding and position ids, cache growth, eos and pad handling, per-row prompt lengths, resume
   without duplicates or gaps, batch budgets.
6. Evaluation: the parser's strictness, what counts as correct, per-class and per-slice aggregation, duplicates,
   empty gold, the unseen-question cell.
7. Config: does the config say what the record says (recipe, budget, data, cuts, control), and are the controls matched?
8. Anything that would make a number in the record wrong without an error: logging counters, throughput, tags.

## Equivalence checks that must exist and pass

- **Data-parallel equivalence**: two ranks against one rank on the same steps give the same loss trajectory and grad
  norm to bf16 precision (`tests/` has the CPU unit test; the GPU check runs as a smoke before the first launch after
  any change to the loop).
- **Accumulation equivalence**: micro-batch splits of one step give the same gradient as the unsplit step.
- **Objective equivalence**: the seq2seq / continuation collates and their denominators agree on every reachable
  combination (`tests/test_train.py`).

## Report template (notes.md §8)

`- **<date>** — Pre-run review <hash8> (Codex gpt-6-astra <tokens>; Opus): <n> findings, <m> fixed (<list>), <k>
rejected (<why>); approved; launching <queue/configs>.`
