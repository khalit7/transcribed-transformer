# CLAUDE.md

Working agreement for this repository. Read this before making changes.

## What this project is

Research towards a **universal encoder for transcribed speech**, and task models built on it that answer compliance questions over collections of call transcripts.

The task shape: input is a case (one or more ASR transcripts of advisor/customer calls, ~32k tokens each, speaker-labelled turns, one turn per line). Input also includes a compliance question and explicit definitions of what constitutes each of its four possible answers. Output is a triple:

- **answer**: `pass | fail | partial_pass | NA`
- **evidence**: the set of line numbers supporting that answer
- **summary**: short reasoning for why the answer is correct

Two hypotheses under test:

1. **Architecture.** Long input, short structured output, with judgements that depend on the whole conversation. Bidirectional attention over the transcript should beat causal attention.
2. **Distribution.** Models trained on written text underperform on disfluent, ASR-transcribed speech. Pretraining on transcript-like text should help.

Questions must work **zero-shot**: the question text and its answer definitions are model input, not a trained-in label set. A question never seen during training must work.

See `README.md` for the research design and the plan in full.

## Hard rules

### Confidentiality

This repository is **public**. The owner works at a company in this domain. Nothing about that employer's internal work enters this repository, ever. Specifically forbidden:

- Production or customer data, in any form, including paraphrased or partial
- Internal model names, architectures, sizes, training recipes or checkpoints
- Customer names, tenant counts, model counts, or any internal metric
- Internal findings, benchmark numbers or evaluation results
- Internal prompt text, taxonomies or question banks

Everything here is built from public datasets and public literature. When describing the motivating problem, describe it generically ("compliance question answering over call transcripts in a regulated domain"), never as a specific employer's system. Run the `confidentiality-check` skill before any commit that touches prose.

If a change would require internal knowledge to justify, it does not belong here.

**Internal source material never enters the working tree, even untracked.** Not in a scratch file, not in a notes file, not "temporarily". An untracked file is one `git add -A` away from public, and it is invisible to the diff that `confidentiality-check` reads, so the one control that would catch it never sees it. Internal reference documents live outside the repository; only their generic, derived conclusions come in, and each one has to survive the justification test above on its own. The `.gitignore` entries for this are a backstop, not the control.

### Dataset licences

Every corpus is assigned to exactly one track, recorded in `data/DATASHEET.md`:

- **Track P (permissive)** — licences allowing commercial use and derivative redistribution. Models trained on Track P only are portable to commercial settings.
- **Track NC (non-commercial)** — CC BY-NC, research-only, or unclear. Best data, but any model touching it is research-only.

**Never mix tracks in a single training run.** Every headline result is reported on both tracks so the cost of the licence restriction is visible. Adding a corpus without a licence decision is not allowed; use the `add-corpus` skill, which enforces this.

### Results are recorded, never estimated

Never write a number into a README, table, docstring or commit message that did not come from an actual run. If a result is not yet measured, write `TBD`. Do not infer, extrapolate or "reasonably estimate" training metrics, throughput or accuracy. A placeholder is honest; a plausible-looking invented number is not.

The same applies to negative results: if an experiment contradicts a hypothesis in this repo, that gets written up, not buried.

## Code conventions

- **Python 3.11+, managed with `uv`.** Dependencies in `pyproject.toml`, never a bare `pip install` into the environment.
- **Config over code.** Adding a model, dataset or head means adding a class and pointing a YAML at it. The training loop does not change. If a change requires editing the training loop to support a new variant, that is a signal the abstraction is wrong.
- **One canonical schema.** Every corpus loader emits the pydantic types in `data/schema.py`. Downstream code never sees a corpus-specific format. If a corpus does not fit the schema, extend the schema deliberately rather than special-casing the consumer.
- **Typed and validated.** pydantic for data, type hints throughout, `mypy` and `ruff` clean.
- Tests with `pytest`. Data loaders get a round-trip test against a small fixture; training code gets a single-step smoke test.

## Experiment tracking — wandb

Every training and evaluation run logs to Weights & Biases. There is no second logging path.

- **Projects by arm family**: `tt-trunk`, `tt-heads`, `tt-scratch`, `tt-encdec`.
- **Log the fully resolved config** as the run config, not a path to a YAML. A run must be reproducible from wandb alone.
- **Mandatory tags on every run**: licence track (`track-p` / `track-nc`), arm (`arm-a` … `arm-e`), and base checkpoint. Results tables are generated from the wandb API filtered on these tags, which is what stops written numbers drifting from real ones.
- **Always logged**: loss, learning rate, grad norm, tokens seen, throughput (tokens/s and MFU), per-GPU memory.
- **Evaluation logs to the same run id** as the training job that produced the checkpoint. A checkpoint's quality and its training curve are never separated.
- **Artefacts** (tokenizer, ASR channel-model parameters, benchmark version) versioned as wandb artifacts, so any result traces back to the exact data that produced it.
- `WANDB_API_KEY` comes from the environment and is never committed. Long unattended runs use `WANDB_MODE=offline` and sync afterwards.

Alongside wandb, each run gets a directory in `experiments/` recording hypothesis, config, result and verdict, cross-linked to the wandb run id. Use the `log-experiment` skill. wandb holds the metrics; `experiments/` holds the reasoning.

## Hardware

Dual RTX 5090 (Blackwell, sm_120, 32GB each, **no NVLink**, PCIe).

- Requires CUDA 12.8+ and a PyTorch build with sm_120 kernels. FlashAttention 2 supports sm_120; FlashAttention 3 is Hopper-only, so do not reach for it.
- No NVLink means all-gather traffic crosses PCIe. **Prefer DDP.** Use FSDP only where memory genuinely forces it, and measure the throughput cost when you do.
- Long-context training relies on FA2 varlen plus unpadding plus activation checkpointing. Expect batch size 1-2 per GPU at 32k with gradient accumulation.
- Every long run must checkpoint and resume cleanly. Runs are measured in days on a machine that is also someone's desktop.

## Writing conventions

- **British spelling** throughout (optimise, anonymisation, behaviour, licence as noun).
- **Go easy on em-dashes.** Do not use them as a default connector. Prefer commas, colons, semicolons, or separate sentences.
- Prose should be legible to someone who does not already know this project. Lead with the outcome, not the plumbing.
- Do not oversell. This project's credibility rests on stated limitations being honest, particularly the benchmark-circularity problem described in the README.

## Git

- Work directly on `main`. No branches unless asked.
- **Commit and push only when explicitly asked.**
- Run `confidentiality-check` before committing prose changes.
- Commit messages describe what changed and why, in the imperative.
