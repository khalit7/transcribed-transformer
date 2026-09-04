# CLAUDE.md

Working agreement for this repository. Read this before making changes.

## What this project is

Research towards models that answer compliance questions over collections of ASR call transcripts, comparing three experiments: a fine-tuned **decoder**, an **encoder-decoder** (bidirectional attention over the transcript), and a prompted **API model** baseline.

The task shape: input is a case (one or more diarised ASR transcripts of calls between staff and customers, speaker-labelled turns with no role labels, one turn per line, up to tens of thousands of tokens each). Input also includes a compliance question and explicit definitions of what constitutes each of its possible answers. Output is a triple:

- **answer**: one of the supplied options (e.g. `pass | fail | partial_pass | NA`)
- **evidence**: the set of line numbers supporting that answer. Line numbers refer to the transcript rendered one turn per line as `<n>: <role>: <text>` (1-based `n`, verbatim corpus role); every labeller, the judge, the benchmark and fine-tuning use that exact rendering (see `data/DATASHEET.md`, "Model input format")
- **summary**: short reasoning for why the answer is correct

Two hypotheses under test:

1. **Architecture.** Long input, short structured output, with judgements that depend on the whole conversation. Bidirectional attention over the transcript should beat causal attention.
2. **Distribution.** Models trained on written text underperform on disfluent, ASR-transcribed speech. Pretraining on transcript-like text should help.

Questions must work **zero-shot**: the question text and its answer definitions are model input, not a trained-in label set. A question never seen during training must work.

See `README.md` for the research design in full.

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

- **Track P (permissive)** — licences allowing commercial use and derivative redistribution. Models trained on Track P only are portable to commercial settings. Share-alike licences (CC BY-SA, CDLA-Sharing) qualify for Track P but carry an **SA flag** in the datasheet: anything released that derives from SA-flagged data must carry the share-alike licence forward, and model cards must say so.
- **Track NC (non-commercial)** — CC BY-NC, research-only, or unclear. Best data, but any model touching it is research-only.

**Never mix tracks in a single training run.** Every headline result is reported on both tracks so the cost of the licence restriction is visible. Adding a corpus without a licence decision is not allowed; use the `add-corpus` skill, which enforces this.

### Data preference hierarchy

The input this project models is **ASR output**: disfluent, full of recognition errors, and systematically unlike written prose. Text that has been cleaned up, or was never spoken, is a substitute for that and is treated as one. Corpus text is preferred in this order, and every `data/DATASHEET.md` entry records which tier it is:

1. **Real ASR output.** Disfluencies, fillers, restarts, misrecognitions, all of it. The target distribution itself. Use directly.
2. **Audio we can run ASR over ourselves.** Produces tier 1. The ASR system used becomes part of the datasheet entry, because different recognisers have different error profiles and mixing them silently is a confound.
3. **Clean written or human-verbatim text.** Only where tiers 1 and 2 are genuinely unavailable *and* the corpus is necessary. Requires the ASR channel model (see `data/SYNTHSHEET.md`) before it is used as training text, and the entry must say why a lower tier was unavoidable.

Human-verbatim transcription is **tier 3, not tier 1**. It is a record of speech, but a transcriber who silently repairs a false start has removed exactly the signal this project exists to model.

A corpus that ships both a human transcript and an ASR transcript is a tier 1 corpus whose human side is **reference data for fitting the channel model**, not training text. Preferring the human side because it is cleaner inverts the whole point.

### Results are recorded, never estimated

Never write a number into a README, table, docstring or commit message that did not come from an actual run. If a result is not yet measured, write `TBD`. Do not infer, extrapolate or "reasonably estimate" training metrics, throughput or accuracy. A placeholder is honest; a plausible-looking invented number is not.

The same applies to negative results: if an experiment contradicts a hypothesis in this repo, that gets written up, not buried.

## Data layout

`data/` holds only data; all code lives in `src/`. Three sheets, disjoint jobs, and they are the only files under `data/` that git tracks:

- `data/DATASHEET.md` — datasets that are **on disk**. Its summary table is the single source of truth for all data.
- `data/SURVEYSHEET.md` — the corpus survey and every acquisition decision, including refusals. A source only earns a DATASHEET row once it is on disk.
- `data/SYNTHSHEET.md` — the recipes for synthetic transcript text (ASR channel model, TTS→ASR) and their QC gates.

Sources are immutable under `data/raw/<dataset>/`, one folder per dataset. Derived artefacts — `data/interim/<name>/{train,val}.jsonl` splits and `data/packed/<stage>/` token streams — are regenerable and not documented in the sheets. Labelled (transcript, question) → (answer, evidence, summary) records live in `data/labelled_data/` (`labelled_data.jsonl` + the derived `questions.jsonl`), generated by `src/synthesis/` (see its README) and described in the DATASHEET's "Labelled data" section; they serve both training and benchmarking (the split is decided downstream), are untracked, and `python -m src.synthesis.export --track p` produces the releasable copy. Raw corpus data is never committed.

## Code conventions

- **Python 3.11+, managed with `uv`.** Dependencies in `pyproject.toml`, never a bare `pip install` into the environment.
- **Config over code.** Adding a model, dataset or experiment means adding a class or a config and pointing YAML at it. If a change requires editing the training loop to support a new variant, the abstraction is wrong.
- **One canonical schema.** Every corpus's preprocessing module in `src/preprocessing/` emits the same document schema into `data/interim/`. Downstream code never sees a corpus-specific format.
- **Typed and validated.** pydantic for data, type hints throughout, `mypy` and `ruff` clean.
- Tests with `pytest`. Data preprocessing gets a round-trip test against a small fixture; training code gets a single-step smoke test.

## Experiment tracking — wandb

Every training and evaluation run logs to Weights & Biases. There is no second logging path.

- **Projects by experiment**: `tt-decoder`, `tt-encdec`, `tt-baselines`.
- **Log the fully resolved config** as the run config, not a path to a YAML. A run must be reproducible from wandb alone.
- **Mandatory tags on every run**: licence track (`track-p` / `track-nc`), experiment (`decoder` / `encdec` / `api`), and base checkpoint. Results tables are generated from the wandb API filtered on these tags, which is what stops written numbers drifting from real ones.
- **Always logged**: loss, learning rate, grad norm, tokens seen, throughput (tokens/s and MFU), per-GPU memory.
- **Evaluation logs to the same run id** as the training job that produced the checkpoint. A checkpoint's quality and its training curve are never separated.
- **Artefacts** (tokenizer, ASR channel-model parameters, benchmark version) versioned as wandb artifacts, so any result traces back to the exact data that produced it.
- `WANDB_API_KEY` comes from the environment and is never committed. Long unattended runs use `WANDB_MODE=offline` and sync afterwards.

Alongside wandb, each run gets a directory in `experiments/` recording hypothesis, config, result and verdict, cross-linked to the wandb run id. Use the `log-experiment` skill. wandb holds the metrics; `experiments/` holds the reasoning.

## Hardware

Dual RTX 5090 (Blackwell, sm_120, 32GB each, **no NVLink**, PCIe).

- Requires CUDA 12.8+ and a PyTorch build with sm_120 kernels. FlashAttention 2 supports sm_120; FlashAttention 3 is Hopper-only, so do not reach for it.
- No NVLink means all-gather traffic crosses PCIe. **Prefer DDP.** Use FSDP only where memory genuinely forces it, and measure the throughput cost when you do.
- Long-context training relies on FA2 varlen plus unpadding plus activation checkpointing. Expect small per-GPU batches at long context with gradient accumulation.
- Every long run must checkpoint and resume cleanly. Runs are measured in days on a machine that is also someone's desktop.

## Writing conventions

- **British spelling** throughout (optimise, anonymisation, behaviour, licence as noun).
- **Go easy on em-dashes.** Do not use them as a default connector. Prefer commas, colons, semicolons, or separate sentences.
- Prose should be legible to someone who does not already know this project. Lead with the outcome, not the plumbing.
- Do not oversell. This project's credibility rests on stated limitations being honest.
- Keep the data sheets current when a dataset is added, removed, or re-scoped.
- **Two terms, never swapped.** *LLM labelling* is a model answering a question about a transcript (producing answer, evidence, summary); that model is the **labeller**. *LLM-as-a-judge* is a model assessing how good such an answer is; that model is the **judge**. `src/synthesis` labels; a judge is only ever the grader of labels. Do not call a labeller a judge, in code, prompts, docs or notes.

## Git

- Work directly on `main`. No branches unless asked.
- **Commit and push only when explicitly asked.**
- Run `confidentiality-check` before committing prose changes.
- Commit messages describe what changed and why, in the imperative.
