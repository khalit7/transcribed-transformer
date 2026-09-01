---
name: asr-noise
description: Fit or apply the ASR channel model that turns clean written text into realistic ASR-style transcript text. Use when synthesising pseudo-transcripts from written corpora, when calibrating the channel against real ASR output, or when validating that synthetic transcripts are indistinguishable from real ones.
---

# ASR channel model

Much of the usable corpus is written text (complaint narratives, written dialogue) while the target distribution is disfluent ASR output. The channel model closes that gap cheaply: fit the error distribution once from paired data, then apply it to millions of clean documents without paying for TTS and ASR at corpus scale.

This is a research contribution in its own right, not a preprocessing utility. Treat its validation seriously.

## Fitting

Fit only from corpora that have **both audio and a human verbatim transcript**. The datasheet flags these.

1. Run open ASR over the audio (Whisper large-v3, Parakeet, and at least one other system, so the channel is not a model-specific artefact).
2. Align each ASR hypothesis to its human reference at word level.
3. Estimate, and store as a versioned wandb artifact:
   - substitution confusion distribution, conditioned on word frequency and phonetic neighbourhood
   - deletion and insertion rates
   - filler distribution (which fillers, at what rate, in what position)
   - disfluency and repair patterns: repetitions, restarts, self-corrections
   - casing and punctuation behaviour of the target system
   - speaker-attribution and turn-boundary error rates

Fit per ASR system as well as pooled. Systems differ, and a channel fitted only to one system will bake in its idiosyncrasies.

## Applying

Apply to clean text to produce pseudo-transcripts. Preserve line structure, because line indices carry the evidence labels: a channel operation must never silently renumber or merge lines. If the channel introduces a turn-boundary error, that must be recorded so the label mapping stays correct.

Record the channel artifact version on every dataset produced. A model trained on channel v1 data and evaluated against channel v2 data is a silent confound.

## Validating

Fitting the channel is easy; proving it produces realistic output is the actual work. Two checks, both required:

**Distributional**: word error rate, filler frequency, mean turn length, type-token ratio and disfluency density of the synthetic output should sit inside the range spanned by the real ASR corpora. Report as a table, not a claim.

**Adversarial**: train a discriminator to separate channel-noised text from real ASR transcripts. **Target is AUC at or below ~0.65.** Near-1.0 AUC means the synthetic text is trivially detectable and models trained on it will not transfer, which is exactly the failure mode this project exists to avoid.

Report the discriminator result honestly, including when it fails. A channel model that does not pass is a finding, and it blocks the phases that depend on it.

## Conventions

- Deterministic given a seed. Same input plus same seed plus same channel version produces byte-identical output.
- Configured by YAML like everything else; the code does not change to add a new channel variant.
- Log fitting runs to the `tt-trunk` wandb project with the corpus and ASR system as tags.
