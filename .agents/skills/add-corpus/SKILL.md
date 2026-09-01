---
name: add-corpus
description: Add a public dataset to the corpus. Use whenever a new dataset is being ingested, a loader is being written, or someone asks to "add", "download", "ingest" or "pull in" a corpus. Enforces the licence-track and ASR-tier decisions before any data is downloaded or any code is written.
---

# Adding a corpus

A corpus is not added until it has a licence decision, a tier assignment, a loader emitting the canonical schema, a datasheet entry and a round-trip test. Work in that order. Do not download data before steps 1 and 2 are answered.

## 1. Licence decision (blocking)

Find the actual licence. Not the paper's claim, not a HuggingFace card summary, the licence itself. Record the URL you read it from.

Assign exactly one track:

| Track | Criteria |
|---|---|
| **P** (permissive) | Commercial use allowed **and** derivatives redistributable. CC-BY, CC0, public domain, ODC-By, Apache/MIT. |
| **NC** (non-commercial) | CC BY-NC, "research use only", "free for non-commercial", bespoke academic agreements, or **anything ambiguous**. |

Ambiguity always resolves to NC. Getting this wrong contaminates every downstream model, and the contamination is invisible once training has started.

Some corpora carry a second restriction beyond commercial use: no redistribution of the raw data, attribution requirements, or a click-through agreement. Record these separately in the datasheet. They govern what may be committed to this public repository, which is a different question from which track the corpus belongs to. **Raw corpus data is never committed** regardless; loaders download to a gitignored local cache.

If the licence cannot be determined, stop and ask. Do not proceed on an assumption.

## 2. Tier decision (blocking)

What kind of text is this, really? The project models ASR output, so text that was cleaned up, or never spoken, is a substitute and gets recorded as one.

| Tier | What | How to use it |
|---|---|---|
| **1** | Real ASR output, disfluencies and recognition errors intact | Directly, as training text |
| **2** | Audio we can run ASR over | Run ASR to produce tier 1. Record which system |
| **3** | Clean written text, or human-verbatim transcription | Only if 1 and 2 are unavailable and the corpus is necessary. Apply the channel model before training use |

Three things people get wrong here, so check each explicitly:

**Human-verbatim transcription is tier 3.** It is a record of speech, but a transcriber who silently repaired a false start removed the signal this project exists to model. Check the corpus documentation for what transcribers were told to do about disfluencies, and quote it in the datasheet rather than assuming.

**Look for an ASR release before settling for the human one.** Corpora that ship human transcripts often ship automatically-derived ones too, in a separate archive that is easy to miss. If both exist, the corpus is **tier 1**, and its human side becomes reference data for fitting the channel model rather than training text. Preferring the human side because it looks cleaner inverts the point.

**Audio changes the answer.** A corpus with audio is tier 2 even if its transcripts are human, because we can produce tier 1 ourselves. Do not file it as tier 3 just because the shipped text is clean.

Record the tier, the evidence for it, and for tier 3 an explicit justification of why nothing better was available.

## 3. Loader

One module per corpus in `data/loaders/`, emitting the pydantic types from `data/schema.py`. Nothing downstream ever sees a corpus-specific format.

The loader is responsible for normalising into the canonical form:

- **Speaker-labelled turns, one turn per line**, with a stable line index. The line index is what evidence labels refer to, so it must be deterministic across runs.
- Speaker roles mapped onto the project's role vocabulary where the corpus supports it, otherwise left as opaque speaker ids.
- Verbatim text preserved. **Do not strip disfluencies, fillers, repetitions or repairs.** They are signal here, not noise.
- Where a corpus ships more than one transcript layer, the loader exposes a `variant` parameter rather than silently picking one, and defaults to the highest tier available. A loader that hard-codes the human transcript when an ASR layer exists is a bug even though nothing downstream will complain.
- Declared `track` field on every record, so a mixed-track batch is detectable at runtime rather than discovered in a results table.

Download to a local cache directory that is gitignored. Loaders must be idempotent and resumable; several of these corpora are large.

## 4. Datasheet entry

Append to `data/DATASHEET.md`. Every entry carries:

source and citation; licence with the URL you verified it from; **track**; **tier, with the evidence for it**; redistribution restrictions; size (documents, turns, approximate tokens); whether transcripts are human-verbatim or ASR output, and if ASR, which system and what era; whether audio is available; what preprocessing the loader applies; and known quality problems.

The "human-verbatim vs ASR output" field matters more than it looks: it decides the tier, and therefore whether the text can be trained on directly or has to go through the channel model first. Corpora shipping **both** are doubly valuable, since they are tier 1 *and* the only thing that can calibrate the channel, so flag them prominently.

**Record the token-length distribution**, not just a total: min, p50, p90, p95, p99 and max over rendered transcripts, using the project tokenizer and `Transcript.render()` so the numbers describe what a model actually receives. A single total hides the shape, and the shape is what decides the benchmark's length buckets, the training sequence length, and how much of the compute budget the long tail consumes. Note explicitly if the distribution is multi-modal, which is common when a corpus mixes short and long recordings, since a p50 in a valley between two modes describes nothing.

## 5. Test

A round-trip test in `tests/` against a small committed fixture (a handful of records, licence permitting, otherwise synthetic records in the same shape). Assert the canonical schema validates, line indices are contiguous from the expected base, speaker labels are populated, and the track field is set.

## 6. Verify before declaring done

- Print a few records and read them. Check that turns are segmented as expected and disfluencies survived.
- Compare the token count against what the source claims. A large discrepancy means the loader is dropping something.
- Confirm nothing under the raw cache path is tracked by git.
- Check the length distribution against the datasheet entry and sanity-check both tails: a p99 far above p95 means the tail is worth designing for, and a minimum near zero usually means empty records the loader should drop rather than emit.
