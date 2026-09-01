# Synthsheet — synthetic transcript recipes

How clean text becomes realistic ASR-style transcript text, and how that is quality-gated. The datasets themselves are recorded in [`DATASHEET.md`](DATASHEET.md) (a synthetic corpus gets a row there once it enters a training mixture; the recipe stays here); acquisition decisions live in [`SURVEYSHEET.md`](SURVEYSHEET.md).

## 1. Why synthesise at all

The distribution this project models is modern commercial ASR over conversational telephone speech. Real permissive examples of that are scarce (see the survey), while good permissive *written* dialogue and *human-verbatim* transcripts are plentiful. Two routes bridge the gap:

- **Route A — text channel model**: fit an error model from corpora that pair human-verbatim reference with ASR hypothesis for the same speech, then apply it to clean text. Cheap, scales to anything written.
- **Route B — TTS→ASR**: speak the text with open-source TTS, degrade to telephone band, and run a real recogniser over it. More expensive per token, but the recognition errors are real rather than sampled, which matters most for the things ASR mangles in structured ways: names, numbers, spelled-out identifiers.

## 2. What the channel must reproduce

The target is the output of current commercial transcription pipelines, whose behaviour is publicly documented and observable on any sample: automatic punctuation and casing restoration (with characteristic errors, e.g. spurious capitals after commas), diarised `SPEAKER_NN` turns with attribution errors in both directions (one person's speech split across two turns; two people merged into one), turns that end mid-sentence, in-band non-speech markers (`<inaudible>`, `<laugh>`, `<affirmative>`, `<crosstalk>`) with inconsistent casing, a vendor filler lexicon (typically American `um`/`uh` regardless of the speaker's dialect), and out-of-vocabulary proper nouns decoded as phonetically-near common words. A channel that only substitutes, deletes and inserts words reproduces none of the structural part of that, so the surface layer below is part of the recipe, not a garnish.

## 3. Channel artifacts

| Artifact | File | Fitted from | State |
|---|---|---|---|
| `channel_v2` | TBD | self-ASR (Whisper large-v3, Parakeet) over AppTek and PriMock57 audio against their verbatim transcripts, per-system and pooled; ACI-Bench's shipped ASR/human pairs join as a third reference | in progress — AppTek audio downloading (2026-09-01), transcription pipeline in `scripts/transcribe.py`; ACI-Bench pairs already on disk |

There is no usable v1: a pre-reset channel fitted from AMI's 2007 ASR/manual pairing was discarded on 2026-09-01, because a 2007 recogniser without punctuation models nothing about the modern target and it will not be used anywhere. AMI's pairing stays excluded from fitting for the same reason. v2 starts the version history.

Fitting requires word-level alignment between reference and hypothesis streams, windowed by time rather than by turn, because ASR turn segmentation is finer than a human transcriber's. Artifacts are versioned as wandb artifacts when runs consume them.

## 4. Route A recipe (channel model)

1. **Lexical layer** (fit from paired corpora): substitution confusions, deletions, insertions, with severity as a scalar so the channel can be calibrated down from the fitted WER to a modern recogniser's; filler and repetition injection fit from tier 1 corpora, never from tier 3 (transcribers repair disfluencies, so tier 3 densities are lower bounds).
2. **Surface layer** (deterministic + sampled): punctuation/casing restoration errors, `um`/`uh` filler lexicon, in-band marker emission with casing noise, speaker split/merge at fitted rates, mid-sentence turn cuts, occasional malformed speaker tags (missing space or colon).
3. **Determinism**: seeded; the channel version is stamped on every synthetic document and carried through to pack `meta.json`.

## 5. Route B recipe (TTS→ASR)

- **TTS**: Dia (Apache-2.0, dialogue-native speaker tags and non-verbal sounds) as primary; Chatterbox-Turbo (MIT) as the fast volume workhorse; Orpheus-3B / MOSS-TTSD / Qwen3-TTS (Apache-2.0) as supplements for tag-driven disfluency and speaker diversity. Not used for Track P outputs: XTTS-v2 (non-commercial), official F5-TTS weights (non-commercial training-data taint), Higgs Audio (Llama licence).
- **Degradation before recognition** (the step that makes it realistic): downsample to 8 kHz telephone band, codec round-trip (AMR-NB/GSM via ffmpeg/sox), additive noise (MUSAN) at varied SNR, cross-talk overlap.
- **Recognition**: Whisper large-v3 and Parakeet, mixed across the corpus, the system logged per document. Known failure mode from the literature: clean TTS into a strong recogniser yields near-zero WER, i.e. unrealistically clean text; degradation plus scripted disfluencies is what prevents that.
- **Scripts**: written dialogue corpora (survey table), plus generated **identifier-dictation episodes**: names spelled letter-by-letter with restarts, phone/account/reference numbers read digit-by-digit (`double`/`treble` conventions), postcodes with phonetic-alphabet codewords, values fragmented across turns and speakers. No public corpus carries this property; speaking such scripts through Route B produces real recognition errors on exactly the token types ASR handles worst. This also feeds any later anonymisation work, since spoken-form identifiers are what standard NER never sees.

## 6. QC gates

A synthetic corpus enters a training mixture only after both gates:

1. **Distributional table**: filler rates, repetition rates, turn-length distribution, marker frequencies, punctuation profile of the synthetic output side-by-side with a held-out tier 1 corpus. Gross mismatches block.
2. **Adversarial discriminator**: a simple classifier trained to separate synthetic from real ASR text must struggle — target AUC ≤ ~0.65 on held-out data. If the discriminator wins, the channel has failed and every phase depending on it blocks.

## 7. Where the output goes

Synthetic documents land in `data/interim/<source>-synth/` with the channel or pipeline version stamped per document, get a DATASHEET row when first mixed, and are packed like any other source. Tracks are inherited from the source text; TTS/ASR tooling licences are recorded alongside.

## 8. State

The self-ASR pipeline is live: `scripts/transcribe.py` (faster-whisper large-v3, float16, word timestamps; telephone degradation per §5 minus codec/noise, which wait on ffmpeg). Measured 2026-09-01 on one RTX 5090: **20x realtime** on a smoke-test call. First production pass (AppTek diarization config, 873 calls) running; the split-channel pass and the channel-v2 fit follow once the AppTek `test` audio finishes downloading. The TTS route is not started. Nothing on this sheet has produced *training* data yet.

Keep this file current when a recipe, model choice, or QC gate changes.
