# Datasheet

Every corpus used in this project is recorded here before it is used. Entries are added by the `add-corpus` skill, which will not proceed without a licence decision.

## Licence tracks

| Track | Criteria | Consequence |
|---|---|---|
| **P** — permissive | Commercial use allowed **and** derivatives redistributable | Models trained on Track P only are commercially portable |
| **NC** — non-commercial | CC BY-NC, research-only, unclear, or bespoke academic agreement | Any model touching this data is research-only |

Ambiguity resolves to **NC**. Tracks are never mixed within a training run. Every headline result is reported on both tracks.

Raw corpus data is never committed to this repository. Loaders download to a gitignored local cache.

## Status

**Ingested: AMI** (licence verified at source, loader written and tested, token count measured).

Everything else below is the planned set, recorded with the licence position established during design. **Each still needs its licence verified against the primary source, its loader written, and its actual token count measured** before it counts as ingested. The AnnoMI entry is what that verification looks like when it fails.

## Track P

### AMI Meeting Corpus
- **Source**: https://groups.inf.ed.ac.uk/ami/corpus/ — annotations from https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip (22MB, sha256 `b56e5bab…bbc9f9d`, pinned in the loader)
- **Licence**: **CC BY 4.0**, verified 2026-07-29 against the `LICENCE.txt` bundled *inside the annotation archive itself*, which is the most authoritative source available: *"The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."* The download page corroborates it for this release: *"annotations unchanged since 16-June 2014 release; license altered to CC BY 4.0"*. Commercial use permitted, derivatives redistributable.
- **Track**: P
- **Restrictions**: **attribution required** — this obligation propagates to anything trained on it and must appear in model cards, not just here. The corpus page scopes the licence as "all of the signals and transcription, and some of the annotations", so **the annotation layers are not uniformly covered**; only the orthographic transcription is used here, and that is covered. Raw data is not committed regardless.
- **Why**: verbatim spontaneous multi-party speech with disfluencies preserved
- **Audio available**: yes, though not downloaded by the loader. This is the only Track P corpus pairing audio with human verbatim transcripts, so it is the reference data for ASR channel-model fitting.
- **Transcripts**: human verbatim, orthographic. Not ASR.
- **Size measured 2026-07-29**: 171 meetings, 83,868 turns, 5,685,219 characters, **1,674,833 tokens** (ModernBERT-large tokenizer, over `Transcript.render()`).
- **Token-length distribution per meeting**: min 1,375 / p25 6,319 / **p50 9,630** / p75 12,550 / p90 15,487 / p95 17,689 / p99 23,641 / max 29,605. **Unimodal**, single peak around 8–11k, right-skewed. Fraction exceeding 4k: 88.9%; 8k: **62.0%**; 16k: 8.8%; 32k: **0.0%**.
- **Consequences**: the whole corpus fits under 32k, so no meeting needs truncation or chunking. But 62% exceed 8k, so an 8k context window would truncate the majority of meetings — this corpus alone argues for the context-extension phase rather than making it optional.
- **Preprocessing the loader applies**: NXT per-speaker word streams resolved against per-speaker segment files, then merged across speakers by `(start time, speaker, segment id)` to produce one chronological turn sequence. Words kept verbatim including filled pauses, repetitions and truncations. Non-lexical annotation markup (`vocalsound`, `disfmarker`, `gap`, `transformerror`) is excluded from the text because it is annotation *about* speech rather than spoken words and no ASR system emits it; counts are recorded per transcript in `meta`. Corpus-wide excluded: 27,395 disfmarker, 27,073 vocalsound, 5,125 gap, 30 transformerror. Segments containing only markup are dropped rather than emitted as empty lines.
- **Known quality problems**: speaker roles are meeting roles (project manager, designer), not advisor/customer, so every turn is `Role.UNKNOWN` — the loader does not guess. 4–5 speakers per meeting, so this is multi-party rather than dyadic and is a **structural mismatch** with two-party advisor/customer calls; useful for disfluency and channel modelling, weaker as a task-shape proxy. One speaker in the archive has segments but no word stream and is skipped. Scenario meetings are role-played rather than genuine.
- **ICSI**: listed in earlier drafts alongside AMI. Not yet assessed, has its own licence, and needs its own entry before use.
- **Status**: **loader written and tested; full corpus parsed and measured.** Not yet used in any training run.

### CFPB Consumer Complaint Database
- **Source**: https://www.consumerfinance.gov/data-research/consumer-complaints/
- **Licence**: US government public domain (to verify)
- **Track**: P
- **Why**: real financial complaint narratives with product taxonomy and company response outcome
- **Audio available**: no
- **Transcripts**: **written, not spoken** — requires the ASR channel model before use as transcript-like text
- **Size**: ~2M records with narratives. Token count: TBD
- **Status**: not ingested

### Earnings-21 / Earnings-22
- **Source**: https://arxiv.org/abs/2203.15591
- **Licence**: free-to-use (to verify)
- **Track**: P
- **Why**: financial spoken English, accent diversity
- **Audio available**: yes — usable for channel-model fitting
- **Transcripts**: human
- **Size**: 119h (Earnings-22). Token count: TBD
- **Status**: not ingested

### FineWeb-Edu / Ettin open corpus
- **Source**: HuggingFace
- **Licence**: ODC-By (to verify)
- **Track**: P
- **Why**: generic web text for the arm E control corpus
- **Audio available**: no
- **Transcripts**: n/a, written text
- **Size**: sampled to match the transcript corpus token count exactly (arm E requires token matching)
- **Status**: not ingested

## Track NC

### AnnoMI
- **Source**: https://github.com/uccollab/AnnoMI
- **Licence**: **none stated — undeterminable.** Checked 2026-07-29: the official repository has no `LICENSE` file, the GitHub API reports `license: None`, and the README states no terms of use, only a request to cite two papers. The Future Internet 2023 article (DOI [10.3390/fi15030110](https://doi.org/10.3390/fi15030110)) is Gold OA under CC BY, but **that licenses the article, not the separately-distributed dataset**.
- **Track**: **NC** — by the ambiguity rule, not by a positive non-commercial licence.
- **Second, independent problem**: the transcripts are of third-party motivational-interviewing demonstration videos that the AnnoMI authors did not create. They can license their own annotations; they cannot unilaterally license the underlying spoken content. So even an explicit CC BY on the repository would not by itself make the transcript text commercially portable, which is what Track P is meant to guarantee.
- **Third-party claims are not evidence**: a HuggingFace mirror tags the data `openrail`. That is not the authors' release, and OpenRAIL is use-restricted, so it would fail the Track P test even if it were authoritative.
- **Why it still matters**: expert-annotated counselling dialogues, the closest public proxy for disclosure of health and life-event difficulty. Losing it from Track P weakens the commercially-portable track precisely on **vulnerability**, one of the three priority question families. That cost is real and is recorded here rather than absorbed quietly.
- **Route to Track P**: an explicit licence statement from the authors covering the dataset, *plus* a resolution of the underlying-video rights. Both would be needed; the first alone is not sufficient.
- **Audio available**: source videos are public; transcripts are the released artefact
- **Transcripts**: human, expert-annotated per utterance
- **Size**: 133 transcripts, ~9.7k utterances. Token count: TBD
- **Status**: not ingested

### CallCenterEN
- **Source**: https://arxiv.org/abs/2507.02958 — `AIxBlock/92k-real-world-call-center-scripts-english`
- **Licence**: **CC BY-NC 4.0**
- **Track**: NC
- **Why**: by far the closest public analogue to the target distribution
- **Audio available**: no (transcripts released)
- **Transcripts**: PII-redacted; whether human or ASR is **unconfirmed** and needs checking, since it determines whether this can serve as channel-fitting reference data
- **Size**: 91,706 conversations / 10,448 audio hours. Token count: TBD
- **Notes**: already PII-redacted, which interacts with the anonymisation head — redaction artefacts must not be learned as PII patterns
- **Status**: not ingested

### SPGISpeech 1.0 + 2.0
- **Source**: https://arxiv.org/abs/2104.02014, https://arxiv.org/abs/2508.05554
- **Licence**: free for non-commercial use
- **Track**: NC
- **Why**: professionally transcribed financial speech at scale
- **Audio available**: yes — usable for channel-model fitting, but only for NC-track outputs
- **Transcripts**: professional human, fully formatted
- **Size**: 5,000h + 3,780h. Token count: TBD
- **Status**: not ingested

### MediaSum
- **Source**: https://arxiv.org/abs/2103.06410
- **Licence**: research use only
- **Track**: NC
- **Why**: the only large transcript-to-summary corpus; trains the summary head
- **Audio available**: no
- **Transcripts**: broadcast interview transcripts
- **Size**: 463.6K transcripts with summaries. Token count: TBD
- **Notes**: summaries are topic descriptions, not justifications of a verdict. Useful for summary-head pretraining, **not** a substitute for task-specific summary supervision
- **Status**: not ingested

### BETOLD
- **Source**: to confirm
- **Licence**: research use (to verify)
- **Track**: NC
- **Why**: human-agent phone dialogues with breakdown labels, as a dissatisfaction proxy
- **Audio available**: no
- **Size**: 13,524 dialogues. Token count: TBD
- **Status**: not ingested
