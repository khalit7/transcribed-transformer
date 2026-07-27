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

No corpora ingested yet. Entries below are the planned set, recorded with the licence position established during design. **Each still needs its licence verified against the primary source, its loader written, and its actual token count measured** before it counts as ingested.

## Planned — Track P

### AMI Meeting Corpus (+ ICSI)
- **Source**: https://groups.inf.ed.ac.uk/ami/corpus/
- **Licence**: CC-BY (to verify against primary source)
- **Track**: P
- **Why**: verbatim spontaneous multi-party speech, disfluencies preserved
- **Audio available**: yes — usable for ASR channel-model fitting
- **Transcripts**: human verbatim
- **Size**: ~100h (AMI). Token count: TBD
- **Status**: not ingested

### AnnoMI
- **Source**: https://github.com/uccollab/AnnoMI
- **Licence**: public domain (to verify)
- **Track**: P
- **Why**: expert-annotated counselling dialogues; closest public proxy for disclosure of health and life-event difficulty
- **Audio available**: source videos are public; transcripts are the released artefact
- **Transcripts**: human, expert-annotated per utterance
- **Size**: 133 transcripts, ~9.7k utterances. Token count: TBD
- **Status**: not ingested

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

## Planned — Track NC

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
