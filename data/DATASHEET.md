# Datasheet

Every corpus used in this project is recorded here before it is used. Entries are added by the `add-corpus` skill, which will not proceed without a licence decision.

## Licence tracks

| Track | Criteria | Consequence |
|---|---|---|
| **P** — permissive | Commercial use allowed **and** derivatives redistributable | Models trained on Track P only are commercially portable |
| **NC** — non-commercial | CC BY-NC, research-only, unclear, or bespoke academic agreement | Any model touching this data is research-only |

Ambiguity resolves to **NC**. Tracks are never mixed within a training run. Every headline result is reported on both tracks.

Raw corpus data is never committed to this repository. Loaders download to a gitignored local cache.

## Data tiers

Licence says whether we *may* use a corpus. Tier says how close it is to the distribution this project actually models, which is ASR output.

| Tier | What | How it is used |
|---|---|---|
| **1** | Real ASR output, disfluencies and recognition errors intact | Directly, as training text |
| **2** | Audio available, so we can produce tier 1 ourselves | Run ASR over it; the system used is recorded |
| **3** | Clean written text, or human-verbatim transcription | Channel model applied first. Requires a justification for why nothing better was available |

**Human-verbatim transcription is tier 3, not tier 1.** A transcriber who silently repairs a false start has removed the signal this project exists to model. A corpus shipping both a human and an ASR transcript is tier 1, and its human side becomes reference data for fitting the channel rather than training text.

## Status

**Ingested: AMI, Taskmaster-1+2** (licence verified at source, loader written and tested, token count measured).

Combined Track P ingested total: **9,777,191 tokens** across 22,978 transcripts. The two are complementary rather than additive: AMI is long multi-party meetings (p50 9,630 tokens), Taskmaster is short dyadic service calls (p50 321). Neither covers long-context *dyadic* conversation, which remains the gap on this track.

Everything else below is the planned set, recorded with the licence position established during design. **Each still needs its licence verified against the primary source, its loader written, and its actual token count measured** before it counts as ingested. The AnnoMI entry is what that verification looks like when it fails.

## Track P

### AMI Meeting Corpus
- **Tier**: **1.** AMI ships a second archive, [`ami_public_auto_1.5.1.zip`](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_auto_1.5.1.zip) (68MB, also CC BY 4.0), containing real ASR output under `ASR/ASR_AS_CTM_v1.0_feb07/` — 664 word-level hypothesis files with timings, named `{meeting}.{speaker}.words.xml` exactly as the manual ones are. Its README states: *"if you want to use the alignment between ASR and manual transcription you need the manual data unzipped into the same directory."* So AMI is a **paired** corpus: ASR output for training, human transcript as channel-fitting reference.
- **Correction, 2026-07-29**: this entry previously described only the manual annotations, and the loader read only those. Under the data-tier rule that was the wrong layer — human-verbatim transcription is tier 3, and a tier 1 layer was sitting unopened in the same distribution. The loader gains a `variant` parameter and the ASR layer becomes the default for training text.
- **ASR system**: `ASR_AS_CTM_v1.0_feb07`, the AMI-ASR system circa **February 2007**. Its word error rate is far above a modern recogniser's, so a channel fitted on it without severity calibration will **over-noise** relative to the ASR systems this project is ultimately aimed at. That is a calibration problem, not a disqualifying one, and the adversarial validation gate is what catches it.
- **Source**: https://groups.inf.ed.ac.uk/ami/corpus/ — annotations from https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip (22MB, sha256 `b56e5bab…bbc9f9d`, pinned in the loader)
- **Licence**: **CC BY 4.0**, verified 2026-07-29 against the `LICENCE.txt` bundled *inside the annotation archive itself*, which is the most authoritative source available: *"The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."* The download page corroborates it for this release: *"annotations unchanged since 16-June 2014 release; license altered to CC BY 4.0"*. Commercial use permitted, derivatives redistributable.
- **Track**: P
- **Restrictions**: **attribution required** — this obligation propagates to anything trained on it and must appear in model cards, not just here. The corpus page scopes the licence as "all of the signals and transcription, and some of the annotations", so **the annotation layers are not uniformly covered**; only the orthographic transcription is used here, and that is covered. Raw data is not committed regardless.
- **Why**: verbatim spontaneous multi-party speech with disfluencies preserved
- **Audio available**: yes, though not downloaded by the loader. This is the only Track P corpus pairing audio with human verbatim transcripts, so it is the reference data for ASR channel-model fitting.
- **Transcripts**: **both**. Human verbatim orthographic (manual archive) *and* real ASR output (auto archive). This is what makes AMI the single most valuable corpus here despite its small size and wrong interaction shape.
- **Size measured 2026-07-29** (manual variant): 171 meetings, 83,868 turns, 5,685,219 characters, **1,674,833 tokens** (ModernBERT-large tokenizer, over `Transcript.render()`). ASR-variant token count: TBD, and expected to differ, since a recogniser drops and inserts words.
- **Token-length distribution per meeting**: min 1,375 / p25 6,319 / **p50 9,630** / p75 12,550 / p90 15,487 / p95 17,689 / p99 23,641 / max 29,605. **Unimodal**, single peak around 8–11k, right-skewed. Fraction exceeding 4k: 88.9%; 8k: **62.0%**; 16k: 8.8%; 32k: **0.0%**.
- **Consequences**: the whole corpus fits under 32k, so no meeting needs truncation or chunking. But 62% exceed 8k, so an 8k context window would truncate the majority of meetings — this corpus alone argues for the context-extension phase rather than making it optional.
- **Preprocessing the loader applies**: NXT per-speaker word streams resolved against per-speaker segment files, then merged across speakers by `(start time, speaker, segment id)` to produce one chronological turn sequence. Words kept verbatim including filled pauses, repetitions and truncations. Non-lexical annotation markup (`vocalsound`, `disfmarker`, `gap`, `transformerror`) is excluded from the text because it is annotation *about* speech rather than spoken words and no ASR system emits it; counts are recorded per transcript in `meta`. Corpus-wide excluded: 27,395 disfmarker, 27,073 vocalsound, 5,125 gap, 30 transformerror. Segments containing only markup are dropped rather than emitted as empty lines.
- **Known quality problems**: speaker roles are meeting roles (project manager, designer), not advisor/customer, so every turn is `Role.UNKNOWN` — the loader does not guess. 4–5 speakers per meeting, so this is multi-party rather than dyadic and is a **structural mismatch** with two-party advisor/customer calls; useful for disfluency and channel modelling, weaker as a task-shape proxy. One speaker in the archive has segments but no word stream and is skipped. Scenario meetings are role-played rather than genuine.
- **ICSI**: listed in earlier drafts alongside AMI. Not yet assessed, has its own licence, and needs its own entry before use.
- **Status**: **loader written and tested; full corpus parsed and measured.** Not yet used in any training run.

### Taskmaster-1 and Taskmaster-2
- **Tier**: **3.** Human transcription of recorded speech, and **no audio is released**, so it cannot be re-ASR'd into tier 1. Disfluencies were only partially preserved (transcribers *"sometimes corrected them"*), which is precisely the tier-3 failure mode. Channel model required before use as training text.
- **Why it is kept anyway**: tier governs its use as *text*, not its use as *structure*. Its interaction shape and advisor/customer role labels are what the benchmark needs, and neither is affected by transcript cleanliness.
- **Source**: https://github.com/google-research-datasets/Taskmaster
- **Licence**: **CC BY 4.0**, verified 2026-07-29 against each release's own README, which states it directly: *"made available under the Creative Commons Attribution 4.0 License."* Commercial use permitted, derivatives redistributable, attribution required.
- **Track**: P
- **Why**: **this is what closes the Track P dyadic gap.** Both are two-person *spoken* dialogue collected by Wizard of Oz, with crowdsourced workers as the customer and **trained call centre operators as the assistant**. That is the advisor/customer interaction shape, from real humans talking, at a scale nothing else on Track P offers.
- **Size measured 2026-07-29**: **22,807 dialogues, 475,398 turns, 8,102,358 tokens** (ModernBERT-large over `Transcript.render()`). Split: tm1-woz 5,503; tm2 flights 2,481, food-ordering 1,050, hotels 2,357, movies 3,056, music 1,603, restaurant-search 3,276, sports 3,481. The source README claims 5,507 spoken for TM-1 and 17,289 for TM-2; the measured figures are 5,503 (four dialogues contain no utterances at all and are dropped) and 17,304. Measured numbers are used here.
- **Roles**: `ASSISTANT` → `Role.ADVISOR`, `USER` → `Role.CUSTOMER`. Measured 257,480 advisor turns, 217,863 customer turns, 55 unknown. **The first corpus here that populates roles at all**, which matters for any question whose answer depends on who said something.
- **Token-length distribution per dialogue**: min 7 / p25 240 / **p50 321** / p75 439 / p90 564 / p95 640 / p99 828 / max 2,389. Unimodal, tight, right-skewed. Fraction exceeding 512: 15.0%; 1,024: **0.3%**; 2,048: **0.0%**.
- **Consequence, and it is the important one**: these dialogues are **short**, and their length range is almost **disjoint from AMI's** (7–2,389 here against 1,375–29,605 there). So Track P now has dyadic conversation, but only at the short end; there is still **no permissively-licensed dyadic corpus at the long context this project targets**. The constructive reading is that this is what makes multi-call *case* construction possible: a realistic case is several calls, and concatenating short real dialogues reaches case-scale length without truncating anything or inventing filler.
- **Domains**: TM-1 pizza ordering, auto repair, ride service, movie tickets, coffee, restaurant reservations. TM-2 restaurants, food ordering, movies, hotels, flights, music, sports.
- **Audio available**: **no.** Transcripts only. So this cannot serve as ASR channel-fitting reference data, which is why it does not on its own replace AMI or HarperValleyBank.
- **Transcripts**: human transcription of recorded speech. Disfluencies are **partially** preserved — the TM-1 README says they were *"usually transcribed as spoken, but sometimes transcribers corrected them."* Treat disfluency density here as a **lower bound**, and do not fit channel-model statistics on it.
- **Preprocessing the loader applies**: only the spoken files are downloaded; `self-dialogs.json` is never fetched and `load_file` refuses a path matching it. Turn indices are rebuilt from position rather than trusting the corpus `index` field. Blank-text utterances are dropped rather than emitted as empty lines; dialogues left with no turns are dropped entirely.
- **Known quality problems**: task-oriented consumer service, not financial advice, so the domain is wrong even though the interaction shape is right. Wizard of Oz means the assistant knew they were role-playing. **55 utterances carry real text under an empty speaker label** — these keep their text and are marked `UNKNOWN` rather than dropped, since dropping would renumber every later line. Four TM-1 dialogues are empty. The written self-dialogue half of TM-1 is the live trap: same record schema, typed prose, and nothing downstream would detect the substitution.
- **Status**: **ingested.** Loader written and tested (10 tests), full corpus parsed and measured. Not yet used in any training run.

### HarperValleyBank
- **Tier**: **2.** Ships per-speaker audio, so we can run ASR over it ourselves and produce tier 1 dyadic banking text. The shipped `human_transcript` is tier 3 and becomes a second channel-fitting reference point, giving a modern recogniser to set against AMI's 2007 one.
- **Source**: https://github.com/cricketclub/gridspace-stanford-harper-valley — paper [arXiv:2010.13929](https://arxiv.org/abs/2010.13929)
- **Licence**: **CC BY 4.0**, verified 2026-07-29 by reading the `LICENSE` file in the repository itself, not the GitHub API's detected label. Commercial use permitted, derivatives redistributable, attribution required. The paper separately describes it as *"a free, public domain spoken dialog corpus"*; the file in the repository is the operative statement and it is CC BY 4.0. Both are permissive, so the track is P either way.
- **Track**: P
- **Why**: the only Track P corpus that is dyadic, spoken, **in a consumer banking domain**, and **ships audio**. It is the closest public analogue on the permissive track to the target task, and the only permissive source that can calibrate an ASR channel model on two-party call audio rather than meeting-room audio.
- **Size (as stated by the source)**: **1,446 conversations, ~23 hours of audio, 59 speakers**, vocabulary about 700 unique words. Token count: TBD.
- **Audio available**: **yes**, per-speaker single-channel WAV, alongside human transcripts. This is what makes it valuable out of proportion to its size.
- **Transcripts**: human, in a `human_transcript` field, with word-level timings and, unusually, a **`speaker_role` field distinguishing agent from caller**. This is the first corpus here that can populate `Role.ADVISOR` / `Role.CUSTOMER` rather than leaving every turn `UNKNOWN`.
- **Known quality problems, and they are serious**: the calls are **simulated**, produced for a Stanford course by speakers following assigned scripts across eight task types. The ~700-word vocabulary is the tell: this is templated speech, not spontaneous conversation. **Use it for acoustics, role structure and channel fitting; do not use it as a language-modelling corpus and do not let its lexical statistics into anything measuring disfluency naturalness.**
- **Status**: **licence decided, not ingested.**

### CFPB Consumer Complaint Database
- **Tier**: **3**, and the weakest kind: written narratives that were never spoken at all. Usable only with the channel model applied, and even then it models *written complaint prose passed through a recogniser*, which nobody ever says out loud. Justification: it is the only large public source of real financial-complaint content, and complaint identification is one of the three priority question families.
- **Source**: https://www.consumerfinance.gov/data-research/consumer-complaints/
- **Licence**: US government public domain (to verify)
- **Track**: P
- **Why**: real financial complaint narratives with product taxonomy and company response outcome
- **Audio available**: no
- **Transcripts**: **written, not spoken** — requires the ASR channel model before use as transcript-like text
- **Size**: ~2M records with narratives. Token count: TBD
- **Status**: not ingested

### Earnings-21 / Earnings-22
- **Tier**: **2.** Audio available, so ASR can be run over it. Worth revisiting if the channel model needs more reference data than AMI provides.
- **Source**: https://arxiv.org/abs/2203.15591
- **Licence**: free-to-use (to verify)
- **Track**: P
- **Why**: financial spoken English, accent diversity
- **Audio available**: yes — usable for channel-model fitting
- **Transcripts**: human
- **Size**: 119h (Earnings-22). Token count: TBD
- **Status**: not ingested

### FineWeb-Edu / Ettin open corpus
- **Tier**: **3.** Written web text, never spoken. Justification: Arm E needs 3B tokens per corpus and all the real ASR text in existence under a permissive licence is roughly 300x short of that, so there is no tier 1 or 2 route to this arm. E1 uses it clean and E2 uses the identical documents channel-noised, which is what makes the comparison controlled.
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
- **Tier**: **3** as distributed (human transcription). Nominally tier 2 since the source videos are public, but the underlying-video rights problem below means we cannot use the audio either, so the tier is moot until the licence question is resolved.
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
- **Tier**: **unresolved, and now blocking.** Whether these transcripts are human or ASR decides whether the corpus is tier 1 (the closest public analogue to the target distribution, usable directly) or tier 3 (needing the channel model). Resolve before writing the loader.
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
- **Tier**: **2.** Audio available. Note the transcripts are *fully formatted* professional output, which is further from ASR than ordinary human-verbatim text, so the human side is a poor channel reference even though the audio is a good tier 1 source.
- **Source**: https://arxiv.org/abs/2104.02014, https://arxiv.org/abs/2508.05554
- **Licence**: free for non-commercial use
- **Track**: NC
- **Why**: professionally transcribed financial speech at scale
- **Audio available**: yes — usable for channel-model fitting, but only for NC-track outputs
- **Transcripts**: professional human, fully formatted
- **Size**: 5,000h + 3,780h. Token count: TBD
- **Status**: not ingested

### MediaSum
- **Tier**: **3.** Interview transcripts, human-produced. Used for summary supervision on Arm B rather than as distribution-matched training text, so the tier matters less here than usual.
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
- **Tier**: **TBD**, with the licence. Resolve both together.
- **Source**: to confirm
- **Licence**: research use (to verify)
- **Track**: NC
- **Why**: human-agent phone dialogues with breakdown labels, as a dissatisfaction proxy
- **Audio available**: no
- **Size**: 13,524 dialogues. Token count: TBD
- **Status**: not ingested
