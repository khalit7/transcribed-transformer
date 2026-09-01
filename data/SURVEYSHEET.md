# Surveysheet — corpus survey and acquisition decisions

Every corpus considered for this project, including the ones refused, and why. A source only earns a [`DATASHEET.md`](DATASHEET.md) row once it is on disk; synthesis recipes live in [`SYNTHSHEET.md`](SYNTHSHEET.md). Survey conducted 2026-08-31 (two structured web sweeps: pretraining/long-context sources, and benchmark/annotation sources), on top of the design-time candidate list.

## Policy — how corpora are assessed

- **Licence verified at the primary source** (a LICENSE file in the distribution, the corpus's own page), never an aggregator or mirror tag. Ambiguity resolves to Track NC. Share-alike (CC BY-SA, CDLA-Sharing) is Track P with the SA flag. Free sources only; paid or member-gated corpora (LDC and similar) are out of scope.
- **Tier before beauty**: real ASR text > audio we can transcribe ourselves (2× RTX 5090; Whisper large-v3 and Parakeet, system recorded per corpus) > clean text needing the channel model. Human-verbatim transcription is tier 3.
- **What the target distribution needs** (all justifiable from how commercial diarised ASR behaves and what regulators describe): spontaneous conversational telephone-register speech, not read or scripted; verbatim or ASR transcripts with disfluencies intact; diarised speaker turns, roles not necessarily labelled and not reliably two-party (regulatory guidance on vulnerable customers expressly anticipates carers and relatives joining calls); long-form sessions rather than utterance clips; British English preferred; spoken identifiers (numbers dictated digit-by-digit, names spelled aloud) are the rarest and most valuable property. Disqualifying for benchmark case text: read speech, cleaned-up transcripts, utterance-level clips without conversational context, LLM-generated dialogue.
- **Transcription-convention fingerprint**: record each corpus's filler convention (`um`/`uh` vs `erm`/`er`), since it identifies the transcription lexicon and mixing conventions silently is a confound.
- **Third-party ASR is recorded, ours is preferred**: where a source ships someone else's ASR output, it is usable tier 1, but for bulk training text one locally-run recogniser is preferred for uniformity, with the shipped transcripts kept as a cross-recogniser reference.
- **YouTube-derived "CC BY" is uploader-asserted.** Keep per-video licence metadata, honour takedowns, and record the residual risk in the datasheet entry.
- **Dedupe by source URL** across the YouTube-derived corpora (YODAS2, Emilia-YODAS, YouTube-Commons, Granary) and between People's Speech and anything harvested from courts or councils.

## The survey

`is_ingested`: yes | queued (in the ingestion queue below) | no (with reason). Licence marked `?` where not yet verified at the primary source; verification happens at ingestion, before download.

### Long-context and conversational case text

| Corpus | Size | Track | Tier | Shape | is_ingested | Notes / decision |
|---|---|---|---|---|---|---|
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) | ~100h, 1.3M tok ASR | P (CC-BY-4.0) | 1 | 4–5 speaker meetings | **yes** | Paired ASR + verbatim; the channel-fitting anchor |
| [Taskmaster-1+2](https://github.com/google-research-datasets/Taskmaster) | 22,807 dialogues, 8.1M tok | P (CC-BY-4.0) | 3 | dyadic spoken service, short | **yes** | Case-packing text with role labels; disfluencies partially repaired |
| [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | 128.6h, 873 calls | P (CC-BY-SA-4.0, verified on card 2026-08-31) | 2 | dyadic agent/customer, 16 scenarios, 14 accents | **yes** | Ingested 2026-08-31. Split-channel audio + verbatim transcripts + roles; eval + channel reference only (card states evaluation-only intent). SA flag |
| [PriMock57](https://github.com/babylonhealth/primock57) | 57 consultations | P (CC-BY-4.0, LICENSE.md read) | 2 | dyadic clinician/patient, UK | **queued (2)** | Mock GP consultations with audio; vulnerability (health) proxy + channel reference |
| [ACI-Bench](https://github.com/microsoft/clinical_visit_note_summarization_corpus) | 207 dialogues | P (CC-BY-4.0, LICENSE read 2026-08-31) | **1** | dyadic doctor/patient, long | **yes** | Ingested 2026-08-31. Better than surveyed: ships paired ASR / ASR-corrected / human transcripts of the same encounters → tier 1 + channel reference |
| [OANC Switchboard](https://anc.org/) | 2,320 calls, ~3M words | P (ANC terms: unrestricted incl. commercial) | 3 | dyadic phone, spontaneous, ~6 min | **queued (4)** | Real disfluency, no audio (audio is LDC-only). anc.org TLS currently broken: archive the licence page at download |
| [ICSI](https://groups.inf.ed.ac.uk/ami/icsi/) | ~72h | P (CC-BY-4.0, licence page) | 2 | multi-party meetings | **queued (5)** | More paired channel-fitting material alongside AMI |
| [CourtListener oral arguments](https://www.courtlistener.com/audio/) | ~53,000h, 102,131 transcribed recordings | P (public domain, bulk-data page) | 1+2 | advocate vs bench, 20–90 min | **partial** | SCOTUS complete 2026-08-31 (1,001 arguments, 11.15M words); federal appeals courts extending. Transcripts are flat text (no speaker turns) |
| [CANDOR](https://betterup-data-requests.herokuapp.com/) | 1,656 convs, 850h | NC (CC-BY-NC-4.0, application) | 1 | dyadic spontaneous, ~31 min | queued (NC) | The exact target shape; AWS Transcribe ASR. Apply early, review takes time |
| [SpokenWOZ](https://spokenwoz.github.io/) | 5.7k dialogues, 249h | NC (CC-BY-NC-4.0) | 1+2 | dyadic task calls, short | queued (NC) | Best NC spoken TOD; test set hidden |
| [MeetingBank](https://meetingbank.github.io/) | 1,366 meetings, 3,579h | NC (CC-BY-NC-SA-4.0) | 1 | council meetings, avg 2.6h (~28k tok) | **yes** | Ingested 2026-08-31 (transcripts only, 19.9M words). HF distribution is per agenda item, flat text; whole meetings reconstruct by uid grouping; speaker layer ships with the audio release, not fetched |
| [SSSD](https://wavlab-speech.github.io/SSSD) | 727h | NC presumed (CMU Flintbox, terms unpublished) | 2 | dyadic unscripted, 25–30 min | no — request terms | Either this or Hume-DaiKon flipping permissive would close the permissive long-dyadic gap; ask |
| Hume-DaiKon | 743h (481h EN) | NC presumed (challenge-gated) | 1 | dyadic video calls, ~47 min | no — request terms | As above |
| [SBCSAE](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus) | 60 recordings, ~249k words | NC (CC-BY-**ND**-3.0) | 2+3 | mixed, many dyadic, ~20 min | no | The near-miss: natural long dyadic conversation, blocked solely by ND (no derivatives) |
| [HarperValleyBank](https://github.com/cricketclub/gridspace-stanford-harper-valley) | 1,446 calls, ~23h | P (CC-BY-4.0, LICENSE read) | 2 | dyadic banking, scripted | no — demoted | Simulated, ~700-word vocabulary. Superseded by AppTek as the modern channel reference; revisit only if more telephone-band audio is needed |
| TalkBank (CABNC, CallHome/CallFriend, DementiaBank) | various | NC (blanket CC-BY-NC-SA, terms preclude LLM training) | 2/3 | conversational | no | Terms explicitly disallow this use |
| Spoken BNC2014 | 11.5M words | NC (registration, research-only) | 3 | UK conversation, `erm` convention | no | Wrong transcription convention for the target and research-only |
| [NPR interviews / MediaSum](https://github.com/zcgzcgzcg1/MediaSum) | 463.6K transcripts | NC (research only) | 3 | broadcast interviews, edited | queued (NC, low) | Summary supervision only; edited transcripts |
| [DailyTalk](https://github.com/keonlee9420/DailyTalk) | 20h | NC (derives from CC-BY-NC-SA DailyDialog) | 2 | dyadic acted | no | Acted readings of written dialogue |
| [MultiDialog](https://huggingface.co/datasets/IVLLab/MultiDialog) | 340h | NC (licence just "cc", unresolved) | 2 | dyadic acted TopicalChat | no | Acted scripts; licence ambiguous |

### Bulk pretraining speech and text

| Corpus | Size | Track | Tier | Shape | is_ingested | Notes / decision |
|---|---|---|---|---|---|---|
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 92.2k h EN | P (CC-BY-4.0 stated for the YODAS-derived portion) | 1+2 | podcasts/interviews, in-the-wild | queued (T2) | Largest permissive spontaneous pool with transcripts attached. Emilia-proper portion is CC-BY-NC: ingest the YODAS portion only. Click-through gate with indemnification language: record it. Uploader-asserted CC risk |
| [YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons) | ~45B words | P (CC-BY, aggregation of CC-BY videos) | 1 | mixed monologue/conversation | queued (T2) | Whisper transcripts at zero GPU cost; uploader-asserted CC risk; filter to conversational subsets |
| [FAMA](https://huggingface.co/collections/FBK-MT/fama) EN data | 14.2k h | P (CC-BY) | 1 | YouTube-Commons re-transcribed, Whisper large-v3 | queued (T2) | Higher-quality uniform slice of the above |
| [YODAS2](https://huggingface.co/datasets/espnet/yodas2) EN | large | P (CC-BY-3.0) | 2 | long-form YouTube | queued (T2, audio pool) | The audio pool for a uniform local ASR pass; dedupe against Emilia-YODAS and YouTube-Commons by URL |
| [People's Speech](https://mlcommons.org/datasets/peoples-speech/) (CC-BY/PD subset) | ~25k+ h | P (per-item; SA slice → P+SA) | 2 (shipped text is forced-aligned: 3) | interviews, proceedings, long | queued (T2) | Re-transcribe locally; rebuild long documents from utterance chunks; exclude or flag the SA slice |
| Congressional hearings (+ [govinfo CHRG](https://www.govinfo.gov/app/collection/chrg) transcripts) | thousands of h; 46,814 transcripts | P (US federal PD) | 2 (+3 reference) | multi-party hearings | queued (T2, engineering-heavy) | Harvest per committee; C-SPAN's own productions are NC, only official feeds are PD |
| FOMC press conferences ([federalreserve.gov](https://www.federalreserve.gov/)) | ~100–120h | P (site content PD per disclaimer) | 2+3 paired | press conference, financial register | queued (T2) | Paired video + verbatim transcript: clean financial-register channel-calibration set |
| White House press briefings | ~1,000–3,000h | P (US federal PD) | 2 | briefings, adversarial Q&A | no — reserve | Available if more PD audio is needed |
| [VoxPopuli](https://github.com/facebookresearch/voxpopuli) | large | P (CC0 / EP reuse) | 2 | parliamentary monologue | no | Formal read/prepared monologue, wrong shape |
| [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel) | 441k h pseudo-labels | P (CC-BY-4.0) | 1 | mixed | no — reserve | Whisper pseudo-labels over permissive audio; mostly non-conversational EN |
| LibriSpeech / Libri-Light / MLS | huge | P | 2 | read audiobooks | no | Read speech, disqualified as case/pretraining text; usable as clean-side calibration only |
| Loquacious Set | 25k h | P (assembled CC0/CC-BY) | 2 | utterance-chunked, half read | no | Wrong shape (clips, read speech) |
| [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) | 10k h | NC (Terms of Access: non-commercial; Apache tag covers code only) | 2 (text is forced-aligned human) | podcasts/YouTube | queued (NC, T3) | Commonly mislabelled permissive; it is not |
| [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) | 1.1M episodes | NC (gated research-only) | 1 | podcasts, diarised turns, long | queued (NC, T3) | Plenty of dyadic interview shows |
| [RadioTalk](https://github.com/social-machines/RadioTalk) | 2.8B words, 284k h | NC (no licence stated → NC) | 1 | talk radio incl. call-ins | queued (NC, T3) | Real Kaldi ASR with speaker turns and a phone/studio flag |
| [CFPB complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/) | ~2M narratives | P (US gov PD, verify) | 3 | written, never spoken | queued (T2, low) | Only large source of real financial-complaint content; channel model required |
| FineWeb-Edu | huge | P (ODC-By, verify) | 3 | web text | no — reserve | Was the from-scratch-ablation control; that arm is gone. Reserve for generic-text contrast experiments |
| [Earnings-21/22](https://github.com/revdotcom/speech-datasets) | 119h+ | P+SA (CC-BY-SA-4.0 **transcripts only**) | 3 | earnings calls | no | **Correction 2026-08-31**: the licence covers "the transcripts and associated text files"; audio rights were never granted, so the planned tier-2 use is void |
| [SPGISpeech](https://datasets.kensho.com/datasets/spgispeech) | 8,780h | NC (Kensho terms: non-commercial, **no derivatives/redistribution**) | 2 | earnings calls, formatted transcripts | no | **Correction 2026-08-31**: terms are stricter than plain NC and arguably incompatible with releasing even NC-track checkpoints |
| [CallCenterEN](https://arxiv.org/abs/2507.02958) (AIxBlock 92k) | 91,706 convs, 10,448h | NC (CC-BY-NC-4.0; **ungated**, contrary to earlier note) | **1** | agent/customer calls, ~1k words, no speaker turns | **yes** | Ingested 2026-08-31. AssemblyAI ASR confirmed; audio withheld, no tier-2 path. Flat text (speaker field null throughout); in-band PII placeholders must not be learned as PII patterns |
| BETOLD | 13,524 dialogues | P (Apache-2.0, repo LICENSE) | — | intent/entity sequences | **no — rejected** | **Correction 2026-08-31**: ships no utterance text and no audio at all, only NLU/NLG intent sequences with breakdown labels. Unusable as transcript text despite the permissive licence |

### Question-family proxies and annotation sources

| Corpus | Family | Size | Track | Tier | is_ingested | Notes / decision |
|---|---|---|---|---|---|---|
| [CompliBench](https://github.com/UCSB-NLP-Chang/CompliBench) | conduct | 318 dialogues | P (Apache-2.0 repo; data in-repo, no separate statement — confirm) | 3 (synthetic) | queued (T2) | Turn-level violation labels via controlled flaw injection; the labels-by-construction template, and the closest published analogue to this benchmark |
| [ABCD](https://github.com/asappresearch/abcd) | conduct | 10k dialogues | P (MIT) | 3 | queued (T2) | Explicit agent policy flows → procedure-adherence questions with labels by construction |
| [Doc2Dial / MultiDoc2Dial](https://github.com/IBM/multidoc2dial) | conduct, evidence | 4,796 convs | P (CC-BY-3.0) | 3 | queued (T2) | Per-turn grounding spans in government-service documents: evidence-supervision seed; benefits-advice register |
| [NatCS / DSTC11-T2](https://github.com/amazon-science/dstc11-track2-intent-induction) | conduct | ~5–7k dialogues | P (CDLA-Permissive-2.0 data licence in-repo) | 3 (spoken-style role-play) | queued (T2, low) | Permissive customer-support register; confirm which domains ship |
| [talkmap banking/telecom](https://huggingface.co/datasets/talkmap/banking-conversation-corpus) | filler | 500k convs | P (MIT) | 3 (synthetic) | no — reserve | Synthetic; filler only, never benchmark text |
| [TweetSumm](https://github.com/guyfe/Tweetsumm) | complaints, evidence | 1,100 dialogues | P+SA (CDLA-Sharing-1.0) | 3 (written) | queued (T2) | 3 extractive + 3 abstractive human summaries per dialogue; extractive = sentence-selection labels, a direct evidence-supervision seed |
| [TWCS](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | complaints | ~800k convs | P? (CC-BY-SA reported; **verify at Kaggle**) | 3 (written) | no — verify first | Real dissatisfied customers, brand agents |
| [USS](https://github.com/sunnweiwei/user-satisfaction-simulation) | complaints | 55k annotations | NC (no licence file) | — (annotations) | no — email authors | Per-utterance satisfaction labels over MultiWOZ/SGD/ReDial/CCPE; a licence statement would be a cheap Track P unlock |
| [EmoWOZ](https://huggingface.co/datasets/hhu-dsml/emowoz) | complaints | 83k labels | NC (CC-BY-NC-4.0) | 3 | queued (NC, T3) | Best dissatisfaction/abuse annotations |
| FINCORP / X-FinCORP | complaints | ~6.3k instances | NC (unclear) | 3 | no — lead | Financial complaint severity + causal spans |
| [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | vulnerability (health) | 1.7k snippets | **NC** (repo CC-BY-4.0, but underlying mtsamples.com terms grant no explicit commercial/derivative use → ambiguity rule) | 3 | no — on disk, not used | Arrives inside the ACI-Bench clone; downgraded 2026-08-31 after reading the repo NOTICE |
| [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) | vulnerability (life events, resilience) | 16k rows | P (MIT, **provenance flags**: clinical-trial-derived + GPT-generated halves) | 3 (Q&A shape) | no — reserve | Register source, not case text; record the provenance caveat if ever used |
| [AnnoMI](https://github.com/uccollab/AnnoMI) | vulnerability | 133 transcripts | NC (no licence stated; underlying-video rights unresolved) | 3 | queued (NC, low) | The 2026-07 verification that moved it from P to NC stands; route back to P requires both an author licence and video-rights resolution |
| [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) | vulnerability | 1,300 dialogues | NC (**"academic research use only"** — stricter than the CC-BY-NC it is usually reported as) | 3 | queued (NC, low) | FailedESConv's 196 failed-support conversations are a rare negative-example source for "was it handled properly" |
| HOPE / MEMO | vulnerability | 212 sessions | NC (agreement + email) | 3 (spoken-origin) | no — lead | Real counselling transcripts; highest NC value if access granted |
| [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) | vulnerability | 189 sessions | NC (EULA) | 2 | no — lead | Clinical distress interviews |
| [EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues) | vulnerability | 25k convs | NC (CC-BY-NC-4.0) | 3 | no | Short written situations |
| [CACTUS](https://github.com/coding-groot/cactus) | vulnerability | 31,577 dialogues | NC (GPL-2.0 applied to data → ambiguity rule) | 3 (synthetic) | no | Largest counselling set, but LLM-generated (disqualified as case text) and licence-awkward |
| [Counsel-Chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) | vulnerability | 2,775 Q&A | NC (MIT tag over scraped third-party content it cannot cover) | 3 | no | The tag is not the licence |
| Crisis Text Line, 7cups sets | vulnerability | — | — | — | no | Access-restricted, not obtainable |
| DBDC1–5 | conduct | ~1,950 dialogues | NC (registration) | 3 | no | Human–bot breakdown labels; wrong interaction kind |
| FED | conduct | 125 dialogues | NC (unverified) | 3 | no | Too small, human–bot |

No public corpus was found that combines conversational telephone speech with dictated identifiers (numbers read digit-by-digit, names spelled aloud); that property is synthesised instead, per [`SYNTHSHEET.md`](SYNTHSHEET.md). The FCA capability driver likewise has no permissive corpus; vulnerability questions on that driver rely on synthesis from the public FG21/1 guidance.

## Ranked ingestion queue

**Tranche 1 (active; re-scoped 2026-08-31 to tier 1 only)**: real ASR text first, audio and clean-text corpora deferred. Done: AppTek transcripts (audio deferred at 5.6/52 GB, resumable via `scripts/download_apptek.py --what all`), ACI-Bench (tier 1). Next: CourtListener SCOTUS transcripts. Deferred mid-download, resumable scripts in `scripts/`: PriMock57 (488 MB partial LFS), OANC (14/326 MB), ICSI (transcript zips complete, unprocessed; licence verified CC BY 4.0 at https://groups.inf.ed.ac.uk/ami/icsi/license.shtml, OANC terms verified and archived at `data/raw/oanc/LICENSE_PAGE.html`: "freely available for download and use for research and development, including commercial development"). Each ingest goes through the `add-corpus` skill: licence re-verified at the primary source before download, download script in `scripts/`, DATASHEET row with provenance and examples.

**Remaining tier 1, ranked by distance to the target distribution** (2026-08-31; conversational service register, diarised turns, modern punctuation-restoring ASR, long documents — in that order of weight):

1. **CallCenterEN** (NC) — real agent/customer calls in modern commercial ASR: the closest public match to the target, full stop. Found ungated on 2026-08-31, contrary to the survey; **downloading** via `scripts/download_callcenteren.py`
2. **CANDOR** (NC) — dyadic ~31 min spontaneous conversation, AWS Transcribe conventions: the only corpus at target length in a casual dyadic register. **Blocked on an access application** (human review; needs Khalid)
3. **SPoRC** (NC) — diarised speaker turns over long conversational episodes at podcast-archive scale; the NC pretraining backbone. Gate accepted 2026-09-01; **downloading** text layers only (`turns/` 13.5 GiB + `episodes/` 14.9 GiB parquet) via `scripts/download_sporc.py`, acoustics and bulk metadata skipped
4. **Emilia-YODAS EN** (P) — tier 1 transcripts exist but are **bundled with the audio in 1.35 TiB of tars, with no text-only path** (measured 2026-09-01), so the cost per text token is enormous. Deferred; for permissive bulk text, FAMA/YouTube-Commons is the cheaper route, and the tars only become worth it if their audio is wanted for self-ASR anyway
5. **MeetingBank** (NC) — ~28k-token diarised Speechmatics ASR documents: the long-context + turn-structure anchor; formal civic register. Ungated; **downloading** via `scripts/download_meetingbank.py` (audio not fetched)
6. **CourtListener, federal appeals courts** (P) — cadc and ca1 pulled 2026-08-31 (see DATASHEET); the remaining ~76k circuit arguments (ca9 alone holds 33,406) stay available via `scripts/download_courtlistener.py --court <id>` once a pretraining token budget fixes how much is worth taking
7. **FAMA EN / YouTube-Commons** (P) — bulk Whisper text, flat, mixed register; take a conversational slice once a token budget exists
8. **RadioTalk** (NC by ambiguity) — huge and telephone-flavoured, but 2019-era Kaldi output (lowercase, unpunctuated) is the wrong transcription convention; use only if bulk NC runs short
9. **SpokenWOZ** (NC) — short task dialogues; marginal

**Supervision + written sources (unranked, small, fetch when the benchmark work starts)**: TweetSumm · MultiDoc2Dial · ABCD · CompliBench · EmoWOZ · AnnoMI · ESConv · FOMC pressers · People's Speech slice · congressional hearings. Separately: request terms for SSSD and Hume-DaiKon; email USS authors for an annotation licence; verify TWCS at Kaggle.

**Deferred tier 2 (needs GPU time, resume when channel work starts)**: AppTek audio remains the highest-value single item in the whole queue once self-ASR runs, because it yields diarised modern-ASR transcripts of dyadic service calls plus paired verbatim reference; then PriMock57, ICSI audio, OANC (tier 3 text).

Keep this file current when a corpus is surveyed, queued, ingested, or refused.
