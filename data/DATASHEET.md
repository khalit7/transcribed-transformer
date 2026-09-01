# Datasheet — transcribed-transformer

The datasets this project holds: what each one is, where it lives, how big it is, what role it plays, and under what licence. The corpus **survey** and every acquisition decision, including refusals, live in [`SURVEYSHEET.md`](SURVEYSHEET.md); a source only earns a row here once it is on disk. The recipes for **synthetic** transcript text live in [`SYNTHSHEET.md`](SYNTHSHEET.md); when a synthetic corpus enters a training mixture, its row goes here and the recipe stays there.

## Overview

`data/` holds only data; all code lives in `src/`. Sources are immutable under `data/raw/<dataset>/`, one folder per dataset. Derived artefacts — `data/interim/<name>/{train,val}.jsonl` and `data/packed/<stage>/` token streams — are regenerable and not documented here.

Measurement provenance: token counts are measured on the local copy with the tokenizer named per entry, never copied from a dataset card. Counts dated 2026-07-29 were produced with the ModernBERT-large tokenizer over a one-turn-per-line render by preprocessing code removed in the 2026-08-31 repo reset; they remain the best measurements until the new preprocessing re-measures them. `TBD` means not yet measured; nothing here is estimated.

No raw sample is committed (all of `data/` except the three sheets is gitignored), so the examples quoted below are the canonical record of what each corpus looks like.

Keep this file current when a dataset is added, removed, or re-scoped.

## Licence tracks

| Track | Criteria | Consequence |
|---|---|---|
| **P** — permissive | Commercial use allowed **and** derivatives redistributable | Models trained on Track P only are commercially portable |
| **NC** — non-commercial | CC BY-NC, research-only, or unclear | Any model touching this data is research-only |

Ambiguity resolves to **NC**. Share-alike licences (CC BY-SA, CDLA-Sharing) qualify for Track P but carry an **SA flag**: released derivatives must carry the licence forward, and model cards must say so. Tracks are never mixed within a training run. Every headline result is reported on both tracks.

## Data tiers

Licence says whether we *may* use a corpus. Tier says how close it is to the distribution this project models, which is ASR output.

| Tier | What | How it is used |
|---|---|---|
| **1** | Real ASR output, disfluencies and recognition errors intact | Directly, as training text |
| **2** | Audio available, so we can produce tier 1 ourselves | Run ASR over it; the system used is recorded |
| **3** | Clean written text, or human-verbatim transcription | Channel model applied first ([`SYNTHSHEET.md`](SYNTHSHEET.md)). Requires a justification for why nothing better was available |

**Human-verbatim transcription is tier 3, not tier 1.** A transcriber who silently repairs a false start has removed the signal this project exists to model. A corpus shipping both a human and an ASR transcript is tier 1, and its human side becomes reference data for fitting the channel rather than training text.

## Summary — the single source of truth for ALL data

Waves: **w1** = 2026-07 (initial ingests), **w2** = the 2026-08-31 acquisition round (tranche 1 of the [survey queue](SURVEYSHEET.md#ranked-ingestion-queue)). Status `raw on disk` means licence verified and source downloaded, preprocessing module not yet written (all preprocessing code was removed in the 2026-08-31 reset). `—` means not applicable.

| Dataset | Added | Status | Role | Track | Tier | SA | Local path | Raw size (MB) | Tokens | License |
|---|---|---|---|---|---|---|---|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | w1 | raw on disk | channel-fitting reference (paired ASR + verbatim); pretraining text (ASR layer) | P | **1** | — | `data/raw/ami/` | 91 | **1,280,265** (asr, 126 meetings) · 1,674,833 (manual, 171) | CC-BY-4.0 |
| [Taskmaster-1 + 2](https://github.com/google-research-datasets/Taskmaster) | w1 | raw on disk | case-packing dialogue text (short dyadic service calls, advisor/customer roles) | P | 3 | — | `data/raw/taskmaster/` | 152 | **8,102,358** (22,807 dialogues) | CC-BY-4.0 |
| [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | w2 | transcripts on disk; ⚠️ audio deferred (tier-1-first policy, 2026-08-31; 5.6 GB of 52 GB fetched, resumable) | **eval + channel reference only, never training text** (source card: evaluation-only intent); benchmark case text, modern-recogniser channel fitting | P | 2 | **SA** | `data/raw/apptek_callcenter/` | 52,224 (stated) | TBD (measured 1,278,110 words, 94,679 turns, 873 calls) | CC-BY-SA-4.0 |
| [ACI-Bench](https://github.com/microsoft/clinical_visit_note_summarization_corpus) | w2 | raw on disk | vulnerability (health) case text; **paired ASR/corrected/human channel reference** | P | **1** | — | `data/raw/aci_bench/data/aci-bench/` | 10 (repo) | TBD (measured 269,523 dialogue words, 207 encounters) | CC-BY-4.0 |
| MTS-Dialog (same repo) | w2 | ⚠️ on disk, not used | short written doctor/patient snippets + summaries | **NC** (by ambiguity) | 3 | — | `data/raw/aci_bench/data/mts-dialog/` | ″ | TBD | CC-BY-4.0 repo / mtsamples terms unclear |
| [CourtListener oral arguments](https://www.courtlistener.com/audio/) (scotus, cadc, ca1) | w2 | raw on disk (3 courts complete) | long-context tier 1 pretraining text | P | **1** | — | `data/raw/courtlistener/<court>/` | 234 | TBD (measured **42,385,111 words**, 6,518 arguments) | Public domain (US federal works; Free Law bulk data) |
| [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank) | w2 | raw on disk | NC long-context tier 1 pretraining text; summary supervision | **NC** | **1** | — | `data/raw/meetingbank/` | 110 | TBD (measured 19,921,133 words, 6,892 agenda-item records) | CC-BY-NC-SA-4.0 |
| [CallCenterEN](https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english) | w2 | raw on disk | NC in-domain tier 1 pretraining text (service calls) | **NC** | **1** | — | `data/raw/callcenteren/` | 1,433 | TBD (95,953 call JSONs; sampled: mean 1,068 words/call, p95 2,850) | CC-BY-NC-4.0 |

**Total on disk** (measured 2026-07-29, ModernBERT-large tokenizer): **9,382,623 training-usable tokens** across 22,933 transcripts (AMI asr + Taskmaster; the AMI manual layer is channel reference, not training text).

Pipeline rule (applies to every row): each source lands in `data/raw/<name>/` via a script in `scripts/`, gets a preprocessing module producing `data/interim/<name>/{train,val}.jsonl` split by container (meeting / dialogue / call — never by turn), and enters training only through a mixture manifest (`configs/mixtures/*.yaml`) whose per-source token counts are recorded in the pack's `meta.json`. Manifests refuse to mix licence tracks.

## AMI Meeting Corpus

- **Role:** the only Track P corpus pairing real ASR output with a human verbatim transcript of the same speech, so it is the channel-fitting reference; its ASR layer is also tier 1 pretraining text · **Path:** `data/raw/ami/` (`ami_public_auto_1.5.1.zip`, `ami_public_manual_1.6.2.zip`) · **Tokens:** see summary · **Size:** 91 MB
- **License:** CC-BY-4.0, verified 2026-07-29 against the `LICENCE.txt` bundled inside the annotation archive itself: *"The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."* Attribution propagates to model cards. The licence covers signals and transcription plus some annotations; only the transcription layers are used. · **Source:** https://groups.inf.ed.ac.uk/ami/corpus/

100+ hours of 4–5 speaker research meetings, spontaneous and disfluent. **Both transcript layers ship**: real ASR output (`ASR/ASR_AS_CTM_v1.0_feb07/`, 664 per-speaker word files with timings) and human verbatim annotation. The ASR layer covers 126 meetings, all with a manual counterpart; 45 manual-only meetings have no ASR. On the 126 paired meetings the ASR side yields 1,280,265 tokens against 1,219,423 manual (ratio 1.05) and 102,014 turns against 58,199 (1.75): finer segmentation, no punctuation. Per-meeting ASR token lengths (2026-07-29): min 1,699 / p50 9,912 / p95 19,202 / max 29,749; 67.5% exceed 8k, none exceed 32k.

The recogniser is the AMI-ASR system of **February 2007** (measured WER 0.395 against the manual layer). That error rate is far above a modern recogniser's and it emits no punctuation or casing, so a channel fitted on it over-noises and under-punctuates relative to the modern commercial ASR this project targets; see the calibration note in [`SYNTHSHEET.md`](SYNTHSHEET.md).

Known quality problems: speaker labels are meeting roles, not service roles, so role identification supervision is absent; scenario meetings are role-played; one speaker has segments but no word stream and is skipped. The meetings are multi-party (4–5 speakers), which the target distribution's own multi-speaker character makes less of a mismatch than a strict two-party framing would suggest, though the register is a meeting, not a service call.

**Examples** (ASR layer, one speaker's decoded stream, verbatim):

1. `Okay Hi how do you do the summary is it So are you basing um certain threshold a half or a special to us Okay Oh that's the old old one okay I Okay Um Uh-huh So you want to send steve the prototype and we should change the data now`

## Taskmaster-1 and Taskmaster-2

- **Role:** the Track P supply of two-party spoken service dialogue with advisor/customer role labels; short, so several compose into one multi-call case without truncation · **Path:** `data/raw/taskmaster/` (8 JSON files, spoken subsets only) · **Tokens:** see summary · **Size:** 152 MB
- **License:** CC-BY-4.0, verified 2026-07-29 against each release's own README: *"made available under the Creative Commons Attribution 4.0 License."* · **Source:** https://github.com/google-research-datasets/Taskmaster

22,807 two-person spoken dialogues collected Wizard-of-Oz style: crowdworkers as customers, **trained call-centre operators as assistants**. Domains: pizza, auto repair, rides, movies, coffee, restaurants (TM-1); restaurants, food ordering, movies, hotels, flights, music, sports (TM-2). Roles map cleanly (257,480 assistant turns, 217,863 user turns, 55 unlabelled with real text, kept as unknown). Per-dialogue token lengths (2026-07-29): min 7 / p50 321 / p99 828 / max 2,389.

**Tier 3**: human transcription, no audio released, and the TM-1 README concedes disfluencies were *"usually transcribed as spoken, but sometimes transcribers corrected them"* — treat its disfluency density as a lower bound and never fit channel statistics on it. The written self-dialogue half of TM-1 (`self-dialogs.json`) is typed prose in the same schema and must never be ingested. Four TM-1 dialogues are empty and are dropped.

**Examples** (one utterance per line in the source; truncated):

1. `ASSISTANT: Hi there! How can I help?` / `USER: Oh well, I've tried to go see Aquaman in Reno, Nevada.` *(tm1-woz, movie-tickets)*
2. `USER: Hello. I'd like to find a round trip commercial airline flight from San Francisco to Denver.` / `ASSISTANT: San Francisco to Denver, got it.` *(tm2-flights)*

## AppTek Call-Center Dialogues (wave 2)

- **Role:** benchmark case text and the modern-recogniser channel reference. **Never training text**: beyond the SA licence, the dataset card states it is *"intended exclusively for evaluation and analysis rather than model training"* and lists training under out-of-scope use; that intent is honoured here even though CC BY-SA does not legally impose it. · **Path:** `data/raw/apptek_callcenter/` · **Tokens:** TBD (tokenizer not yet chosen post-reset; measured 2026-08-31: 1,278,110 words over 94,679 speaker-labelled segments in 873 calls) · **Size:** 52.2 GB stated (audio downloading; transcripts 25 MB on disk)
- **License:** CC-BY-SA-4.0, verified 2026-08-31 in the dataset card's own metadata and Dataset Description (`license: cc-by-sa-4.0`; "License: CC BY 4.0-SA"). **SA flag**: derivatives released from this data must carry the licence forward. Ungated. · **Source:** https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues — paper arXiv:2604.27543
- **Procured:** found in the 2026-08-31 corpus survey (tranche 1, item 1). Downloaded pinned to revision `b98967d9` (2026-08-25) via `scripts/download_apptek.py` (metadata first, then `--what all` for audio). Two configs: `test/<accent>/` split-channel WAV, one file per speaker (1,746 files), and `diarization/<accent>/` merged mono per call (873 files) with timestamped, role-labelled segments. Processing module: TBD.

Spontaneous **role-played agent–customer calls**, newly recorded for the benchmark (no overlap with public web corpora), 128.6 h of speech across 156 speakers, 14 accent groups (~8–11 h each, incl. en-GB, Scottish, Welsh, Irish) and 16 service domains (banking, insurance, energy, telecoms, health…). Strictly two speakers per call with stable `agent`/`customer` role labels; 5–15 min per call, mean 10.4 min. Measured words per call: min 273 / p25 1,090 / p50 1,380 / p75 1,790 / p95 2,414 / max 3,413 — so single calls are short-to-mid length and several compose into one multi-call case.

Transcripts are **fully manual verbatim transcription** (no pre-generated ASR), with disfluency markup preserved: hesitations as `(um)`, partial words with `~`. Tier 2: the per-speaker split-channel audio is what self-ASR runs over (planned: Whisper large-v3 and Parakeet, logged per file), producing tier 1 text whose alignment against the verbatim side fits the v2 channel model ([`SYNTHSHEET.md`](SYNTHSHEET.md)).

Known quality problems: role-played rather than real customers, so scenario stakes are simulated; the transcription normalisation used for the corpus's own WER scoring (`word_mappings.py`) removes selected hesitation tokens, so channel fitting must use the raw `text` field, not the normalised form; exactly two speakers per call, so multi-party structure has to come from other sources.

**Examples** (diarization config, `en-GB`, verbatim incl. disfluency markup; truncated):

1. `customer: Okay, nice to meet you. I'm just calling in regards to the new (um) new program that you have, could I please ask some questions about that?`
2. `agent: So the n~ new (um) organize sale , organic certification program that'll be providing (um) of.`
3. `customer: Yeah, so I'm just calling to ask about the new organic certification program for cro~ crops that you're providing?`

## ACI-Bench (wave 2)

- **Role:** vulnerability (health-driver) benchmark case text, and a second paired channel reference: the `src_experiment_data` folder ships **the same encounters as real ASR output, ASR-corrected text, and human transcripts** · **Path:** `data/raw/aci_bench/data/aci-bench/` · **Tokens:** TBD (measured 2026-08-31: 269,523 dialogue words over 207 encounters; words per encounter in `train` min 628 / p50 1,240 / max 3,050) · **Size:** 9.5 MB (whole repo clone)
- **License:** CC-BY-4.0, verified 2026-08-31 by reading the `LICENSE` file in the clone (Creative Commons Attribution 4.0 International, full text). The GitHub API reports "Other/NOASSERTION", which is wrong; the file governs. ACI-Bench encounters are synthetic role-plays created by clinicians and annotators, so no third-party text rights lurk underneath. · **Source:** https://github.com/microsoft/clinical_visit_note_summarization_corpus — paper: Nature Scientific Data 10, 586 (2023)
- **Procured:** found in the 2026-08-31 corpus survey (tranche 1, item 3). Cloned by `scripts/download_aci_bench.sh`, commit `293e454` pinned by the kept `.git`. The repo states it is static and will not change. Processing module: TBD.

207 doctor–patient visit dialogues (the longest public visit dialogues), role-played and then actually recorded and ASR-transcribed for the `aci` subset. **Tier 1 evidence**: `src_experiment_data/*_aci_asr.csv` is raw recogniser output (lowercase, unpunctuated, visible attribution errors), with `*_aci_asrcorr.csv` the corrected version of the same encounters and `*_virtscribe_asr.csv` / `*_virtscribe_humantrans.csv` a second paired style; joinable by `encounter_id`. The `challenge_data` splits carry cleaned dialogue plus the full clinical note (a summary-supervision bonus) and metadata (patient name, age, complaints).

Known quality problems: US-clinical register rather than UK financial advice; role-played patients; `[doctor]`/`[patient]` speaker tags rather than diarised `SPEAKER_NN`; small (207 encounters).

**Examples** (`test1_aci_asr`, verbatim ASR layer; truncated):

1. `[doctor] hey charles i'm using this cool new recording device to help me with my documentation is that okay with you [patient] sure [doctor] awesome how are you doing today`
2. `[doctor] so jerry is a 45 -year-old male who came in today with an ankle injury jerry what happened` *(asrcorr layer of the same subset)*

## MTS-Dialog (wave 2, on disk, not used)

- **Role:** ⚠️ not used · **Path:** `data/raw/aci_bench/data/mts-dialog/` (arrives inside the ACI-Bench clone) · **License:** repo CC-BY-4.0, but the `NOTICE` file records the underlying mtsamples.com terms: *"feel free to print, share, link, and distribute… please notify us, and please give credit"*. That grants distribution with attribution but says nothing explicit about commercial use or derivatives, so by the ambiguity rule this subcollection is **Track NC** despite the repo licence.
- **Why it ended up unused.** 1.7k short written doctor/patient snippets (tier 3) whose only edge over ACI-Bench was being permissive; with the track downgraded it adds nothing ACI-Bench and the NC counselling corpora don't cover better. Revisit only if mtsamples grants explicit terms.

## CourtListener oral arguments (wave 2)

- **Role:** the Track P long-context tier 1 source: real ASR text at the document lengths this project targets, which nothing else permissive provides · **Path:** `data/raw/courtlistener/<court>/transcripts.jsonl` · **Tokens:** TBD. Measured 2026-08-31: **scotus** complete, 11,150,036 words over 1,001 arguments (per-argument words min 5,327 / p25 9,589 / p50 10,327 / p75 11,889 / p95 16,772 / p99 20,558 / max 26,864 — unimodal, right-skewed, none under 500 words); **cadc** (DC Circuit) complete, 17,896,752 words over 2,604 arguments (min 237 / p50 6,048 / p95 13,528 / max 24,721; one record under 500 words); **ca1** (First Circuit) complete, 13,338,323 words over 2,913 arguments (p50 4,303 / max 21,889; five records under 500 words — inspect and drop empties at preprocessing). Combined: **42,385,111 words over 6,518 arguments**, no duplicate ids within any court. Roughly 76k further transcribed circuit arguments remain available; pulling them is a token-budget decision recorded in the survey queue · **Size:** 234 MB
- **License:** public domain. Free Law Project's bulk-data page marks its data free of known copyright restrictions, and Supreme Court oral-argument recordings are US federal government works. Sourced from the CourtListener API directly, not from Oyez (whose wrapper is CC BY-NC). · **Source:** https://www.courtlistener.com/api/rest/v4/audio/ (`docket__court=scotus`, `stt_status=1`)
- **Procured:** found in the 2026-08-31 corpus survey. Pulled by `scripts/download_courtlistener.py` (cursor-resumable, honours `Retry-After` on the anonymous rate limit; polite 10 s page interval). `stt_source=1` on every record: CourtListener's own Whisper-family transcription. MP3 `download_url` kept per record so a local diarised ASR pass remains possible later. Processing module: TBD.

Sustained 20–90 minute two-sided argument between advocates and the bench: spontaneous, interrupted, high-register spoken English, transcribed by a real recogniser with restored punctuation. **Limitations, stated plainly**: the transcript field is flat text with no speaker turns (diarisation requires the audio, deferred under the tier-1-first policy), the register is formal advocacy rather than service calls, and it is not advisor/customer in role. It is used as long-context pretraining distribution, never as benchmark case text.

**Examples** (record 104587, verbatim; truncated):

1. `We'll hear argument next in Case 24-889, Hikma Pharmaceuticals v. Amarin Pharma. Mr. Klein. Thank you, Mr. Chief Justice, and may it please the Court. Under Section 271B of the Patent Act, selling a product suited for both infringing and substantial non-infringing uses is lawful, unless the seller actively induces the infringing use.`

## MeetingBank (wave 2)

- **Role:** the NC track's long-context tier 1 text, and summary supervision (each record pairs a transcript with a human summary) · **Path:** `data/raw/meetingbank/` (`{train,validation,test}.json`, JSONL despite the extension) · **Tokens:** TBD (measured 2026-08-31: 19,921,133 words over 6,892 records; words per record min 69 / p50 986 / p95 12,093 / max 67,634) · **Size:** 110 MB
- **License:** CC-BY-NC-SA-4.0 (dataset card) → **Track NC**. · **Source:** https://huggingface.co/datasets/huuuyeah/meetingbank — paper ACL 2023
- **Procured:** ranked 5th of the remaining tier 1 sources 2026-08-31; downloaded via `scripts/download_meetingbank.py` (ungated). Audio (`huuuyeah/MeetingBank_Audio`) deliberately not fetched. Processing module: TBD.

1,366 US city-council meetings (3,579 h) transcribed by Speechmatics: modern ASR with restored punctuation and casing. **This HF distribution is per agenda item, not per meeting** — `uid` encodes `council_date_item`, so whole meetings are reconstructable by grouping, which is where the ~28k-token documents come from. **No speaker labels in these files**: the diarised segment layer ships with the audio release, so as downloaded this is long flat tier 1 text. Register is formal civic proceedings, not service calls.

**Examples** (train, verbatim; truncated):

1. `Please refrain from profane or obscene speech. Direct your comments to council as a whole and refrain from individual or personal attacks. Councilwoman Gilmore, will you please put Council Bill 161 on the floor? Yes, President Brooks, I move that council bill 161 as amended, be placed upon final consideration and do pass.`

## CallCenterEN (wave 2)

- **Role:** the NC track's in-domain anchor: real customer-service telephone calls in modern commercial ASR, the closest public match to the target distribution · **Path:** `data/raw/callcenteren/` (11 domain zips of per-call JSON) · **Tokens:** TBD (95,953 call JSONs on disk; 1,650-call sample: mean 1,068 words and 589 s audio per call, p50 752 / p95 2,850 / max 12,633 words) · **Size:** 1.4 GB
- **License:** CC-BY-NC-4.0 (dataset card) → **Track NC**. Found **ungated** 2026-08-31, contrary to the survey's expectation. · **Source:** https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english — paper arXiv:2507.02958
- **Procured:** ranked 1st of the remaining tier 1 sources 2026-08-31; downloaded via `scripts/download_callcenteren.py`. Domains: medicare (largest, 61,513 calls), home services, automotive, insurance inbound/outbound, general customer service, medical equipment. Audio is withheld by the publisher (biometric privacy), so there is no tier-2 path. Processing module: TBD.

Per-call JSON: flat `text`, overall `confidence`, `audio_duration`, word-level timings, and the list of redacted PII policies. Confirms the survey's tier resolution: this is AssemblyAI ASR output with restored punctuation and casing (the paper states 0.1% was human-reviewed at WER 3.87%).

Known quality problems, and they matter: **no speaker turns** — the `speaker` field is null on every sampled word, so agent and customer sentences are interleaved in one stream and turn structure would have to be inferred; **PII is replaced in-band with bracketed placeholders** (`[PERSON_NAME]`, `[ORGANIZATION]`, `[DATE]`…), often repeated per word of the redacted span, which is both a redaction artefact that must not be learned as a PII pattern and a systematic distortion of exactly the identifier-dictation phenomena this project cares about; the file count (95,953) exceeds the paper's 91,706 conversations because two "(reupload)" zips overlap the originals — dedupe by transcript id at preprocessing.

**Examples** (medicare_inbound, verbatim; truncated):

1. `Thank you for calling [ORGANIZATION] [ORGANIZATION] [ORGANIZATION]. This is [PERSON_NAME]. How can I help you today? Hi, I'm looking to get a grooming appointment. I had filled out paperwork yesterday and left a message to be called or texted back. Sure, just give me one second. Let me pull up your account.`

## Channel artifact v1 (pipeline artifact)

- **Role:** ⚠️ reference only — superseded for synthesis once a modern-recogniser channel is fitted · **Path:** `data/channel/ami_channel_v1.json` · **Size:** 4 MB
- **Generated by:** the pre-reset `asr_channel` fit over the 126 paired AMI meetings (942,880 reference words): WER 0.3946 (sub 0.2524, del 0.0791, ins 0.0631), 88,849 distinct substitution pairs. Kept as the measured record of a 2007-era recogniser's error profile and as a severity upper bound; **why it is not used for synthesis**: the target distribution is modern commercial ASR with restored punctuation and casing, which this channel neither models nor approximates. The refit plan is in [`SYNTHSHEET.md`](SYNTHSHEET.md).

Keep this file current when a dataset is added, removed, or re-scoped.
