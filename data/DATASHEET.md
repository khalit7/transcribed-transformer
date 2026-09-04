# Datasheet — transcribed-transformer

The datasets this project holds: what each one is, where it lives, how big it is, what role it plays, and under what licence. The corpus **survey** and every acquisition decision, including refusals, live in [`SURVEYSHEET.md`](SURVEYSHEET.md); a source only earns a row here once it is on disk. The recipes for **synthetic** transcript text live in [`SYNTHSHEET.md`](SYNTHSHEET.md); when a synthetic corpus enters a training mixture, its row goes here and the recipe stays there.

## Overview

`data/` holds only data; all code lives in `src/`. Sources are immutable under `data/raw/<dataset>/`, one folder per dataset. Derived artefacts — `data/interim/<name>/{train,val}.jsonl` and `data/packed/<stage>/` token streams — are regenerable and not documented here.

Measurement provenance: token counts are measured on the local copy with the tokenizer named per entry, never copied from a dataset card. Counts dated 2026-07-29 (AMI, Taskmaster) were produced with the ModernBERT-large tokenizer by pre-reset code and stand until those modules are rewritten; counts dated 2026-09-02 come from `src/preprocessing/pack.py` with the provisional Qwen3 tokenizer. `TBD` means not yet measured; nothing here is estimated.

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

Waves: **w1** = 2026-07 (initial ingests), **w2** = the 2026-08-31 acquisition round (tranche 1 of the [survey queue](SURVEYSHEET.md#ranked-ingestion-queue)). Status vocabulary: `raw on disk` = licence verified and downloaded, no preprocessing module yet; `interim` = `data/interim/<name>/` written by its `src/preprocessing/` module; `packed (<stage>)` = in a built mixture under `data/packed/`. `—` means not applicable. Token counts name their tokenizer (*Qwen3* = provisional pack tokenizer, see below; *ModernBERT* = pre-reset 2026-07-29 measurement).

| Dataset | Added | Status | Role | Track | Tier | SA | Local path | Raw size (MB) | Tokens | License |
|---|---|---|---|---|---|---|---|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | w1 | raw on disk (NXT parser not yet rewritten post-reset) | tier 1 pretraining text (asr layer); paired manual layer kept but its 2007 recogniser is not a channel-fitting reference | P | **1** | — | `data/raw/ami/` | 91 | **1,280,265** ModernBERT (asr, 126 meetings) · 1,674,833 (manual, 171) | CC-BY-4.0 |
| [Taskmaster-1 + 2](https://github.com/google-research-datasets/Taskmaster) | w1 | interim (tier 3: excluded from packs until a channel passes the gate) | case-packing dialogue text (short dyadic service calls, advisor/customer roles) | P | 3 | — | `data/raw/taskmaster/` | 152 | **8,102,358** ModernBERT (22,807 dialogues; 3,698,542 words in interim) | CC-BY-4.0 |
| [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | w2 | complete on disk (49 GB, 2,619 WAVs); **self-ASR done** (3 passes → `data/interim/apptek_callcenter_selfasr/`); channel v2 fitted from it | **eval + channel reference only, never training text** (source card: evaluation-only intent); benchmark case text, modern-recogniser channel fitting | P | 2 | **SA** | `data/raw/apptek_callcenter/` | 49,152 (on disk) | not packed by policy (1,278,110 words, 94,679 turns, 873 calls) | CC-BY-SA-4.0 |
| [ACI-Bench](https://github.com/microsoft/clinical_visit_note_summarization_corpus) | w2 | raw on disk (module pending) | vulnerability (health) case text; **paired ASR/corrected/human channel reference** | P | **1** | — | `data/raw/aci_bench/data/aci-bench/` | 10 (repo) | TBD (measured 269,523 dialogue words, 207 encounters) | CC-BY-4.0 |
| MTS-Dialog (same repo) | w2 | ⚠️ on disk, not used | short written doctor/patient snippets + summaries | **NC** (by ambiguity) | 3 | — | `data/raw/aci_bench/data/mts-dialog/` | ″ | TBD | CC-BY-4.0 repo / mtsamples terms unclear |
| [CourtListener oral arguments](https://www.courtlistener.com/audio/) (scotus, cadc, ca1) | w2 | **packed (p_v1)** | long-context tier 1 pretraining text | P | **1** | — | `data/raw/courtlistener/<court>/` | 234 | **51,936,162** Qwen3 (42,384,093 words, 6,512 arguments) | Public domain (US federal works; Free Law bulk data) |
| [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank) | w2 | **packed (nc_v1)** | NC long-context tier 1 pretraining text; summary supervision | **NC** | **1** | — | `data/raw/meetingbank/` | 110 | **23,534,451** Qwen3 (19,921,133 words; 1,249 whole meetings from 6,892 items) | CC-BY-NC-SA-4.0 |
| [CallCenterEN](https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english) | w2 | **packed (nc_v1)** | NC in-domain tier 1 pretraining text (service calls) | **NC** | **1** | — | `data/raw/callcenteren/` | 1,433 | **99,437,272** Qwen3 (66,897,886 words; 90,831 calls after dedupe of 95,953 files) | CC-BY-NC-4.0 |
| [SPoRC](https://huggingface.co/datasets/blitt/SPoRC) | w2 | **packed (nc_v1)**, conversational subset | NC pretraining backbone: diarised conversational turns at scale | **NC** | **1** | — | `data/raw/sporc/` | 29,184 | **4,373,956,442** Qwen3 (conversational subset: 364,970 of 1,124,058 episodes, 2,810,792,362 words); full source 4,585,128,049 by its own token_count | research gate (accepted 2026-09-01) |

**Provisional tokenizer (2026-09-02)**: token counts marked *Qwen3* are measured with the Qwen3 tokenizer (`Qwen/Qwen3-1.7B`, ungated, a plausible decoder base) via the pack step. It is provisional: each arm repacks with its own tokenizer and `meta.json` records which. Earlier counts marked *ModernBERT* (2026-07-29) predate the reset.

**Interim layer measured 2026-09-02** (`src/preprocessing/*`, whitespace words; Qwen3 tokens where packed):

| Interim source | Docs (train / val) | Words | Qwen3 tokens | Words per doc p50 / max |
|---|---|---|---|---|
| `courtlistener` | 6,400 / 112 | 42,384,093 | **51,936,162** (p_v1 pack) | 5,375 / 26,864 |
| `callcenteren` | 88,933 / 1,898 | 66,897,886 | **99,437,272** (nc_v1) | 574 / 19,077 |
| `meetingbank` | 1,218 / 31 | 19,921,133 | **23,534,451** (nc_v1) | 10,819 / 77,276 |
| `taskmaster` | 22,364 / 443 | 3,698,542 | — (tier 3, not packed) | 147 / 1,050 |
| `sporc` | 357,690 / 7,280 | 2,810,792,362 | **4,373,956,442** (nc_v1) | 6,983 / 186,327 |

**Packs built 2026-09-02** (train tokens, Qwen3, EOS-separated): `p_v1` = 51,046,342 (CourtListener); `nc_v1` = **4,496,928,165** (SPoRC 4,373,956,442 + CallCenterEN 99,437,272 + MeetingBank 23,534,451), val 91,326,467; 914 s to build. Tokens per word: 1.23 on flat court text, 1.56 on turn-rendered SPoRC (the `SPEAKER_NN: ` prefixes on 135M turns cost roughly a fifth of the tokens).

Pipeline rule (applies to every row): each source lands in `data/raw/<name>/` via a script in `scripts/`, gets a preprocessing module (`src/preprocessing/<name>.py`) producing `data/interim/<name>/{train,val}.jsonl` — one document per line with `source, doc_id, track, tier, asr_system, has_speakers, n_turns, n_words, text, meta`; `text` is `SPEAKER_NN: utterance` one turn per line where the source has turns, and flat text with `has_speakers=false` where it does not (no invented turn boundaries) — split 2% val by container hash (call / episode / argument / meeting, never by turn), and enters training only through a mixture manifest (`configs/mixtures/*.yaml`) packed by `src/preprocessing/pack.py` into `data/packed/<stage>/` with per-source token counts and a tokenizer fingerprint in `meta.json`. The packer refuses a manifest whose sources carry mixed licence tracks.

## Labelled data (training and benchmarking)

Labelled (transcript, question) → (evidence, answer, summary) records live in `data/labelled_data/`, generated by `src/synthesis/` (`uv run python -m src.synthesis.synth_data <backend:model> <generation_size>`; backends `claude:` via `claude -p` or `ollama:` for local models). Nothing under the directory is tracked in git (the question bank's source of truth is `src/synthesis/question_bank.py`, which is); a releasable Track P copy is produced by `python -m src.synthesis.export --track p`.

**The flow**: pick a real call at random, then ask it every bank question whose `dataset_allow_list` includes the call's dataset (`<generation_size>` counts calls, not labels), and have a labeller answer each. No text is written into calls (an earlier injection design was removed on 2026-09-03; every label is produced by LLM labelling of a real conversation). Each label records **evidence first, then answer, summary, confidence** and any **tags**; optional checks store a second labeller's blind verdict (`verification`) and an evidence **ablation** (necessary / sufficient). The records serve both fine-tuning and benchmarking; the generator does not split them — that decision is taken downstream, per call, when sets are assembled.

- **`labelled_data.jsonl`** — one self-contained record per label, `id = <call>::<question_id>`: `dataset`; `source_id`; `track`; `question` (`id`, `source`, `family` ∈ `vulnerability | complaint | eod | general_qa`, `text`, `description`, `options` with criteria, `tags` vocabulary, `dataset_allow_list`); `transcript` (line-aligned `variants` with `origin`, each line `<role>: text` with the role verbatim from the corpus; `speakers`, `role_source`); `label` (`evidence` 1-based lines, `answer` ∈ `pass | fail | partial_pass | NA`, `summary`, `tags`, `confidence`); `verification` and `ablation` (optional); `generation_info` (`name` = backend:model, `labelled_variant`, `cost_usd`, `timestamp`); `meta`.
- **`questions.jsonl`** — derived from the bank; regenerated after each run.

**Model input format (fixed 2026-09-04).** Every labeller and the judge receive the transcript rendered as one turn per line, each line `<n>: <role>: <text>` with `n` a 1-based line number and `role` the corpus's verbatim speaker label, e.g. `1: agent: Hi. How can I help you?` / `2: customer: Hi, I'm calling about my policy.` The stored record keeps the unnumbered lines (`transcript.variants[].lines`, `<role>: <text>`); numbering is applied at prompt time by `label.numbered()`. Evidence is a list of those line numbers, so **the same rendering (number, role, text) is the input format for the benchmark and for fine-tuning**: any model evaluated or trained on these labels must see the transcript numbered exactly this way, or the evidence keys stop meaning what they meant. Clean and messy variants are line-aligned, so line *i* is the same turn in both.

**Evidence keys are never guaranteed complete**; score evidence as precision against the key. `ablation.necessary`/`sufficient` say how far a key can be trusted.

**Question bank (v2, 146 questions after the second-pass audit below; v0 had 79, v1 73)**. Every question carries a `dataset_allow_list`, and every question that allows a call's dataset is asked of that call. **Every question is written in the speaker vocabulary of the transcripts it is asked of, and a question is shared across corpora only when its wording already applies verbatim** (2026-09-03): `gen-` (48 service-call conduct questions, agent/customer) and the four family questions (`vul-01-present`, tagged with the FCA FG21/1 characteristics; `vul-02-handled`; `cmp-01-complaint`; `eod-01-dissatisfaction`) are AppTek's; `tm-` (40: the four families plus 36 conduct behaviours, assistant/user) are Taskmaster's, derived from the `gen-` text by substitution because the conduct is identical; `aci-` (26: complaint and dissatisfaction about care plus 24 conduct behaviours written for a clinical visit, doctor/patient) are ACI-Bench's; `spk-` (22, phrased in terms of "the speakers") are shared across all four corpora per the audit, and include one podcast-only rare-event question, whether any speaker discloses a vulnerability about themselves; `hst-` (7, host) are SPoRC host-behaviour questions (sponsor message read, host named, guest introduced, floor shared, restating, self-promotion, listener messages). Labels per call: AppTek 68, Taskmaster 46, ACI-Bench 39, SPoRC 29. Vulnerability has no ACI-Bench form (a patient is trivially in the health driver). The 50 original `general_qa` questions were authored 2026-09-03 after surveying the case pool (16 service domains in AppTek — retail, banking, aviation, food, health, travel, hospitality, property, insurance, agriculture, telecoms, technology, energy, finance, entertainment, delivery; Taskmaster bookings; ACI-Bench clinical visits; SPoRC interviews) and the *shape* of real call-QA practice, in original wording, covering opening, identity and data, needs, explanation and information, manner, handling and closing; the family questions were defined to match how such assessments are scored in practice (for detection questions `fail` means the thing was detected).

**Allow-list audit, first pass (2026-09-03)**. Because every allowed question is asked of every call, the allow list decides which labels exist, so every (question, dataset) cell of the then 79-question bank was probed on the same 40 real calls per dataset (seed 3) by three labellers, qwen3:32b, gemma3:27b and Claude Sonnet, and reconciled by majority (`audit_bank.py`). Decision rules: a dataset leaves a question's list when the majority answers NA on ≥75% of its calls, or when one answer covers ≥90% and the rare answer carries no compliance weight; skew alone never removes a question whose rare answer matters, and such questions are kept as rare-event detection tests. Outcome: six questions removed (`gen-38-apology-where-due`, `gen-42-redirected-appropriately`, `spk-02-topic-stated`, `spk-09-disagreement-respectful`, `spk-19-examples-given`, `spk-25-resource-named`), 39 allow lists narrowed, nothing added (labellers will answer service-call questions about podcasts, but a question about *the agent* is ill-posed where there are no roles); the resulting per-dataset coverage is what the `tm-` and `aci-` sets encode. Measured rare-event rates on 40 calls per dataset (Sonnet; the other labellers within ±2): vulnerability present 5 AppTek / 1 Taskmaster / 39 ACI-Bench (every patient counts under the health driver) / 8 SPoRC; complaint made 2 / 0 / 2 / 0; dissatisfaction expressed 3 / 1 / 4 / 3. Labellers gave the same majority answer on 63–68% of cells pairwise (mean L1 0.57 qwen–Sonnet, 0.67–0.70 with gemma); gemma3:27b was the lenient outlier (mean pass share 0.62 vs 0.44 and 0.41) and on three questions (`spk-19`, `spk-22`, `spk-25`) answered the opposite of qwen on every service call, which reads as an ambiguous fail criterion rather than a data property: `spk-19` and `spk-25` were removed and `spk-22` lost Taskmaster, where it is trivially `fail`. Probe cost with Sonnet: $35.42 for 160 calls × 79 questions (one prompt per call); the local labellers cost GPU time only. That pass rendered transcripts as `SPEAKER_NN` with a sentence telling the labeller which tag was the agent; Probe files: `data/labelled_data/probes/audit_<dataset>_<labeller>[-roles]_seed3.json`.

**Second pass (2026-09-03, aligned bank).** The aligned bank was re-probed on the identical 40 calls per dataset under the verbatim-role rendering with the host identified, each call seeing only its allowed questions (`--dataset`), same three labellers, 480 calls, 0 failures, Sonnet $28.92. AppTek and Taskmaster cells were stable against the first pass (only threshold-edge changes), so the substitution-derived `tm-` wording behaves like the original; ACI-Bench moved most, as expected where the questions were actually rewritten, and mostly towards being answerable (`aci-28-clarifying-question` 75% NA → 78% pass, `aci-49-review-narrated` 65% NA → 97% pass). Applied by the same rules: `spk-08`, `spk-11`, `spk-13`, `spk-16` left Taskmaster (90–100% pass under all labellers: bookings have no figures to be vague about, no advice, no promotion, no call to action); `aci-17`, `aci-48`, `tm-48`, `hst-05`, `hst-07` were dropped as trivially skewed; `aci-06` and `aci-25` were dropped and `spk-22` narrowed to SPoRC because the labellers contradicted each other on every call. Kept although skewed: every family question, `vul-02`/`tm-vul-02` (NA by construction), `gen-31`/`tm-31`, `spk-12` on AppTek and SPoRC, `spk-17`, `spk-23`, `tm-23`, `tm-29`, `aci-14`, `aci-16`, `aci-37`. Labeller agreement was unchanged (same majority answer 64–66% pairwise; mean pass share gemma 0.63, qwen 0.46, Sonnet 0.49). Rare-event counts per 40 calls (Sonnet): vulnerability 6 AppTek / 0 Taskmaster; complaint 3 / 0 / 1 (ACI); dissatisfaction 4 / 2 / 3; a speaker disclosing a vulnerability about themselves on SPoRC 17 (qwen 3, gemma 9: the labellers read "disclosure" very differently there); the host reading a sponsor message 8 (local labellers 26–27). Final bank: **146 questions**; labels per call AppTek 68, Taskmaster 46, ACI-Bench 39, SPoRC 29. Reconciled tables: `audit_seed3_reconciled.txt` (first pass), `audit_seed3_roles_reconciled.txt` (second), `audit_seed3_pass1_vs_pass2.txt` (per-labeller diff).

**Case sources** (any conversation between different people qualifies; **speaker roles are always the corpus's own labels, rendered verbatim on every line**, 2026-09-03): AppTek (P+SA; roles `agent`/`customer`; clean = verbatim, **messy = real Whisper output time-aligned to the verbatim turns**); Taskmaster (P; roles `assistant`/`user`) and ACI-Bench (P; roles `doctor`/`patient`/`patient_guest`), messy = channel v2.2, synthetic; SPoRC (NC; real diarised ASR, single variant), which records no roles, so the **host is identified first** by Claude Sonnet from the opening and closing turns (`src/synthesis/identify_speakers.py`; cached per episode in `data/labelled_data/speakers/sporc.jsonl` with confidence and reason; 240 episodes identified 2026-09-03, 239 with a host, median confidence 0.98, $21.24) and rendered `host:`, the other speakers keeping their diarisation tags; an episode without an identifiable host is skipped, never rendered role-less. The labeller reads the clean variant by default (`--labeller-variant messy` to change); labels apply to both since variants are line-aligned.

**State (2026-09-03)**: generator live on both backends; smoke test of 3 labels with `ollama:qwen3:32b` as labeller, `claude:sonnet` as verifier and ablation on: 39 s, $0.29 (all verification cost). `labelled_data.jsonl` is empty until a sized run is requested. AppTek cases are never training text.

## AMI Meeting Corpus

- **Role:** tier 1 pretraining text (ASR layer). It also pairs that ASR with a human verbatim transcript of the same speech, but the recogniser is from 2007, so the pairing is a historical record rather than a channel-fitting reference (channel v2 fits from modern pairs: AppTek, PriMock57, ACI-Bench) · **Path:** `data/raw/ami/` (`ami_public_auto_1.5.1.zip`, `ami_public_manual_1.6.2.zip`) · **Tokens:** see summary · **Size:** 91 MB
- **License:** CC-BY-4.0, verified 2026-07-29 against the `LICENCE.txt` bundled inside the annotation archive itself: *"The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."* Attribution propagates to model cards. The licence covers signals and transcription plus some annotations; only the transcription layers are used. · **Source:** https://groups.inf.ed.ac.uk/ami/corpus/

100+ hours of 4–5 speaker research meetings, spontaneous and disfluent. **Both transcript layers ship**: real ASR output (`ASR/ASR_AS_CTM_v1.0_feb07/`, 664 per-speaker word files with timings) and human verbatim annotation. The ASR layer covers 126 meetings, all with a manual counterpart; 45 manual-only meetings have no ASR. On the 126 paired meetings the ASR side yields 1,280,265 tokens against 1,219,423 manual (ratio 1.05) and 102,014 turns against 58,199 (1.75): finer segmentation, no punctuation. Per-meeting ASR token lengths (2026-07-29): min 1,699 / p50 9,912 / p95 19,202 / max 29,749; 67.5% exceed 8k, none exceed 32k.

The recogniser is the AMI-ASR system of **February 2007** (measured WER 0.395 against the manual layer). That error rate is far above a modern recogniser's and it emits no punctuation or casing, which is why this pairing is excluded from channel fitting ([`SYNTHSHEET.md`](SYNTHSHEET.md)); a channel v1 fitted from it pre-reset was discarded on 2026-09-01 for that reason. The ASR layer remains valid tier 1 text, with its age recorded as a convention caveat.

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

- **Role:** benchmark case text and the modern-recogniser channel reference. **Never training text**: beyond the SA licence, the dataset card states it is *"intended exclusively for evaluation and analysis rather than model training"* and lists training under out-of-scope use; that intent is honoured here even though CC BY-SA does not legally impose it. · **Path:** `data/raw/apptek_callcenter/` · **Tokens:** not packed by policy (measured 2026-08-31: 1,278,110 words over 94,679 speaker-labelled segments in 873 calls) · **Size:** 49 GB on disk (2,619 WAVs + 25 MB transcripts)
- **License:** CC-BY-SA-4.0, verified 2026-08-31 in the dataset card's own metadata and Dataset Description (`license: cc-by-sa-4.0`; "License: CC BY 4.0-SA"). **SA flag**: derivatives released from this data must carry the licence forward. Ungated. · **Source:** https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues — paper arXiv:2604.27543
- **Procured:** found in the 2026-08-31 corpus survey (tranche 1, item 1). Downloaded pinned to revision `b98967d9` (2026-08-25) via `scripts/download_apptek.py` (metadata first, then `--what all` for audio). Two configs: `test/<accent>/` split-channel WAV, one file per speaker (1,746 files), and `diarization/<accent>/` merged mono per call (873 files) with timestamped, role-labelled segments. Processing module: none (excluded from training by policy); self-ASR outputs in `data/interim/apptek_callcenter_selfasr/{diarization-degraded,test-degraded,test-clean}/`.

**Self-ASR done 2026-09-01/02** (`scripts/transcribe.py`, faster-whisper large-v3): all 873 merged calls and all 1,746 split-channel files, each in a telephone-degraded and (split-channel only) a clean pass, 18–28x realtime per RTX 5090. Aligning the split-channel hypotheses against the verbatim transcripts gave channel v2: pooled WER 0.2764 degraded / 0.2677 clean, accents from 0.236 (en-AU) to 0.346 (en-CN); details and the failed QC gate in [`SYNTHSHEET.md`](SYNTHSHEET.md).

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
- **Procured:** found in the 2026-08-31 corpus survey (tranche 1, item 3). Cloned by `scripts/download_aci_bench.sh`, commit `293e454` pinned by the kept `.git`. The repo states it is static and will not change. Processing module: see the interim table above.

207 doctor–patient visit dialogues (the longest public visit dialogues), role-played and then actually recorded and ASR-transcribed for the `aci` subset. **Tier 1 evidence**: `src_experiment_data/*_aci_asr.csv` is raw recogniser output (lowercase, unpunctuated, visible attribution errors), with `*_aci_asrcorr.csv` the corrected version of the same encounters and `*_virtscribe_asr.csv` / `*_virtscribe_humantrans.csv` a second paired style; joinable by `encounter_id`. The `challenge_data` splits carry cleaned dialogue plus the full clinical note (a summary-supervision bonus) and metadata (patient name, age, complaints).

Known quality problems: US-clinical register rather than UK financial advice; role-played patients; `[doctor]`/`[patient]` speaker tags rather than diarised `SPEAKER_NN`; small (207 encounters).

**Examples** (`test1_aci_asr`, verbatim ASR layer; truncated):

1. `[doctor] hey charles i'm using this cool new recording device to help me with my documentation is that okay with you [patient] sure [doctor] awesome how are you doing today`
2. `[doctor] so jerry is a 45 -year-old male who came in today with an ankle injury jerry what happened` *(asrcorr layer of the same subset)*

## MTS-Dialog (wave 2, on disk, not used)

- **Role:** ⚠️ not used · **Path:** `data/raw/aci_bench/data/mts-dialog/` (arrives inside the ACI-Bench clone) · **License:** repo CC-BY-4.0, but the `NOTICE` file records the underlying mtsamples.com terms: *"feel free to print, share, link, and distribute… please notify us, and please give credit"*. That grants distribution with attribution but says nothing explicit about commercial use or derivatives, so by the ambiguity rule this subcollection is **Track NC** despite the repo licence.
- **Why it ended up unused.** 1.7k short written doctor/patient snippets (tier 3) whose only edge over ACI-Bench was being permissive; with the track downgraded it adds nothing ACI-Bench and the NC counselling corpora don't cover better. Revisit only if mtsamples grants explicit terms.

## CourtListener oral arguments (wave 2)

- **Role:** the Track P long-context tier 1 source: real ASR text at the document lengths this project targets, which nothing else permissive provides · **Path:** `data/raw/courtlistener/<court>/transcripts.jsonl` · **Tokens:** **51,936,162** Qwen3 (p_v1: 51,046,342 train + 889,820 val; 6 near-empty arguments dropped). Measured 2026-08-31: **scotus** complete, 11,150,036 words over 1,001 arguments (per-argument words min 5,327 / p25 9,589 / p50 10,327 / p75 11,889 / p95 16,772 / p99 20,558 / max 26,864 — unimodal, right-skewed, none under 500 words); **cadc** (DC Circuit) complete, 17,896,752 words over 2,604 arguments (min 237 / p50 6,048 / p95 13,528 / max 24,721; one record under 500 words); **ca1** (First Circuit) complete, 13,338,323 words over 2,913 arguments (p50 4,303 / max 21,889; five records under 500 words — inspect and drop empties at preprocessing). Combined: **42,385,111 words over 6,518 arguments**, no duplicate ids within any court. Roughly 76k further transcribed circuit arguments remain available; pulling them is a token-budget decision recorded in the survey queue · **Size:** 234 MB
- **License:** public domain. Free Law Project's bulk-data page marks its data free of known copyright restrictions, and Supreme Court oral-argument recordings are US federal government works. Sourced from the CourtListener API directly, not from Oyez (whose wrapper is CC BY-NC). · **Source:** https://www.courtlistener.com/api/rest/v4/audio/ (`docket__court=scotus`, `stt_status=1`)
- **Procured:** found in the 2026-08-31 corpus survey. Pulled by `scripts/download_courtlistener.py` (cursor-resumable, honours `Retry-After` on the anonymous rate limit; polite 10 s page interval). `stt_source=1` on every record: CourtListener's own Whisper-family transcription. MP3 `download_url` kept per record so a local diarised ASR pass remains possible later. Processing module: see the interim table above.

Sustained 20–90 minute two-sided argument between advocates and the bench: spontaneous, interrupted, high-register spoken English, transcribed by a real recogniser with restored punctuation. **Limitations, stated plainly**: the transcript field is flat text with no speaker turns (diarisation requires the audio, deferred under the tier-1-first policy), the register is formal advocacy rather than service calls, and it is not advisor/customer in role. It is used as long-context pretraining distribution, never as benchmark case text.

**Examples** (record 104587, verbatim; truncated):

1. `We'll hear argument next in Case 24-889, Hikma Pharmaceuticals v. Amarin Pharma. Mr. Klein. Thank you, Mr. Chief Justice, and may it please the Court. Under Section 271B of the Patent Act, selling a product suited for both infringing and substantial non-infringing uses is lawful, unless the seller actively induces the infringing use.`

## MeetingBank (wave 2)

- **Role:** the NC track's long-context tier 1 text, and summary supervision (each record pairs a transcript with a human summary) · **Path:** `data/raw/meetingbank/` (`{train,validation,test}.json`, JSONL despite the extension) · **Tokens:** **23,534,451** Qwen3 (nc_v1) over 1,249 reconstructed meetings (19,921,133 words; per meeting p50 10,819 / max 77,276 words — the tail exceeds any planned context and needs a chunking policy). Per agenda item (2026-08-31): min 69 / p50 986 / p95 12,093 / max 67,634 words · **Size:** 110 MB
- **License:** CC-BY-NC-SA-4.0 (dataset card) → **Track NC**. · **Source:** https://huggingface.co/datasets/huuuyeah/meetingbank — paper ACL 2023
- **Procured:** ranked 5th of the remaining tier 1 sources 2026-08-31; downloaded via `scripts/download_meetingbank.py` (ungated). Audio (`huuuyeah/MeetingBank_Audio`) deliberately not fetched. Processing module: see the interim table above.

1,366 US city-council meetings (3,579 h) transcribed by Speechmatics: modern ASR with restored punctuation and casing. **This HF distribution is per agenda item, not per meeting** — `uid` encodes `council_date_item`, so whole meetings are reconstructable by grouping, which is where the ~28k-token documents come from. **No speaker labels in these files**: the diarised segment layer ships with the audio release, so as downloaded this is long flat tier 1 text. Register is formal civic proceedings, not service calls.

**Examples** (train, verbatim; truncated):

1. `Please refrain from profane or obscene speech. Direct your comments to council as a whole and refrain from individual or personal attacks. Councilwoman Gilmore, will you please put Council Bill 161 on the floor? Yes, President Brooks, I move that council bill 161 as amended, be placed upon final consideration and do pass.`

## CallCenterEN (wave 2)

- **Role:** the NC track's in-domain anchor: real customer-service telephone calls in modern commercial ASR, the closest public match to the target distribution · **Path:** `data/raw/callcenteren/` (11 domain zips of per-call JSON) · **Tokens:** **99,437,272** Qwen3 (nc_v1) over 90,831 calls after removing 5,070 reupload duplicates and 43 sub-50-word fragments (66,897,886 words; p50 574 / max 19,077 words per call). Earlier 1,650-call sample: mean 1,068 words and 589 s audio per call · **Size:** 1.4 GB
- **License:** CC-BY-NC-4.0 (dataset card) → **Track NC**. Found **ungated** 2026-08-31, contrary to the survey's expectation. · **Source:** https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english — paper arXiv:2507.02958
- **Procured:** ranked 1st of the remaining tier 1 sources 2026-08-31; downloaded via `scripts/download_callcenteren.py`. Domains: medicare (largest, 61,513 calls), home services, automotive, insurance inbound/outbound, general customer service, medical equipment. Audio is withheld by the publisher (biometric privacy), so there is no tier-2 path. Processing module: see the interim table above.

Per-call JSON: flat `text`, overall `confidence`, `audio_duration`, word-level timings, and the list of redacted PII policies. Confirms the survey's tier resolution: this is AssemblyAI ASR output with restored punctuation and casing (the paper states 0.1% was human-reviewed at WER 3.87%).

Known quality problems, and they matter: **no speaker turns** — the `speaker` field is null on every sampled word, so agent and customer sentences are interleaved in one stream and turn structure would have to be inferred; **PII is replaced in-band with bracketed placeholders** (`[PERSON_NAME]`, `[ORGANIZATION]`, `[DATE]`…), often repeated per word of the redacted span, which is both a redaction artefact that must not be learned as a PII pattern and a systematic distortion of exactly the identifier-dictation phenomena this project cares about; the file count (95,953) exceeds the paper's 91,706 conversations because two "(reupload)" zips overlap the originals — dedupe by transcript id at preprocessing.

**Examples** (medicare_inbound, verbatim; truncated):

1. `Thank you for calling [ORGANIZATION] [ORGANIZATION] [ORGANIZATION]. This is [PERSON_NAME]. How can I help you today? Hi, I'm looking to get a grooming appointment. I had filled out paperwork yesterday and left a message to be called or texted back. Sure, just give me one second. Let me pull up your account.`

## SPoRC (wave 2)

- **Role:** the NC track's pretraining backbone: the only source at scale whose text has the target's structural shape — diarised `SPEAKER_NN` turns over long spontaneous conversations · **Path:** `data/raw/sporc/` (`turns/text/` 127 parquet, `episodes/` 140 parquet, manifest, READMEs; the acoustics and bulk metadata layers were deliberately not fetched) · **Tokens:** **4,373,956,442** Qwen3 (nc_v1) for the conversational subset — 364,970 episodes (357,690 train / 7,280 val), 135,269,142 turns, 2,810,792,362 words, p50 6,983 / max 186,327 words per episode. Full source: 4,585,128,049 by its own `token_count` over 185,218,224 turns and 1,124,058 episodes · **Size:** 28.5 GB
- **License:** gated research release (HF gate accepted by the account 2026-09-01) → **Track NC**. · **Source:** https://huggingface.co/datasets/blitt/SPoRC — the Structured Podcast Research Corpus
- **Procured:** ranked 3rd of the remaining tier 1 sources; text-only download via `scripts/download_sporc.py` 2026-09-01. Processing module: see the interim table above.

Whisper transcription with diarisation over open-RSS podcast audio. Per-turn schema: `episode_id`, `podcast_id`, `speaker` (`SPEAKER_00` convention, the same as the target distribution's), `turn_text`, start/end times, `token_count`, plus inferred speaker name/role where the pipeline could establish them. Turn lengths are strongly bimodal (sampled median 7 words, mean 40): backchannels interleaved with long monologues, which also matches the target's shape.

**Selection applied in `src/preprocessing/sporc.py` (2026-09-02)**: English; 2–7 distinct speaker labels per episode (drops 227,574 monologue episodes / 612M source tokens, and the 8+ label tail, which is mostly diarisation over-segmentation); ≥ 20 turns; turns of ≥ 15 tokens recurring verbatim in ≥ 3 episodes of the same podcast dropped (only 3.0M tokens, 0.07% — advertising is rarely verbatim-identical after ASR). SPoRC's own `num_main_speakers` is 0 for 751k episodes and was not usable. The label count is a proxy for true speaker count: a two-person interview fragmented into 9 labels is excluded, a monologue with one spurious label included; recorded as a limitation.

Known quality problems: podcast register (ads, intros, music segments transcribed as speech); diarisation quality varies with episode audio; `speaker` can list multiple ids on a merged turn (the first is used); the `SPEAKER_NN: ` prefixes cost roughly a fifth of the tokens (1.56 tokens/word vs 1.23 on flat text).

**Examples** (turns/text, verbatim; truncated):

1. `SPEAKER_00:  Welcome to the Big Mx Radio Podcast. Brought to you by MedTerra CBD. You can go to MedTerra CBD.com right now and enter discount code Big Mx Radio 15 to say 15% on every single one of your orders.`

Keep this file current when a dataset is added, removed, or re-scoped.
