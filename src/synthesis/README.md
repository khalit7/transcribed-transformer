# src/synthesis — LLM labelling

**Terminology.** *LLM labelling* is a model answering a question about a transcript, producing answer, evidence and summary; that model is the **labeller**, and it is all this package does. *LLM-as-a-judge* is a different step: a model assessing how good a label is; that model is the **judge**. The blind second labeller behind `--verify-model` and the ablation re-labelling are still labelling, not judging. The judging step is `grade_labels.py` (Opus grades each labeller's outputs) and `analyse_labellers.py` (the report and the human-calibration loop); see `experiments/2026-09-03-labeller-selection/`.

Produces (transcript, question) → (evidence, answer, summary) examples from real public call transcripts. The same records serve fine-tuning and benchmarking; how they are split between the two is decided downstream, not by this package. The flow is deliberately simple: **pick a transcript at random, then ask it every bank question whose `dataset_allow_list` includes the transcript's dataset, and have a labeller answer each.** Questions are not sampled: if a question makes sense for a dataset it is asked of every call drawn from it, so each call carries a complete answer sheet and re-running after a bank change fills in only the new pairs. No text is written into calls; every label is produced by LLM labelling of a real conversation.

## Which labeller labels what

Decided 2026-09-04 after the labeller-selection study (`experiments/2026-09-03-labeller-selection/`): five labellers (Claude Sonnet and four local models) labelled the same 1,820 (call, question) pairs, Opus judged 400 of them on answer, evidence, summary and tags, and Khalid read the judgements and accepted them. Sonnet was best on every criterion (answers 0.85, evidence 1.60/2, summary 1.81/2, tags 1.80/2); qwen3.8 (Qwen3.5-27B, run locally with Ollama) was closest (0.81 / 1.45 / 1.72 / 1.54) at 47 labels per minute for nothing, but clearly weaker on rare-event recall (0.53 vs 0.73) and on vulnerability tags. So:

- **`general_qa` is labelled by `ollama:qwen3.8`** (the bulk of the labels, free).
- **`vulnerability`, `complaint` and `eod` are labelled exclusively by `claude:sonnet`** (the rare-event labels the compliance use case exists for; ~5% of all labels).

The two runs, in either order, over the same transcripts:

```
uv run python -m src.synthesis.synth_data ollama:qwen3.8 200 general_qa
uv run python -m src.synthesis.synth_data claude:sonnet 200 vulnerability,complaint,eod --claude-account w
uv run python -m src.synthesis.export --track p                                          # releasable copy
```

`synth_data` takes three required arguments: the labeller, the number of transcripts, and the comma-separated **question families** to label (`vulnerability`, `complaint`, `eod`, `general_qa`). Transcript selection depends only on the seed and the pool (`--seed`, `--pool-per-source`, defaults 0 and 200), never on the model or the families, so both runs cover the same calls and each call ends up with its general_qa labels from qwen3.8 and its rare-event labels from Sonnet; `generation_info.name` on every record says which. Both append to the same `labelled_data.jsonl`. **No pair is ever labelled twice**: ids (`<call>::<question>`) already in the file are skipped whoever wrote them, which also means the first model to label a pair keeps it, so run each family with its intended labeller only. Use the same `--seed` and `--pool-per-source` on both runs and on any resume. **Two Claude accounts**: every entry point that calls `claude -p` (`synth_data`, `probe_bank`, `identify_speakers`, `grade_labels`) takes `--claude-account p|w` (personal or work; it sets `CLAUDE_CONFIG_DIR` the way the `p_claude`/`w_claude` shell aliases do), and the account used is recorded on each record (`generation_info.claude_account`, and on judgements and host identifications). Without the flag the environment's default account is used and `null` is recorded.

The size argument counts **transcripts**, not labels: with the current bank a call yields 110 labels on AppTek, 68 on Taskmaster, 61 on ACI-Bench and 53 on SPoRC. The model argument names the backend: `claude:<model>` runs `claude -p` (cost tracked from the CLI, ~$0.06–0.10 a label with Sonnet); `ollama:<model>` runs a local model (cost 0; `qwen3:32b`, `llama3.3:70b`, `gemma3:27b`, `deepseek-r1:70b` are pulled). Prompts put the transcript first and the question last so a caching server can reuse the prefix across questions on the same call. Labeller calls run concurrently: `--workers` defaults to 16 for `claude:` (independent API calls). For `ollama:` **every GPU is used automatically**: `synth_data` starts (or reuses) one Ollama server pinned to each GPU (`ollama_servers.py`; user processes, logs and pids under `/tmp/ollama-pinned`, left running and idle afterwards), runs one worker per server, and sends consecutive calls to alternate servers so each call's questions stay on one server and its transcript prefix stays cached. Measured 2026-09-04 with qwen3.8 on general_qa: 47 labels/min on one GPU, **83 on two**. Parallel slots on one server were tried and removed: Ollama 0.32 serves Qwen3.5 one request at a time. Set `OLLAMA_URLS` to use specific servers instead. Records are appended as they complete, so file order is not label order; ids are what matter. A Claude rate or usage limit does not fail the run: the worker that hits it sleeps until the reset the CLI reports (or backs off 30 s → 10 min, for up to 4 h), prints one line per wait, and resumes; only malformed output is retried immediately and then counted as a failure. A killed or failed run is resumed by re-running the same command: existing ids are skipped. `labelled_data.jsonl` is the ledger of what has been labelled: ids are `<call>::<question>`, call ids derive from the source files, and a call's rendering is a pure function of the source (verbatim roles, no run-dependent choices), so the same call always appears identically no matter which run produced each of its labels.

## Train, val and benchmark

The ledger serves both training and benchmarking; `split.py` decides which record goes where, and `benchmark.py` builds the three files. Rules (2026-09-08):

- **Split by transcript, never by (transcript, question) pair.** A call in both train and benchmark leaks the whole conversation. SPoRC groups by podcast: every episode of one podcast lands on the same side (31 podcasts contribute more than one episode).
- **Only fully labelled calls can be benchmark or val**: every question allowed for the call's dataset has a label. Calls that lost a pair to a labeller failure, and the Taskmaster/SPoRC calls with family labels only, go to train.
- **Benchmark calls are the best of 3,000 seeded draws plus 20,000 swap refinements**, scored on proportional coverage of the dataset's strata (AppTek domain and locale; SPoRC category; ACI-Bench source split), a floor of 30 rare-event positive calls per family (answer `fail` on a family question) where the pool allows, and, for SPoRC, at least 15 of the 60 episodes above the pool's p90 length (Qwen3 tokens of the rendered transcript).
- **About 20% of the general_qa question lineages are held out of training per dataset** (a lineage is a question plus its Taskmaster word-substituted `tm-` variant, which is the same question). Pairs with a held-out question are dropped from train and val; on benchmark calls they form the **unseen-question cell**, the one the zero-shot claim rests on. The vulnerability, complaint and eod questions are never held out (Khalid's decision: they stay in training).
- **Val** is a small in-distribution slice of the remaining fully labelled calls, same question set as train, for early stopping and model selection only. It is never reported.
- **Benchmark labels are reconciled, train labels are not.** Every benchmark general_qa pair gets a second, independent local label (`relabel.py ollama:gemma4:12b`); where it agrees with the ledger's qwen3.8 answer the label stands with the union of both evidence keys, where it disagrees Opus labels the pair and its label is the gold label (`relabel.py claude:opus --disagree-with`). The family labels (Sonnet) get an Opus re-label on every rare-event positive. Opus, not Sonnet, adjudicates because Sonnet is the API baseline and must not be graded against labels it wrote. Every benchmark record carries every labeller's answer (`provenance.labels`) and how the gold label was settled (`single | agreement | adjudicated`). The ground truth is model-generated and the benchmark write-up says so.

Frozen assignment (`splits.json`, seed 0, 2026-09-08), calls per dataset:

| Dataset | Fully labelled | Benchmark | Val | Train | Excluded |
|---|---|---|---|---|---|
| AppTek | 870 | 100 | 40 | 733 | 0 |
| ACI-Bench | 206 | 40 | 10 | 156 | 0 |
| Taskmaster | 206 | 60 | 15 | 1,145 | 0 |
| SPoRC | 201 | 60 | 16 | 1,130 | 3 |

Excluded = partially labelled sibling episodes of a benchmark podcast (neither side). Held out: 41 general_qa lineages, 51 question ids (AppTek 20 of 100 general_qa questions, Taskmaster 13 of 64, ACI-Bench 12 of 58, SPoRC 10 of 52). Pairs: benchmark 20,700 (seen-question cell 16,840, unseen 3,860), val 5,561, train 91,502, dropped 20,808 (held-out questions on train/val calls). Benchmark positives: AppTek complaint 40 / eod 33 / vulnerability 31 calls, ACI-Bench 6 / 5, Taskmaster eod 5 / vulnerability 12 (complaint has 2 positives in the whole corpus and is not measurable there), SPoRC vulnerability 23. SPoRC benchmark median 6,786 tokens, max 15,018 (cases are capped at 160 turns), 15 above the pool's p90 of 10,595.

```
uv run python -m src.synthesis.split                      # once; refuses to overwrite (--force rebuilds = a new benchmark)
uv run python -m src.synthesis.relabel ollama:gemma4:12b --families general_qa --out data/labelled_data/benchmark/second_gemma4-12b.jsonl
uv run python -m src.synthesis.relabel claude:opus --families general_qa --disagree-with data/labelled_data/benchmark/second_gemma4-12b.jsonl --out data/labelled_data/benchmark/adjudicate_opus_general.jsonl --claude-account p
uv run python -m src.synthesis.relabel claude:opus --families vulnerability,complaint,eod --answers fail --out data/labelled_data/benchmark/adjudicate_opus_families.jsonl --claude-account p
uv run python -m src.synthesis.benchmark                  # train.jsonl, val.jsonl, benchmark.jsonl, benchmark_summary.json
```

Anything labelled after the freeze joins train (`split.assign` sends unknown calls there). Multi-call cases, when they are composed, must be composed from one side only, and `splits.json` is the single file both composers read. Reconciliation results (2026-09-09). Second pass, gemma4:12b on the 19,280 benchmark general_qa pairs: 19,276 labels, 4 loop failures, 3.0 h on two GPUs (107 labels/min); agreed with qwen3.8 on **15,462 (80.2%)**: AppTek 79.2%, Taskmaster 82.0%, ACI-Bench 79.6%, SPoRC 81.5%; seen-question cell 80.0% vs unseen 81.1%. Disagreements have no dominant shape (fail/pass, NA/fail and pass/fail each about 645 pairs). Evidence keys of agreeing labellers overlap at Jaccard 0.55, hence the union rule. Opus adjudicated the 3,818 disagreements and missing pairs ($587.02, $0.154 per label): it sided with qwen3.8 on 2,155 (56%), with gemma4 on 1,284 (34%) and with neither on 379 (10%). Opus also re-labelled the 194 family positives ($32.13): it agreed with Sonnet's `fail` on 126 (65%). Net effect: 1,663 general_qa gold answers (8.6%) and 68 family gold answers differ from the ledger. Gold positives per family after adjudication (calls; Sonnet's count in brackets): AppTek complaint 30 (40), dissatisfaction 26 (33), vulnerability 18 (31); SPoRC vulnerability 21 (23); ACI-Bench 3 (6) and 3 (5); Taskmaster dissatisfaction 2 (5), vulnerability 3 (12). Gold labels with no evidence: 2,045 NA, 669 pass, 193 fail. **Policy (2026-09-09): empty means empty.** An absence answer carries no evidence; an empty gold key is scored literally (the model must emit `[]`, any cited line is a false positive), training records are used as labelled, nothing is re-labelled or filled with near-miss lines. Labellers were not told this, so some absence answers do cite near-miss lines; that inconsistency is a known limitation of the gold, not corrected. Opus total for the benchmark: $619.15.

## What a label is, and the checks around it

The labeller is asked for **evidence first, then the answer, then the summary**, plus its own 0–1 **confidence** and, for questions that define one, **tags** from a closed vocabulary (the vulnerability question tags the FCA FG21/1 characteristic(s) present). Evidence-first ordering is what improved citation quality in the literature; confidence is what routes items to human audit later.

Two optional checks, each stored on the record:

- `--verify-model M` — a second labeller (use a different model family) answers the same pair **blind**; `verification.agrees` records whether the answers match. Disagreement is a routing signal for human review, not a vote.
- `--ablate` — the primary labeller re-answers with the cited lines blanked (**necessary** if the answer changes) and with only the cited lines kept (**sufficient** if the answer holds). Evidence keys are never guaranteed complete, so this is how far an evidence key can be trusted beyond the labeller's word.

## Files

| Module | Role |
|---|---|
| `schema.py` | `Question`, `Transcript`/`Variant`, `Case`, `Label`, `Verification`, `Ablation`, `Generation`, `LabelledRecord`, `Provenance`, `BenchmarkRecord` — pydantic, validated |
| `question_bank.py` | the bank (source of truth); `write_questions()` derives `questions.jsonl` |
| `cases.py` | builds calls from what is on disk, each as line-aligned `clean`/`messy` variants whose lines carry the corpus's own speaker role labels verbatim; raises `NoSpeakerRoles` for a corpus without them |
| `label.py` | the labelling prompt, the JSON schema (enforced server-side on Ollama), `label()`, `verify()` (second labeller, blind), `ablate()` (re-labelling with evidence removed / kept) |
| `llm.py` | `ask_json()` over the two backends |
| `synth_data.py` | the CLI loop |
| `identify_speakers.py` | names the host among a SPoRC episode's diarised speakers (Sonnet), cached per episode; the SPoRC builder renders that speaker as `host` |
| `probe_bank.py` | answers every bank question on a sample of real calls (one call per prompt; `--dataset` to restrict to a dataset's allowed questions; `--workers N` for concurrent `claude -p` calls) and reports each question's answer distribution, to catch questions that always get the same answer |
| `audit_bank.py` | reconciles probe files from several labellers per dataset: flags each (question, dataset) cell NA-dominant or skewed by labeller majority, and diffs the proposal against the bank's allow lists |
| `grade_labels.py` | **LLM-as-a-judge**: for a stratified sample of pairs labelled by several labellers, the judge (Opus) answers the question itself, then grades every label's answer, evidence, summary and tags, labels anonymised and shuffled per pair; one call per pair, ~$0.25–0.30 with Opus |
| `analyse_labellers.py` | per-labeller report from the judgements (answer accuracy, rare-event recall, NA↔fail confusion, evidence and summary grades, tag grades, cost and speed); blind human-review export/import and judge-vs-human agreement |
| `split.py` | freezes the call assignment (train / val / benchmark, grouped by podcast for SPoRC) and the held-out question lineages into `splits.json`; `assign()` is the one place the rule lives |
| `relabel.py` | labels the benchmark pairs again with another labeller: a second opinion (`ollama:gemma4:12b`) or an adjudication of the disagreements and the rare-event positives (`claude:opus`); one file per labeller under `benchmark/` |
| `benchmark.py` | writes `train.jsonl`, `val.jsonl` and `benchmark.jsonl` (gold label reconciled from the labellers, provenance and seen/unseen-question cell on every record) plus `benchmark_summary.json` |
| `export.py` | track-filtered release copy |

## Data format

`data/labelled_data/labelled_data.jsonl` — one self-contained record per label:

```
id               "<call_id>::<question_id>"
dataset          apptek | taskmaster | aci_bench | sporc
source_id        corpus/config/locale/document the call came from
track            track-p | track-nc
question         {id, source, family: vulnerability|complaint|eod|general_qa, text, description,
                  options: [{value: pass|fail|partial_pass|NA, criteria}], tags: [allowed qualifiers],
                  dataset_allow_list: [datasets the question may be asked of]}
transcript       {variants: [{kind: clean|messy, origin, lines: ["<role>: ...", ...]}],
                  speakers: [role labels that occur, verbatim from the corpus], role_source}
label            {evidence: [1-based lines], answer, summary, tags: [from question.tags], confidence: 0-1}
verification     {model, answer, evidence, tags, agrees} | null
ablation         {model, necessary, sufficient} | null
generation_info  {name: backend:model, labelled_variant: clean|messy, cost_usd, timestamp, claude_account: p|w|null}
meta             call metadata from the source corpus
```

**Model input format (fixed 2026-09-04).** Every labeller and the judge receive the transcript rendered as one turn per line, each line `<n>: <role>: <text>` with `n` a 1-based line number and `role` the corpus's verbatim speaker label, e.g. `1: agent: Hi. How can I help you?` / `2: customer: Hi, I'm calling about my policy.` The stored record keeps the unnumbered lines (`transcript.variants[].lines`, `<role>: <text>`); numbering is applied at prompt time by `label.numbered()`. Evidence is a list of those line numbers, so **the same rendering (number, role, text) is the input format for the benchmark and for fine-tuning**: any model evaluated or trained on these labels must see the transcript numbered exactly this way, or the evidence keys stop meaning what they meant. Clean and messy variants are line-aligned, so line *i* is the same turn in both.

Variants are line-aligned, so one evidence key serves both and the clean-vs-messy gap per system is measurable directly; `generation_info.labelled_variant` says which one the labeller read (default clean, the more reliable label). Speaker roles are the corpus's own labels, rendered on every line. `questions.jsonl` is derived from the bank and regenerated after each run.

## Case sources

| Source | Track | speaker labels (verbatim) | clean variant | messy variant |
|---|---|---|---|---|
| AppTek Call-Center Dialogues | P (SA) | `agent`, `customer` | verbatim segments | **real** Whisper output over telephone-degraded audio, time-aligned to the verbatim turns |
| Taskmaster-1/2 | P | `assistant`, `user` | human transcription | channel v2.2 noised (synthetic) |
| ACI-Bench | P | `doctor`, `patient`, `patient_guest` | cleaned dialogue | channel v2.2 noised (synthetic) |
| SPoRC | NC | `host` (identified, see below) + `SPEAKER_NN` for the others | — | real diarised ASR (the only variant) |

**Speaker roles are always the corpus's own labels, rendered verbatim on every line** (decision 2026-09-03). No remapping to `agent`/`customer`, no `SPEAKER_NN` randomisation, no glossing note to the labeller. SPoRC records no roles, so its **host is identified first** (`identify_speakers.py`: Sonnet reads the opening 60 and closing 12 turns and names the host tag; cached in `data/labelled_data/speakers/sporc.jsonl` with confidence and reason; identified on demand for uncached episodes, ~$0.09 each) and rendered as `host:`; the other speakers keep their diarisation tags. An episode whose host cannot be identified is skipped, never rendered role-less; a corpus with no roles and no identification raises `NoSpeakerRoles`.

## The bank

**Every question is written in the speaker vocabulary of the transcripts it is asked of, and a question is shared across corpora only when its wording already applies verbatim** (decision 2026-09-03; the target setting has a different speaker set and question set per deployment, so nothing is gained by forcing one wording onto every corpus). 256 questions (bank v3, 2026-09-04); every question carries a **`dataset_allow_list`**, and since every allowed question is asked of every call, that list is a hard gate on which labels exist:

| Set | Vocabulary | Datasets | Count | What |
|---|---|---|---|---|
| `vul-`, `cmp-`, `eod-` | agent / customer | apptek | 10 | vulnerability present (tagged with FCA FG21/1 characteristics), handled, support offered; complaint made, acknowledged, apologised for, escalation offered, timescale given; dissatisfaction expressed, its cause asked |
| `gen-` | agent / customer | apptek | 82 | service-call conduct: opening, identity and data, needs and sales conduct, explanation, handling (holds, callbacks, ownership), manner, closing, plus domain-conditional items (rates, contract length, excess, first payment, visit windows) |
| `tm-` | assistant / user | taskmaster | 58 | the four family questions and 41 conduct behaviours in Taskmaster's vocabulary (derived from the `gen-` text by substitution where the conduct is identical) plus 13 booking-specific questions (`tm-101…`: party size, date confirmed, alternatives, price before booking, special requirements, final confirmation, ambiguity resolved, changed mind, repeated questions, options, restrictions, payment details, search narrated) |
| `aci-` | doctor / patient | aci_bench | 48 | complaint made and acknowledged, dissatisfaction, and 45 conduct behaviours written for a clinical visit (side effects, the plan, consent before tests or prescriptions, when to seek urgent care, …) |
| `spk-` | "the speakers" | all four, per audit | 32 | role-agnostic conversation conduct (introductions, speaking time, interruptions, clarification, jargon, claims, figures, disclosures, advice and caveats, promotion, summary, calls to action, questions answered, topic shifts, sensitive topics, listener address, expertise, offensive language, claims challenged) plus, for podcasts, whether any speaker discloses a vulnerability about themselves |
| `hst-` | host | sporc | 21 | host behaviour: sponsor message read and separated, host named, guest introduced, thanked and pointed to, floor shared, restating, self-promotion, listener messages, conflicts disclosed, guest corrected, open vs leading questions, balance, opinion marked, content warning, episode context, preparation, time management, talking over guests |

Labels per call: AppTek 110, Taskmaster 68, ACI-Bench 61, SPoRC 53. Vulnerability was not given an ACI-Bench form: a patient is trivially in the FCA health driver, and a "beyond the presenting condition" version would be forcing the question.

**Audit (2026-09-03, first pass).** Before the vocabulary alignment, the 79 then-shared questions were probed on the same 40 real calls per dataset with three labellers (qwen3:32b, gemma3:27b, Claude Sonnet; `probe_bank.py`, seed 3, `--family all`) and reconciled with `audit_bank.py`. Rules: a dataset leaves a question's allow list when the labellers' majority answers NA on ≥75% of calls (the question has no occasion to arise there) or when one answer covers ≥90% of calls *and* the rare answer carries no weight (trivial skew). Skew alone is not grounds for removal: questions whose rare answer matters (vulnerability, complaint, dissatisfaction, offensive language, uncaveated advice, an unexplained refusal) are kept as rare-event detection tests, and their positives will need oversampling when sets are assembled. Six questions left the bank (NA-dominant or trivially skewed everywhere, or labellers contradicting each other on every call: `gen-38`, `gen-42`, `spk-02`, `spk-09`, `spk-19`, `spk-25`) and 39 allow lists were narrowed; that per-dataset coverage is what the `tm-` and `aci-` sets now encode. Labeller agreement on the same majority answer was 63–68% pairwise; gemma3:27b was markedly more lenient (mean pass share 0.62 vs 0.44 qwen and 0.41 Sonnet), which is why decisions are taken by majority. Probe files are in `data/labelled_data/probes/`.

**Second pass (2026-09-03, aligned bank).** The aligned bank was re-probed on the identical 40 calls per dataset under the verbatim-role rendering with the host identified, each call seeing only its allowed questions (`--dataset`), same three labellers, 480 calls, 0 failures, Sonnet $28.92. AppTek and Taskmaster cells were stable against the first pass (only threshold-edge changes), so the substitution-derived `tm-` wording behaves like the original; ACI-Bench moved most, as expected where the questions were actually rewritten, and mostly towards being answerable (`aci-28-clarifying-question` 75% NA → 78% pass, `aci-49-review-narrated` 65% NA → 97% pass). Applied by the same rules: `spk-08`, `spk-11`, `spk-13`, `spk-16` left Taskmaster (90–100% pass under all labellers: bookings have no figures to be vague about, no advice, no promotion, no call to action); `aci-17`, `aci-48`, `tm-48`, `hst-05`, `hst-07` were dropped as trivially skewed; `aci-06` and `aci-25` were dropped and `spk-22` narrowed to SPoRC because the labellers contradicted each other on every call. Kept although skewed: every family question, `vul-02`/`tm-vul-02` (NA by construction), `gen-31`/`tm-31`, `spk-12` on AppTek and SPoRC, `spk-17`, `spk-23`, `tm-23`, `tm-29`, `aci-14`, `aci-16`, `aci-37`. Labeller agreement was unchanged (same majority answer 64–66% pairwise; mean pass share gemma 0.63, qwen 0.46, Sonnet 0.49). Rare-event counts per 40 calls (Sonnet): vulnerability 6 AppTek / 0 Taskmaster; complaint 3 / 0 / 1 (ACI); dissatisfaction 4 / 2 / 3; a speaker disclosing a vulnerability about themselves on SPoRC 17 (qwen 3, gemma 9: the labellers read "disclosure" very differently there); the host reading a sponsor message 8 (local labellers 26–27). Bank v2: **146 questions**; labels per call AppTek 68, Taskmaster 46, ACI-Bench 39, SPoRC 29. Reconciled tables: `audit_seed3_reconciled.txt` (first pass), `audit_seed3_roles_reconciled.txt` (second), `audit_seed3_pass1_vs_pass2.txt` (per-labeller diff).

**Bank v3 (2026-09-04).** 148 further questions were authored after the labeller-selection study and the 600-call labelling, in each corpus's vocabulary and from what those calls actually contain: sales conduct, data handling, holds and callbacks, domain-conditional items for AppTek's 16 service domains, booking mechanics for Taskmaster, clinical communication (medication instructions, allergies, safety-netting, shared decisions, red flags) for ACI-Bench, and host conduct for podcasts, plus six rare-event family questions (complaint acknowledged / apologised / escalated / timescale, dissatisfaction cause asked, support offered on a disclosed vulnerability). Probed on the same 40 calls per dataset with qwen3.8 (both GPUs) and Sonnet, reconciled by `audit_bank.py`. Rules as before, with one refinement: a domain-conditional question is dropped only when both labellers answer NA on ≥ 90% of calls (applicability under ~10%), while 75–90% NA is kept, since a rate or an excess can only arise on the banking or insurance calls. 38 dropped (22 NA-dominant, 16 trivially skewed such as a recording notice that role-played calls never give, a goodbye they always give, or a teach-back doctors never do), 110 kept, six family questions kept regardless. Files: `data/labelled_data/probes/audit_<dataset>_{qwen3_8,sonnet}-v3_seed3.json`, `v3_reconciled.txt`. Sonnet probe cost $24.94.

To add a question: a `q(...)` entry in `question_bank.py` in the vocabulary of its dataset (pass and fail required; `partial_pass`/`NA` where meaningful; optional `description`, `tags`; `datasets=` defaults to `apptek`), then `python -m src.synthesis.probe_bank --ids <id> --sources <ds>:40` to check it varies, then `python -m src.synthesis.question_bank`.

## Known limitations

- Labels are LLM labels; on the audit probe three labellers gave the same majority answer on only 63–68% of (question, dataset) cells, and a local 27B labeller was markedly more lenient than the other two. A human gold slice is the only accuracy estimate that counts; the labeller-selection study (`experiments/2026-09-03-labeller-selection/`) is the closest thing so far.
- Rare-event questions (complaint, dissatisfaction, vulnerability on service calls) fire on roughly 1 call in 10 to 1 in 40 of the public corpora; none of the on-disk data is complaint-heavy.
- Evidence is precision-only; `ablation` says how far a key can be trusted, it does not complete it.
- Single-call cases; multi-call cases with any/all semantics are future work.
- Benchmark ground truth is model-generated (two local labellers plus Opus adjudication for general_qa; Sonnet plus an Opus re-label of the positives for the families). Agreement between labellers is not correctness; the labeller-selection study's judge-vs-unanimous-labellers figure (196/204) is the only calibration of that.
