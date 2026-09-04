# src/synthesis — LLM labelling

**Terminology.** *LLM labelling* is a model answering a question about a transcript, producing answer, evidence and summary; that model is the **labeller**, and it is all this package does. *LLM-as-a-judge* is a different step: a model assessing how good a label is; that model is the **judge**. The blind second labeller behind `--verify-model` and the ablation re-labelling are still labelling, not judging. The judging step is `grade_labels.py` (Opus grades each labeller's outputs) and `analyse_labellers.py` (the report and the human-calibration loop); see `experiments/2026-09-03-labeller-selection/`.

Produces (transcript, question) → (evidence, answer, summary) examples from real public call transcripts. The same records serve fine-tuning and benchmarking; how they are split between the two is decided downstream, not by this package. The flow is deliberately simple: **pick a transcript at random, then ask it every bank question whose `dataset_allow_list` includes the transcript's dataset, and have a labeller answer each.** Questions are not sampled: if a question makes sense for a dataset it is asked of every call drawn from it, so each call carries a complete answer sheet and re-running after a bank change fills in only the new pairs. No text is written into calls; every label is produced by LLM labelling of a real conversation.

```
uv run python -m src.synthesis.synth_data claude:sonnet 20        # 20 transcripts, every applicable question each
uv run python -m src.synthesis.synth_data ollama:qwen3:32b 200 --verify-model claude:sonnet --ablate
uv run python -m src.synthesis.export --track p            # releasable copy
```

The size argument counts **transcripts**, not labels: with the current bank a call yields 68 labels on AppTek, 46 on Taskmaster, 39 on ACI-Bench and 29 on SPoRC. The model argument names the backend: `claude:<model>` runs `claude -p` (cost tracked from the CLI, ~$0.06–0.10 a label with Sonnet); `ollama:<model>` runs a local model (cost 0; `qwen3:32b`, `llama3.3:70b`, `gemma3:27b`, `deepseek-r1:70b` are pulled). Prompts put the transcript first and the question last so a caching server can reuse the prefix across questions on the same call. Labeller calls run concurrently: `--workers` defaults to 16 for `claude:` (independent API calls; one SPoRC episode's 25 labels took 38 s, $2.66) and 1 for `ollama:` (one GPU, extra workers only queue). Records are appended as they complete, so file order is not label order; ids are what matter. A Claude rate or usage limit does not fail the run: the worker that hits it sleeps until the reset the CLI reports (or backs off 30 s → 10 min, for up to 4 h), prints one line per wait, and resumes; only malformed output is retried immediately and then counted as a failure. A killed or failed run is resumed by re-running the same command: existing ids are skipped. `labelled_data.jsonl` is the ledger of what has been labelled: ids are `<call>::<question>`, call ids derive from the source files, and a call's rendering is a pure function of the source (verbatim roles, no run-dependent choices), so the same call always appears identically no matter which run produced each of its labels.

## What a label is, and the checks around it

The labeller is asked for **evidence first, then the answer, then the summary**, plus its own 0–1 **confidence** and, for questions that define one, **tags** from a closed vocabulary (the vulnerability question tags the FCA FG21/1 characteristic(s) present). Evidence-first ordering is what improved citation quality in the literature; confidence is what routes items to human audit later.

Two optional checks, each stored on the record:

- `--verify-model M` — a second labeller (use a different model family) answers the same pair **blind**; `verification.agrees` records whether the answers match. Disagreement is a routing signal for human review, not a vote.
- `--ablate` — the primary labeller re-answers with the cited lines blanked (**necessary** if the answer changes) and with only the cited lines kept (**sufficient** if the answer holds). Evidence keys are never guaranteed complete, so this is how far an evidence key can be trusted beyond the labeller's word.

## Files

| Module | Role |
|---|---|
| `schema.py` | `Question`, `Transcript`/`Variant`, `Case`, `Label`, `Verification`, `Ablation`, `Generation`, `LabelledRecord` — pydantic, validated |
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
generation_info  {name: backend:model, labelled_variant: clean|messy, cost_usd, timestamp}
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

**Every question is written in the speaker vocabulary of the transcripts it is asked of, and a question is shared across corpora only when its wording already applies verbatim** (decision 2026-09-03; the target setting has a different speaker set and question set per deployment, so nothing is gained by forcing one wording onto every corpus). 146 questions; every question carries a **`dataset_allow_list`**, and since every allowed question is asked of every call, that list is a hard gate on which labels exist:

| Set | Vocabulary | Datasets | Count | What |
|---|---|---|---|---|
| `vul-`, `cmp-`, `eod-` | agent / customer | apptek | 4 | vulnerability present (tagged with FCA FG21/1 characteristics) and handled; complaint; dissatisfaction |
| `gen-` | agent / customer | apptek | 48 | service-call conduct: opening, identity and data, needs, explanation, manner, handling, closing |
| `tm-` | assistant / user | taskmaster | 40 | the four family questions and the 36 conduct behaviours the audit found meaningful in task-oriented bookings, in Taskmaster's vocabulary (derived from the `gen-` text by substitution, since the conduct is identical) |
| `aci-` | doctor / patient | aci_bench | 26 | complaint and dissatisfaction about care, and 24 conduct behaviours written for a clinical visit (side effects, the plan, consent before tests or prescriptions, when to seek urgent care, …) |
| `spk-` | "the speakers" | all four, per audit | 22 | role-agnostic conversation conduct (introductions, speaking time, interruptions, clarification, jargon, claims, figures, disclosures, advice and caveats, promotion, summary, calls to action, questions answered, topic shifts, sensitive topics, listener address, expertise, offensive language, claims challenged) plus, for podcasts, whether any speaker discloses a vulnerability about themselves |
| `hst-` | host | sporc | 7 | host behaviour: sponsor message read, host named, guest introduced, floor shared, restating, self-promotion, listener messages |

Labels per call: AppTek 68, Taskmaster 46, ACI-Bench 39, SPoRC 29. Vulnerability was not given an ACI-Bench form: a patient is trivially in the FCA health driver, and a "beyond the presenting condition" version would be forcing the question.

**Audit (2026-09-03, first pass).** Before the vocabulary alignment, the 79 then-shared questions were probed on the same 40 real calls per dataset with three labellers (qwen3:32b, gemma3:27b, Claude Sonnet; `probe_bank.py`, seed 3, `--family all`) and reconciled with `audit_bank.py`. Rules: a dataset leaves a question's allow list when the labellers' majority answers NA on ≥75% of calls (the question has no occasion to arise there) or when one answer covers ≥90% of calls *and* the rare answer carries no weight (trivial skew). Skew alone is not grounds for removal: questions whose rare answer matters (vulnerability, complaint, dissatisfaction, offensive language, uncaveated advice, an unexplained refusal) are kept as rare-event detection tests, and their positives will need oversampling when sets are assembled. Six questions left the bank (NA-dominant or trivially skewed everywhere, or labellers contradicting each other on every call: `gen-38`, `gen-42`, `spk-02`, `spk-09`, `spk-19`, `spk-25`) and 39 allow lists were narrowed; that per-dataset coverage is what the `tm-` and `aci-` sets now encode. Labeller agreement on the same majority answer was 63–68% pairwise; gemma3:27b was markedly more lenient (mean pass share 0.62 vs 0.44 qwen and 0.41 Sonnet), which is why decisions are taken by majority. Probe files are in `data/labelled_data/probes/`.

**Second pass (2026-09-03, aligned bank).** The aligned bank was re-probed on the identical 40 calls per dataset under the verbatim-role rendering with the host identified, each call seeing only its allowed questions (`--dataset`), same three labellers, 480 calls, 0 failures, Sonnet $28.92. AppTek and Taskmaster cells were stable against the first pass (only threshold-edge changes), so the substitution-derived `tm-` wording behaves like the original; ACI-Bench moved most, as expected where the questions were actually rewritten, and mostly towards being answerable (`aci-28-clarifying-question` 75% NA → 78% pass, `aci-49-review-narrated` 65% NA → 97% pass). Applied by the same rules: `spk-08`, `spk-11`, `spk-13`, `spk-16` left Taskmaster (90–100% pass under all labellers: bookings have no figures to be vague about, no advice, no promotion, no call to action); `aci-17`, `aci-48`, `tm-48`, `hst-05`, `hst-07` were dropped as trivially skewed; `aci-06` and `aci-25` were dropped and `spk-22` narrowed to SPoRC because the labellers contradicted each other on every call. Kept although skewed: every family question, `vul-02`/`tm-vul-02` (NA by construction), `gen-31`/`tm-31`, `spk-12` on AppTek and SPoRC, `spk-17`, `spk-23`, `tm-23`, `tm-29`, `aci-14`, `aci-16`, `aci-37`. Labeller agreement was unchanged (same majority answer 64–66% pairwise; mean pass share gemma 0.63, qwen 0.46, Sonnet 0.49). Rare-event counts per 40 calls (Sonnet): vulnerability 6 AppTek / 0 Taskmaster; complaint 3 / 0 / 1 (ACI); dissatisfaction 4 / 2 / 3; a speaker disclosing a vulnerability about themselves on SPoRC 17 (qwen 3, gemma 9: the labellers read "disclosure" very differently there); the host reading a sponsor message 8 (local labellers 26–27). Final bank: **146 questions**; labels per call AppTek 68, Taskmaster 46, ACI-Bench 39, SPoRC 29. Reconciled tables: `audit_seed3_reconciled.txt` (first pass), `audit_seed3_roles_reconciled.txt` (second), `audit_seed3_pass1_vs_pass2.txt` (per-labeller diff).

To add a question: a `q(...)` entry in `question_bank.py` in the vocabulary of its dataset (pass and fail required; `partial_pass`/`NA` where meaningful; optional `description`, `tags`; `datasets=` defaults to `apptek`), then `python -m src.synthesis.probe_bank --ids <id> --sources <ds>:40` to check it varies, then `python -m src.synthesis.question_bank`.

## Known limitations

- Labels are LLM labels; on the audit probe three labellers gave the same majority answer on only 63–68% of (question, dataset) cells, and a local 27B labeller was markedly more lenient than the other two. A human gold slice is the only accuracy estimate that counts (see `synth_plan.md`).
- Rare-event questions (complaint, dissatisfaction, vulnerability on service calls) fire on roughly 1 call in 10 to 1 in 40 of the public corpora; none of the on-disk data is complaint-heavy.
- Evidence is precision-only; `ablation` says how far a key can be trusted, it does not complete it.
- Single-call cases; multi-call cases with any/all semantics are future work.
