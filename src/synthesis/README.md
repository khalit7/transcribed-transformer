# src/synthesis — benchmark data generator

Turns public call transcripts into labelled compliance-QA examples: a call, a question with grading rules, and the answer + evidence lines + rationale that a model is scored against. Labels come either **by construction** (the generator writes the phenomenon into the call at known lines) or by **LLM judgement** (the generator assesses the call as it stands).

```
uv run python -m src.synthesis.synth_data <model_name> <generation_size>
uv run python -m src.synthesis.export --track p        # releasable copy
```

`<model_name>` goes straight to `claude -p --model` (e.g. `sonnet`, `opus`); `<generation_size>` is the number of calls to label. Every generator call is a non-interactive `claude -p` with JSON output; cost comes back from the CLI and is stored per label. Each call carries the CLI's cached system prompt, so a label costs roughly $0.05–0.10 with Sonnet.

## The 2×2

Every label is one cell of two axes, and one prompt template with two optional blocks produces all four:

| | question from the **bank** (reusable) | question **written** for this call |
|---|---|---|
| **as_is** — the call is judged as it stands | a bank question judged | the generator writes a question that fits the conversation, then judges it |
| **injected** — turns are written in so a target answer holds | the generator picks an injectable target from the question's own criteria and writes 1–4 turns | the generator invents a question *and* the turns that make its target answer true |

`injected` labels hold by construction: the evidence is exactly the inserted lines. `as_is` labels are LLM judgements. No evidence key is guaranteed complete (an injected phenomenon may sit in a call that already contained supporting lines), so evidence is scored as precision only.

Per call the orchestrator plans: one injected label (bank, or written with probability `--written-fraction`), `--as-is-per-call` bank questions judged, and with probability `--open-fraction` one written question judged. Calls rotate over `--sources` so both licence tracks fill.

## Files

| Module | Role |
|---|---|
| `schema.py` | `Question`, `Transcript`/`Variant`, `Case` (in memory), `Label`, `Generation`, `LabelledRecord` — pydantic, validated |
| `question_bank.py` | the reusable bank (`source="bank"`); source of truth for those questions. `write_questions()` derives `questions.jsonl` |
| `cases.py` | builds calls from what is on disk, each as line-aligned `clean`/`messy` variants with `SPEAKER_NN` tags and hidden roles |
| `generate.py` | the one prompt template and `generate()`, plus `insert_turns()` |
| `llm.py` | `ask_json()` over `claude -p` |
| `synth_data.py` | the CLI orchestrator |
| `export.py` | track-filtered release copy |

## Data format

`data/benchmark/labelled_data.jsonl` — one self-contained record per label:

```
id              "<call_id>::<question_id>"
source_id       corpus/config/locale/document the call came from
track           track-p | track-nc
question        {id, source: bank|written, family: vulnerability|complaint_and_eod|general_qa,
                 text, description, options: [{value: pass|fail|partial_pass|NA, criteria}]}
transcript      {variants: [{kind: clean|messy, origin, lines: ["SPEAKER_NN: ...", ...]}],
                 speaker_roles: {SPEAKER_NN: agent|customer|other}, tag_policy: random|agent_first}
label           {answer, evidence: [1-based line numbers], summary}
generation_info {name: <model>, mode: as_is|injected, cost_usd, timestamp}
meta            call metadata from the source corpus
```

Variants are line-aligned: line *i* is the same turn in every variant, so one evidence key serves both, and the clean-vs-messy accuracy gap per system is measurable directly. Speaker roles are hidden from any evaluated model; the model must infer them, as with real diarisation. Which lines of a call are synthetic is the evidence of that call's `injected` record.

`data/benchmark/questions.jsonl` — derived: the bank plus every distinct written question in the labelled data. Regenerated after each run and by `export`. Neither file is tracked in git; the bank is (in `question_bank.py`).

## Case sources

| Source | Track | clean variant | messy variant |
|---|---|---|---|
| AppTek Call-Center Dialogues | P (SA) | verbatim segments with roles | **real** Whisper output over telephone-degraded audio, time-aligned to the verbatim turns |
| Taskmaster-1/2 | P | human transcription | channel v2.2 noised (synthetic) |
| ACI-Bench | P | cleaned dialogue | channel v2.2 noised (synthetic) |
| SPoRC | NC | — | real diarised ASR (the only variant) |

Injected turns in messy variants are channel-noised with a crude surface restoration — visibly rougher than surrounding real-ASR lines in AppTek cases; TTS→ASR is the planned upgrade. Podcast (SPoRC) calls suit vulnerability injections better than service-conduct questions.

## Adding a bank question

Add a `q(...)` entry to `QUESTIONS` in `question_bank.py`: id, family, text, options as `(value, criteria)` pairs with at least `pass` and `fail`. Write criteria the generator can act on — for injection it chooses a target whose rule can be made true by *adding* turns, so an absence-shaped rule ("no call-back was promised") is never injected, only judged. Run `python -m src.synthesis.question_bank` to refresh `questions.jsonl`; `pytest tests/test_synthesis.py` validates the bank.

## Known limitations

- Evidence keys are precision-only (see above); a pre-injection absence check would restore recall at about one extra generator call per call.
- LLM-judged labels carry the usual circularity and are occasionally wrong in plausible ways (a judge once answered a *hold* question "pass" citing an injected *call-back* offer). By-construction labels exist to bypass that.
- Single-call cases only; multi-call cases with any/all semantics are v1.
