# transcribed-transformer

Research towards a **universal encoder for transcribed speech**, and task models built on it that answer compliance questions over collections of call transcripts.

> **Status: early.** The design is settled, the implementation is not. No results yet. Every number in this README is `TBD` until it comes from a run, and will stay that way rather than being filled in with an estimate.

## The problem

In regulated industries, conversations between staff and customers are recorded, transcribed, and reviewed against compliance questions. The questions vary in kind and are usually written by the reviewing organisation: *did the advisor communicate clearly*, *did the customer show signs of vulnerability*, *did the customer make a complaint*, *was it handled properly*.

The task has a specific and unusual shape:

- **Input is long.** A single transcript runs to ~32k tokens. A *case* is the whole interaction with one customer: several transcripts, judged together.
- **Output is short and structured.** For each question: an answer (`pass | fail | partial_pass | NA`), the set of transcript line numbers that constitute evidence, and a short justification.
- **Questions are open-ended.** New questions are written all the time. The model receives the question text and the definitions of its four answers as input; it cannot rely on a fixed, trained-in label set.
- **Case semantics differ per question.** Some questions pass if *any* call satisfies them ("was the customer's name established"). Others pass only if *every* call does ("did the advisor greet the customer appropriately"). The model has to infer which from the question.
- **The text is speech.** ASR output, full of disfluencies, fillers, restarts, self-corrections and recognition errors. Not written prose.

The standard approach is to prompt a large causal language model per transcript and aggregate. It works. This project asks whether something meaningfully better is available.

## Two hypotheses

**1. The architecture is mismatched.** Long input, short structured output, and judgements that depend on the conversation as a whole. Causal attention means the opening of a call never sees its ending, yet "was this complaint handled properly" is not answerable from a prefix. Bidirectional attention over the transcript should be the better inductive bias.

**2. The pretraining distribution is mismatched.** Models are pretrained on written text. The input here is disfluent transcribed speech, where fillers, repairs and repetition carry real signal, and recognition errors are systematic rather than random. Pretraining on transcript-like text should help.

Hypothesis 1 has external support. [Ettin](https://arxiv.org/abs/2507.11412) is, as far as I know, the only compute-matched comparison of encoders and decoders trained on identical data with identical recipes, and it finds a **400M encoder beating a 1B decoder on classification**. Hypothesis 2 has, to my knowledge, no clean test in this setting. Producing one is a goal of this work.

## Why an encoder, and why "universal"

A single encoder that genuinely understands transcribed speech is more useful than a single task model, because other things can be built on it. Compliance QA is one consumer. **Anonymisation** is another: PII span detection over transcripts is a prerequisite for using recorded conversations as training data at all, and it is a completely different task shape on the same input distribution.

The universality claim is only credible if it is demonstrated rather than asserted, so the plan puts two unrelated heads on one frozen trunk and reports both.

## Design

Five arms. Arms A and B produce a model; arm E answers the scientific question; arms C and D exist so the comparison means something.

| Arm | What | Why |
|---|---|---|
| **A** | Adapted native encoder ([Ettin-1B](https://huggingface.co/jhu-clsp/ettin-encoder-1b) / [ModernBERT-large](https://arxiv.org/abs/2412.13663)): domain-adaptive MLM on transcripts, context extension 8k→32k, [GLiClass](https://arxiv.org/abs/2508.07662)-style joint encoding of question and transcript, answer + per-line evidence heads, learned case aggregator | Primary deliverable |
| **B** | Encoder-decoder ([T5Gemma 2](https://arxiv.org/abs/2512.14856) 1B-1B): UL2/PrefixLM adaptation, then multi-task answer + evidence + generated summary | Full stack, and the direct test of encoder-decoder adaptation on long context |
| **C** | Bidirectional-prefix decoder ([LLM2Vec](https://arxiv.org/abs/2404.05961)-style mask flip on a 1-4B causal model) | The cheap conversion. Ettin predicts it loses to a native encoder; [T5Gemma](https://arxiv.org/abs/2504.06225) predicts adaptation works. Resolving that on this task is worth knowing either way |
| **D** | Baselines: frontier API model with a faithful prompt pipeline, size-matched causal decoder SFT'd on the same data, off-the-shelf zero-shot encoders | Nothing above means anything without these |
| **E** | **From-scratch controlled ablation.** A ~150-400M encoder trained twice under a [biphasic CLM→MLM](https://arxiv.org/abs/2507.00994) recipe, on token-matched corpora: generic web text vs transcript-heavy text. Everything else held identical | The only arm that cleanly isolates hypothesis 2. Every adapted arm confounds it with the base model's pretraining |

Arm E is where the actual science is. It is also the arm most likely to produce a result worth publishing, independent of whether the deliverable arms win.

### Bridging written text to ASR text

Several of the most useful corpora are written, not spoken. Rather than paying for TTS-then-ASR at corpus scale, the plan fits an **ASR channel model** from corpora that have both audio and human verbatim transcripts (word-level confusion distributions, deletion and insertion rates, filler and repair patterns, casing and punctuation behaviour), then applies it cheaply to clean text.

The channel is validated adversarially, not assumed: a discriminator should struggle to separate synthetic transcripts from real ASR output. If it does not, the channel has failed and that blocks the phases depending on it.

## Data, and the licence trap

The best public analogues for this input are **non-commercial**, which matters if the point is eventual transfer to a commercial setting. So the corpus is split into two tracks, and **every headline result is reported on both**, making the cost of the restriction visible instead of hidden.

**Track P — permissive, commercially portable**

| Source | Why |
|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) + ICSI | Verbatim spontaneous multi-party speech with audio. Disfluencies preserved, so usable for channel fitting |
| [AnnoMI](https://github.com/uccollab/AnnoMI) | Expert-annotated counselling dialogues. The closest public proxy for disclosure of health and life-event difficulty |
| [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) | ~2M real financial complaint narratives with product taxonomy and resolution outcome |
| [Earnings-21/22](https://arxiv.org/abs/2203.15591) | Financial spoken English, accent diversity |
| FineWeb-Edu, Ettin's open corpus | Generic text for the arm E control |

**Track NC — research only**

| Source | Why |
|---|---|
| [CallCenterEN](https://arxiv.org/abs/2507.02958) (91,706 conversations, 10,448 audio hours) | By far the closest public analogue to the target distribution |
| [SPGISpeech 1.0 + 2.0](https://arxiv.org/abs/2508.05554) | ~8,800 hours of professionally transcribed financial speech |
| [MediaSum](https://arxiv.org/abs/2103.06410) (463.6K transcripts with summaries) | The only large transcript-to-summary corpus; trains the summary head |
| BETOLD | Human-agent phone dialogues with breakdown labels, as a dissatisfaction proxy |

Per-corpus licences, restrictions and preprocessing live in [`data/DATASHEET.md`](data/DATASHEET.md). Raw corpus data is never committed.

## Benchmark

No public benchmark exists for compliance QA over transcripts, so one has to be built: multi-call cases, a question bank in the production shape (question plus explicit definitions of all four answers), held-out questions never seen in training, evidence labelled as line indices. Vulnerability questions are structured on the [FCA's four drivers](https://www.fca.org.uk/publications/finalised-guidance/guidance-firms-fair-treatment-vulnerable-customers) (health, life events, resilience, capability) so the taxonomy is a real regulatory one rather than an invented one.

Scoring covers the answer (accuracy, macro-F1), the evidence (set F1, plus sufficiency and comprehensiveness, to catch right answers reached through wrong evidence), and the summary (faithfulness to the cited lines).

### The limitation this project has to be honest about

**Silver labels are generated by large language models, so "beating a frontier model" on them is partly circular.** This is the central methodological weakness and it is stated here rather than buried in a discussion section.

What is done about it, in order of importance: the **human-annotated gold slice is the headline metric** and silver is dev signal only; silver labels come from multi-model consensus in a different family from the baseline being compared against; inter-annotator agreement on the gold slice is published alongside the results.

This mitigates the problem. It does not eliminate it, and results should be read with that in mind.

## Results

`TBD`. Nothing has been run yet.

Tables will be generated from the wandb API rather than written by hand, reported on both licence tracks, split by seen and held-out questions, with inference cost alongside quality. Negative results are reported the same way as positive ones; if the architecture hypothesis turns out to be wrong, that is the finding.

## Hardware

Two RTX 5090s (Blackwell, 32GB each, no NVLink), part-time over 3-6 months. Every design choice is sized to that budget, which is why the from-scratch arm is 150-400M rather than something larger, and why it is a controlled ablation rather than an attempt at a competitive pretrained model.

Measured on this machine with ModernBERT-large (395M) under DDP, bf16, micro batch 1 per GPU ([full write-up](experiments/2026-07-27-p0-throughput/)):

| seq_len | activation ckpt | tokens/s | effective TFLOP/s | MFU | peak mem |
|---:|:---:|---:|---:|---:|---:|
| 8,192 | no | 36,625 | 121.3 | 28.9% | 19.60 GiB |
| 8,192 | yes | 29,451 | 97.5 | 23.3% | 8.94 GiB |
| 16,384 | yes | 18,464 | 77.9 | 18.6% | 15.05 GiB |
| 32,768 | yes | 10,481 | 63.2 | 15.1% | 27.63 GiB |

Both 16k and 32k run out of memory without activation checkpointing. At 32k the peak is 27.6 GiB of 31.4 GiB available, so a 1B encoder will not fit at that length on these cards: ModernBERT-large is the long-context trunk and larger models are restricted to 8k ablations. Throughput at 32k is 29% of the 8k rate, which is the real cost of the long-context design and is budgeted for rather than assumed away.

## Layout

```
data/         canonical schema, per-corpus loaders, DATASHEET, ASR channel model
bench/        question bank, case construction, labelling, scoring
models/       encoder trunks, task heads, encoder-decoder and prefix-LM arms
training/     YAML configs, launchers, distributed setup, wandb integration
eval/         harness, results tables, faithfulness metrics
experiments/  one directory per run: hypothesis, config, result, verdict
```

## Notes

This is independent research on public data. It is motivated by a class of problem that is common in regulated industries, and it contains no proprietary data, systems or findings from any employer.

Licensed under [Apache 2.0](LICENSE). Model weights will be released per licence track: Track P models freely, Track NC models under the restrictions their training data carries.
