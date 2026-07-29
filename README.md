# transcribed-transformer

Research towards a **universal encoder for transcribed speech**, and task models built on it that answer compliance questions over collections of call transcripts.

> **Status: early.** The design is settled, the implementation is not. No results yet. Every number in this README is `TBD` until it comes from a run, and will stay that way rather than being filled in with an estimate.

## The problem

In regulated industries, conversations between staff and customers are recorded, transcribed, and reviewed against compliance questions. The questions vary in kind and are usually written by the reviewing organisation: *did the advisor communicate clearly*, *did the customer show signs of vulnerability*, *did the customer make a complaint*, *was it handled properly*.

The task has a specific and unusual shape:

- **Input is long, but not uniformly.** A single transcript can run to ~32k tokens, while a great many are far shorter. A *case* is the whole interaction with one customer: several transcripts, judged together.
- **Output is short and structured.** For each question: one answer, the set of transcript line numbers that constitute evidence, and a short justification.
- **Questions are open-ended, and so are their answers.** New questions are written all the time, by different organisations. The model receives the question text, the permitted answers, and the grading rule for each one, all as input. It cannot rely on a fixed, trained-in label set, because there isn't one: the same underlying judgement is variously expressed as `{yes, no}`, as `{pass, partial pass, fail, NA}`, or as opaque codes, and which side of the judgement is the interesting one varies too.
- **Case semantics differ per question.** Some questions pass if *any* call satisfies them ("was the customer's name established"). Others pass only if *every* call does ("did the advisor greet the customer appropriately"). The model has to infer which from the question.
- **The text is speech.** ASR output, full of disfluencies, fillers, restarts, self-corrections and recognition errors. Not written prose.

Scope: **speech transcripts only.** The same class of system is often asked about chat logs, scanned documents and emails. Those are excluded here because the hypotheses below are specifically about ASR distribution and disfluency, which do not apply to written or OCR'd input. Generalising off that distribution is a later question, and a good test of the trunk, but it is not what this project is measuring.

The standard approach is to prompt a large causal language model per transcript and aggregate. It works. This project asks whether something meaningfully better is available.

## Two hypotheses

**1. The architecture is mismatched.** Long input, short structured output, and judgements that depend on the conversation as a whole. Causal attention means the opening of a call never sees its ending, yet "was this complaint handled properly" is not answerable from a prefix. Bidirectional attention over the transcript should be the better inductive bias.

**2. The pretraining distribution is mismatched.** Models are pretrained on written text. The input here is disfluent transcribed speech, where fillers, repairs and repetition carry real signal, and recognition errors are systematic rather than random. Pretraining on transcript-like text should help.

Hypothesis 1 has external support. [Ettin](https://arxiv.org/abs/2507.11412) is, as far as I know, the only compute-matched comparison of encoders and decoders trained on identical data with identical recipes, and it finds a **400M encoder beating a 1B decoder on classification**. Hypothesis 2 has, to my knowledge, no clean test in this setting. Producing one is a goal of this work.

## Why an encoder, and why "universal"

Beyond the attention argument, three properties of this task favour an encoder for structural reasons rather than empirical ones. They are worth stating separately because they hold by construction, not on average.

**An option-scoring model cannot emit an invalid answer.** Since the permitted answers arrive as input, the natural encoder design scores each supplied option and returns the winner's text verbatim. There is no decoding path that produces a label nobody offered. A generative model has to *learn* to copy one of the strings it was shown, and copying-under-instruction is a behaviour that can fail: emitting the grading rule instead of the label it describes, or echoing the scaffolding it was shown the options in. For the encoder that failure is unreachable; for the decoder it is a thing you measure and hope stays low. How much this matters in practice is [an open question this project measures](#format-validity) rather than assumes.

**Per-line evidence tagging emits a fixed-size output.** Asked for supporting line numbers, a generative model has to produce a variable-length list of integers over a document that may be tens of thousands of tokens long. [Measured](experiments/2026-07-29-evidence-index-probe/) on real ASR text, that degrades sharply: format validity falls from 88% at 512 tokens to **0% at 8k and beyond** for Qwen3-1.7B, and overall runs 39% / 52% / 61% for Qwen3-1.7B, Qwen3-8B and Mistral-7B-Instruct.

The mechanism was not the one expected. Indices never went out of range: **not once in 336 generations**, across three models, two model families and two transcript layers. Instead the models stop *selecting* lines and start *enumerating* them, emitting contiguous runs counting upward until the output budget is exhausted and the JSON is cut off mid-integer. An encoder tags pooled line representations instead: one decision per line, fixed size, no list to overrun and no counting fallback when selection fails. Notably, **constrained decoding would not fix this** — a grammar forcing integers in range accepts a degenerate enumeration happily, because the failure is semantic rather than syntactic.

The same probe found that **ASR text is markedly harder than human transcription**: the smallest model scores 38% at long context on human-verbatim transcripts of the same corpus and 0% on ASR of it. Difficulty estimates taken from clean transcripts are optimistic.

**Zero-shot over labels and zero-shot over questions are the same operation.** Both are text the model reads rather than structure it was trained into, so one mechanism covers both.

Beyond compliance QA, a single encoder that genuinely understands transcribed speech is more useful than a single task model, because other things can be built on it. **Anonymisation** is the second consumer: PII span detection over transcripts is a prerequisite for using recorded conversations as training data at all, and it is a completely different task shape on the same input distribution.

The universality claim is only credible if it is demonstrated rather than asserted, so the plan puts two unrelated heads on one frozen trunk and reports both.

## Design

Five arms. Arms A and B produce a model; arm E answers the scientific question; arms C and D exist so the comparison means something.

| Arm | What | Why |
|---|---|---|
| **A** | Adapted native encoder ([Ettin-1B](https://huggingface.co/jhu-clsp/ettin-encoder-1b) / [ModernBERT-large](https://arxiv.org/abs/2412.13663)): domain-adaptive MLM on transcripts, context extension 8k→32k, [GLiClass](https://arxiv.org/abs/2508.07662)-style joint encoding of question, answer options and transcript in one forward pass; **option-scoring** answer head plus per-line evidence head, learned case aggregator | Primary deliverable |
| **B** | Encoder-decoder ([T5Gemma 2](https://arxiv.org/abs/2512.14856) 1B-1B): UL2/PrefixLM adaptation, then multi-task answer + evidence + generated summary | Full stack, and the direct test of encoder-decoder adaptation on long context |
| **C** | Bidirectional-prefix decoder ([LLM2Vec](https://arxiv.org/abs/2404.05961)-style mask flip on a 1-4B causal model) | The cheap conversion. Ettin predicts it loses to a native encoder; [T5Gemma](https://arxiv.org/abs/2504.06225) predicts adaptation works. Resolving that on this task is worth knowing either way |
| **D** | Baselines: frontier API model with a faithful prompt pipeline, size-matched causal decoder SFT'd on the same data, off-the-shelf zero-shot encoders, and a majority-class predictor | Nothing above means anything without these. The majority-class row is cheap insurance: if the benchmark's answer distribution turns out skewed, accuracy will look strong for no reason, and this is the row that shows it |
| **E** | **From-scratch controlled ablation.** A ~150-400M encoder trained twice under a [biphasic CLM→MLM](https://arxiv.org/abs/2507.00994) recipe, on token-matched corpora: generic web text vs transcript-heavy text. Everything else held identical | The only arm that cleanly isolates hypothesis 2. Every adapted arm confounds it with the base model's pretraining |

Arm E is where the actual science is. It is also the arm most likely to produce a result worth publishing, independent of whether the deliverable arms win.

If arms have to be cut for time, **C goes first and E never goes.** B stays ahead of C because the written justification is what a human reviewer actually reads: a system that produces a correct verdict it cannot explain is not a replacement for one that explains itself, whatever its accuracy.

### The answer head has no fixed label set

Worth spelling out, because it is the constraint that shapes the architecture. The obvious design, a softmax over `pass | fail | partial_pass | NA`, cannot work: answer vocabularies vary in arity, in wording, and in which side of the judgement carries the evidence, and a question written after training can use a vocabulary no one has seen.

So the answer head **scores the options it is given**. Question, description, and each option's label and grading rule are encoded jointly with the transcript, one forward pass, one score per supplied option, softmax over that set alone, and the winning option's text is returned verbatim. Variable arity, nothing trained in, and the same mechanism whether a question offers two answers or five.

This is the shape [GLiClass](https://arxiv.org/abs/2508.07662) already demonstrates for zero-shot classification with label text as input, applied to a setting where the label set genuinely changes per example rather than per dataset.

### Bridging written text to ASR text

Several of the most useful corpora are written, not spoken. Rather than paying for TTS-then-ASR at corpus scale, the plan fits an **ASR channel model** from corpora that have both audio and human verbatim transcripts (word-level confusion distributions, deletion and insertion rates, filler and repair patterns, casing and punctuation behaviour), then applies it cheaply to clean text.

The channel is validated adversarially, not assumed: a discriminator should struggle to separate synthetic transcripts from real ASR output. If it does not, the channel has failed and that blocks the phases depending on it.

## Data, and the licence trap

The best public analogues for this input are **non-commercial**, which matters if the point is eventual transfer to a commercial setting. So the corpus is split into two tracks, and **every headline result is reported on both**, making the cost of the restriction visible instead of hidden.

### Corpora are ranked by how close they are to ASR, not just by licence

Licence decides whether a corpus *may* be used. A second axis decides how much it is worth: how close its text is to real recogniser output.

1. **Real ASR output**, disfluencies and misrecognitions intact. Used directly.
2. **Audio we can run ASR over ourselves**, producing tier 1. The recogniser used is recorded, since error profiles differ and mixing them silently is a confound.
3. **Clean written or human-verbatim text.** Only where nothing better exists, and only after the channel model has been applied.

**Human-verbatim transcription is tier 3, not tier 1**, which is the counter-intuitive part. It is a faithful record of speech, but a transcriber who quietly repairs a false start has deleted the exact signal this project models. Applying that rule caught a mistake already made here: AMI was ingested from its manual annotations, while a tier 1 ASR layer sat unopened in a second archive of the same distribution.

**Track P — permissive, commercially portable**

| Source | Why |
|---|---|
| [Taskmaster-1 + 2](https://github.com/google-research-datasets/Taskmaster) | 22,807 **two-party spoken** dialogues, 8.1M tokens, CC BY 4.0: crowdworker as customer, trained call-centre operator as assistant. The right interaction shape at scale, and the only corpus here with advisor/customer roles. Short (p50 321 tokens) and no audio |
| [HarperValleyBank](https://arxiv.org/abs/2010.13929) | 1,446 dyadic **consumer banking** calls with audio and agent/caller role labels, CC BY 4.0. Small and scripted, but the closest permissive domain match and channel-fittable |
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) + ICSI | The only **paired** corpus here: ships real 2007-era ASR output *and* the human transcript of the same speech, both CC BY 4.0. That pairing is what the ASR channel model is fitted from |
| [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) | ~2M real financial complaint narratives with product taxonomy and resolution outcome |
| [Earnings-21/22](https://arxiv.org/abs/2203.15591) | Financial spoken English, accent diversity |
| FineWeb-Edu, Ettin's open corpus | Generic text for the arm E control |

**Track NC — research only**

| Source | Why |
|---|---|
| [AnnoMI](https://github.com/uccollab/AnnoMI) | Expert-annotated counselling dialogues, the closest public proxy for disclosure of health and life-event difficulty. Here **because it has no stated licence at all**, not because it is marked non-commercial: see below |
| [CallCenterEN](https://arxiv.org/abs/2507.02958) (91,706 conversations, 10,448 audio hours) | By far the closest public analogue to the target distribution |
| [SPGISpeech 1.0 + 2.0](https://arxiv.org/abs/2508.05554) | ~8,800 hours of professionally transcribed financial speech |
| [MediaSum](https://arxiv.org/abs/2103.06410) (463.6K transcripts with summaries) | The only large transcript-to-summary corpus; trains the summary head |
| BETOLD | Human-agent phone dialogues with breakdown labels, as a dissatisfaction proxy |

Per-corpus licences, restrictions and preprocessing live in [`data/DATASHEET.md`](data/DATASHEET.md). Raw corpus data is never committed.

**Track P has to carry the interaction shape, not just the licence.** The first corpus ingested here was AMI, which is 4-5 speaker meetings: excellent for disfluency and channel work, structurally wrong for two-party advisor/customer calls. A permissive track made only of meetings, written complaints and web text would have compared badly against the non-commercial track for reasons of corpus *shape* rather than licence, which would have made the two-track comparison meaningless. Taskmaster and HarperValleyBank exist in the table above to close that gap: dyadic spoken service interaction at scale, and dyadic banking calls with audio, both CC BY 4.0.

It is closed only partly, and the measurements say where. AMI spans 1,375–29,605 tokens per transcript; Taskmaster spans 7–2,389. Those ranges barely overlap, so **Track P has dyadic conversation and it has long context, but not both at once.** Since a *case* is several calls rather than one, short real dialogues compose into case-scale input without truncation or invented filler, which is how the benchmark gets there. But no permissively-licensed corpus here is a single long two-party call, and that is stated rather than papered over.

**A worked example of the rule costing something.** AnnoMI was planned as Track P on the assumption it was public domain. Verifying that against the primary source found no licence at all: no `LICENSE` file, no terms in the README, and the CC BY that turns up in searches belongs to the authors' *paper* rather than to the separately-distributed dataset. There is a second problem underneath: the transcripts are of third-party demonstration videos the authors did not create, so they can license their annotations but not the underlying speech.

Ambiguity resolves to NC, so it moved. The cost is not cosmetic — AnnoMI is the best public proxy for **vulnerability**, one of the three priority question families, so the commercially-portable track is now weakest exactly where it can least afford to be. That is the sort of thing the two-track split exists to make visible instead of letting it be discovered after training.

## Benchmark

No public benchmark exists for compliance QA over transcripts, so one has to be built: multi-call cases, a question bank in the real shape (question, description, and the permitted answers each with its grading rule), evidence labelled as line indices. Vulnerability questions are structured on the [FCA's four drivers](https://www.fca.org.uk/publications/finalised-guidance/guidance-firms-fair-treatment-vulnerable-customers) (health, life events, resilience, capability) so the taxonomy is a real regulatory one rather than an invented one.

Four design decisions do most of the work, and each of them is cheap now and expensive to retrofit after labelling.

**Answer vocabularies vary across questions.** Two-way, three-way and four-way; different spellings of the same idea; one that uses opaque codes rather than words; and at least one where the *pass*-side label is the evidence-bearing one. A question bank sharing a single label set cannot tell a model that reads its options apart from one that memorised them during training, and telling those apart is the entire zero-shot claim. The polarity variation earns its place separately: if evidence always sits on the failing side, a model can learn "found something → fail" and score well without reading the options at all.

**The held-out split is by question family, never by case.** Whole families and themes are held out, not individual questions drawn from a family that stays in training: questions within a family are frequently near paraphrases, so a finer split leaks. Holding out *cases* for questions the model trained on would measure memorisation and would read as success. Seen and unseen criteria are reported as separate columns and **the unseen column is the headline**.

**Criterion diversity over example count.** Many questions with few cases each, rather than few questions with many. Generality over questions is what is being measured, so the labelling budget buys breadth.

**Cases are length-bucketed, not uniformly maximal.** Real transcript collections vary enormously in length, and a benchmark where every case is ~32k would misrepresent both accuracy and cost. Cases are binned by rendered token length and results reported per bucket. The [measured throughput](#hardware) makes the cost side concrete: 32k runs at 29% of the 8k token rate, so the mix of operating points dominates the compute budget. The model must *support* 32k; it will usually run well below it.

### Scoring

The answer (accuracy and macro-F1 against that question's own vocabulary, with a majority-class baseline that any arm must beat), the evidence, the summary (faithfulness to the cited lines, with the judge model version pinned, since judge drift invalidates every earlier score invisibly), and format validity.

<a id="format-validity"></a>**Format validity is scored as a first-class metric**, separately from judgement. An answer that is byte-equal to a permitted value scores full credit; one that is only recoverable by matching a grading rule back to the label it describes scores partial credit **and counts as a failure**; one that is unrecoverable scores zero. Evidence must be a JSON array of integers, in range, de-duplicated, ascending, and empty rather than a sentinel when nothing was found.

This is scored strictly and reported openly because a repair layer downstream of the model makes the raw error rate invisible, and because the encoder arms are structurally immune to a failure the generative baselines have to earn their way out of. How large that gap actually is on public data is a question this benchmark answers rather than assumes.

**Evidence is scored as precision against a partial key**, plus sufficiency and comprehensiveness from the rationale-faithfulness literature. Set-level recall and F1 are reported only where a key is marked exhaustive. Human evidence keys are usually not: a question often has several genuinely correct supporting lines and an annotator marks the ones they noticed, so scoring recall against such a key punishes a model for finding a line the annotator missed and understates every system equally, baselines included.

### The limitation this project has to be honest about

**Silver labels are generated by large language models, so "beating a frontier model" on them is partly circular.** This is the central methodological weakness and it is stated here rather than buried in a discussion section.

What is done about it, in order of importance: the **human-annotated gold slice is the headline metric** and silver is dev signal only; silver labels come from multi-model consensus in a different family from the baseline being compared against; inter-annotator agreement on the gold slice is published alongside the results.

This mitigates the problem. It does not eliminate it, and results should be read with that in mind.

## P1a: two cheap probes first

Before the benchmark is built and before any training recipe is committed to, two measurements that need no labels and take days rather than weeks. Both exist to catch a wrong assumption while it is still cheap to be wrong. Both are done.

**Can a model emit valid evidence line indices at all?** [Run](experiments/2026-07-29-evidence-index-probe/) on AMI ASR transcripts truncated to seven length buckets, with Qwen3-1.7B, Qwen3-8B and Mistral-7B-Instruct on identical items, scored strictly on format validity. Validity degrades sharply with length in every model. The predicted failure mode — out-of-range indices — never occurred; the real one is degenerate enumeration overrunning the output budget. The hypothesis was registered before the run and the prediction it got wrong is [written up as wrong](experiments/2026-07-29-evidence-index-probe/#verdict), as is a flawed first comparison that had to be re-run.

Still missing: a frontier baseline, since no API key was available.

**What lengths do the corpora actually have?** Measured for AMI in [`data/DATASHEET.md`](data/DATASHEET.md): p50 9,630 tokens, p95 17,689, max 29,605, with 62% of meetings exceeding 8k and none exceeding 32k. Benchmark length buckets come from this rather than from a guess.

## Results

`TBD`. Nothing has been run yet.

Tables will be generated from the wandb API rather than written by hand, reported on both licence tracks, split by seen and held-out criteria and by length bucket. **Cost is a gate, not a footnote:** an arm has to be at least as cheap per case as the baseline it beats before the result is written up as a win, because efficiency is half the reason for preferring a small encoder over a prompted frontier model in the first place.

Negative results are reported the same way as positive ones; if the architecture hypothesis turns out to be wrong, that is the finding.

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

`hardware_performance.md` is appended to by every run, recording GPU and CPU utilisation, temperature, clocks, power and throttling. It exists for what a loss curve cannot show: whether the machine sustained full speed. It distinguishes hitting a **power cap**, which is the expected steady state at full load and extrapolates fine, from **thermal** limiting, which is a cooling fault that worsens as a multi-day run heat-soaks.

## Notes

This is independent research on public data. It is motivated by a class of problem that is common in regulated industries, and it contains no proprietary data, systems or findings from any employer.

Licensed under [Apache 2.0](LICENSE). Model weights will be released per licence track: Track P models freely, Track NC models under the restrictions their training data carries.
