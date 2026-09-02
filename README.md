# transcribed-transformer

Research towards models that answer compliance questions over collections of ASR call transcripts, and a measurement of how much architecture and pretraining distribution matter for that task.

> **Status: early.** The design is settled, the implementation is not. No results yet. Every number in this README is `TBD` until it comes from a run, and will stay that way rather than being filled in with an estimate.

## TLDR — context and motivation

Compliance question answering over call transcripts is a task where reported results and deployed results tend to diverge. Models that look strong on benchmarks built from clean written text do noticeably worse once the input is genuine ASR output, and the usual response is to reach for a frontier model behind an API. That works. It is also an expensive answer to a task that ought to be tractable at a fraction of the size.

The wager here is that the cause is the input itself. **Transcribed speech is disfluent, and it is long.** Fillers, restarts, repairs and recognition errors are not noise to be cleaned away before the real work starts; they *are* the distribution. And a compliance judgement usually depends on the whole conversation rather than a prefix of it, which is a poor fit for causal attention.

This repository exists to put numbers on that rather than assert it, and to train models that work on ASR-mangled text instead of in spite of it.

## The problem

In regulated industries, conversations between staff and customers are recorded, transcribed, and reviewed against compliance questions. The questions vary in kind and are usually written by the reviewing organisation: *did the advisor communicate clearly*, *did the customer show signs of vulnerability*, *did the customer make a complaint*, *was it handled properly*.

The task has a specific and unusual shape:

- **Input is long, but not uniformly.** A single transcript can run to tens of thousands of tokens, while a great many are far shorter. A *case* is the whole interaction with one customer: several transcripts, judged together.
- **The text is diarised ASR output.** Speaker-labelled turns (`SPEAKER_00`-style tags, one turn per line), with disfluencies, recognition errors, and machine-restored punctuation. Roles are not labelled: who is the advisor and who is the customer has to be inferred from the conversation, and calls are not reliably two-party — spouses, relatives and colleagues join advice calls, which regulatory guidance on vulnerable customers explicitly anticipates.
- **Output is short and structured.** For each question: one answer, the set of transcript line numbers that constitute evidence, and a short justification.
- **Questions are open-ended, and so are their answers.** New questions are written all the time, by different organisations. The model receives the question text, the permitted answers, and the grading rule for each one, all as input. It cannot rely on a fixed, trained-in label set: the same underlying judgement is variously expressed as `{yes, no}`, as `{pass, partial pass, fail, NA}`, or as opaque codes. A question never seen during training must work.
- **Case semantics differ per question.** Some questions pass if *any* call satisfies them; others only if *every* call does. The model has to infer which from the question.

Scope: **speech transcripts only.** Chat logs, scanned documents and emails are excluded, because the hypotheses below are specifically about ASR distribution and disfluency.

## Two hypotheses

**1. The architecture is mismatched.** Long input, short structured output, and judgements that depend on the conversation as a whole. Causal attention means the opening of a call never sees its ending, yet "was this complaint handled properly" is not answerable from a prefix. Bidirectional attention over the transcript should be the better inductive bias.

**2. The pretraining distribution is mismatched.** Models are pretrained on written text. The input here is disfluent transcribed speech, where fillers, repairs and repetition carry real signal, and recognition errors are systematic rather than random. Pretraining on transcript-like text should help.

## Design — three experiments

| Experiment | What | Why |
|---|---|---|
| **Decoder** | A causal LM fine-tuned to emit the structured output (answer, evidence lines, justification) | The standard approach, trained on the same data as the alternative |
| **Encoder-decoder** | Bidirectional encoder over the transcript, decoder for the structured output | The test of hypothesis 1: identical data and budget, different attention over the input |
| **API model** | A frontier model behind an API with a faithful prompt pipeline | The baseline both trained arms must beat, on quality *and* on cost per case |

Both trained arms are continued-pretrained on transcript-like text before task fine-tuning, which is where hypothesis 2 enters; the data, its licence tracks and its provenance are recorded in [`data/DATASHEET.md`](data/DATASHEET.md), with the corpus survey in [`data/SURVEYSHEET.md`](data/SURVEYSHEET.md) and the transcript-synthesis recipes in [`data/SYNTHSHEET.md`](data/SYNTHSHEET.md).

All three are scored on the same benchmark: cases built from public call corpora, questions with explicit answer options and grading rules, labels partly by construction (phenomena written into calls at known lines) and partly by LLM judgement, each call in a clean and a real-ASR variant so the cost of transcription noise is measured directly. The generator and its data format are documented in [`src/synthesis/README.md`](src/synthesis/README.md) and the DATASHEET's Benchmarking section.

Results are recorded, never estimated; negative results get written up the same as positive ones.

This is independent research on public data. It is motivated by a class of problem common in regulated industries, and it contains no proprietary data, systems or findings from any employer. Licensed under [Apache 2.0](LICENSE).
