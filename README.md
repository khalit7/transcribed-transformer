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
- **The text is diarised ASR output.** Speaker-labelled turns, one turn per line (rendered to the model as `<line number>: <speaker>: <text>`, so evidence can be cited by line), with disfluencies, recognition errors, and machine-restored punctuation. Roles are not labelled: who is the advisor and who is the customer has to be inferred from the conversation, and calls are not reliably two-party — spouses, relatives and colleagues join advice calls, which regulatory guidance on vulnerable customers explicitly anticipates.
- **Output is short and structured.** For each question: one answer, the set of transcript line numbers that constitute evidence, and a short justification.
- **Questions are open-ended, and so are their answers.** New questions are written all the time, by different organisations. The model receives the question text, the permitted answers, and the grading rule for each one, all as input. It cannot rely on a fixed, trained-in label set: the same underlying judgement is variously expressed as `{yes, no}`, as `{pass, partial pass, fail, NA}`, or as opaque codes. A question never seen during training must work.
- **Case semantics differ per question.** Some questions pass if *any* call satisfies them; others only if *every* call does. The model has to infer which from the question.

Scope: **speech transcripts only.** Chat logs, scanned documents and emails are excluded, because the hypotheses below are specifically about ASR distribution and disfluency.

## Two hypotheses

**1. The architecture is mismatched.** Long input, short structured output, and judgements that depend on the conversation as a whole. Causal attention means the opening of a call never sees its ending, yet "was this complaint handled properly" is not answerable from a prefix. Bidirectional attention over the transcript should be the better inductive bias.

**2. The pretraining distribution is mismatched.** Models are pretrained on written text. The input here is disfluent transcribed speech, where fillers, repairs and repetition carry real signal, and recognition errors are systematic rather than random. Pretraining on transcript-like text should help.

## Design — the experiments

Five trained designs were considered and one frontier API baseline. All trained arms come from one model family, Qwen3, so that an arm differs from the control in exactly one thing: the attention mask, or the architecture built from the same weights. Every arm runs at 1.7B first (full fine-tuning fits one GPU), then 4B and 8B with LoRA for the arms that survive; same data, same split, same `<n>: <role>: <text>` rendering, loss on the answer only with zero prompt loss, 32k sequence length. Verdicts come from a literature review (2026-09-10); results are `TBD` until a run produces them and are recorded per run in `experiments/`.

| # | Design | Models | Status |
|---|---|---|---|
| E0 | **API baseline.** Frontier model, same prompt and rendering, prompt caching, structured output | Claude Sonnet 5 (primary), Claude Opus 5 (ceiling), one non-Anthropic model (TBD) | Run. Sonnet labelled the family questions and Opus adjudicated the benchmark, so both partly grade their own work; the non-Anthropic model is the uncontaminated point. |
| E1 | **Causal SFT.** Standard decoder fine-tuning, causal attention, loss on the answer only | Qwen3-1.7B (full FT); Qwen3-4B-Instruct-2507 and Qwen3-8B (LoRA) | Run. The control. |
| E2 | **Prefix-LM SFT.** Same as E1 with bidirectional attention over the prompt, causal over the answer | Same checkpoints as E1, same everything, mask only | Run. The cleanest test of hypothesis 1: a causal-pretrained decoder given bidirectional input attention at fine-tuning gained 2.3 to 6.4 points at 1B to 8B and beat a full encoder-decoder (arXiv 2510.26622). Needs FlexAttention (FlashAttention-2 cannot express the mask) and gives up KV sharing across the questions asked of one call. |
| E3 | **Encoder-decoder, jointly trained.** Bidirectional encoder over the prompt, decoder with cross-attention, loss on the answer | Built from the E1 checkpoint by the T5Gemma recipe: encoder and decoder both initialised from Qwen3-1.7B (then 4B), cross-attention from self-attention, all trained. T5Gemma 2 1b-1b as a non-comparable reference | Run after E2, if E2 moves the metric. Reported against both the parameter-matched and the inference-matched decoder. Risk: T5Gemma reports parity with its decoder-only origin only after tens of billions of adaptation tokens. |
| E4 | **Frozen encoder, soft tokens.** A frozen bidirectional encoder runs over the transcript; the decoder reads the question, the encoder's per-token states projected into its input space, and the answer, through ordinary self-attention. The decoder never sees the transcript text | Encoder: ModernBERT-large, frozen (8k context; SPoRC cases above 8k are encoded in chunks and reported separately). Decoder: the E1 base checkpoints, trained as in E1 | Run at 1.7B after E2, gated: a linear probe must recover line indices from the projected encoder tokens before any decoder training. The literature expects this to fail (recall collapses under compression, arXiv 2412.17483; the frozen-backbone autoregressive design was the worst of four in IDEFICS2, arXiv 2405.02246). |
| E5 | **Frozen encoder, cross-attention.** Same two stages, but the decoder reads the encoder's states through new cross-attention layers; its self-attention runs over question and answer only | Same encoder and decoder as E4, zero-initialised cross-attention every fourth layer, initialised from the decoder's own self-attention | Run at 1.7B after E2, gated by the same probe. Only the cross-attention layers are new parameters. Known negatives: a frozen encoder lost to a jointly trained one in CEPE (arXiv 2402.16617); cross-attention lost about ten points on fine-detail document reading in NVLM (arXiv 2409.11402). |

Order: E0 and E1 at 1.7B (which also settle the training stack and the evaluation pipeline); E2 against E1; E4 and E5 if the probe passes, E3 if E2 moved the metric; scale the survivors; then hypothesis 2, E1 and E2 repeated from a checkpoint continued-pretrained on transcript text, on Track NC first because the Track P pack is too small to test it. The data, its licence tracks and its provenance are recorded in [`data/DATASHEET.md`](data/DATASHEET.md), with the corpus survey in [`data/SURVEYSHEET.md`](data/SURVEYSHEET.md) and the transcript-synthesis recipes in [`data/SYNTHSHEET.md`](data/SYNTHSHEET.md).

Every arm is scored on the same benchmark: cases built from public call corpora, questions with explicit answer options and grading rules, labels produced by LLM labelling of real, unmodified calls (a labeller answers each question with evidence first; a separate LLM-as-a-judge step assesses label quality), each call in a clean and a real-ASR variant so the cost of transcription noise is measured directly. The generator and its data format are documented in [`src/synthesis/README.md`](src/synthesis/README.md) and the DATASHEET's "Labelled data" section.

Results are recorded, never estimated; negative results get written up the same as positive ones.

This is independent research on public data. It is motivated by a class of problem common in regulated industries, and it contains no proprietary data, systems or findings from any employer. Licensed under [Apache 2.0](LICENSE).
