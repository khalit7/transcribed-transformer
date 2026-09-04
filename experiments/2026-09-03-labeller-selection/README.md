# Deciding which LLM is best for automatic labelling

Terminology (CLAUDE.md): the models under test are **labellers** (LLM labelling: answer, evidence, summary, tags for a question about a transcript); the model that grades their outputs is the **judge** (LLM-as-a-judge). This entry records the labeller comparison; the judging and human review are the second half of it.

## Hypothesis

A locally run open-weight model can label (transcript, question) pairs well enough to replace the API model for large-scale labelling, where "well enough" means: the judge rates its **answers** correct about as often as Sonnet's on every question family, it does not miss the **rare events** (complaint, dissatisfaction, vulnerability disclosure) more often than Sonnet does, its **evidence** lines are rated as supporting the answer, its **summaries** are rated faithful to the transcript, and its **vulnerability tags** match the disclosed characteristics. The audits gave reason to expect qwen3:32b close to Sonnet on answers and gemma-family models lenient; nothing is known yet about evidence and summary quality for any of them.

## Prediction

- qwen3:32b and qwen3.8 within a few points of Sonnet on answer correctness overall, but measurably worse on evidence completeness and on rare-event recall; llama3.3:70b comparable to the Qwens on answers and slower; gemma4:12b clearly worse on rare events and NA-vs-fail discipline.
- **Disconfirming outcomes**: every local model at least 10 points below Sonnet on answer correctness in some family (then large-scale labelling needs the API model, or a cascade); or a local model matching Sonnet on all five criteria (then the API model is only needed as verifier).

## Setup

- Labellers: `ollama:qwen3:32b`, `ollama:qwen3.8` (Qwen3.5-27B), `ollama:gemma4:12b`, `ollama:llama3.3:70b`, `claude:sonnet`. Left out: gemma3:27b (characterised as a lenient outlier in the bank audits), deepseek-r1:70b (reasoning model, run without thinking is not representative), coder and 8B models.
- Data: the same **40 calls, 10 per dataset** (AppTek, Taskmaster, ACI-Bench, SPoRC with the host identified from the cached 240), seed 11, clean variant; every question the bank v2 allows for the call's dataset → **1,820 labels per labeller** (68 / 46 / 39 / 29 per call). `synth_data <labeller> 40 --seed 11 --out data/labelled_data/labellers/<labeller>.jsonl`; no verification, no ablation (the judge grades evidence directly).
- Judge: `claude:opus`, grading each labeller's answer, evidence, summary and (vulnerability) tags on a stratified sample of pairs, all labellers' outputs for a pair in one call; sample size set once labels exist. Then a human review by Khalid of a subset of the judged pairs, to calibrate the judge before its verdicts are trusted.
- Costs measured per step below. Local labelling measured at 0.6–2.0 s per label after the transcript prefix is cached (first label on a call pays the prefill).
- wandb: not used; this is a data-pipeline study, not a training or arm-evaluation run.

## Result

**Labelling** (2026-09-03/04): all five labellers completed the same 1,820 pairs. Sonnet 0 failures, **$148.87** ($0.082/label), ~30 min at 16 workers; qwen3:32b 0 failures, 53 min; qwen3.8 0 failures, 39 min; llama3.3:70b 0 failures, 83 min (spans both GPUs); gemma4:12b **2 permanent failures** (one SPoRC episode on which it emits a repetition loop, reproduced after restart with a 2,048-token cap). Two pipeline fixes came out of the run: an output cap on Ollama calls, and transcript selection independent of what is already on disk (a resumed run had drifted onto 23 extra calls; those 529 off-sample labels were discarded).

Descriptive, on the 1,818 common pairs: local models report near-constant confidence (gemma4 1.00 on every label, qwen3.8 0.98; Sonnet 0.78 with spread), so only Sonnet's confidence can route anything; qwen3:32b leaves the evidence list empty on 24% of labels (Sonnet 16%, gemma4 11%); Qwen3.5 and Gemma4 use `partial_pass` a third as often as Sonnet.

**Judging** (2026-09-04, Opus): 400 pairs stratified by dataset × family with disagreement pairs over-represented (196 of 400 have at least two different answers; on the 204 unanimous pairs the judge agreed with the labellers on 196), all five labels per pair in one call, **0 failures, $94.63** ($0.24/pair). Full report: `data/labelled_data/judgements/opus_400_report.txt`.

| labeller | answer correct | rare-event recall (n=15) | NA where judge said fail | evidence (0–2) | Jaccard with judge evidence | summary (0–2) | tags (0–2) | labels/min | $/label |
|---|---|---|---|---|---|---|---|---|---|
| sonnet | **0.85** | 0.73 | 2 | **1.60** | **0.51** | **1.81** | **1.80** | 60 (16 workers) | 0.082 |
| qwen3.8 | 0.81 | 0.53 | 1 | 1.45 | 0.43 | 1.72 | 1.54 | **47** | 0 |
| gemma4:12b | 0.79 | 0.60 | 0 | 1.50 | 0.45 | 1.57 | 1.72 | 29 | 0 |
| llama3.3:70b | 0.77 | **0.80** | 3 | 1.17 | 0.35 | 1.35 | 1.46 | 22 | 0 |
| qwen3:32b | 0.71 | 0.47 | 14 | 1.00 | 0.28 | 1.18 | 1.48 | 35 | 0 |

Per family (answer correct): general_qa (290) Sonnet 0.83 / qwen3.8 0.77 / gemma4 0.74 / llama 0.72 / qwen3:32b 0.66; vulnerability (50) gemma4 0.92 / Sonnet 0.90 / llama 0.84 / qwen3.8 0.80 / qwen3:32b 0.78; complaint (30) all ≥ 0.97; eod (30) llama and qwen3.8 0.93, Sonnet 0.87, gemma4 and qwen3:32b 0.83, but on the 5 pairs where the judge found dissatisfaction gemma4 caught none.

**Human review**: 40 judged pairs exported blind (`review_opus_400.md`); TBD.

## Verdict

Provisional, pending the human calibration of the judge: the hypothesis is **partly confirmed**. qwen3.8 (Qwen3.5-27B) is within 4 points of Sonnet on answers and close on summaries and evidence, at 47 labels/min for nothing, but is clearly worse on rare-event recall (0.53 vs 0.73 on 15 pairs) and on vulnerability tags; gemma4:12b is close behind and best on vulnerability answers, but misses dissatisfaction and has a repetition-loop failure mode on long podcast episodes; llama3.3:70b has the best rare-event recall and the weakest evidence; qwen3:32b, the local model the audits relied on, is the weakest labeller of the five, mainly through empty evidence lists and answering NA where the rules give fail. No local model matches Sonnet on all five criteria, so the disconfirming case "local model needs only the API model as verifier" did not occur; nor did the other ("every local model ≥10 points below Sonnet in some family"): qwen3.8 is within 6 points everywhere except rare events. Caveats: the judge is the same family as Sonnet; 15 rare-event pairs cannot rank models; the sample over-represents disagreements, so absolute accuracies understate performance on typical pairs.

## Follow-ups

- Human calibration of the judge on the exported 40 pairs (`analyse_labellers --import-review`, then `--human`); if the human agrees with Opus at ≥ 0.85 on answer correctness the ranking stands.
- Rare events need a dedicated test: a set of calls the audits flagged as positives, so recall is measured on ≥ 50 positives rather than 15.
- Candidate production design: qwen3.8 labels everything locally; Sonnet re-labels the family questions and a random 10% as verifier; disagreements and any local label with an empty evidence list go to a Sonnet re-label. Cost then scales with the family questions (~5% of labels) rather than all of them.
- Confidence from local models is unusable as a routing signal; use verifier disagreement and evidence emptiness instead.
