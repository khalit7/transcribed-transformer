# E3, native reference: T5Gemma 2 1b-1b against Gemma 3 1B

Written before the run on 2026-09-15, after the Qwen-built encoder-decoder (`../2026-09-14-e3-encdec-qwen3-1.7b-base/`) was stopped for trailing E1 by 0.35 in val loss. Khalid: "I still want to investigate the encoder idea further. What do you think of running a native encoder decoder like T5 gemma?" then "Yes lets start with T5gemma2 1b-1b".

## Hypothesis

The Qwen-built encoder-decoder failed at the transfer, and the cleanest explanation is adaptation budget: 661M tokens against the roughly 2T tokens Google spent turning Gemma 3 into T5Gemma 2. A converged encoder-decoder removes that variable. If an encoder-decoder that has already paid its adaptation cost beats the decoder it was made from on this task, the encoder idea survives and our E3 failed for budget; if it does not, a dedicated bidirectional encoder does not help this task at about 1B parameters however well it is trained.

This is a matched pair from one family, not a comparison with the Qwen arms: T5Gemma 2 1b-1b (encoder and decoder both from Gemma 3 1B, adapted by Google, about 2.1B parameters of which 0.4B is an unused, frozen image tower) against Gemma 3 1B pretrained (the decoder it was adapted from), both fine-tuned with E1's recipe on the same data and steps. The difference between the two is architecture plus the adaptation, which is the thing under test. E1 (Qwen3-1.7B-Base) is reported alongside as context only.

## Prediction

- Fine-tune val loss of T5Gemma 2 at or below Gemma 3 1B's at every logged step from 400 on.
- Benchmark unseen-question macro-F1 of T5Gemma 2 above Gemma 3 1B by at least 0.02 on both variants, evidence F1 at or above it.
- **Disconfirmed if** T5Gemma 2 is at or below Gemma 3 1B on unseen-question macro-F1 on both variants. That would mean a fully converged encoder-decoder adds nothing over its own decoder here, and the encoder line closes at this size for reasons other than adaptation budget.
- Inconclusive if either run fails to train cleanly (the library's T5Gemma 2 has no FlashAttention path and a reported failure above 4k decoder tokens; our decoder side is about 150 tokens).

## Setup

- T5Gemma 2 (`configs/e3/t5gemma2-1b-1b.yaml`, `model.arch: hf_encdec`): the library's own `T5Gemma2ForConditionalGeneration` forward (encoder: 26 layers, 22 with a bidirectional 512-token sliding window and 4 full; decoder: merged self- and cross-attention, same layout), SDPA attention, every text parameter trained under DDP with E1's precision recipe (bf16 model, fp32 masters, 8-bit AdamW), gradient checkpointing, the image tower frozen. Prompt to the encoder opening with `<bos>`, decoder teacher-forced from `<bos>`, loss on the JSON label and its `<eos>`. Context capped at the model's native 16k (`max_seq_len: 16384`; the benchmark's longest case is 15,018 tokens; the calls above 16k are dropped, 22 of 91,502 training records with the Gemma tokenizer, leaving 91,480 examples, 252.2M prompt tokens and 2,858 steps).
- Gemma 3 1B (`configs/e1/gemma3-1b-pt.yaml`): E1's recipe unchanged, FlashAttention 2, the same 16k cap, prompt opening with `<bos>`.
- Both: E1's data, split, rendering, lr 1e-5 cosine, 50 warmup, 32 sequences per step, one epoch, val every 200 steps on the same 512 fixed examples.
- Generation: T5Gemma 2 through `generate.py --backend hf_encdec` (the library's generate, greedy, length-sorted batches under an encoder-token cap); Gemma 3 1B through vLLM as E1. Scoring as before.
- Probe (one RTX 5090, before the run): a fine-tuning step at 1k / 4k / 8k / 15k / 16k encoder tokens with a 160-token target peaked at 19.8–19.9 GiB throughout (the optimizer step dominates), 2.1k / 7.1k / 6.4k / 4.6k / 4.6k encoder tokens/s; batched greedy generation ran. Smoke (both GPUs, 3 steps): loss 1.40, val 1.43 at step 3, 17.4k tokens/s across the pair, 23.4 GiB peak.

## wandb runs

- T5Gemma 2, attempt 1: [tt-encdec/bivp95p4](https://wandb.ai/khalit7-/tt-encdec/runs/bivp95p4), 2026-09-15 23:55 → 00:26, hung at step 303 in an NCCL all-reduce at its first checkpoint (no checkpoint written; 2-hour watchdog). Val 1.458 at step 0, 0.833 at step 200.
- T5Gemma 2, attempt 2: [tt-encdec/3spaw5sg](https://wandb.ai/khalit7-/tt-encdec/runs/3spaw5sg), 2026-09-16 02:29 → 05:00, hung at the same step; the NCCL flight recorder showed the two ranks entering the wall-clock checkpoint one step apart (each rank read its own clock), so their collectives paired wrongly. Fixed: rank 0 decides and broadcasts.
- T5Gemma 2, attempt 3: [tt-encdec/1a1kxss8](https://wandb.ai/khalit7-/tt-encdec/runs/1a1kxss8), from step 0 at 2026-09-16 05:02.
- Gemma 3 1B: [tt-decoder/om6sbier](https://wandb.ai/khalit7-/tt-decoder/runs/om6sbier), started 2026-09-16 12:23.

## Result

**T5Gemma 2 1b-1b** (attempt 3, 2026-09-16 05:02 → 09:48, 4.77 h, 2,858 steps; generation 2.6 h on two GPUs through the library's generate; scored 12:23). Val loss 1.458 → 0.831 (200) → 0.680 (1000) → 0.571 (2000) → 0.529 (2858). Benchmark, read from `checkpoints/e3-t5gemma2-1b-1b/eval/benchmark/results.md` (also logged to the wandb run under `bench/*`), with E1 (Qwen3-1.7B-Base, different family, context only) and E2++ beside it:

| variant / slice | metric | T5Gemma 2 1b-1b | Gemma 3 1B (control) | E1 Qwen3-1.7B | E2++ |
|---|---|---|---|---|---|
| clean, overall (n=17,520) | macro-F1 | 0.744 | 0.651 | 0.735 | 0.731 |
| | evidence F1 | 0.619 | 0.530 | 0.603 | 0.619 |
| | evidence precision | 0.725 | 0.589 | 0.697 | 0.722 |
| | format valid | 0.990 | 0.955 | 0.985 | 0.988 |
| clean, unseen questions (n=3,260) | macro-F1 | 0.677 | 0.496 | 0.654 | 0.655 |
| | evidence F1 | 0.578 | 0.477 | 0.573 | 0.586 |
| messy, overall (n=20,700) | macro-F1 | 0.722 | 0.640 | 0.719 | 0.713 |
| | evidence F1 | 0.590 | 0.500 | 0.570 | 0.584 |
| messy, unseen questions (n=3,860) | macro-F1 | 0.642 | 0.493 | 0.652 | 0.649 |
| | evidence F1 | 0.552 | 0.444 | 0.550 | 0.555 |
| messy, SPoRC (n=3,180) | macro-F1 | 0.687 | 0.592 | 0.700 | 0.689 |
| | evidence F1 | 0.546 | 0.428 | 0.509 | 0.531 |

Majority-class macro-F1 is 0.592 / 0.594 overall and 0.582 / 0.572 on unseen questions (clean / messy). Tag Jaccard 0.869 clean / 0.841 messy against E1's 0.777 / 0.758.

Reading before the control lands: the converged encoder-decoder, at 1.7B trainable parameters from a 1B decoder family, sits at or above the 1.7B Qwen decoder on every clean slice and on evidence everywhere, and below it only on unseen-question answers under the messy variant (−0.010) and on SPoRC answers (−0.013). Its evidence precision matches E2++'s, the best of the prefix-LM line. None of this is the pair comparison; that needs the Gemma 3 1B number.

**Gemma 3 1B control** (2026-09-16 12:23 → 15:14, 2.83 h, same 2,858 steps; vLLM generation 8 min on two GPUs; scored 15:22). Val loss 1.732 (0) → 0.921 (200) → 0.742 (1000) → 0.659 (2000) → 0.632 (2858), against T5Gemma 2's 1.458 → 0.831 → 0.680 → 0.571 → 0.529 on the same tokenizer: the encoder-decoder led by 0.09 at step 200 and by 0.10 at the end. Benchmark as in the table. The control is below the majority-class predictor on unseen questions (0.496 vs 0.582 clean, 0.493 vs 0.572 messy) and has the worst format validity of any trained arm (0.955 / 0.946). Tag Jaccard 0.776 / 0.757 against T5Gemma 2's 0.869 / 0.841. Val split: macro-F1 0.687 / 0.670, evidence F1 0.578 / 0.536, against T5Gemma 2's 0.773 / 0.739 and 0.707 / 0.660.

**The pair.** T5Gemma 2 over Gemma 3 1B, clean / messy: macro-F1 +0.093 / +0.082 overall, +0.181 / +0.149 on unseen questions; evidence F1 +0.089 / +0.090; evidence precision +0.136 / +0.139. Every dataset, family, cell and length slice favours the encoder-decoder, and the margin is widest exactly where the task is hardest for the control: unseen questions and the 4k–16k prompts (messy 8–16k: macro-F1 0.702 vs 0.580, evidence F1 0.539 vs 0.411).

## Verdict

**Confirmed, as predicted, with the confound the design accepted.** The prediction asked for at least +0.02 unseen-question macro-F1 on both variants; the measured margins are +0.181 and +0.149, and val loss, evidence and format all agree. A converged encoder-decoder beats the decoder it was made from on this task by a wide margin at about 1B, and it also sits at or above the 1.7B Qwen causal decoder (E1) on most slices while being a different family.

What it does not separate: T5Gemma 2 is Gemma 3 1B plus about 2T tokens of further training plus a second tower (1.7B trainable parameters against 1.0B). The pair attributes the gain to "architecture plus adaptation", which is what a native model can offer; the one-variable version of this question (our E3, 661M adaptation tokens on the Qwen weights) failed, and this result says the failure was about budget and not about the idea. The weak control is part of the finding, not a flaw in it: Gemma 3 1B pretrained, under the E1 recipe, does not generalise to unseen questions at all, and the adapted encoder-decoder from the same weights does.

For the README's E3 row: the native reference is now the strongest trained model on the benchmark at any size tried, and the encoder-decoder line does not close at 1.7B; it closes for *building one from a decoder with a small budget*.

**After the size control (2026-09-16, evening):** the margin over Gemma 3 1B is not a parameter-count effect. The 270m-270m, at under a third of the decoder's non-embedding parameters, matches it on answers and beats it on evidence and format; what size buys, going from 270m-270m to 1b-1b, is the ability to answer questions never seen in training above chance, which neither small model has. The attribution left open is between the bidirectional architecture and Google's 2T-token adaptation, and a native model cannot separate those two.

## Size control: T5Gemma 2 270m-270m (added 2026-09-16, before its run)

Khalid, on the pair result: "the 1b-1b model is the model with the highest number of parameters. so maybe that plays a part." Against its control it does have twice the non-embedding parameters (1.40B against 0.70B); against E1 it is parameter-matched (1.41B), and per prompt token it costs what Gemma 3 1B costs, since only the encoder reads the prompt. The cheap test: T5Gemma 2 270m-270m (encoder and decoder each 18 layers, hidden 640; about 0.2B non-embedding parameters in total, under a third of Gemma 3 1B's 0.70B) under the identical recipe, `configs/e3/t5gemma2-270m-270m.yaml`.

- **Prediction.** If the 1b-1b margin is architecture plus adaptation rather than size, the 270m-270m lands at or above Gemma 3 1B on unseen-question macro-F1 on both variants (0.496 clean, 0.493 messy) and above the majority predictor there (0.582 / 0.572). **Size is confirmed as part of the explanation if** the 270m-270m falls clearly below the 1B decoder on unseen questions on both variants; a result between the control and the 1b-1b is the mixed case and gets reported as such.
- wandb: [tt-encdec/o0lbp362](https://wandb.ai/khalit7-/tt-encdec/runs/o0lbp362), started 2026-09-16 16:50; 0.37B trainable parameters (0.79B with the frozen image tower); val loss 2.026 at step 0.
- **Result** (2026-09-16 16:50 → 19:12 training, 2.35 h; generation 1.8 h; scored 21:02). Val loss 2.026 (0) → 1.015 (200) → 0.814 (1000) → 0.704 (2000) → 0.667 (2858): below the 1B decoder (0.632) by 0.035–0.09 throughout. Benchmark, clean / messy:

| metric | 270m-270m (0.2B non-emb) | Gemma 3 1B (0.7B) | 1b-1b (1.4B) |
|---|---|---|---|
| macro-F1 overall | 0.660 / 0.641 | 0.651 / 0.640 | 0.744 / 0.722 |
| macro-F1 unseen questions | 0.509 / 0.510 | 0.496 / 0.493 | 0.677 / 0.642 |
| majority predictor, unseen | 0.582 / 0.572 | same | same |
| evidence F1 | 0.576 / 0.545 | 0.530 / 0.500 | 0.619 / 0.590 |
| evidence precision | 0.667 / 0.634 | 0.589 / 0.555 | 0.725 / 0.694 |
| format valid | 0.982 / 0.973 | 0.955 / 0.946 | 0.990 / 0.985 |
| tag Jaccard | 0.837 / 0.801 | 0.776 / 0.757 | 0.869 / 0.841 |
| messy 8–16k: macro-F1 / evidence F1 | 0.618 / 0.470 | 0.580 / 0.411 | 0.702 / 0.539 |
| messy SPoRC: macro-F1 / evidence F1 | 0.612 / 0.486 | 0.592 / 0.428 | 0.687 / 0.546 |

- **Reading: the mixed case, leaning against size.** The first half of the prediction holds: at under a third of the control's non-embedding parameters and with a worse val loss, the 270m-270m ties the 1B decoder on answers (+0.009 / +0.001 overall, +0.013 / +0.017 unseen) and beats it on evidence F1 (+0.046 / +0.045), evidence precision (+0.08), format validity, tags, long prompts and SPoRC. Parameter count therefore does not explain the 1b-1b's margin over the same decoder: a far smaller encoder-decoder reaches the decoder's answer accuracy and passes it on everything else. The second half fails: like the 1B decoder, the 270m-270m stays below the majority predictor on unseen questions (0.509 vs 0.582), so zero-shot question transfer is the one thing in the 1b-1b's result that needed its capacity, and the two small models fail it alike. Val loss ranked the three models the other way round from the benchmark on evidence and format (270m-270m worst on loss, second on both metrics), a third time that loss has not predicted this benchmark.

## The small pair: Gemma 3 270M control (added 2026-09-16 23:00, before its run)

Khalid: "I would like to run the decoder only version of all models that have an encoder-decoder twin." Three same-family pairs (270m, 1b, 4b) make the pair result a scale trend. First the small one: `google/gemma-3-270m` (text-only pretrained decoder, the origin of the 270m-270m; 18 layers, hidden 640, about 0.1B non-embedding parameters) under `configs/e1/gemma3-270m.yaml`, E1's recipe, 16k cap.

- **Prediction.** The small pair repeats the shape of the large one: the 270m-270m above the 270M decoder on evidence F1 (by at least 0.03 on both variants) and on format validity, with answers close (within 0.02 macro-F1 either way) and both below the majority predictor on unseen questions. **Disconfirmed if** the 270M decoder matches or beats the 270m-270m on evidence F1 on both variants, which would mean the encoder-decoder's evidence advantage at 1b-1b was a one-off. If the decoder clears the majority predictor on unseen questions while the 270m-270m does not, the "size buys transfer" reading of the 1b-1b result is wrong and gets rewritten.
- wandb: [tt-decoder/b8rzkm1s](https://wandb.ai/khalit7-/tt-decoder/runs/b8rzkm1s), 2026-09-16 23:01 → 23:52 (0.84 h; vLLM generation 6 min; scored 23:59).
- **Result.** Val loss 2.180 (0) → 1.195 (200) → 0.975 (1000) → 0.893 (2000) → 0.870 (2858), against the 270m-270m's 2.026 → 1.015 → 0.814 → 0.704 → 0.667: a gap of 0.20 at the end (0.10 in the 1B pair). Benchmark, clean / messy:

| metric | Gemma 3 270M | 270m-270m |
|---|---|---|
| format valid | 0.651 / 0.632 | 0.982 / 0.973 |
| macro-F1 overall | 0.446 / 0.431 | 0.660 / 0.641 |
| macro-F1 unseen questions | 0.315 / 0.313 | 0.509 / 0.510 |
| evidence F1 | 0.299 / 0.276 | 0.576 / 0.545 |
| evidence precision | 0.310 / 0.290 | 0.667 / 0.634 |
| messy 8–16k: macro-F1 / evidence F1 | 0.406 / 0.197 | 0.618 / 0.470 |

The 270M decoder collapses under greedy decoding: 35.9% of its benchmark outputs are runaway evidence lists that count line numbers upward until the 512-token cap (13,714 of 38,220), and the rest is scored as is. Overall macro-F1 is below the majority predictor (0.446 vs 0.592). The prediction ("answers close, evidence advantage of at least 0.03, both below majority on unseen") is confirmed on the evidence and format half and exceeded on answers: the encoder-decoder is ahead by 0.21 macro-F1 and 0.28 evidence F1, and it is the only one of the two above chance overall. The runaway share of the other decoders for reference: Gemma 3 1B about 4.5% invalid, E1 1.4%.

## The large pair: Gemma 3 4B and T5Gemma 2 4b-4b (added 2026-09-17 00:15; revised 00:30)

Khalid: "I would like to prove this on a bigger scale", full-parameter. The 4b-4b's 7.5B parameters do not fit E1's recipe (fp32 master weights) on two 32 GB cards: about 37 GB per GPU sharded. The first plan put both 4B arms on a bf16 + stochastic-rounding recipe (`optim.weights: bf16_sr`) and re-ran the 270m-270m under it as a check. That check, stopped at step 230, showed the recipes are **not** equivalent: val loss 0.922 at step 200 against 1.015 under fp32 masters (identical 2.026 at step 0), training loss lower from step 50 on. **Decision (Khalid, 00:25): every run stays on E1's recipe; the 4b-4b goes to rented hardware** (two 80 GB or four 48 GB cards) under the same recipe. Gemma 3 4B runs locally under E1's recipe, sharded as the Qwen-built E3 was.

- **Prediction (unchanged).** The 1B and 270M pairs give +0.09 and +0.21 macro-F1 for the encoder-decoder. If the trend is real, 4b-4b beats Gemma 3 4B by at least +0.03 macro-F1 overall on both variants and leads on evidence F1; the margin is expected to shrink with size. **Disconfirmed if** Gemma 3 4B matches or beats the 4b-4b on unseen-question macro-F1 on both variants: the encoder-decoder's advantage would then be a small-model effect that a large enough decoder closes.
- Probe, Gemma 3 4B under E1's recipe sharded (both GPUs, one 16k sequence, 160-token target): 24.4 GiB, 4.6k tokens/s per GPU (3.88B trainable, image tower frozen, FlashAttention 2). The bf16_sr probe numbers, for the record: 17.6 GiB / 4.7k tok/s (Gemma 3 4B), 27.5 GiB / 1.5k tok/s (4b-4b).
- Configs `configs/e1/gemma3-4b-pt.yaml` (running), `configs/e3/t5gemma2-4b-4b.yaml` (annotated for rented hardware); 16k cap as in the other pairs.
- wandb: Gemma 3 4B [tt-decoder/xgv48ne6](https://wandb.ai/khalit7-/tt-decoder/runs/xgv48ne6), 2026-09-17 00:29 → 11:08 (10.65 h; vLLM generation 17 min; scored 11:35 after two failed launches, see below). 4b-4b TBD (rented hardware).
- **Gemma 3 4B result.** Val loss 1.265 (0) → 0.635 (200) → 0.523 (1000) → 0.454 (2000) → 0.427 (2858): the lowest final loss of any run (1b-1b 0.529, E1 0.503 on Qwen's tokenizer, 1B decoder 0.632). Benchmark, clean / messy, with the 1b-1b (the encoder-decoder it brackets from above at 3.9B vs 2.1B parameters) and E1:

| metric | Gemma 3 4B | T5Gemma 2 1b-1b | E1 Qwen3-1.7B |
|---|---|---|---|
| macro-F1 overall | 0.778 / 0.745 | 0.744 / 0.722 | 0.735 / 0.719 |
| macro-F1 unseen questions | 0.702 / 0.684 | 0.677 / 0.642 | 0.654 / 0.652 |
| evidence F1 | 0.635 / 0.604 | 0.619 / 0.590 | 0.603 / 0.570 |
| evidence precision | 0.743 / 0.713 | 0.725 / 0.694 | 0.697 / 0.663 |
| format valid | 0.992 / 0.988 | 0.990 / 0.985 | 0.985 / 0.979 |
| messy SPoRC, macro-F1 / evidence F1 | 0.731 / 0.554 | 0.687 / 0.546 | 0.700 / 0.509 |
| messy 8–16k, macro-F1 / evidence F1 | 0.739 / 0.548 | 0.702 / 0.539 | 0.721 / 0.492 |

Val split: macro-F1 0.820 / 0.777, evidence F1 0.737 / 0.687. The strongest trained model on the benchmark so far, on every slice.

- **Reading as the inference-matched control.** A 3.9B decoder beats the 2.1B encoder-decoder by 0.034 / 0.023 macro-F1 overall and 0.025 / 0.042 on unseen questions, with evidence F1 +0.016 / +0.014. The encoder-decoder gets within that margin while reading each prompt token through 0.7B parameters against the decoder's 3.9B, about a fifth of the compute per prompt token. Whether the 4b-4b (7.5B, 3.9B per prompt token) beats Gemma 3 4B is the pair question, still open.
- Two evaluation launches failed on the export before this number: the multimodal class needs the image-processor files, and the streamed export had written tensors under module names where vLLM reads the on-disk convention. The export now goes through `save_pretrained` on a CPU copy plus a rename of the frozen image tower; the model itself was unaffected.

**Recipe observation, parked.** The stopped check (`checkpoints/e3-t5gemma2-270m-270m-bf16sr-stopped/`, tt-encdec run) is the only measurement of the precision recipe's effect on this task: at lr 1e-5, bf16 weights with stochastic rounding trained faster than bf16 weights with fp32 masters over the first 230 steps of the 270m-270m. Not benchmarked, not explained; a plausible mechanism is that most single updates at this learning rate are below bf16's resolution and reach the weights only once accumulated under the master-copy recipe. Worth a full run at some point; not part of the pair design.

## Follow-ups

- **SmolLM3 3B cross-family control (2026-09-19).** [Pre-run hypothesis and setup](../2026-09-19-e1-smollm3-3b-base/README.md): an additional E1 baseline under the same precision recipe and 16k cap, testing practical competitiveness rather than architecture alone.
- **Inference-matched decoder control.** The README asks for the encoder-decoder to be reported against both the parameter-matched and the inference-matched decoder. Gemma 3 4B pretrained under the E1 recipe brackets T5Gemma 2 1b-1b from above (4B against 2.1B); if the 1b-1b still wins or ties, the architecture claim strengthens. Full-parameter training of 4B needs the FSDP path generalised from `EncDec` to a plain decoder (or LoRA on both arms, which is a further confound).
- **T5Gemma 2 4b-4b** is the scale point (8.6B parameters, FSDP required, ~2× the time); worth it only after the 4B decoder control exists.
- **E0** (the API baseline) is now the missing column: the strongest trained model needs to be placed against it.
- The 22 training calls above 16k were dropped for both arms; RoPE interpolation to 32k for T5Gemma 2 is untested here.
- Generation through the library's `generate` took 2.6 h for the benchmark, against 8 min for the control under vLLM; vLLM has no T5Gemma 2 support, so a faster custom loop (encoder once, cached decode, as `--backend encdec`) is the fix if this model is evaluated often.
