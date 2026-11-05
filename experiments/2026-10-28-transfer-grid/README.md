# Transfer grid: does "the capacity goes in the encoder" hold off the compliance task?

**Status: nine cells scored 2026-11-01, observations recorded, analysis pending; pre-registered 2026-10-28 with three datasets; the ECtHR arms have been running since 2026-10-28; re-scoped to
ECtHR alone on 2026-10-30 (see the Setup).** Everything below the Setup was written before any number existed, apart from
the dated additions. Results are filled in as the queue scores each run, from wandb, never from memory.

## Hypothesis

On the compliance benchmark the sound results put the capacity in the encoder: at equal total size the arm with the
bigger encoder wins (1b-enc + 270m-dec over 270m-enc + 1b-dec), a 4b encoder behind a 270M decoder matches the 4B
decoder alone, and the 270M decoder goes from below the floor to the native 2B pair's level as its encoder grows
(`README.md`, "Three questions", question 3). That is one benchmark, built from one labeller's labels over four
conversational corpora. I want to know whether the ordering is a property of the architecture at this input/output shape
(a long document in, a short structured answer out) or of this task. If it is the architecture, the same grid trained on
other long-input, short-output tasks should reproduce the same ordering; if it is the task, it will not.

## Prediction

One dataset, ECtHR (pre-registered as three; see the scope change under Setup), nine E1/E4 arms (plus, from 2026-10-30, seven E1++/E5-soft-tokens++ arms; see the Setup), one seed. I expect:

1. **Equal size, different split**: `e4mm-1b-enc-270m-dec` scores above `e4mm-270m-enc-1b-dec` on the headline metric.
2. **Big encoder, tiny decoder**: `e4mm-4b-enc-270m-dec` scores within noise of `e1-gemma3-4b-pt` and above `e1-gemma3-1b-pt`.
3. **Encoder axis beats decoder axis**: with the 270m encoder fixed, growing the decoder 270m → 1b → 4b recovers less of
   the 270M-decoder-to-4B-decoder gap than growing the encoder 270m → 1b → 4b does with the 270m decoder fixed.
4. **An encoder helps at every size**: each native pair (`270m-270m`, `1b-1b`) scores above its decoder-only control.

What disconfirms: prediction 1 reversed, or prediction 2 failing (the 4b encoder + 270M decoder below the 1B decoder)
where the 4B decoder itself clears the floor. I also expect the absolute numbers to be lower than on the compliance
benchmark, since this task is not what the arms were designed for; absolute level is not what this record tests.

Noise is not estimated in advance beyond the planning note that ECtHR's benchmark (10,000 binary pairs) resolves
differences of a point or two. The second seed on the equal-size pair, which would turn prediction 1 into a measured
interval, is deferred.

## Setup

**Dataset** (`data/DATASHEET.md`, "Transfer dataset"): ECtHR task A from LexGLUE (Convention articles violated, one binary
question per article, Articles 8 and 10 held out of training: 72,000 / 8,000 / 10,000 records, 2,000 of them the
unseen-question cell). Written text (tier 3), Track NC; nothing derived from it is released. It has its own
`data/labelled_data/ecthr/` built by `src/preprocessing/ecthr.py`, in the labelled-data record schema so the trainer, the
generator and the evaluator read it unchanged. Each cell is one arm, trained from the same base checkpoint its compliance
counterpart started from; nothing is mixed with the compliance data.

**Scope change, 2026-10-30 (the ECtHR block was at its seventh cell; no arm of the other two had trained).** I pre-registered
three datasets: ECtHR, SCOTUS from LexGLUE (the issue area of a full Supreme Court opinion, one of thirteen; 5,000 / 1,400 /
1,400 records; the long-input arm, benchmark median prompt 12k tokens) and QuALITY v1.0.1 (four-way reading comprehension
over 2k–9k-token articles; 2,388 / 135 / 2,086 records). I dropped both and removed them from the repository (loaders,
configs, tests, records, datasheet and survey rows) after four independent analyses of the first six ECtHR cells and of the
two pilots showed that neither would test the question. SCOTUS does not need its long input: a bag-of-words classifier
trained on the first 500 words of each opinion scores 0.695 accuracy on the benchmark against the 1B decoder pilot's 0.770,
0.719 on the full text, and the curve is flat beyond 2,000 words; the LexGLUE authors describe it as a topic task. QuALITY is
unlearnable at these sizes from its 2.4k training questions: the 1B pilot's 0.309 accuracy is matched by a classifier that
never sees the article (0.330 from the question and option texts alone), the pilot answered `A` on 54% of questions against
23.5% in gold, and its validation loss never left the prior entropy (an estimate that attributes the whole loss to the
answer token); the published QuALITY-only baselines sit at 24–31% for
the same reason. Both pilots ran (tt-decoder: SCOTUS `s5e5b4ql`, which chose 2 epochs; QuALITY `ztiu0bvt`, which chose 1) and their
checkpoints were deleted. The long-input test moves to a dataset built from this project's own long ASR transcripts, labelled
by a stronger labeller than the compliance data's, with a pre-registered admission test that a dataset must pass before any
arm trains: a bag-of-words on its first 500 words must fall well short of the pilot. That dataset gets its own record.

**Arms**, nine E1/E4 arms (the seven E5-soft-tokens arms are tabled further down), each the compliance config with only the data paths, over-long policy, epochs and
`val_every` changed (`scripts/make_transfer_configs.py`; `tests/test_transfer_configs.py` checks the configs differ from
their sources in nothing else):

| Arm | Encoder | Decoder | Params | Compliance counterpart |
|---|---|---|---|---|
| `e1-gemma3-270m` | — | Gemma 3 270M | 0.27B | `configs/e1/gemma3-270m.yaml` |
| `e1-gemma3-1b-pt` | — | Gemma 3 1B | 1B | `configs/e1/gemma3-1b-pt.yaml` |
| `e1-gemma3-4b-pt` | — | Gemma 3 4B | 4B | `configs/e1/gemma3-4b-pt.yaml` (FSDP) |
| `e4-t5gemma2-270m-270m` | T5Gemma 2 270m | 270m | 0.54B | `configs/e4/t5gemma2-270m-270m.yaml` |
| `e4-t5gemma2-1b-1b` | 1b | 1b | 2B | `configs/e4/t5gemma2-1b-1b.yaml` |
| `e4mm-270m-enc-1b-dec` | 270m | 1b | ~1.3B | `configs/e4/mm-270m-enc-1b-dec.yaml` |
| `e4mm-270m-enc-4b-dec` | 270m | 4b | ~4.3B | `configs/e4/mm-270m-enc-4b-dec.yaml` (FSDP) |
| `e4mm-1b-enc-270m-dec` | 1b | 270m | ~1.3B | `configs/e4/mm-1b-enc-270m-dec.yaml` |
| `e4mm-4b-enc-270m-dec` | 4b | 270m | ~4.3B | `configs/e4/mm-4b-enc-270m-dec.yaml` (FSDP) |

**Added 2026-10-30, before any encoder-decoder cell of the grid had scored beyond the 270m-270m pair: the E5-soft-tokens arms with the
compliance adaptation stage, seven, run after the nine E4 arms rather than as a separate follow-up.** Each starts from the compliance stage's export
(`checkpoints/adapt-continuation-corpora-*/final`), so both sides of every pair went through the identical stage and the
comparison is matched exactly as it was on compliance; no new stages are run. The interface they test was learnt on speech
transcripts, so a transfer here also says whether that interface carries to written documents.

| Arm | Encoder | Decoder | Starts from | Compliance counterpart |
|---|---|---|---|---|
| `e1pp-gemma3-1b-pt` | — | Gemma 3 1B, adapted | `adapt-continuation-corpora-gemma3-1b-pt/final` | `configs/e1/gemma3-1b-pt-plusplus.yaml` |
| `e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-1b` | T5Gemma 2 270m | Gemma 3 1B | `adapt-continuation-corpora-e5-soft-tokens-t5gemma2-270m-enc-gemma3-1b/final` | `configs/e5-soft-tokens/t5gemma2-270m-enc-gemma3-1b-plusplus.yaml` |
| `e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-1b` | 1b | Gemma 3 1B | `…-e5-soft-tokens-t5gemma2-1b-enc-gemma3-1b/final` | `…1b-enc-gemma3-1b-plusplus.yaml` |
| `e1pp-gemma3-270m` | — | Gemma 3 270M, adapted | `adapt-continuation-corpora-gemma3-270m/final` | `configs/e1/gemma3-270m-plusplus.yaml` |
| `e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-270m` | 270m | Gemma 3 270M | `…-e5-soft-tokens-t5gemma2-270m-enc-gemma3-270m/final` | `…270m-enc-gemma3-270m-plusplus.yaml` |
| `e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-270m` | 1b | Gemma 3 270M | `…-e5-soft-tokens-t5gemma2-1b-enc-gemma3-270m/final` | `…1b-enc-gemma3-270m-plusplus.yaml` |
| `e5-soft-tokens-pp-t5gemma2-4b-enc-gemma3-270m` | 4b | Gemma 3 270M | `…-e5-soft-tokens-t5gemma2-4b-enc-gemma3-270m/final` | `…4b-enc-gemma3-270m-plusplus.yaml` (FSDP) |

Predictions for these, written before any of them ran, from the compliance results (E5-soft-tokens++ 1b → 1B +0.036 answers over
E1++; E5-soft-tokens++ 4b → 270M +0.262 over E1++ 270M and within 0.014 of the 4B decoder; the soft stitch below the native stitch at
the same sizes): (5) every E5-soft-tokens++ arm scores above its adapted decoder control, by more at the 270M decoder
than at the 1B; (6) at the 270M decoder the E5-soft-tokens encoder axis orders 270m < 1b < 4b, and sits at or below the E4 native-stitch
axis (`e4mm-{270m,1b,4b}-enc-270m-dec`) cell for cell; (7) `e5-soft-tokens-pp-t5gemma2-4b-enc-gemma3-270m` lands within 0.03 of
`e1-gemma3-4b-pt` on the headline. What disconfirms: an E5-soft-tokens++ arm at or below its E1++ control on the held-out cell; if that happens, the register caveat applies and a fresh stage on ECtHR's own documents is the
next run, not a conclusion.

**Prediction 8, added 2026-10-30 after six ECtHR cells: the encoder-decoder advantage over a same-size decoder is a
long-input advantage.** (Refuted at its premise the same day; see the Result. Kept as dated.) Evidence that prompted it: on compliance (median prompt
2.3k tokens on the call corpora, 6k on the podcasts, cap 16k) the native 1b-1b pair led the 1B decoder by 0.097 macro-F1;
on ECtHR (median prompt 1.8k) it trails by 0.004, and the whole deficit sits in the prompts of 2k tokens or fewer (−0.016
on 4,529 pairs) while every bucket above 2k reads level or ahead (+0.009, +0.001, +0.003). The prediction, as written: on a dataset
with long inputs both native pairs beat their decoder controls by more than they did on ECtHR, and within each dataset the
pair-minus-decoder difference rises with the length bucket (`length=` slices in `metrics.json`). If confirmed, the capacity
finding should be stated with its scope: for long inputs. The per-bucket reading stays as a standing rule whatever becomes of
the prediction: the Result section of every dataset's record carries a table of arm-minus-control macro-F1 by length bucket
(≤2k, 2–4k, 4–8k, 8–16k rendered-prompt tokens, the evaluator's `length=` slices) for the native pairs and the stitches,
with the compliance benchmark re-read the same way from its existing metrics files, so the length effect can be isolated
from the dataset.

The three cells the compliance grid left for rented hardware (1b-enc + 4b-dec, 4b-enc + 1b-dec, 4b-4b) stay out. The
stitches start from the affine maps fitted for the compliance runs (`checkpoints/stitch/*.pt`, ridge regression on
compliance transcripts); they are an initialisation that trains, and I did not refit them on the transfer text.

**Recipe**: the compliance recipe unchanged (lr 1e-5, warmup 50, weight decay 0.1, batch 32 sequences, 16k cap, full
parameters, the stitch at 10× the base rate), one seed (0). The one knob is the epoch count, set by a **pilot**: the 1B
decoder trains for the top of a candidate range (2 epochs) with `val_every` a divisor of the steps per epoch, the epoch
with the lowest validation loss becomes the count, and every arm then trains for it. The pilot runs at a constant learning
rate after warm-up (`min_lr_ratio: 1.0`), not the compliance cosine, so that where its validation loss bottoms out says
how many passes over the data help rather than where the cosine happened to be; the arms then anneal fully over the
chosen count. That makes the pilot an imperfect stand-in for the arms (a point a reviewer raised), and the record will
say which epoch the curve chose and how flat it was around it. The pilot is a separate run from the
1B decoder's own cell (the trainer saves by wall-clock, not at epoch ends, so the pilot's checkpoint at the chosen epoch
is not recoverable); it gives the 1B decoder no advantage over the other arms beyond having been looked at. If the
validation loss is still falling at the top of a range, the range is extended rather than stopped.

**What the datasets forced** (all under the two-model review, `pre-run-review`):

- *Document prompt.* A `transfer` question gets the preamble "You answer questions about a document. One paragraph per
  line, numbered. Each line starts with the fixed label text." in place of the conversation preamble; the rest of the
  prompt (evidence-first instructions, options with criteria, the JSON shape) is identical (`src/train/data.task_prompt`).
- *Target.* The three-field JSON with an empty evidence list and a blank summary, since neither is annotated. Only the
  answer field is scored here; evidence metrics are reported but degenerate (gold empty everywhere).
- *The 16k cap.* The compliance loader drops a record longer than the cap. The transfer configs set `data.overlong:
  truncate` instead, so that every benchmark pair is scorable whatever its length: a document is cut from the tail to fit
  (`data.fit_prompt`, a binary search over the number of lines kept), and generation applies the same cut (`generate.py
  --truncate-to 16320`). The evaluator flags each record whose full rendered prompt exceeds the cap (`--cap-tokens
  --tokenizer`, re-rendering with the run's tokenizer) and slices by it (`over_cap=True/False`), so the under-cap reading
  comes from the scored pipeline, not from an ad-hoc script. On ECtHR this touches 58 training cases and 3 benchmark cases.
- *Floors.* The evaluator's floor is each question's majority gold answer. A question asked of a single pair in the whole
  split would make that floor the gold itself, so such a question takes the majority gold over every pair offering the same
  options (the option-position prior) instead; not exercised on ECtHR, where every question is asked of every case.
- *Metrics.* Pooled macro-F1 over the binary questions (headline), per-article slices (the evaluator slices by question
  when a benchmark has at most 32 questions; `qid=ecthr-a-6` and so on), the held-out cell (Articles 8 and 10).

**Queue** (`scripts/queue_transfer.sh`, tracked; the gate hashes it with the tree). Before anything runs on a dataset
the queue checks that the benchmark on disk is the build this record describes (record count and `unseen_q` count;
ECtHR's `summary.json` also records the task and held-out articles), so a rebuild with other arguments cannot silently
change a cell. An interrupted training run resumes from its last checkpoint rather than restarting under the same wandb
id.

1. *(Deferred 2026-10-28 after two of its 27 jobs, nine checkpoints on each of the three then-registered datasets: the grid comes first; the zero-shot baselines will be run only if the trained numbers need them. The two scored, the 270M and 1B compliance decoders on ECtHR, read 0.406 and 0.595 macro-F1 against the 0.471 majority floor with 56% and 65% of outputs parsing.)* Zero-shot: the nine existing compliance checkpoints generate on the benchmark with no training
   (`queue_transfer.sh zeroshot ecthr`), with the compliance answer budget of 512 tokens (they write an
   evidence list and a summary; their compliance outputs have a median of 80 to 110 tokens, so the transfer runs' 64-token
   budget would cut most of them off and score the cap), the prompt cut to 15,872 tokens to fit; scored into each run's
   `eval/zeroshot-<dataset>/` and logged to its wandb run under `zeroshot-<dataset>/bench/`. A pipeline smoke on documents and a baseline for how much the
   compliance training transfers; I expect it to be weak and it is not a result about capacity.
2. The pilot, `pilot-e1-gemma3-1b-pt-ecthr`; then the epoch count is written into the arm configs
   (`make_transfer_configs.py grid ecthr=2`), the configs re-hashed and approved.
3. The nine E4 arms then the seven E5-soft-tokens arms: train, generate on the benchmark (vLLM for decoders, the `hf_encdec`,
   `stitched` and `soft` backends otherwise), check every pair is present exactly once, score, log to the run. (The queue
   ran with the E4 arms only until 2026-10-30, when the E5-soft-tokens arms were added.)

Every stage passes the review gate first. wandb projects: `tt-decoder` (E1 and E1++ arms), `tt-encdec` (E4 and E5-soft-tokens arms);
every run tagged `transfer-grid` and `transfer-ecthr`. The transfer runs keep their compliance counterparts' experiment,
base and size tags (the comparison tags `pair-*`, `grid-*`, `control-*`, `size-control` are dropped), so any results table
built from wandb for the compliance benchmark must exclude `transfer-grid`, and one for this grid must require it.

## wandb runs

Filled in as each run starts.

| Run | Project | Run id | Epochs | Note |
|---|---|---|---|---|
| `pilot-e1-gemma3-1b-pt-ecthr` | tt-decoder | `5e2lwxbi` | 2 (constant lr) | trained 16:14 to 20:48 on 2026-10-28, 4,500 steps, 374M tokens. Validation loss at epoch 0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0 / 1.2 / 1.4 / 1.6 / 1.8 / 2.0: 0.936 / 0.0102 / 0.0103 / 0.0103 / 0.0084 / 0.0103 / 0.0097 / 0.0085 / 0.0091 / 0.0078 / 0.0081. Every epoch-2 reading below the epoch-1 readings but one; minimum at 1.8 to 2.0, flat at the top. **ECtHR arms: 2 epochs.** The metric is diluted by the fixed JSON tokens (the target is 17 tokens of which one is uncertain), so the absolute differences are small; the direction is consistent across five readings. |

## Result

Filled as the sixteen cells score. Cells so far (gold = dataset, macro-F1; floor 0.471):

| arm | macro-F1 | seen articles | held-out articles | wandb |
|---|---|---|---|---|
| `e1-gemma3-1b-pt-ecthr` | 0.840 | 0.876 | 0.655 | tt-decoder `zx3mau2p` |
| `e1-gemma3-270m-ecthr` | 0.827 | 0.867 | 0.556 | tt-decoder `w0aqocq9` |
| `e1-gemma3-4b-pt-ecthr` | 0.852 | 0.880 | 0.741 | tt-decoder `ejh3gmpg` |
| `e4-t5gemma2-270m-270m-ecthr` | 0.831 | 0.871 | 0.587 | tt-encdec `jzz1ne9t` |
| `e4-t5gemma2-1b-1b-ecthr` | 0.836 | 0.870 | 0.618 | tt-encdec `coyg3hy7` |
| `e4mm-270m-enc-1b-dec-ecthr` | 0.825 | 0.868 | 0.555 | tt-encdec `3inlwpqm` |
| `e4mm-270m-enc-4b-dec-ecthr` | 0.811 | 0.864 | 0.473 | tt-encdec `uwd9glew` |
| `e4mm-1b-enc-270m-dec-ecthr` | 0.825 | 0.869 | 0.620 | tt-encdec `kry1agat` |
| `e4mm-4b-enc-270m-dec-ecthr` | 0.801 | 0.862 | 0.465 | tt-encdec `356ct1iv` |

Running notes on the predictions, written as cells land and before the dataset is complete: prediction 4 (each native pair
above its decoder control) holds for the 270m-270m pair (+0.004 overall, +0.031 held-out) and **fails for the 1b-1b pair**
(−0.004 overall, −0.006 seen, −0.037 held-out, with twice the parameters; on compliance the same pair led the same control
by 0.097). One dataset, one seed, on a task that has compressed every difference so far; the equal-size contrast
(prediction 1) is the one the grid turns on and is still open. `e4mm-270m-enc-1b-dec` (0.825 / 0.868 / 0.555) sits level with the 270M decoder alone and below the 270m-270m pair: behind a fixed 270m encoder, growing the decoder from 270m to 1b bought nothing on the held-out articles (−0.032). Every arm with a 270m encoder is at 0.555–0.587 held-out and the arm with a 1b encoder at 0.618, so the encoder-decoder arms' held-out score follows the encoder's size so far, although all of them sit below the plain decoder of the same total size. Where the 1b-1b pair loses to the 1B decoder: on the prompts of 2k tokens or fewer (45% of the benchmark; −0.016), while on every bucket above 2k it is level or ahead (+0.009, +0.001, +0.003), and on held-out Article 8 (0.485 against 0.524). On compliance, whose inputs were long, the same pair led by 0.097. Hypothesis written here on 2026-10-30 (prediction 8 above): the encoder's advantage over a same-size decoder is a long-input advantage, and ECtHR has few long inputs. Refuted the same day; see below.

**Seventh cell, 2026-10-31 (scored 05:28).** `e4mm-270m-enc-4b-dec` (270m encoder, 4B decoder, FSDP, 16 h 20 min of
training over the power loss and resume): macro-F1 0.811 overall, 0.864 seen, **0.473 held-out, at the 0.471 floor**: Article 8
0.468 and Article 10 0.474, with accuracies 0.839 and 0.901 equal to the base rates, so on the held-out articles this arm
answers `not_violated` throughout. The analysts' prediction ("well below the 4B decoder on held-out") holds in the
strongest form: the 4B decoder alone reads 0.741 held-out, the 270M decoder alone 0.556, and putting the 4B decoder behind
a 270m encoder reads the floor on the held-out articles. What that says about where the capacity for an unseen article
sits is left to the analysis (see Verdict); the observation is recorded as it stands. On the seen articles the arm is the lowest of the seven (0.864 against 0.867–0.880), within the ±0.009 resolution.
By length bucket: 0.836 / 0.802 / 0.755 / 0.807 (≤2k / 2–4k / 4–8k / 8–16k), a steeper fall with length than the
decoders (the 1B: 0.84-level across buckets in the earlier reading). The ninth cell (`e4mm-4b-enc-270m-dec`) was pre-registered as
the one to read for the capacity question; its result is above.

**Eighth cell, 2026-10-31 (scored 13:36).** `e4mm-1b-enc-270m-dec` (1b encoder, 270M decoder, DDP, 7 h 53 min of training):
macro-F1 0.825 overall, 0.869 seen, **0.620 held-out**. The analysts' prediction (near the 1b-1b pair, 0.60 to 0.63, and
above the 270m-encoder + 1B-decoder stitch) holds: the native 1b-1b pair reads 0.618, the mirror stitch 0.555. This is the
equal-size, different-split contrast the grid turns on (prediction 1): the same 1.3B parameters score 0.620 held-out with
the capacity in the encoder against 0.555 with it in the decoder, +0.065, well outside the ±0.009 resolution; on the seen
articles the two are level (0.869 against 0.868). By held-out article: Article 10 at 0.739 (accuracy 0.912, both answers
predicted), Article 8 at 0.495 (accuracy 0.837, `violated` predicted 54 times against 123 gold), so the difference between the two arms sits
in Article 10, and Article 8 stays near the floor for every arm so far except the 4B decoder. By length bucket 0.836 / 0.812 / 0.802 / 0.861 (≤2k / 2–4k / 4–8k / 8–16k), the longest bucket highest, unlike
the 270m-encoder + 4B-decoder arm's fall with length. Against the controls: between the 270M decoder alone (0.556) and the
1B decoder alone (0.655) on held-out, as the native pair is. The ninth cell (`e4mm-4b-enc-270m-dec`, FSDP, launched 13:36)
asks whether a 4b encoder behind the same 270M decoder reaches or passes the 4B decoder's 0.741.

**Ninth cell, 2026-11-01 (scored 15:13).** `e4mm-4b-enc-270m-dec` (4b encoder, 270M decoder, FSDP, 25 h of training):
macro-F1 0.801 overall, 0.862 seen, **0.465 held-out, at the 0.474 floor** on both articles (Article 8 0.459 with
`violated` predicted 29 times against 123 gold; Article 10 0.470, accuracy 0.845 equal to the base rate). Prediction 2
("within noise of the 4B decoder and above the 1B decoder") is not met: the 4B decoder alone reads 0.741 on the same cell
while this arm reads the floor. This is the outcome the pre-registration named as disconfirming; what it means for the
capacity question is left to the analysis (see Verdict). The seen score is the lowest of the nine (0.862; the grid spans 0.862 to 0.880,
within the ±0.009 resolution). By length bucket 0.826 / 0.790 / 0.742 / 0.807. The queue stopped itself after this score
(the seven ECtHR E5-soft-tokens arms do not run, D65).

**Audit and analysis of the first six ECtHR cells (2026-10-30; three independent read-only auditors, then three
independent analysts, two Claude Opus and one Claude Fable in each round).** Every number above reproduces from the raw
generations and raw gold (gold re-derived from the LexGLUE files with 0 mismatches; no leakage; prompts token-identical
between training and generation; exports carry the trained weights; training identical across arms). What follows is how
the numbers should be read, recorded here so later cells are read the same way.

*What ECtHR can and cannot resolve.* The seen-article cell (8,000 of 10,000 pairs) has no headroom: six arms from 270M to
4B sit within 0.013 (0.867–0.880), the decoder-size axis moves 0.013 there against 0.274 on compliance, 1B → 4B is +0.004,
and the one-seed paired-bootstrap resolution is about ±0.009. No ordering among the 1B-class arms can be read off it, and
every remaining cell, E5-soft-tokens++ included, is expected to land within ±0.01 of the others there. The held-out cell (2,000
pairs, Articles 8 and 10) resolves about ±0.035 and can show large effects (the decoder axis moved 0.185 there), but it
is two articles, one of which (Article 8) every arm below 4B answers at the floor (phi with gold 0.015–0.087; the 4B 0.445)
through a shared case-level shortcut (saying `violated` on Article 8 mostly where the case already has another violation,
although in gold Article 8 violations fall mostly in cases with no other violation, P = 0.357 against 0.050). Run-to-run
variance is as large as the architectural differences: the pilot 1B decoder (same model and data, constant learning rate)
disagrees with the 1B arm on 173 held-out pairs; the 1b-1b pair disagrees with the 1B arm on 87, and per-article held-out
scores move by 0.08–0.10 between two runs of one architecture. Bootstrap intervals over cases do not contain this.
Consequence: on ECtHR, predictions are read on the held-out cell only, with intervals and per-article numbers, as
"resolved" only beyond about 0.04, and small contrasts (prediction 1 unless the gap is large, prediction 7's 0.03, any
E5-soft-tokens++-vs-E1++ difference at the 1B decoder) are expected to be ties here.

*The 1b-1b pair against the 1B decoder, re-read.* Overall −0.004 [−0.013, +0.006] and seen −0.006 [−0.015, +0.004]:
unresolved. Held-out −0.037 [−0.070, −0.004], pooled; averaging per article instead gives −0.024 [−0.056, +0.008]. The
deficit decomposes into an operating-point difference on Article 10 (the two arms rank cases identically, phi 0.563 each;
the pair predicts `violated` 46 times against the decoder's 94, precision 0.76 / recall 0.45 against 0.54 / 0.66; per-article
F1 0.770 against 0.779) and the Article 8 floor (3 against 10 true positives of 123). On held-out the pair is more accurate
(0.906 against 0.895, McNemar p ≈ 0.03) and lower only on pooled macro-F1 because it predicts half as many positives (65
against 132; every encoder-decoder arm predicts fewer violations per case, 0.905–0.931 against the decoder's 0.983 and
gold's 1.090). So "prediction 4 fails for the 1b-1b pair" is corrected to: unresolved on seen and overall; on held-out the
pair trails on pooled macro-F1 through its operating point on Article 10 and the shared Article 8 floor, within the
range two runs of one architecture span.

*Why the compliance lead did not appear.* On compliance the pair tracked the 4B decoder on every slice (overall 0.795 vs
0.804; unseen questions +0.195 vs +0.193 over the 1B; families, datasets, definition-length terciles and length buckets
alike): a generic "behaves like a bigger model" lead on a reading task with operational answer criteria and 205 training
questions. ECtHR's held-out cell measures applying a legal concept named by its title (the criteria are tautological:
"the Court found a violation of Article 8") after training on eight fixed questions, which takes knowledge of what the
article covers; that grows with total model size (decoders 0.556 / 0.655 / 0.741) and neither 1B half of the pair has it.
The compliance-trained 1B decoder scored higher on Article 8 zero-shot (0.607) than after two epochs on ECtHR (0.524): the
eight-question fine-tune narrowed it, while the 4B learnt the task and kept the concept (0.530 → 0.722). A smaller
contributor: the compliance 270M decoder's lead was inflated by its 21% invalid outputs (the 270m pair's +0.201 shrinks to
+0.144 on both-valid pairs; the 1b pair's +0.097 to +0.090), and ECtHR's 17-token target cannot fail format.

*Prediction 8 (input length) is refuted at its premise.* ECtHR's benchmark median prompt is 2,264 tokens, not 1.8k (that
was the training split), and the compliance clean benchmark's is 2,250 with the same share at ≤2k (44% vs 45%) and no
prompts above 8k; its lead was largest on short unseen prompts (+0.240 at ≤2k). On ECtHR's seen articles the
pair-minus-decoder difference has no length trend (−0.012, +0.008, −0.012, −0.013); the "≤2k deficit" is held-out Article
10 cases. The compliance messy variant's lead does rise with length, but the 4B decoder's lead over the 1B rises
identically: a capacity-by-length trend, not an encoder one. The prediction stays as dated and refuted; the per-length-bucket reading stays as a standing rule
for every dataset.

*Capacity accounting on ECtHR so far.* Among the encoder-decoder arms held-out follows the encoder (270m encoders
0.555–0.587, 1b encoder 0.618) and growing the decoder behind a 270m encoder bought nothing (−0.032); a 1b encoder alone
(0.618) sits between the 270M decoder (0.556) and the 1B decoder (0.655). The discriminating cell is
`e4mm-4b-enc-270m-dec`: Article 8 phi well above 0.1 means concept knowledge is usable from the encoder side; still at the
floor means it has to sit in the decoder and the decoder-only's unseen-concept advantage is real. Analysts' predictions,
written before those cells ran: `e4mm-1b-enc-270m-dec` near the 1b-1b pair on held-out (0.60–0.63), above the 270m-enc +
1b-dec stitch; `e4mm-270m-enc-4b-dec` well below the 4B decoder on held-out.

*Smaller corrections.* The resumed 4B run's wandb curve is a seamless splice (wandb dropped the re-logged steps
2,670–2,700), not a gap; training itself is continuous. 58, not 59, ECtHR training cases exceed the cap. The `cut to cap`
slice is 30 pairs from 3 documents and is not a result. Training and generation cut over-long documents at paragraph
boundaries under budgets that differ by about 46 tokens, so a cut generation prompt can lack one paragraph training kept
(22 of ECtHR's 30 cut pairs; identical for every arm). The compliance and ECtHR
macro-F1 scales are not comparable (pooled 3–4-option questions under teacher gold against ten binary questions under
dataset gold), so lead sizes are not compared across them.

*Options for the decision (2026-10-30 evening):* (1) finish the three remaining E4 stitches, then decide
on the seven ECtHR E5-soft-tokens arms by whether the 4b encoder moves held-out; (2) read ECtHR on held-out with intervals and
per-article numbers, and add forced-choice answer-token log-probability scoring (AUROC) to remove the operating-point
confound (small evaluator and generation change, a review round, retroactive on the existing exports); (3) rebuild the
ECtHR task with four or five held-out articles for a larger, more diverse unseen cell (retrains ECtHR, about four days);
(4) second seeds of the 1B decoder and the 1b-1b pair (about 13 h) to measure the variance the
intervals miss. Cheapest decisive tests for the 1b-1b question: zero-shot ECtHR with the compliance-trained pairs (the
deferred stage, 30–60 min each) and teacher-forced answer log-probabilities on the 2,000 held-out pairs (minutes per arm).

*Decision, 2026-10-30 (late evening).* The nine E1/E4 arms run to completion as pre-registered; the seven E1++/E5-soft-tokens++ arms
follow them in the running queue unless I stop it after the ninth cell (open); options (2) to (4) stay open.
The other two datasets are dropped (the scope change under Setup): on this evidence they would have cost about ten days of
training (an estimate from the ECtHR arm times) to produce one saturated topic benchmark and one at chance. Four analyses fed the decision, each written without
sight of the others: a bootstrap forecast of each benchmark's resolution (SCOTUS resolves about 0.03 accuracy with one
seed, QuALITY 0.03–0.04, both without run-to-run variance, which on ECtHR exceeded the architectural differences); the
shortcut probes quoted under Setup; a literature calibration (the SCOTUS pilot's 0.770 sits at the published fine-tuned
encoder ceiling of 68–78 micro-F1, and QuALITY-only small models published at 24–31%); and a survey of replacements
(ContractNLI the only cheap human-gold candidate, CC BY 4.0, 2,091 benchmark pairs, four hypotheses holdable out, but
compliance-length at a 1.8k-token median; CUAD-as-clause-presence the long reserve). Both pilots scored their benchmarks
from their final checkpoints, past the epoch the curve chose, so neither pilot score is what an arm would have scored.

Interruption log: the machine lost power or hard-reset at about 15:40 on 2026-10-29 during `e1-gemma3-4b-pt-ecthr` (step
~2,690 of 4,500); the queue was relaunched at 15:44 and the run resumed from its step-2665 checkpoint under the same wandb
id, so its curve has a 25-step gap there. No other run was affected. On resume the run went out of memory in backward at step ~2,681 (allocator fragmentation at a 0.5 GB margin; the queue's training stages now set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), and its vLLM generation failed at load until the base model's processor files were copied into the export (the trainer writes model and tokenizer only); 16 h 47 min of training in all, two epochs complete, nothing retrained.

**All nine cells, read from each run's `metrics.json` (macro-F1, clean variant, one seed; `violated` predicted = how many of
the 2,000 held-out pairs the arm answered `violated`, against 200 gold):**

| Arm | Overall | Seen | Held-out | Article 8 | Article 10 | `violated` predicted | ≤2k | 2–4k | 4–8k | 8–16k |
|---|---|---|---|---|---|---|---|---|---|---|
| `e1-gemma3-270m` | 0.827 | 0.867 | 0.556 | 0.487 | 0.639 | 50 | 0.842 | 0.818 | 0.789 | 0.844 |
| `e1-gemma3-1b-pt` | 0.840 | 0.876 | 0.655 | 0.524 | 0.779 | 132 | 0.859 | 0.825 | 0.805 | 0.861 |
| `e1-gemma3-4b-pt` | 0.852 | 0.880 | 0.741 | 0.722 | 0.764 | 241 | 0.867 | 0.846 | 0.833 | 0.823 |
| `e4-t5gemma2-270m-270m` | 0.831 | 0.871 | 0.587 | 0.469 | 0.720 | 70 | 0.842 | 0.822 | 0.804 | 0.859 |
| `e4-t5gemma2-1b-1b` | 0.836 | 0.870 | 0.618 | 0.485 | 0.770 | 65 | 0.843 | 0.834 | 0.806 | 0.864 |
| `e4mm-270m-enc-1b-dec` | 0.825 | 0.868 | 0.555 | 0.530 | 0.591 | 70 | 0.840 | 0.819 | 0.779 | 0.847 |
| `e4mm-1b-enc-270m-dec` | 0.825 | 0.869 | 0.620 | 0.495 | 0.739 | 163 | 0.836 | 0.812 | 0.802 | 0.861 |
| `e4mm-270m-enc-4b-dec` | 0.811 | 0.864 | 0.473 | 0.468 | 0.474 | 64 | 0.836 | 0.802 | 0.755 | 0.807 |
| `e4mm-4b-enc-270m-dec` | 0.801 | 0.862 | 0.465 | 0.459 | 0.470 | 111 | 0.826 | 0.790 | 0.742 | 0.807 |
| majority floor | 0.471 | 0.471 | 0.474 | 0.467 | 0.480 | | 0.470 | 0.473 | 0.472 | 0.461 |

Two readings the table adds to the verdict. Article 8 is near its floor for every arm except the 4B decoder (0.722): the
whole held-out spread below that arm is Article 10. And the length buckets fall from ≤2k to 4–8k for every arm and recover
at 8–16k (480 pairs from the longest documents, mostly `not_violated`), so no arm reads long documents better than short
ones here; the 4B decoder is the only arm whose 8–16k score is below its 4–8k score.

## Verdict

**Pending.** The nine cells are measured and tabulated above; the interpretation waits for a proper analysis (mine,
with the measurements listed under Follow-ups). What is recorded here is what was observed, against the predictions written
before the runs, without a conclusion about where capacity should go.

Observations, 2026-11-01, one seed each, two held-out articles:

- *Seen articles.* All nine arms sit between 0.862 and 0.880 macro-F1 on the eight trained articles, a spread within the
  ±0.009 bootstrap resolution; the headline metric does not rank the designs on this task.
- *Held-out articles.* The decoder-only arms read 0.556 at 270M, 0.655 at 1B, 0.741 at 4B. The native pairs read 0.587
  and 0.618. Among the stitches, the equal-size contrast reads 0.620 (1b encoder, 270M decoder) against 0.555 (270m
  encoder, 1B decoder), as prediction 1 said; both stitches with a 4b component read at the floor, 0.473 and 0.465, which
  is the outcome prediction 2 named as disconfirming. Prediction 3 cannot be read from these cells; prediction 4 holds for
  the 270m pair (+0.031 held-out) and not for the 1b pair (−0.037).
- *Per article and per length.* Article 8 is at or near its floor for every arm except the 4B decoder; every arm's score
  falls from the shortest bucket to 4–8k and recovers at 8–16k.

Candidate explanations that these runs do not separate, to be weighed in the analysis: the affine stitch between spaces
of very different width at 4b; FSDP training (both floor arms trained under FSDP, as did the 4B decoder control, which
read 0.741); two epochs over eight articles in the two largest stitched models (their seen scores are also the lowest two);
the operating point of an arm that answers `not_violated` on nearly every held-out pair, which a forced-choice
log-probability reading would remove; one seed and a two-article held-out cell. The compliance dataset's own finding is
recorded in the main README on its own evidence; the v2 podcast dataset (`experiments/2026-10-31-v2-long-podcast/`) is the
second dataset built for this question.

## Follow-ups

Open after the nine cells (2026-11-01), cheapest first, as inputs to the analysis: forced-choice answer-token log-probability scoring of the 2,000
held-out pairs for every arm, from the existing exports (an evaluator change, one review round); a second seed of
`e4mm-4b-enc-270m-dec` and of `e1-gemma3-1b-pt`; the native T5Gemma 2 4b-4b pair on ECtHR if it fits the cards; a rebuilt
task with four or five held-out articles; the nine compliance checkpoints' zero-shot ECtHR baselines (two of 27 jobs ran).
Housekeeping: the queue script still carries the SCOTUS and QuALITY keys of the dropped datasets; remove them under a
review round before the script is used again.

Deferred from the design, to be taken up only if the grid holds: a second seed on the two equal-size mix-and-match
arms on ECtHR; task B of ECtHR as a second question set over the same cases; refitting the stitches on the transfer text.

**Superseded 2026-10-30: the E5-soft-tokens arms joined the grid itself, seven, run after the E4 arms (see the Setup).** The earlier
plan, kept for the record: four E5-soft-tokens fine-tunes from the compliance adaptation checkpoints, queued to run once the E4 arms
were scored. The question it answers: does marrying an encoder to a decoder-only model through
an adapted interface beat the decoder alone off the compliance task, as it did on it (+0.036 answers at the 1B decoder,
+0.262 with the 4b encoder behind the 270M decoder)? On the compliance benchmark the soft-token stitch into a raw decoder-only model was within noise of the decoder
on answers without the continuation stage (+0.010 with the 1b encoder, −0.004 with the 270m) and lifted answers only
with it (+0.036 for 1b → 1B over E1++), so an E5-soft-tokens arm here needs the stage. Rather than new stages, I would start both
sides of each pair from the compliance stage's exports (`adapt-continuation-corpora-gemma3-{1b-pt,270m}` as the E1++
controls; `adapt-continuation-corpora-e5-soft-tokens-t5gemma2-{1b-enc-gemma3-1b,4b-enc-gemma3-270m}` as the E5-soft-tokens++ arms), which keeps
the comparison matched exactly as it was on compliance and asks a question of its own: does an interface learned on
speech transcripts transfer to legal prose? Four runs: E1++ 1B, E5-soft-tokens++ 1b-enc → 1B, E1++ 270M, E5-soft-tokens++ 4b-enc → 270M. Caveat to carry: a null result cannot separate "the
interface does not transfer across register" from "the interface does not help here"; only a fresh stage on ECtHR's own
documents (9,000 cases, about 20M tokens, enough for a stage of the compliance size) could separate them, at 4 to 10 hours
per stage per side.
