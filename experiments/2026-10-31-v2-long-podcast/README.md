# Labelled data v2: long podcast episodes, and the capacity grid on them

**Status: labelled (264,000 labels) and split (frozen 2026-11-05); configs for the 270M-decoder stage of the grid
written, the grid not yet pre-registered, no arm trained. Everything below the Pilot section was written before the
pilot ran; the pilot's measured results are filled in from the files it writes, never from memory. The sections are in
the order the decisions were made, and later ones supersede earlier ones; "The dataset as used" below states the final
state.**

## The dataset as used (2026-11-05)

- **Episodes:** 6,000 SPoRC podcast episodes (real Whisper ASR with diarisation, as distributed), 1,500 in each of four
  length bands by Gemma 3 tokens of the rendered transcript: `0-4k`, `4-8k`, `8-12k`, `12-16k` (the top band stops at
  14,808 so the longest prompt and its answer fit 16,384). None is a v1 episode or comes from a podcast title v1 used.
- **Rendering:** one corpus turn per line, verbatim, `<n>. <speaker>:<utterance>`, hosts and guests renamed `host_N` /
  `guest_N` by an identifier, other voices keeping their diarisation tag.
- **Questions:** 44, every one asked of every episode in the same words; 32 trained, 12 held out of training as the
  zero-shot cell. No question takes a per-episode argument.
- **Gold:** Claude Opus, one call per episode answering all 44 questions, for train, val and benchmark alike: one gold and
  one circularity statement (every arm is judged against Opus's reading). Opus agreed with itself on 0.914 of answers
  between two runs, the noise floor under every comparison.
- **Split:** 1,150 train / 50 val / 300 benchmark episodes per band (below, "The split").

## Why a second dataset

The compliance benchmark (v1) put the capacity in the encoder. The public transfer datasets I tried could not test
whether that holds when the input is long: ECtHR saturates on its seen articles and its held-out cell is two articles;
SCOTUS is a topic task a bag-of-words solves from its first 500 words; QuALITY is unlearnable at 1B from 2.4k questions
(`experiments/2026-10-28-transfer-grid/`, Setup, scope change). So v2 builds the long-input dataset itself, from real
diarised ASR podcast transcripts (SPoRC), labelled by a stronger labeller than v1's, with a question bank written for
podcasts whose answers depend on the whole episode. v2 is kept apart from v1 in every file (`src/synthesis_v2/`,
`data/labelled_data_v2/`, this record), so the two iterations can be read side by side.

## The dataset

**Pool** (`data/DATASHEET.md`, "Labelled data v2"): 7,000 SPoRC episodes from 6,326 podcasts, none shared with any v1
episode or show, in four length strata by the exact Gemma 3 token count of the rendered transcript: `2-4k`, `4-8k`,
`8-16k` (2,000 each; these fit the 16k training cap) and `16-32k` (1,000, labelled only if a longer-context run is
planned). Filters: 2 to 4 speakers, at least 40 turns, a known duration and at least 90 words per minute (partial
transcripts out), second speaker at least 10% of the words (monologues out), no repetition loops or collapsed lines.
Three strata inside one corpus with one question bank and one labeller is the design point: it isolates the length
effect from the dataset, which comparing ECtHR with compliance could not.

**Rendering** (`src/synthesis_v2/cases.py`): one turn per line, `n: host_1: text`, `n: host_2: text` for the
identified hosts and `n: guest_1: text`, `n: guest_2: text` for the identified guests, each numbered in order of first
appearance, and `n: SPEAKER_NN: text` for any other voice (the pilot ran with every host as `host` and guests as their
tags; changed 2026-10-31, see the labelling method section); consecutive same-speaker fragments merged before numbering (the median pool
episode had 38% of its diarised turns at two words or fewer), bracketed non-speech tokens removed, Whisper repetition
loops collapsed, merged turns over 150 words split at sentence ends. Hosts and guests are identified by Sonnet reading the opening
80 and closing 20 turns and naming every host tag and every guest tag (co-hosted shows are common: SPoRC itself names two or more hosts for
9% of pool episodes and none for 79%); SPoRC's own host label is stored beside the answer as a cross-check where it
exists (it agreed with v1's Sonnet identification on 306 of 349 episodes).

**Gold**: Claude Opus labels train, val and benchmark alike, through the `claude -p` route, so
v2 has one gold and one circularity statement: every arm is judged against Opus's reading of the episode. v1 had a
split gold (local labellers for training, Opus only on adjudicated benchmark pairs).

**Question bank** (`src/synthesis_v2/question_bank.py`, family `podcast`): 37 candidate questions entering the pilot,
of which 10 are proposed as the held-out (zero-shot) cell; my target after the pilot is 30 trained and 10 held
out. Every question pairs two places in the episode (announced against covered, asked against answered, raised against
resolved, disclosed against acknowledged, early against later) or asks a global negative, so the opening or closing
300 words cannot settle it; no question refers to a line range; every term of degree is defined once in a shared
definition block that opens every description; the "never happens" option is always `none`, defined positively;
13 questions are parametrised templates whose argument (a topic phrase paraphrased from the episode's description or
drawn from its body, a proper-noun name, or two of either) is generated per episode and stored on the record. The bank
was written by three independent analyses of the pool and the literature and rewritten by three independent clarity
audits in each of two rounds; the reasoning is in the module docstring and the notes (D66 to D68).

## Pilot

**Purpose.** Replace the bank's estimated answer distributions with measured ones, measure labeller agreement by
stratum before any production label is bought, measure the real cost per call on the `claude -p` route, and decide the
final bank.

**Design.** 120 episodes (`data/labelled_data_v2/pilot_ids.txt`): 40 per in-cap stratum, one per podcast, six per
category. Host identification (Sonnet) and template arguments (Sonnet for the paraphrased topic phrases, extraction
for the names) on those 120; a 5-episode smoke comparing one question per call with seven questions per call under
Sonnet, cache reads and cost read from the CLI's usage block; then one **Opus pass** (the gold) and one **Sonnet pass**
over all 120 episodes and all 37 candidates, seven questions per call, grouped and ordered by a seed.

**Pre-registered rules** (`src/synthesis_v2/pilot_report.py`, D67):

- A question is measurable only when every in-cap stratum holds at least 20 pairs whose question text matches between
  the passes.
- A question is **cut** if Opus-against-Sonnet Cohen's kappa is under 0.40 in any stratum, or Opus answers `none` on
  over 75% of episodes, or one answer on over 90%. Declared rare-event detectors (`clm-01` fact corrected, `dis-02`
  pushback on the host) are exempt from the two skew rules and use raw agreement under 0.80 instead of kappa.
- The **project stops** if the bank-mean kappa in the 8–16k stratum is under 0.40, or more than 0.20 below the 2–4k
  stratum's: labeller consistency would then be a function of length, the confound the dataset exists to measure.
- I spot-check about 100 pilot pairs, weighted to the 8–16k stratum and the held-out lineages, against a
  shared blind spot between the two models.
- Evidence position (share of cited lines in the last third) and cost per episode are reported per question and
  stratum; the shortcut probes (bag-of-words on the first 500 words, question-and-options-only baseline,
  argument-only baseline for templates) need more episodes than 120 and run on the first 1,000 production episodes as
  the gate before any arm trains. (Superseded: the probes as run are the ones in "The grid" below, on the frozen split.)

**Cost before measuring** (estimates, to be replaced): a bare `claude -p` call costs about $0.05 before content on
this route (measured once, 2026-10-30); the pilot is about 120 identification calls, 500 to 700 argument calls, and
1,440 labelling calls over the two passes, so some tens of dollars of overhead plus content.

**Results (measured 2026-10-31, 00:25; `data/labelled_data_v2/pilot/report.md` has the per-question table).**

Run: 117 of the 120 pilot episodes (3 had no identifiable host); 37 candidate questions, 13 of them templates whose
argument existed for 94 to 115 episodes; Opus 4,139 labels in 663 calls, Sonnet 4,139 labels in 664 calls (one call
failed to parse and was refilled on resume), seven questions per call, the same grouping and order for both labellers.
Cost as the CLI reports it: Opus $112.86 ($0.965 per episode, $0.170 per call, $0.027 per label), Sonnet $55.31 ($0.473
per episode); every call showed cache reads (the harness prefix), so content tokens were mostly cache creation. Host
identification cost $6.59 and the arguments $16.51, both Sonnet.

Stop rule: **not triggered**. Bank-mean Opus-against-Sonnet kappa by stratum 0.678 (2–4k), 0.709 (4–8k), 0.671 (8–16k):
labeller consistency does not fall with length on this bank. Over the 23 surviving questions the means are 0.762 /
0.761 / 0.687.

| | Count | Questions |
|---|---|---|
| Survivors | 23 | adv-01, adv-02, adv-03 (HO), adv-04, adv-05, arc-05, clm-01, clm-03, clm-04, cmt-02 (HO), dis-02, ent-01, ent-02, ent-03, ord-02 (HO), qa-01, qa-03, seg-01, top-03 (HO), top-04 (HO), top-05, top-06 (HO), top-07 |
| Cut by agreement (kappa under 0.40 in a stratum) | 7 | arc-01 (also 77% `none`), arc-03 (kappa 0.00 at 8–16k; Opus answered `specific` on 99 of 117), cmt-01, dis-01, dis-03, dis-04, per-01 |
| Cut by skew only (over 75% `none` or over 90% one answer, agreement fine) | 6 | per-02 (88% `none`, kappa 0.78), per-04 (97%), qa-02 (88%, kappa 0.71), qa-05 (75%, kappa 0.76), top-01 (93% `discussed`: the descriptions were accurate), top-02 (94% `none`: the sibling topic is almost always absent) |
| Unmeasurable | 1 | ord-01 (47 episodes had a topic pair; under 20 pairs in a stratum) |

What the pilot says about the bank. The questions that pair two places in the episode through a *rare trigger* (a
deferral, a disagreement, a disclosure, a question turned on the host, a lapse of memory) fail on podcasts not because
labellers disagree but because the trigger is absent in 75 to 97% of episodes: podcasts of 2 to 4 speakers rarely argue,
rarely defer, rarely disclose. The questions that survive are those with a trigger present in most episodes (advice,
recommendations, predictions, sources, alternatives, a question answered or not, the episode's shape) and every
parametrised template except the two topic-presence ones, whose argument generators produced near-constant answers.
The templates are also the most consistent (name first speaker kappa 1.00, topic first speaker 0.96, topic personal
0.93, topic advice 0.87). The `arc-03` callback question collapsed because Opus found a specific callback in nearly every
episode while Sonnet did not: the criterion "a specific thing said earlier" is too easily met in conversation and is not
the long-range test it was meant to be.

Evidence position over the survivors: a mean 0.378 (median 0.36) of cited lines fall in the last third of the episode;
the gold answers `none` on 36% of survivor labels. Empty evidence follows `none` answers (0.0 to 0.58 by question).

Held-out survivors: 6 (adv-03, cmt-02, ord-02, top-03, top-04, top-06), against the target of 10.

**Decision pending my review** of the per-question table and a 100-pair spot check: accept 23 trained plus 6
held out as measured; or author a second small round of template-based questions (the pattern the pilot shows to work)
and pilot them on the same 117 episodes (about $170 for both labellers) to reach the 30/10 target; or relax the 75%
`none` rule for the four skew-only cuts whose agreement is high (per-02, qa-02, qa-05, per-04), which is a change to a
pre-registered rule and would be recorded as one.

**Second pilot (measured 2026-10-31, 12:52; `data/labelled_data_v2/pilot/report_r3.md`).** After the first pilot I removed the
13 cut questions and the unmeasurable one from the bank (they stay listed in the bank file as `REMOVED`, with reasons)
and ran two more candidate rounds, each drafted by an Opus analyst against the first pilot's lessons (common triggers,
two-place relations, templates) and rewritten by three independent clarity audits (two Opus, one Fable, each bounding
trigger rates with regex probes over the pilot transcripts). The 18 unpiloted candidates then went through the same
pilot as before on the 114 pilot episodes the identifier classes as conversations: one Opus and one Sonnet pass, seven
questions per call, the pre-registered rules unchanged.

| | Count | Questions |
|---|---|---|
| Survivors | 17 | adv-06 (HO), adv-08, cos-01 (HO), ent-04 (HO), fam-01, hst-01, med-01 (HO), num-01, num-02, org-01 (HO), per-03 (HO), plc-01 (HO), rep-01 (HO), self-01, top-08, top-09, top-10 |
| Cut by skew | 1 | qa-04 term explained (81% `none`, kappa 0.77) |

Bank-mean Opus-against-Sonnet kappa by stratum 0.729 / 0.770 / 0.741: flat with length again. Cost: Opus $61.99 (2,009
labels, $0.54 per episode), Sonnet $31.18. Closest calls: adv-06 at 74.6% `none`, one episode under the cut line; num-02
at kappa 0.46 to 0.55 per stratum. Strongest: ent-04 (0.98) and fam-01 (0.97), the two questions with the most mechanical
criteria. **The bank is now 40 questions, 26 trained and 14 held out**, which meets the target of at least 25 and 10, so
the candidate loop stops; six further trained candidates drafted and audited for a second batch are kept as a reserve and
were not piloted.

## Labelling method change: Claude Code subagents (D69, 2026-10-31)

After the pilot I moved the gold labelling from headless `claude -p` prompts to Opus running as Claude Code subagents:
one subagent per batch of four episodes of one stratum, each reading its episode packages (the numbered transcript
exactly as the models see it, the instantiated questions with criteria and the shared definitions) and writing one
`labels.json` per episode, which `src/synthesis_v2/agent_label.py` validates and converts to records. The reasoning: a
labeller with file tools can check every mention of a name or topic instead of trusting one pass of attention, and the
harness already carries the cached prefix the session pays for. The limitation, as in v1's 2026-09-29 note: a subagent
runs inside this repository and is told to read only its episode directories; it cannot be prevented from reading more.

**First sample (measured 2026-10-31, 10:50):** 12 pilot episodes (4 per in-cap stratum), 501 labels over the 43
candidates, three subagents. Against the pilot's prompt-Opus labels on the same 429 shared pairs: agreement 0.925
(kappa 0.90); against prompt-Sonnet 0.860 (kappa 0.81); the two prompt passes agree with each other on 0.876 of those
pairs. By stratum the harness-against-prompt-Opus agreement is 0.964 / 0.917 / 0.897. The harness cites fewer evidence
lines per positive answer (2.83 against 4.22). Two things the subagents reported: Grep was not available to them, so the
verification the method was chosen for did not happen (they read the transcripts in full), and the shared definition
"all hosts together count as one speaker", meant for the two counting questions, made them answer `monologue` for
two-host chats and `none` for disagreements between co-hosts. Both are fixed before production: the subagent's brief
loads the search tool; the counting rule is removed rather than scoped (a rule that makes two people one speaker is wrong
for every question, counting included); and the identifier now names guests as well as hosts, so the rendering tells
every participant apart (`host_1`, `host_2`, `guest_1`, `guest_2`; advert voices and clips keep their tags) and the
conversation structure survives. The identifier also classes each recording (conversation, scripted, other): of
the 120 pilot episodes it found 114 conversations, three radio or audio dramas and three private calls, which leave the
set, and 12 episodes with one person split across two diarisation tags (recorded as `same_person`, not yet merged). The
host-only identification is kept beside the new one (`data/labelled_data_v2/speakers/sporc.hosts_only.2026-10-31.jsonl`),
the template arguments were rebuilt against the new labels, and the pilot labels, produced under the old rendering, are
not re-used as gold. The review file is `data/labelled_data_v2/agent/sample12/review.md`.

**Speaker identification over the pool (2026-11-01).** All 6,000 in-cap episodes went through the identifier: 419 from
complete SPoRC metadata (host and guest labels covering every speaker with at least 2% of the words, my
decision), the rest by Sonnet; 5,999 identified, 5,901 conversations, 74 other, 24 scripted, 7 conversations without a
host, so **5,891 buildable conversations** (1,945 / 1,961 / 1,985 by stratum). 3,511 have one host and 2,423 two or more;
4,182 have a guest; 10.8% carry a same-person group. Where SPoRC has host labels, Sonnet's host set agrees with them on
374 of 877 episodes, so the metadata-sourced records follow a different labeller from the rest and are to be spot-checked
before production. Cost $325.56 through `claude -p`. The strata were then topped up to at least 2,000 buildable
conversations each (120 more episodes drawn with a new seed under the same filters and podcast cap, identified for $5.77):
**2,004 / 2,004 / 2,002 buildable conversations**, 6,010 of 6,120 in-cap episodes.

Every prompt in the pipeline is a template file under `data/labelled_data_v2/prompts/` (README there), loaded through
`src/prompts.py`; the v1 and transfer prompts moved likewise to `data/labelled_data/prompts/`. The renderings are
byte-identical to the strings they replaced.

## Rendering and strata revised: no preprocessing, four strata (2026-11-01)

I decided to render every v2 transcript with no preprocessing at all: each line is the corpus turn exactly as SPoRC
distributes it, `n: <speaker tag>: <utterance>`, with no merging of same-speaker fragments, no removal of bracketed
non-speech tokens, no collapsing of repetition loops and no splitting of long turns. This replaces the rendering rules
described under "The dataset" above (still implemented in `src/synthesis_v2/cases.py` until the code is changed). The
input this project models is diarised ASR as it comes out of the recogniser; fragments, stray tokens and loops are part
of that distribution, and every cleaning step is a choice the deployed model would not get to make.

With it I replaced the three length strata by four, assigned on that same unprocessed rendering in Gemma 3 tokens:
`0-4k`, `4-8k`, `8-12k`, `12-16k`, keeping about 6,000 episodes in all and re-bucketing the existing pool before drawing
anything new. Two measurements prompted it. On the merged rendering the old `4-8k` stratum was a narrow band at 4k
(p10 / p50 / p90 3,427 / 4,076 / 4,649 tokens, 44% under 4k), because the pool draw filled it from the `2-4k`
candidates (every episode at most 3,333 words) and merging shrank them; and one `8-16k` stratum mixed episodes a factor
of two apart. Re-bucketed on the raw rendering, the 6,010 buildable conversations fall 2,004 / 2,004 / 1,217 / 785
(raw `SPEAKER_NN` tags; median 3,228 / 4,715 / 9,835 / 13,564 tokens, median 66 / 135 / 193 / 297 lines). Still open:
whether the speaker tag is the raw diarisation tag or the identified `host_N` / `guest_N` label, and the per-stratum
counts, since the top stratum is short.

Line format, decided the same day. The transcript the arms are trained and tested on is
`<line_number>. <speaker tag>:<utterance>`, the speaker tag being the identified `host_N` or `guest_N` (other voices keep
their diarisation tag), e.g. `12. host_1:yeah so that's the thing`. The labeller sees the same format, so evidence line
numbers mean the same thing everywhere in v2; this departs from the `n: role: text` rendering v1 used. I size the strata
on `<line_number>. <SPEAKER_NN>:<utterance>`, the corpus's own tags, so that which stratum an episode falls in does not
depend on the speaker identifier. On that form the 6,010 buildable conversations fall **2,028 / 1,996 / 1,209 / 777**
in `0-4k` / `4-8k` / `8-12k` / `12-16k` (median 3,224 / 4,713 / 9,836 / 13,524 tokens, 66 / 137 / 193 / 296 lines,
longest 15,445).

### The selection (2026-11-01)

I fixed 1,500 episodes per stratum (`src/synthesis_v2/strata.py`), chosen so each stratum reproduces its band rather than
whatever the pool happened to hold. A seeded sample of 3,000 eligible SPoRC episodes per stratum, measured exactly on the
sizing rendering, gives the population's distribution; each stratum is cut into cells (1,000-token sub-bins crossed with
tokens-per-word quintiles, since fragment density is what the old `4-8k` draw was biased on), the 1,500 are allocated to
cells by population share, and each cell is filled from already identified episodes first. The top stratum stops at
14,808 sizing tokens: 16,384 less the end token, one token of slack, the widest prompt frame (874 tokens) and a 700-token
answer allowance, so no training example is dropped for length unless an answer exceeds that allowance. 2,467 new
episodes were drawn and identified (two draws, seeds 11 and 12); the identifier rejected 31 of them.

| Stratum | Episodes | Reused / new | Podcasts | Sizing tokens min / p10 / median / p90 / max | Population median | Share per 1k sub-bin, % (selected / population) | Training tokens median / max | Lines median | Words median | Tokens per word (selected / population) | Speakers 2 / 3 / 4 | Audio minutes median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0-4k` | 1,500 | 1,282 / 218 | 1,384 | 609 / 1,773 / 3,054 / 3,849 / 3,999 | 3,039 | 1/14/33/52 vs 1/14/33/52 | 2,976 / 3,950 | 65 | 1,877 | 1.58 / 1.58 | 806 / 454 / 239 | 11.3 |
| `4-8k` | 1,500 | 579 / 921 | 1,464 | 4,005 / 4,369 / 6,248 / 7,638 / 7,999 | 6,271 | 19/25/27/29 vs 19/24/27/30 | 6,104 / 7,928 | 106 | 4,053 | 1.46 / 1.46 | 603 / 577 / 320 | 23.9 |
| `8-12k` | 1,500 | 1,197 / 303 | 1,473 | 8,000 / 8,372 / 9,808 / 11,448 / 11,995 | 9,839 | 27/27/24/21 vs 27/27/25/21 | 9,606 / 11,899 | 194 | 6,364 | 1.51 / 1.52 | 434 / 619 / 447 | 36.8 |
| `12-16k` (to 14,808) | 1,500 | 646 / 854 | 1,468 | 12,000 / 12,236 / 13,311 / 14,456 / 14,808 | 13,283 | 39/37/24 vs 39/37/24 | 12,995 / 14,744 | 294 | 8,441 | 1.58 / 1.58 | 294 / 649 / 557 | 47.5 |

6,000 episodes, 3,704 of them reused. In `0-4k` one cell (under 1,000 tokens, fourth density quintile) could not
be filled: the population sample holds a single episode there, from a podcast already at its two-episode cap. I filled
it with a spare from the nearest cell (same length sub-bin, next density quintile; a 962-token episode), marked `fills`
in `selection.jsonl`; the other 5,999 choices are unchanged. The `0-4k` row above was measured before that one
substitution. Every stratum matches its population sample within 35 tokens at
the median, one point per sub-bin and 0.01 tokens per word; the `4-8k` median moved from 4,715 to 6,248. Training
tokens are the transcript in the training rendering (`host_N` / `guest_N` labels), always at most the sizing count.

## No per-episode arguments (2026-11-01)

I removed every question that takes a per-episode argument (a name or topic drawn from the episode and written into the
question, such as "Who first mentions this name in the episode? Name: ..."), with the code and data that produced the
arguments. Every question is now asked of every episode in the same words. The argument machinery (name extraction,
Sonnet topic paraphrases, planted-absent negatives from sibling episodes) was the most fragile part of the pipeline: its
records went stale whenever the rendering or the speaker sets changed, and a planted-absent argument fixes the answer
by construction. The cost is the bank: 14 of the 40 surviving questions were templates (eight trained, six held out),
so the approved bank is **26 questions, 18 trained and 8 held out**, below my target of 30 and 10. Their wording and
argument recipes are kept in `removed_argument_templates.jsonl` beside this record. Argument-free replacements will be
drafted as provisional questions (`questions_provisional.jsonl`) and join the bank only after a pilot; the widest prompt,
and so the 14,808-token ceiling, is unchanged.

## Round four: the bank on the verbatim rendering (2026-11-02)

The earlier pilots ran on merged turns; the bank is now asked of the verbatim corpus lines, where 47% of lines are under
four words. So I re-piloted the 26 approved questions, alongside 26 new argument-free candidates in two rounds (each
drafted by one Opus agent, rewritten by three independent clarity audits, merged under the 874-token prompt budget, and
reviewed before the paid pilot). All three passes used the same 120 episodes, 30 per stratum from the 6,000 selected, one
per podcast, with Opus as gold against Sonnet and the cut rules registered before any result.

| Pass | Questions | Survive | Bank-mean kappa 0-4k / 4-8k / 8-12k / 12-16k |
|---|---|---|---|
| Re-pilot of the approved bank | 26 | 19 | 0.669 / 0.637 / 0.674 / 0.691 |
| Round four candidates | 16 | 15 | 0.796 / 0.766 / 0.720 / 0.721 |
| Round five candidates | 10 | 10 | 0.867 / 0.873 / 0.828 / 0.814 |

The stop rule never fired. The re-pilot is a negative result worth stating: seven questions that passed on merged turns
failed on the verbatim rendering, each by agreement in one stratum. Five need a judgement about intent or a relation
between distant turns (adv-01, arc-05, hst-01, qa-03, and the held-out cmt-02), and the two rare-event detectors (clm-01,
dis-02) lost agreement at 12-16k. Round four lost hyp-01 (kappa 0.29 at 0-4k). The new questions that pass best have a
closed, formal trigger (a name, an abbreviation, a stated age, a named platform, a date) and a mechanical outcome. Two
survivors sit close to the line: per-06 (held out, 12-16k kappa 0.41) and trn-01 (`none` on 73%). Per-question tables are
in `data/labelled_data_v2/pilot/report_r4a.md`, `report_r4b.md` and `report_r5.md`; the pilots cost $323.30 as the CLI
reports it.

The approved bank is **44 questions, 32 trained and 12 held out**, meeting my target of 30 and 10; every held-out question
keeps a trained sibling that tests the same skill over different evidence. A further round of eight drafts was prepared
as a buffer and not piloted, since the target was met.

## Production labelling method: one lean call per episode (2026-11-02)

I replaced the subagent method (above) with the cheapest route that leaves the labels unchanged. The pilot's calls sent
every 7-question batch through Claude Code's full frame, with the transcript and the shared definitions written to a
one-hour cache that was never read back: 84% of the cost was those cache writes. The production route asks all 44
questions of an episode in one `claude -p` call, prints the shared definitions once instead of in every question,
replaces Claude Code's system prompt with one sentence, offers no tools and turns prompt caching off. The model stays
Opus at its default effort, and the questions, rules and transcript rendering are unchanged.

I tested it on 60 pilot episodes (15 per stratum) against the pilot's Opus labels, with a second run of 20 episodes to
measure how much Opus disagrees with itself:

| | 0-4k | 4-8k | 8-12k | 12-16k | All |
|---|---|---|---|---|---|
| Answer agreement, new route vs pilot Opus | 0.942 | 0.926 | 0.900 | 0.894 | 0.916 |
| Answer agreement, new route vs itself (20 episodes) | 0.936 | 0.905 | 0.923 | 0.891 | 0.914 |
| Cost per episode, new route (USD, as the CLI reports it) | 0.230 | 0.273 | 0.299 | 0.320 | 0.281 |
| Cost per episode, pilot route | 1.296 | 1.611 | 1.952 | 2.308 | 1.792 |

The new route moves Opus's answers no more than running Opus twice does, at 6.4 times lower cost; for scale, Sonnet
agreed with Opus at 0.864 on the same items. Cited evidence lines vary more than answers in either route (Jaccard 0.55
against the pilot, 0.72 between two runs of the new route). All 6,000 selected episodes, the 120 pilot episodes among
them, are labelled this way (`scripts/label_v2.sh`, output `data/labelled_data_v2/labelled_data.jsonl`). What I did not
test: a lower thinking effort, which could cut up to about a quarter of the remaining cost but might cost accuracy.

## After the pilot

The approved bank is 44 questions (32 trained, 12 held out; see above) and the labelled set the 6,000 selected episodes
above (1,500 per stratum). Production labelling finished on 2026-11-05: 264,000 labels, none missing, $1,718.72 of
usage on the labels as the CLI reports it ($1,719.59 over all 6,132 calls, failed and retried calls included). The pre-registration of the grid was meant to come before production labelling and did
not; it is written below before any arm trains.

## The split (2026-11-05)

I split by episode into 1,150 train, 50 val and 300 benchmark episodes in every band (4,600 / 200 / 1,200). At 300
episodes per band the 95% half-width of an accuracy difference between two arms is about ±0.01 per band on the trained
questions and about ±0.015 to ±0.02 on the held-out ones (*estimate*: it assumes the two arms disagree on a fifth of
answers). Going to 350 per band would narrow that by about 0.001 to 0.002 at the cost of 200 training episodes, and a
bigger benchmark buys little against Opus's own run-to-run disagreement, so compute goes to seeds before benchmark
size. Val only selects checkpoints.

Val and benchmark episodes are drawn only from episodes connected to nothing a model has trained on or the bank was
written against. Episodes are linked into groups by any of: feed id, normalised show title, episode title with near-equal
duration, a shared full host name, a shared iTunes owner or author name, and shared transcript text (sampled 8-word
passages). Every group of more than one selected episode goes to train, and so does every group touching
(a) the 236 episodes the question bank was drafted, probed and cut against, (b) every SPoRC document the earlier
adaptation stages may have read, replayed exactly from their configs and verified against their cached samples (35,050
documents), or (c) v1's 1,209 labelled SPoRC episodes. Of the 6,000, 575 had been read by an adaptation stage and 2,656
sit in a group with such a document or a v1 episode; all are in train. Without this, about a tenth of the first draft's
benchmark had been read during adaptation, and the same episode or show had reached both train and benchmark under
different feed ids. `src/synthesis_v2/split.py` holds the rules. I checked the frozen split with leak audits written
apart from its own code (other title normalisations, RSS and image URLs, descriptions, another passage hash) and found
no val or benchmark episode connected to a trained-on, design or train show.

The benchmark of each band is the best of 300 seeded draws refined by swaps, matched to the eligible pool on category and
length/density cell, with a floor of min(10, pool count // 2) benchmark episodes for every (question, answer) class.
Against the whole band, its category, cell and per-question answer mixes sit within the spread of a random 300 drawn from
the band. Two leans are by construction and stated rather than corrected:

- It holds no show from a network whose owner name has two or more words and few shows with several episodes: those are
  all connected to something in train. Shows of one-word-owner station networks are not linked, and eight sister shows of
  such networks with different hosts sit in val or benchmark.
- The rare-answer floor over-represents the rarest classes, up to about 2.2 times their share of the eligible pool where the floor binds (a class of one or two
  episodes can show a larger ratio by chance); the
  split report lists each class it inflates by 1.25 times or more. Benchmark class rates are not population rates.

| | Episodes | Label pairs |
|---|---|---|
| train | 4,600 | 147,200 (32 trained questions each) |
| val | 200 | 6,400 |
| benchmark | 1,200 | 52,800: 38,400 trained questions, 14,400 held out |

The held-out questions' labels on train and val episodes (57,600 pairs) are not used. `splits.json` (seed 0) and the
three files are in `data/labelled_data_v2/`; `benchmark_summary.json` records the hashes of the question file and of the
three split files and the `splits.json` they were built from, and training and evaluation refuse files that no longer
match.

## The held-out set after a twin audit (2026-11-05)

Before freezing the split I checked that no held-out question is a variation of a trained one, by reading the questions
and by measuring, over all 264,000 labels, how well each trained question's answer predicts each held-out answer on the
same episode. There was one twin: per-03 ("story answered") is a narrower version of trained
per-05 ("storytellers"). Per-05's answer settles per-03's on 71% of episodes (best-mapping accuracy 0.860 against a 0.624
majority baseline). I moved per-03 to training and held out agr-01 ("agreement stated") in its place, the trained question
least predictable from the rest (the other trained questions add 0.009 over its 0.611 baseline) with the steadiest pilot
agreement of the candidates (kappa 0.85 overall, at least 0.75 in every band). No label changed; still 32 trained and 12
held out.

Two held-out questions share a gate with training: per-06 ("story questioned") answers `none` exactly when trained
per-05 does, and nam-02 ("name taken up") is partly settled by whether any speaker is not a host. The evaluator reports a
`cell=unseen_q_ungated` slice without those gated pairs (per-06 only on episodes with a story, nam-02 only on episodes with
a non-host speaker) beside the full held-out cell. The audit also bounds what the zero-shot cell can claim:
every held-out answer format and many of the shared definitions also occur in trained questions, so it tests new
triggers and relations over definitions and formats the model has seen, not unseen formats.

## The grid (pre-registered 2026-11-05, before any arm trained)

### Question and hypotheses

For a long input and a short structured output, does a separate encoder in front of a decoder help, does it help more
than spending the same parameters on a bigger decoder, and does the answer depend on input length? The compliance
results put the capacity in the encoder; v2 tests that on one corpus, one bank and one labeller across four lengths.

- **H1 (encoder in front of the same decoder).** At every length band, each encoder arm beats its own decoder-only
  control: E5++ arms beat E1++, and E4 arms beat E1. Read first on E5++ against E1++, the clean comparison: the decoder
  is the same Gemma 3 checkpoint, after the same adaptation stage. E4 against E1 also changes the decoder's weights
  (T5Gemma 2's decoder is not Gemma 3's), so it is supporting evidence.
- **H2 (encoder size).** Within a decoder size, a larger encoder scores higher: 4b > 1b > 270m encoders in front of the
  270M decoder.
- **H3 (same total size).** An encoder arm beats the decoder-only model of about its total size: the 1b-encoder +
  270M-decoder arms against E1 1B, the 4b-encoder + 270M-decoder arms against E1 4B.
- **H4 (length).** The encoder arms' advantage over their control grows with length: the 12-16k gap exceeds the 0-4k gap.

### Arms, in run order (one seed, seed 0)

| # | Run | Exp | Encoder | Decoder | Read against |
|---|---|---|---|---|---|
| 1 | `e1-gemma3-270m-v2` | E1 | — | Gemma 3 270M | (control) |
| 2 | `e1pp-gemma3-270m-v2` | E1++ | — | Gemma 3 270M, adapted | (control) |
| 3 | `e1-gemma3-1b-pt-v2` | E1 | — | Gemma 3 1B | (control, H3) |
| 4 | `e1-gemma3-4b-pt-v2` | E1 | — | Gemma 3 4B | (control, H3) |
| 5 | `e4-t5gemma2-270m-270m-v2` | E4 | 270m | T5Gemma 2 270m | #1 |
| 6 | `e5pp-t5gemma2-270m-enc-gemma3-270m-v2` | E5++ | 270m | Gemma 3 270M | #2 |
| 7 | `e4mm-1b-enc-270m-dec-v2` | E4 | 1b | T5Gemma 2 270m | #1, #3 |
| 8 | `e5pp-t5gemma2-1b-enc-gemma3-270m-v2` | E5++ | 1b | Gemma 3 270M | #2, #3 |
| 9 | `e4mm-4b-enc-270m-dec-v2` | E4 | 4b | T5Gemma 2 270m | #1, #4 |
| 10 | `e5pp-t5gemma2-4b-enc-gemma3-270m-v2` | E5++ | 4b | Gemma 3 270M | #2, #4 |
| 11 | `e1pp-gemma3-1b-pt-v2` | E1++ | — | Gemma 3 1B, adapted | (control) |
| 12 | `e4-t5gemma2-1b-1b-v2` | E4 | 1b | T5Gemma 2 1b | #3 |
| 13 | `e4mm-270m-enc-1b-dec-v2` | E4 | 270m | T5Gemma 2 1b | #3 |
| 14 | `e5pp-t5gemma2-270m-enc-gemma3-1b-v2` | E5++ | 270m | Gemma 3 1B | #11 |
| 15 | `e5pp-t5gemma2-1b-enc-gemma3-1b-v2` | E5++ | 1b | Gemma 3 1B | #11 |
| 16 | `e4mm-270m-enc-4b-dec-v2` | E4 | 270m | T5Gemma 2 4b | #4 |

These are every combination that trains on two 32 GB cards. The native 4b-4b, the 4B E1++, and the E4 / E5++ arms that
pair a 4b encoder with the 1B decoder or a 1b encoder with the 4B decoder need 80 GB cards and are not in this grid.
The E1++ / E5++ arms start from the existing adaptation-stage exports (no new adaptation); those read SPoRC text
during adaptation, which is why the split keeps every episode and show they read out of val and benchmark.

### Recipe and budget

Each arm copies its compliance config's recipe and model block unchanged (`configs/v2/`, `scripts/make_v2_configs.py`;
queue `scripts/queue_v2.sh`). Each training episode contributes 8 of its 32 trained questions, a seeded choice fixed
per episode: 36,800 examples, one epoch of 1,150 steps of 32, 318.0M tokens (316.4M prompt, 1.6M answer; Gemma 3
tokenizer, measured), longest example 15,626 tokens, none dropped for length. Val keeps all 6,400 pairs, of which the
trainer reads a fixed 512 for the val loss. Generation renders the training prompt, never cuts a transcript (the longest
benchmark prompt is 15,428 tokens) and allows a 512-token answer (the longest in the data is 211).

**Budget check on the first arm.** Arm 1 runs alone first. If its val loss at the last val point (step 1,150) is more
than 2% below the one before it (step 920), the budget is not enough for the control and I raise it for every arm (16
questions per episode, the same one epoch) before arm 2; otherwise the budget stands. An arm whose own val loss is
still falling by more than 2% over its last interval is reported with that flag, since its gap to a control may be
undertraining.

### Measures and how they are read

- **Headline:** answer macro-F1 per length band, on the trained-question cell and on the held-out cell, each read
  against its control in the same band and against the band's own majority-answer floor
  (`band=<b>/cell=<c>/floor=majority_answer_in_band`; in-sample majorities, as the global floor). Also accuracy and
  evidence F1. The held-out cell is also read as `cell=unseen_q_ungated`.
- **Noise floor:** Opus agreed with itself on 0.914 of answers (0.936 / 0.905 / 0.923 / 0.891 by band), and one seed
  per arm leaves run-to-run variance unmeasured (on ECtHR it exceeded the differences between architectures). So a gap
  under 0.02 macro-F1 in a band is read as no difference; a gap of 0.02 to 0.05 as provisional, to be confirmed by a
  second seed before it enters a conclusion; above 0.05 as a difference.
- **H1, H2, H3** hold in a band if the gaps they predict are at least 0.02 there; each is reported band by band, with
  the number of bands it holds in.
- **H4** holds if an encoder arm's gap over its control at 12-16k exceeds its gap at 0-4k by at least 0.02, and the
  four bands are ordered the same way or within 0.02 of it.
- **What would refute the compliance finding:** E5++ arms within 0.02 of E1++ in every band, or below it.

### Checks run before any arm trains, and their rules

- **Shortcut probes**, on the frozen split (`src/synthesis_v2/probes.py`): for each question, (a) its majority answer;
  (b) a classifier on the speaker structure alone (number of hosts, guests and other speakers, and the band), which
  reads no transcript content; (c) a bag-of-words classifier on the first 500 words of the transcript, a
  partial-reading baseline. Each classifier is trained on the train episodes, scored on the benchmark episodes per band,
  and compared with the same classifier trained on shuffled labels (mean of five shuffles). A question is
  **shortcut-prone** in a band if the speaker-structure probe beats its null by at least 0.15 macro-F1 there and is at
  least as accurate as the majority answer. Shortcut-prone questions stay in the benchmark; every headline number is also
  reported without them, and a hypothesis that holds only with them is not claimed. The opening-words probe is reported
  but excludes nothing.
- **Evidence-position audit** of the gold: where cited lines fall in the episode (first, middle, last third), per band
  and question. If at 12-16k the middle third holds under half the share of cited lines that the first third holds,
  while at 0-4k it does not, I read that as the labeller under-reading the middle of long episodes, and state it beside
  every 12-16k number.

**Revision before any arm trained (2026-11-05).** The rule as first written compared the best probe with the
constant majority answer, by macro-F1. A check of the rule ran it on the frozen split and it flagged 138 of the 176
(question, band) pairs. It could not tell a shortcut from an artefact: a constant answer's macro-F1 is capped near one
over the number of options, so a class-balanced classifier clears a 0.15 margin with little information (an uninformed
guesser cleared it on 17 pairs); and the first 500 words are about a quarter of a 0-4k episode, real content rather
than a shortcut. I replaced the comparison with a shuffled-label null and limited the exclusion to the probe that reads
no transcript content, as above; I had seen the old rule's count but no result of the revised rule. The same check
showed the evidence-position rule does not fire (middle third 0.326 of cited lines at 0-4k, 0.312 at 12-16k; first third
0.369 and 0.396), so that outcome was known before this revision and is unchanged by it. Two limits of the revised rule, stated rather than tuned away once its output was seen: the classifier is fitted
once, so a flag within a few hundredths of the 0.15 margin could be fit noise; and the speaker-structure probe counts
the voices the identifier left as diarisation tags, which grows with length, so a flag can reflect the identifier's
behaviour as much as the question. The model reads the same tags, so a flagged pair is still one it could answer
without reading the content.

Results: TBD.
