---
name: confidentiality-check
description: Scan changes for material that must not appear in this public repository before committing or publishing. Use before any commit touching prose (README, docs, experiment write-ups, comments), before publishing anything externally, and whenever writing about the motivation for this work.
---

# Confidentiality check

This repository is public. The owner works at a company in this domain, and this project exists precisely because that employer's data cannot be used. The value of the repository depends on that boundary being visibly intact.

Run this before committing prose, and before anything is published externally.

## What must not appear

- **Production or customer data**, in any form, including paraphrased, partial, or "illustrative" examples that originated internally
- **Internal model names, architectures, sizes, training recipes, checkpoints or benchmark numbers**
- **Customer or client names**, tenant counts, model counts, deployment scale, or any internal metric
- **Internal findings** — evaluation results, error analyses, or conclusions drawn from internal data
- **Internal prompts, taxonomies, question banks or label definitions**
- **Colleagues' names** in connection with internal work
- Anything whose justification would require internal knowledge to explain

## What is fine

The generic problem statement, stated as a class of problem rather than one company's system: compliance question answering over call transcripts in a regulated domain; long input and short structured output; the answer/evidence/summary triple as a task formulation; publicly documented regulatory frameworks. Public literature, public datasets, and results measured on them.

## The working tree is in scope, not just the diff

**Internal source material must not sit in the repository at all, even untracked.** This is the failure mode this check is least likely to catch by its own design: an untracked file does not appear in a diff, so reading the diff carefully will not find it. `git add -A` stages it, and history is permanent.

So before reading the diff, run `git status --porcelain --untracked-files=all` and account for every untracked file. Anything that arrived from an internal system, or that you cannot explain the provenance of, gets moved outside the repository before anything else happens. `.gitignore` is a backstop for this, never the control.

## How to check

1. **Check the untracked files first**, per the section above.
2. **Read the diff, all of it.** Not a grep. The failure mode here is a sentence that reads as harmless but only makes sense if you know something internal, and pattern matching does not catch that.
3. **Grep as a backstop**, not as the check: employer name, internal system and model names, client names, colleague names.
4. **Apply the reconstruction test** to every specific claim: could a reader use this to infer something about the employer's internal systems that is not already public? Numbers, architectural details and evaluation results are the usual carriers.
5. **Apply the justification test**: if asked "why is this design choice right?", does the honest answer require citing internal experience? If so, either cite public evidence instead or drop the claim.
6. **Check the motivation prose specifically.** README introductions and experiment write-ups are where internal framing leaks, because that is where the real reason for a decision wants to be explained.

## Converting a blocked claim rather than dropping it

A claim that fails the justification test is often a good claim with the wrong warrant. Before dropping it, try restating it as a **hypothesis this project measures on public data**. "Internal measurement shows X" is unpublishable; "X is expected, here is the metric that tests it, here is the number we measured" is both publishable and better science, because it produces a citable public result instead of an assertion no reader can check.

Use this deliberately. It is the difference between a repository that quietly depends on private knowledge and one that stands on its own evidence.

## When something is found

Say what was found and where, and propose the generic rewrite rather than just deleting. Usually the point survives when restated as a class of problem instead of a specific system.

If already committed, flag it immediately and note that git history is public too, so removing it in a later commit is not sufficient.

## When uncertain

Stop and ask. A false positive costs a question. A false negative is published permanently.
