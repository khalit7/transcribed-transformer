"""P1a: can a generative model emit valid evidence line indices, and does it degrade with length?

Measures **format validity only**. Whether the cited lines are the *right* lines
is a different question needing labels; this asks the cheaper prior question of
whether the output is even well formed and in range. That is enough to test the
claim in the README that per-line tagging avoids a copy-and-count problem
generative models have.

Length is manipulated directly rather than sampled: each source transcript is
truncated at whole-turn boundaries to a series of token budgets, so the same
meeting appears at several lengths. Sampling naturally-long and naturally-short
transcripts instead would confound length with whatever else differs between
meetings.

Run::

    python -m tt.bench.index_probe --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tt.bench.format_validity import parse_json_object, score_evidence
from tt.data.loaders import ami
from tt.data.schema import RenderStyle, Transcript

EVIDENCE_CAP = 15

QUESTIONS: list[str] = [
    "Did any speaker raise a concern about cost, budget or price?",
    "Did the group agree on a decision or an action to take next?",
    "Did any speaker disagree with, or push back on, something another speaker said?",
]
"""Generic enough to plausibly apply to any meeting.

Judgement quality is not scored, but the questions still have to be answerable,
because a model that concludes the question is unanswerable will correctly return
``[]`` and tell us nothing about whether it can emit indices.
"""

PROMPT = """\
{transcript}

---
The transcript above has {n_lines} numbered lines.

Question: {question}

Reply with a single JSON object and nothing else, in exactly this form:
{{"reasoning": "<one or two sentences>", "evidence": [<line numbers>]}}

Rules for "evidence":
- Line numbers only, as JSON integers. Not strings.
- Each must be between 1 and {n_lines}.
- Ascending order, no duplicates, at most {cap} of them.
- If nothing in the transcript bears on the question, use an empty list: []
"""


@dataclass
class Item:
    """One prompt: a transcript truncated to a length budget, plus a question."""

    item_id: str
    transcript_id: str
    question_index: int
    target_tokens: int
    actual_tokens: int
    n_lines: int
    style: RenderStyle
    prompt: str


@dataclass
class Result:
    """One generation, scored."""

    item_id: str
    transcript_id: str
    target_tokens: int
    actual_tokens: int
    n_lines: int
    style: str
    latency_s: float
    output_tokens: int
    raw_output: str
    emitted_think_block: bool
    json_ok: bool
    evidence_structurally_valid: bool
    evidence_fully_valid: bool
    in_range_fraction: float | None
    n_evidence: int | None
    longest_run: int | None
    coverage_fraction: float | None
    failures: list[str]

    @property
    def fully_valid(self) -> bool:
        return self.json_ok and self.evidence_fully_valid


def longest_consecutive_run(items: list[int]) -> int:
    """Length of the longest run of consecutive integers.

    The diagnostic that distinguishes *selecting* lines from *enumerating* them.
    A model that has lost track of which lines matter tends to emit a contiguous
    block counting upward, which is well-typed and in range and therefore invisible
    to every other gate here.
    """
    if not items:
        return 0
    best = run = 1
    for previous, current in zip(items, items[1:], strict=False):
        run = run + 1 if current == previous + 1 else 1
        best = max(best, run)
    return best


def truncate(transcript: Transcript, max_tokens: int, tokenizer: Any, style: RenderStyle) -> str:
    """Render the longest whole-turn prefix fitting inside ``max_tokens``.

    Whole turns only. Cutting mid-turn would produce a final line that is not a
    real turn, which is exactly the kind of malformed input that makes a model's
    failure uninterpretable.
    """
    lines = transcript.render(style=style).split("\n")
    # Binary search the line count rather than tokenising every prefix.
    lo, hi = 1, len(lines)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        n = len(tokenizer("\n".join(lines[:mid]), add_special_tokens=False)["input_ids"])
        if n <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return "\n".join(lines[:best])


def build_items(
    tokenizer: Any,
    *,
    targets: list[int],
    per_target: int,
    seed: int,
    cache_dir: Path,
    styles: list[RenderStyle],
) -> list[Item]:
    """Build the probe set, spanning the length range at whole-turn boundaries."""
    rng = random.Random(seed)
    transcripts = [t for t in ami.load(cache_dir) if t.n_turns >= 40]
    transcripts.sort(key=lambda t: t.id)

    items: list[Item] = []
    for target in targets:
        chosen = rng.sample(transcripts, k=min(per_target, len(transcripts)))
        for i, transcript in enumerate(chosen):
            style = styles[i % len(styles)]
            text = truncate(transcript, target, tokenizer, style)
            n_lines = len(text.split("\n"))
            question_index = i % len(QUESTIONS)
            prompt = PROMPT.format(
                transcript=text,
                n_lines=n_lines,
                question=QUESTIONS[question_index],
                cap=EVIDENCE_CAP,
            )
            items.append(
                Item(
                    item_id=f"{transcript.id}@{target}#{question_index}",
                    transcript_id=transcript.id,
                    question_index=question_index,
                    target_tokens=target,
                    actual_tokens=len(tokenizer(text, add_special_tokens=False)["input_ids"]),
                    n_lines=n_lines,
                    style=style,
                    prompt=prompt,
                )
            )
    return items


def score(item: Item, raw: str, latency_s: float, output_tokens: int) -> Result:
    """Apply the format-validity gates to one generation."""
    body = raw
    emitted_think = "<think>" in raw
    if emitted_think and "</think>" in raw:
        # Recorded, not repaired. Reported separately so a configuration artefact
        # is never mistaken for an inability to emit indices.
        body = raw.split("</think>", 1)[1]

    obj, json_failures = parse_json_object(body.strip())
    if obj is None:
        return Result(
            item_id=item.item_id,
            transcript_id=item.transcript_id,
            target_tokens=item.target_tokens,
            actual_tokens=item.actual_tokens,
            n_lines=item.n_lines,
            style=item.style,
            latency_s=latency_s,
            output_tokens=output_tokens,
            raw_output=raw[:2000],
            emitted_think_block=emitted_think,
            json_ok=False,
            evidence_structurally_valid=False,
            evidence_fully_valid=False,
            in_range_fraction=None,
            n_evidence=None,
            longest_run=None,
            coverage_fraction=None,
            failures=json_failures,
        )

    failures = list(json_failures)
    if "evidence" not in obj:
        failures.append("evidence_present")
        report = score_evidence(None, item.n_lines, cap=EVIDENCE_CAP)
    else:
        report = score_evidence(obj["evidence"], item.n_lines, cap=EVIDENCE_CAP)
    failures.extend(report.failures)

    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        failures.append("reasoning_nonempty")

    return Result(
        item_id=item.item_id,
        transcript_id=item.transcript_id,
        target_tokens=item.target_tokens,
        actual_tokens=item.actual_tokens,
        n_lines=item.n_lines,
        style=item.style,
        latency_s=latency_s,
        output_tokens=output_tokens,
        raw_output=raw[:2000],
        emitted_think_block=emitted_think,
        json_ok=True,
        evidence_structurally_valid=report.structurally_valid,
        evidence_fully_valid=report.fully_valid and "evidence_present" not in failures,
        in_range_fraction=report.in_range_fraction,
        n_evidence=len(report.parsed) if report.parsed is not None else None,
        longest_run=longest_consecutive_run(report.parsed) if report.parsed is not None else None,
        coverage_fraction=(
            len(report.parsed) / item.n_lines if report.parsed is not None else None
        ),
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/ami"))
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 28000],
    )
    parser.add_argument("--per-target", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/processed/index_probe"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import wandb

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    styles: list[RenderStyle] = ["colon", "bracket", "dotted"]
    items = build_items(
        tokenizer,
        targets=args.targets,
        per_target=args.per_target,
        seed=args.seed,
        cache_dir=args.cache_dir,
        styles=styles,
    )
    print(f"{len(items)} items across targets {args.targets}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()  # type: ignore[no-untyped-call]

    run = wandb.init(
        project="tt-heads",
        job_type="probe",
        name=f"p1a-index-{args.model.split('/')[-1]}",
        tags=["arm-p1a", "track-p", f"base:{args.model}"],
        mode="offline" if args.offline else "online",
        config={
            "model": args.model,
            "targets": args.targets,
            "per_target": args.per_target,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "evidence_cap": EVIDENCE_CAP,
            "questions": QUESTIONS,
            "render_styles": styles,
            "corpus": "ami",
            "decoding": "greedy",
            "n_items": len(items),
        },
    )

    results: list[Result] = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item.prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        generated = out[0][inputs["input_ids"].shape[1] :]
        raw = str(tokenizer.decode(generated, skip_special_tokens=True))

        result = score(item, raw, latency, int(generated.shape[0]))
        results.append(result)
        run.log(
            {
                "item": i,
                "actual_tokens": result.actual_tokens,
                "fully_valid": int(result.fully_valid),
                "json_ok": int(result.json_ok),
                "evidence_structurally_valid": int(result.evidence_structurally_valid),
                "in_range_fraction": result.in_range_fraction,
                "latency_s": result.latency_s,
            }
        )
        if (i + 1) % 10 == 0 or i == len(items) - 1:
            valid = sum(r.fully_valid for r in results)
            print(
                f"  {i + 1}/{len(items)}  fully_valid={valid}/{len(results)}"
                f"  ({latency:.1f}s last)",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_")
    path = args.out / f"{slug}.jsonl"
    with path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")
    print(f"wrote {path}", flush=True)

    _summarise(run, results, args.targets)
    run.finish()


def _summarise(run: Any, results: list[Result], targets: list[int]) -> None:
    """Per-length-bucket validity, which is the whole point of the probe."""
    import wandb

    rows = []
    header = (
        f"\n{'target':>8} {'n':>4} {'fully_valid':>12} {'json_ok':>8} "
        f"{'in_range':>9} {'run':>6} {'cover':>7}"
    )
    print(header)
    for target in targets:
        bucket = [r for r in results if r.target_tokens == target]
        if not bucket:
            continue
        n = len(bucket)
        valid = sum(r.fully_valid for r in bucket) / n
        json_ok = sum(r.json_ok for r in bucket) / n
        ranged = [r.in_range_fraction for r in bucket if r.in_range_fraction is not None]
        mean_range = sum(ranged) / len(ranged) if ranged else float("nan")
        runs = [r.longest_run for r in bucket if r.longest_run is not None]
        mean_run = sum(runs) / len(runs) if runs else float("nan")
        covers = [r.coverage_fraction for r in bucket if r.coverage_fraction is not None]
        mean_cover = sum(covers) / len(covers) if covers else float("nan")
        rows.append([target, n, valid, json_ok, mean_range, mean_run, mean_cover])
        print(
            f"{target:>8} {n:>4} {valid:>11.1%} {json_ok:>7.1%} "
            f"{mean_range:>8.1%} {mean_run:>6.1f} {mean_cover:>6.1%}"
        )
        run.summary[f"valid@{target}"] = valid
        run.summary[f"longest_run@{target}"] = mean_run

    failures: dict[str, int] = {}
    for result in results:
        for failure in set(result.failures):
            failures[failure] = failures.get(failure, 0) + 1
    print("\nfailure modes:", dict(sorted(failures.items(), key=lambda kv: -kv[1])))

    run.summary["fully_valid_overall"] = sum(r.fully_valid for r in results) / len(results)
    run.log(
        {
            "by_length": wandb.Table(
                columns=[
                    "target_tokens",
                    "n",
                    "fully_valid",
                    "json_ok",
                    "mean_in_range",
                    "mean_longest_run",
                    "mean_coverage",
                ],
                data=rows,
            )
        }
    )


if __name__ == "__main__":
    main()
