"""E0, the API baseline: a frontier model on the benchmark, writing the same output records as
src/train/generate.py so src/train/evaluate.py scores it like every trained arm. Two backends:

    uv run python -m src.train.api_baseline <split.jsonl> <out_dir> --backend batch [--model claude-sonnet-5]
        [--variants clean,messy] [--limit N] [--batch-size 5000] [--no-cache] [--wandb-project tt-baselines]
    uv run python -m src.train.api_baseline <split.jsonl> <out_dir> --backend cli [--model sonnet] [--account p|w]
        [--workers 16] ...

`batch`: the Message Batches API with the transcript block cached (needs ANTHROPIC_API_KEY, never in the
repository). `cli`: `claude -p --model <name> --output-format json`, the route the labelling pipeline used
(src/synthesis/llm.py), billed to a subscription account; every call carries Claude Code's own context (tens
of thousands of tokens), so the reported cost is the CLI's notional list-price figure and is dominated by that
overhead. Under `cli` the questions of one transcript go to one worker consecutively, so the CLI's automatic
prompt caching can reuse the transcript prefix; `prompt_tokens` on each record is the task prompt counted with
the Qwen3 tokenizer (E1's), so the length buckets line up with E1's, and the API's own input/cache counts are
recorded beside it.

The prompt is src/train/data.task_prompt, the text the trained models were fine-tuned on, sent as one user turn
in two blocks: the instructions and the numbered transcript (cache_control: ephemeral, reused by every question
on the same call, 83 per transcript in the benchmark) and the question with its options. Greedy (temperature 0),
max 512 output tokens, no system prompt, no structured-output tooling: the model's raw text is scored under the
same strict parser as the trained models.

Needs ANTHROPIC_API_KEY in the environment (never in the repository). Resumable: batch ids and the custom-id
map live in <out_dir>/batches.json; re-running polls the outstanding batches and only submits what is missing.
Per-request usage (input, cache write, cache read, output tokens) and list-price cost are recorded on every
output line and summed into <out_dir>/cost.json; the wandb run (project tt-baselines, tags e0 + model) carries
the config and the totals, and evaluate.py logs the benchmark numbers to the same run id.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from src.train.generate import load_pairs

SPLIT_MARK = "\nQUESTION: "  # task_prompt puts the transcript before this and the question after it
MAX_TOKENS = 512
# list prices, USD per million tokens (platform.claude.com/docs/en/about-claude/pricing, 2026-09-17), batch = 50%
PRICES = {
    "claude-sonnet-5": {"input": 2.0, "cache_write": 2.5, "cache_read": 0.2, "output": 10.0},
    "claude-opus-5": {"input": 5.0, "cache_write": 6.25, "cache_read": 0.5, "output": 25.0},
    "claude-fable-5-1": {"input": 10.0, "cache_write": 12.5, "cache_read": 0.25, "output": 50.0},
    "claude-haiku-4-5": {"input": 1.0, "cache_write": 1.25, "cache_read": 0.1, "output": 5.0},
}


def custom_id(id_: str, variant: str) -> str:
    return hashlib.sha1(f"{id_}|{variant}".encode()).hexdigest()[:40]


def request_params(prompt: str, model: str, cache: bool) -> dict:
    """Messages API params for one pair: the transcript block cached, the question block not."""
    if SPLIT_MARK in prompt:
        head, tail = prompt.split(SPLIT_MARK, 1)
        blocks = [{"type": "text", "text": head}, {"type": "text", "text": SPLIT_MARK + tail}]
        if cache:
            blocks[0]["cache_control"] = {"type": "ephemeral"}
    else:
        blocks = [{"type": "text", "text": prompt}]
    return {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0.0,
            "messages": [{"role": "user", "content": blocks}]}


def cost_usd(model: str, usage: dict, batch: bool = True) -> float:
    p = PRICES.get(model.split("@")[0])
    if p is None:  # an unknown model: no price recorded, the wandb run says so
        return float("nan")
    scale = 0.5 if batch else 1.0
    return scale / 1e6 * (usage.get("input_tokens", 0) * p["input"] + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
                          + usage.get("cache_read_input_tokens", 0) * p["cache_read"] + usage.get("output_tokens", 0) * p["output"])


def output_record(id_: str, variant: str, text: str, usage: dict, model: str) -> dict:
    prompt_tokens = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    return {"id": id_, "variant": variant, "text": text, "prompt_tokens": prompt_tokens, "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0), "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
            "cost_usd": cost_usd(model, usage)}


CLI_CWD = Path(os.environ.get("E0_CLI_CWD", "/tmp/e0-empty-cwd"))  # claude -p loads CLAUDE.md and memory from its cwd: run it from an empty directory


def cli_call(prompt: str, name: str, timeout: int = 1800) -> tuple[str, dict, float, str, str]:
    """One `claude -p` call from an empty working directory (so no project instructions join the prompt):
    (raw result text, usage, notional cost, resolved model id, stop reason). A limit or overload refusal is
    waited out as src/synthesis/llm.ask_json does; other failures raise LLMError."""
    import random
    import subprocess

    from src.synthesis.llm import (
        LIMIT_MAX_WAIT,
        LLMError,
        LLMLimit,
        _claude_env,
        _raise_claude,
    )
    CLI_CWD.mkdir(parents=True, exist_ok=True)
    waited, delay = 0.0, 30.0
    while True:
        try:
            proc = subprocess.run(["claude", "-p", "--model", name, "--output-format", "json"],  # noqa: PLW1510
                                  input=prompt, capture_output=True, text=True, timeout=timeout, env={**os.environ, **_claude_env}, cwd=CLI_CWD)
            if proc.returncode != 0:
                _raise_claude((proc.stderr + "\n" + proc.stdout).strip())
            env = json.loads(proc.stdout)
            text = (env.get("result") or "").strip()
            if env.get("is_error"):
                _raise_claude(text or json.dumps(env)[-500:])
            usage = env.get("usage") or {}
            used = env.get("modelUsage") or {}
            resolved = max(used, key=lambda m: used[m].get("inputTokens", 0) + used[m].get("cacheReadInputTokens", 0)) if used else name
            return text, usage, float(env.get("total_cost_usd") or 0.0), resolved, str(env.get("stop_reason") or "")
        except LLMLimit as e:
            if waited >= LIMIT_MAX_WAIT:
                raise LLMError(f"limit not cleared after {waited / 60:.0f} min: {e}") from e
            pause = (e.retry_after + 15) if e.retry_after is not None else delay
            delay = min(delay * 2, 600)
            pause = min(pause, LIMIT_MAX_WAIT - waited) + min(60.0, 0.2 * pause) * random.random()
            print(f"  limit hit, waiting {pause / 60:.1f} min: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(pause)
            waited += pause


def run_cli(pairs: list[tuple[str, str, str]], out_path: Path, model: str, account: str | None, workers: int) -> None:
    """The whole split through `claude -p`, one transcript's questions per worker in sequence, appending records."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoTokenizer

    from src.synthesis.llm import LLMError, set_claude_account
    set_claude_account(account)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for id_, v, p in pairs:
        groups.setdefault((id_.split("::")[0], v), []).append((id_, v, p))
    lock = threading.Lock()
    n_done = [0]
    t0 = time.time()

    def work(items: list[tuple[str, str, str]]) -> None:
        counts = [len(x) for x in tok([p for _, _, p in items], add_special_tokens=False)["input_ids"]]
        for (id_, v, p), n_prompt in zip(items, counts):
            try:
                text, usage, cost, resolved, stop = cli_call(p, model)
                rec = {"id": id_, "variant": v, "text": text, "prompt_tokens": n_prompt, "output_tokens": usage.get("output_tokens", 0),
                       "api_input_tokens": usage.get("input_tokens", 0), "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                       "cache_write_tokens": usage.get("cache_creation_input_tokens", 0), "cost_usd": cost, "model": resolved, "stop_reason": stop}
            except LLMError as e:
                rec = {"id": id_, "variant": v, "text": "", "prompt_tokens": n_prompt, "output_tokens": 0, "error": str(e)[:500]}
            with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_done[0] += 1
                if n_done[0] % 200 == 0:
                    print(f"[{n_done[0]}/{len(pairs)}] {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, groups.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--backend", choices=["batch", "cli"], default="batch")
    ap.add_argument("--model", default=None, help="batch: an API model id (default claude-sonnet-5); cli: a claude -p model name (default sonnet)")
    ap.add_argument("--account", default=None, help="cli: which subscription account claude -p bills, p or w (default: the environment's)")
    ap.add_argument("--workers", type=int, default=16, help="cli: concurrent claude -p calls")
    ap.add_argument("--variants", default="clean,messy")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (a pilot)")
    ap.add_argument("--batch-size", type=int, default=5000, help="requests per batch (API limit 100,000 / 256 MB)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status polls")
    ap.add_argument("--wandb-project", default="tt-baselines")
    args = ap.parse_args()
    if args.model is None:
        args.model = "sonnet" if args.backend == "cli" else "claude-sonnet-5"
    pairs = load_pairs(args.split, args.variants.split(","), args.limit)
    # pairs from the same call sit together, so the cached transcript prefix is reused within a batch or a worker
    pairs.sort(key=lambda p: (p[0].split("::")[0], p[1]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "gen.jsonl"
    if args.backend == "cli":
        done_cli: set[tuple[str, str]] = set()
        if out_path.exists():
            done_cli = {(json.loads(l)["id"], json.loads(l)["variant"]) for l in out_path.open()}
        todo_pairs = [p for p in pairs if (p[0], p[1]) not in done_cli]
        print(f"{len(pairs)} pairs: {len(done_cli)} done, {len(todo_pairs)} to run through claude -p (account {args.account or 'default'})", flush=True)
        if todo_pairs:
            run_cli(todo_pairs, out_path, args.model, args.account, args.workers)
        finish(args, out_path, cache=True, batch=False)
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    import anthropic
    client = anthropic.Anthropic()
    state_path = args.out_dir / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": [], "ids": {}}
    done: set[str] = set()
    if out_path.exists():
        done = {custom_id(json.loads(l)["id"], json.loads(l)["variant"]) for l in out_path.open()}
    by_cid = {custom_id(i, v): (i, v, p) for i, v, p in pairs}
    submitted = {cid for b in state["batches"] for cid in b["custom_ids"]}
    todo = [cid for cid in by_cid if cid not in done and cid not in submitted]
    print(f"{len(pairs)} pairs: {len(done)} done, {len(submitted - done)} in flight, {len(todo)} to submit", flush=True)

    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        reqs = [{"custom_id": cid, "params": request_params(by_cid[cid][2], args.model, not args.no_cache)} for cid in chunk]
        batch = client.messages.batches.create(requests=reqs)
        state["batches"].append({"id": batch.id, "custom_ids": chunk, "created": batch.created_at.isoformat(), "collected": False})
        state["ids"].update({cid: [by_cid[cid][0], by_cid[cid][1]] for cid in chunk})
        state_path.write_text(json.dumps(state))
        print(f"submitted {batch.id}: {len(chunk)} requests", flush=True)

    t0 = time.time()
    with out_path.open("a") as f:
        while any(not b["collected"] for b in state["batches"]):
            for b in state["batches"]:
                if b["collected"]:
                    continue
                info = client.messages.batches.retrieve(b["id"])
                c = info.request_counts
                print(f"  {b['id']}: {info.processing_status} (succeeded {c.succeeded}, errored {c.errored}, "
                      f"expired {c.expired}, processing {c.processing}) {time.time() - t0:.0f}s", flush=True)
                if info.processing_status != "ended":
                    continue
                n_ok = n_err = 0
                for res in client.messages.batches.results(b["id"]):
                    id_, variant = state["ids"][res.custom_id]
                    if res.result.type == "succeeded":
                        msg = res.result.message
                        text = "".join(blk.text for blk in msg.content if blk.type == "text")
                        usage = msg.usage.model_dump()
                        f.write(json.dumps(output_record(id_, variant, text, usage, args.model)) + "\n")
                        n_ok += 1
                    else:  # errored / expired / canceled: an empty output, scored as invalid, and recorded
                        err = res.result.model_dump()
                        f.write(json.dumps({"id": id_, "variant": variant, "text": "", "prompt_tokens": 0, "output_tokens": 0,
                                            "error": json.dumps(err)[:500]}) + "\n")
                        n_err += 1
                f.flush()
                b["collected"] = True
                state_path.write_text(json.dumps(state))
                print(f"collected {b['id']}: {n_ok} ok, {n_err} failed", flush=True)
            if any(not b["collected"] for b in state["batches"]):
                time.sleep(args.poll)

    finish(args, out_path, cache=not args.no_cache, batch=True)


def finish(args, out_path: Path, cache: bool, batch: bool) -> None:
    """Totals to cost.json and a wandb run (project tt-baselines, tags e0 + model) that evaluate.py then logs to."""
    rows = [json.loads(l) for l in out_path.open()]
    models = sorted({r["model"] for r in rows if r.get("model")})
    totals = {"requests": len(rows), "failed": sum("error" in r for r in rows),
              "prompt_tokens": sum(r["prompt_tokens"] for r in rows), "output_tokens": sum(r["output_tokens"] for r in rows),
              "api_input_tokens": sum(r.get("api_input_tokens", 0) for r in rows),
              "cache_read_tokens": sum(r.get("cache_read_tokens", 0) for r in rows),
              "cache_write_tokens": sum(r.get("cache_write_tokens", 0) for r in rows),
              "cost_usd": sum(r.get("cost_usd", 0.0) for r in rows), "model": args.model, "resolved_models": models,
              "backend": args.backend, "cache": cache, "batch": batch}
    (args.out_dir / "cost.json").write_text(json.dumps(totals, indent=1))
    print(json.dumps(totals, indent=1))
    import wandb
    wandb_id_path = args.out_dir / "wandb_id"
    label = models[0] if len(models) == 1 else args.model
    run = wandb.init(project=args.wandb_project, name=f"e0-{label}", tags=["e0", label, args.backend, "cached" if cache else "uncached"],
                     id=wandb_id_path.read_text().strip() if wandb_id_path.exists() else None, resume="allow",
                     config={"experiment": "e0", "model": label, "backend": args.backend, "account": getattr(args, "account", None),
                             "split": str(args.split), "variants": args.variants, "limit": args.limit, "max_tokens": MAX_TOKENS if batch else "cli default",
                             "temperature": 0.0 if batch else "cli default", "cache": cache, "batch": batch, "prompt": "src/train/data.task_prompt"})
    run_id = run.id
    wandb_id_path.write_text(run_id)
    run.log({f"cost/{k}": v for k, v in totals.items() if isinstance(v, (int, float))})
    run.finish()
    print(f"wandb run {run_id}; score with: uv run python -m src.train.evaluate {args.split} {out_path} {args.out_dir}/benchmark "
          f"--wandb-project {args.wandb_project} --wandb-run {run_id}")


if __name__ == "__main__":
    main()
