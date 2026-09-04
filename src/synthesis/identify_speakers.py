"""Identify the host among the diarised speakers of a SPoRC episode.

SPoRC records `SPEAKER_NN` diarisation tags and no roles. Speaker roles are always
rendered verbatim from the corpus (D19), so for SPoRC the role has to be established
first: an LLM reads the opening and closing turns of the episode and names the tag
that is the HOST (runs the show: introduces it and the guests, asks the questions,
reads the ads, closes). Only the host is identified; the other speakers keep their
diarisation tags, which keeps them distinguishable without claiming a role for them.

Results are cached in data/labelled_data/speakers/sporc.jsonl (one line per episode:
doc_id, host, confidence, reason, model, cost_usd, timestamp) and reused by
cases.sporc_cases, which identifies on demand for any episode not yet cached.

    uv run python -m src.synthesis.identify_speakers --limit 240            # pre-identify, in builder order
    uv run python -m src.synthesis.identify_speakers --limit 240 --workers 16
"""

import argparse
import concurrent.futures
import datetime as dt
import json

from src.synthesis import llm
from src.synthesis.llm import ask_json, set_claude_account
from src.synthesis.question_bank import OUT_DIR

DEFAULT_MODEL = "claude:sonnet"
CACHE = OUT_DIR / "speakers" / "sporc.jsonl"
HEAD, TAIL = 60, 12  # turns shown to the identifier: hosts open and close the show

SCHEMA = {
    "type": "object",
    "properties": {
        "host": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["host", "confidence", "reason"],
}


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.open():
        rec = json.loads(line)
        out[rec["doc_id"]] = rec
    return out


def prompt(turns: list[tuple[str, str]]) -> str:
    tags = list(dict.fromkeys(t for t, _ in turns))
    shown = turns if len(turns) <= HEAD + TAIL else turns[:HEAD] + [("...", "[middle of the episode omitted]")] + turns[-TAIL:]
    body = "\n".join(f"{t}: {x}" for t, x in shown)
    return f"""Below are the opening and closing turns of a podcast episode, transcribed with automatic speaker diarisation. The speaker tags are {', '.join(tags)}.

Which tag is the HOST: the person who runs the show, introduces the episode and any guests, asks the questions, reads sponsor messages, and closes? If there are co-hosts, name the one who does most of the running. If no tag can be identified as the host, answer null.

TRANSCRIPT
{body}

Respond with a single JSON object and nothing else:
{{"host": "<one of {tags}> or null", "confidence": <0-1>, "reason": "<one sentence>"}}"""


def identify(doc_id: str, turns: list[tuple[str, str]], model: str = DEFAULT_MODEL) -> dict:
    """Identify the host for one episode; returns the cache record (not yet written)."""
    tags = {t for t, _ in turns}
    resp, cost = ask_json(prompt(turns), model, SCHEMA)
    host = resp.get("host")
    if host not in tags:
        host = None
    conf = resp.get("confidence")
    return {
        "doc_id": doc_id, "host": host,
        "confidence": min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else None,
        "reason": str(resp.get("reason", "")).strip()[:300], "model": model, "cost_usd": round(cost, 5),
        "claude_account": llm.CLAUDE_ACCOUNT,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def append(rec: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    from src.synthesis.cases import sporc_episodes

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=240, help="episodes to identify, in builder order (cache is skipped)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--claude-account", "--claude_account", dest="claude_account", choices=["p", "w"], default=None,
                   help="which Claude account claude -p bills: p (personal) or w (work); default: the environment's")
    args = p.parse_args()
    set_claude_account(args.claude_account)

    cache = load_cache()
    todo = []
    for seen, (doc_id, turns, _meta) in enumerate(sporc_episodes(), 1):
        if seen > args.limit:
            break
        if doc_id not in cache:
            todo.append((doc_id, turns))
    print(f"first {args.limit} episodes in builder order: {len(todo)} to identify with {args.model}", flush=True)

    total = 0.0
    hosts = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(identify, d, t, args.model): d for d, t in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            doc_id = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {doc_id}: failed ({str(e)[:100]})", flush=True)
                continue
            append(rec)
            total += rec["cost_usd"]
            hosts += rec["host"] is not None
            print(f"[{i}/{len(todo)}] {doc_id}: host={rec['host']} conf={rec['confidence']} cost=${total:.2f}", flush=True)
    print(f"done: {len(todo)} identified, {hosts} with a host, ${total:.2f} -> {CACHE}")


if __name__ == "__main__":
    main()
