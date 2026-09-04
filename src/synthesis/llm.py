"""Two generation backends behind one call: `ask_json(prompt, model, schema)`.

Model strings name the backend explicitly:
  claude:<model>   -> `claude -p --model <model> --output-format json` (stdin prompt); cost from the CLI
  ollama:<model>   -> local Ollama /api/chat with the JSON schema enforced by `format`; cost 0

Both return (parsed_json, cost_usd). Prompts put the transcript first and the
question last so that a caching server can reuse the transcript prefix across
questions on the same call.
"""

import json
import random
import re
import subprocess
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
# rate limit (429), usage-cap ("usage limit reached|<unix ts>"), overloaded (529) and similar transient refusals
_LIMIT = re.compile(r"rate.?limit|usage limit|limit reached|too many requests|\b429\b|\b529\b|overloaded|capacity|try again later", re.IGNORECASE)
_RESET_TS = re.compile(r"limit reached\|(\d{9,11})")
LIMIT_MAX_WAIT = 4 * 3600  # give up on one call after this much cumulative backoff


class LLMError(RuntimeError):
    pass


class LLMLimit(LLMError):
    """A limit or overload refusal; `retry_after` is seconds until the reported reset, if any."""

    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


def split_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise LLMError(f"model must be 'claude:<name>' or 'ollama:<name>', got {model!r}")
    backend, name = model.split(":", 1)
    if backend not in ("claude", "ollama"):
        raise LLMError(f"unknown backend {backend!r}")
    return backend, name


def _raise_claude(msg: str) -> None:
    if _LIMIT.search(msg):
        m = _RESET_TS.search(msg)
        raise LLMLimit(msg[-300:], max(0.0, int(m.group(1)) - time.time()) if m else None)
    raise LLMError(msg[-500:])


def _claude(prompt: str, name: str, timeout: int) -> tuple[dict, float]:
    proc = subprocess.run(["claude", "-p", "--model", name, "--output-format", "json"],  # noqa: PLW1510
                          input=prompt, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        _raise_claude((proc.stderr + "\n" + proc.stdout).strip())
    envelope = json.loads(proc.stdout)
    result = (envelope.get("result") or "").strip()
    if envelope.get("is_error"):
        _raise_claude(result or json.dumps(envelope)[-500:])
    try:
        parsed = json.loads(_FENCE.sub("", result))
    except json.JSONDecodeError:
        _raise_claude(result)  # a limit message in place of JSON is waited out; anything else is a plain retry
        raise
    return parsed, float(envelope.get("total_cost_usd") or 0.0)


def _ollama(prompt: str, name: str, schema: dict | None, timeout: int, num_ctx: int = 32768,
            num_predict: int = 2048) -> tuple[dict, float]:
    # num_predict caps a repetition loop (seen: gemma4:12b emitting a 29k-character summary) so a runaway
    # call fails fast on the JSON parse instead of running to the timeout; a label needs a few hundred tokens
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "format": schema or "json", "think": False,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    return json.loads(_FENCE.sub("", content.strip())), 0.0


def ask_json(prompt: str, model: str, schema: dict | None = None, retries: int = 2,
             timeout: int = 1800) -> tuple[dict, float]:
    """One labelling prompt. Malformed output is retried `retries` times immediately; a limit or
    overload refusal (claude backend) is waited out instead: sleep until the reported reset if the
    CLI gives one, otherwise exponential backoff from 30 s to 10 min, for up to LIMIT_MAX_WAIT in
    total, so a run pauses on a limit rather than failing every remaining pair in minutes."""
    backend, name = split_model(model)
    last = ""
    attempts = 0
    waited = 0.0
    delay = 30.0
    while True:
        try:
            return _claude(prompt, name, timeout) if backend == "claude" else _ollama(prompt, name, schema, timeout)
        except LLMLimit as e:
            if waited >= LIMIT_MAX_WAIT:
                raise LLMError(f"limit not cleared after {waited / 60:.0f} min: {e}") from e
            if e.retry_after is not None:
                pause = e.retry_after + 15
            else:
                pause, delay = delay, min(delay * 2, 600)
            pause = min(pause, LIMIT_MAX_WAIT - waited) * (1 + 0.2 * random.random())  # jitter spreads the workers
            print(f"  [{model}] limit hit, waiting {pause / 60:.1f} min: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(pause)
            waited += pause
        except (json.JSONDecodeError, TypeError, KeyError, LLMError, OSError) as e:
            last = f"{type(e).__name__}: {str(e)[:300]}"
            attempts += 1
            if attempts > retries:
                raise LLMError(last) from e
