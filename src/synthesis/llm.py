"""Two generation backends behind one call: `ask_json(prompt, model, schema)`.

Model strings name the backend explicitly:
  claude:<model>   -> `claude -p --model <model> --output-format json` (stdin prompt); cost from the CLI
  ollama:<model>   -> local Ollama /api/chat with the JSON schema enforced by `format`; cost 0

Both return (parsed_json, cost_usd). Prompts put the transcript first and the
question last so that a caching server can reuse the transcript prefix across
questions on the same call.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

# Ollama endpoints, one per GPU (ollama_servers.ensure_servers() starts them on demand; OLLAMA_URLS overrides).
# A caller picks one per call (`ollama_url(key)`) so a call's questions share that server's prefix cache.
# Until `set_ollama_urls` is called, the system service is used.
OLLAMA_URLS = [u.strip().rstrip("/") + "/api/chat" for u in
               os.environ.get("OLLAMA_URLS", "http://localhost:11434").split(",") if u.strip()]
OLLAMA_URL = OLLAMA_URLS[0]


def set_ollama_urls(bases: list[str]) -> None:
    global OLLAMA_URL
    OLLAMA_URLS[:] = [b.rstrip("/") + "/api/chat" for b in bases]
    OLLAMA_URL = OLLAMA_URLS[0]


def ollama_url(key: str | int | None = None) -> str:
    """The Ollama endpoint for a routing key (a call id): the same key always maps to the same server."""
    if key is None or len(OLLAMA_URLS) == 1:
        return OLLAMA_URLS[0]
    h = int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16) if isinstance(key, str) else int(key)
    return OLLAMA_URLS[h % len(OLLAMA_URLS)]
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)
# rate limit (429), usage-cap ("usage limit reached|<unix ts>"), overloaded (529) and similar transient refusals
_LIMIT = re.compile(r"rate.?limit|usage limit|limit reached|too many requests|\b429\b|\b529\b|overloaded|capacity|try again later", re.IGNORECASE)
_RESET_TS = re.compile(r"limit reached\|(\d{9,11})")
LIMIT_MAX_WAIT = 4 * 3600  # give up on one call after this much cumulative backoff


# Which Claude account `claude -p` bills: Khalid's shell aliases select an account by CLAUDE_CONFIG_DIR
# (p_claude / w_claude); `set_claude_account("p"|"w")` does the same for every call from this process.
CLAUDE_ACCOUNTS = {"p": "~/.claude-personal-config", "w": "~/.claude-work-config"}
_claude_env: dict[str, str] = {}
CLAUDE_ACCOUNT: str | None = None


def set_claude_account(account: str | None) -> None:
    """Route claude -p calls to account "p" (personal) or "w" (work); None keeps the environment's default."""
    global CLAUDE_ACCOUNT
    _claude_env.clear()
    CLAUDE_ACCOUNT = account
    if account is None:
        return
    if account not in CLAUDE_ACCOUNTS:
        raise LLMError(f"claude account must be one of {sorted(CLAUDE_ACCOUNTS)}, got {account!r}")
    _claude_env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(CLAUDE_ACCOUNTS[account])


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
                          input=prompt, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **_claude_env})
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
            num_predict: int = 2048, url: str | None = None) -> tuple[dict, float]:
    # num_predict caps a repetition loop (seen: gemma4:12b emitting a 29k-character summary) so a runaway
    # call fails fast on the JSON parse instead of running to the timeout; a label needs a few hundred tokens
    body = {"model": name, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "format": schema or "json", "think": False,
            "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}}
    req = urllib.request.Request(url or OLLAMA_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        content = json.loads(r.read())["message"]["content"]
    return json.loads(_FENCE.sub("", content.strip())), 0.0


def ask_json(prompt: str, model: str, schema: dict | None = None, retries: int = 2,
             timeout: int = 1800, route: str | int | None = None) -> tuple[dict, float]:
    """One labelling prompt. `route` (e.g. the call id) picks the Ollama server when several are configured. Malformed output is retried `retries` times immediately; a limit or
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
            if backend == "claude":
                return _claude(prompt, name, timeout)
            return _ollama(prompt, name, schema, timeout, url=ollama_url(route))
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
