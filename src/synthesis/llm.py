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
from pathlib import Path

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
# The CLI's own wording since 2026: "You've hit your 5-hour limit · resets 3:40am (Europe/London)" (also weekly, Opus,
# Sonnet limits), in an envelope with "api_error_status": 429 and "api_error": "usage_limit_reached".
_LIMIT = re.compile(r"rate.?limit|usage limit|session limit|limit reached|hit your [\w -]{0,30}limit|usage_limit_reached|"
                    r"too many requests|\b429\b|\b529\b|overloaded|capacity|try again later", re.IGNORECASE)
_RESET_TS = re.compile(r"limit reached\|(\d{9,11})")
# "resets 10:20pm (Europe/London)", or with a date for a longer (weekly) limit: "resets Nov 5, 3pm", "resets Nov 5 at 3pm"
_RESET_CLOCK = re.compile(r"resets? (?:at )?(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* (\d{1,2}),? (?:at )?)?"
                          r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.IGNORECASE)
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
LIMIT_MAX_WAIT = 6 * 3600  # give up on one call after this much cumulative backoff (session limits reset within 5 h)
# A reset time the CLI reports is waited out in full, even days away (a weekly limit), outside the backoff budget above;
# each such wait is capped at RESET_MAX_WAIT and a call waits for at most RESET_MAX_WAITS of them.
RESET_MAX_WAIT = 8 * 86400
RESET_MAX_WAITS = 4  # long waits only: a reset that has just passed (a 60 s retry) does not count
# What the CLI itself prints in place of a reply when a limit is hit; used for a short non-JSON reply, where the broad
# _LIMIT (which matches 'capacity', '429' and the like) would read a model's own prose as a limit.
_LIMIT_NOTICE = re.compile(r"hit your [\w -]{0,30}limit|usage limit|session limit|limit reached|usage_limit_reached|"
                           r"rate.?limit|too many requests", re.IGNORECASE)


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
    """`cost` is what the failed attempt was billed, when the CLI reported it (0.0 otherwise)."""

    def __init__(self, msg: str, cost: float = 0.0):
        super().__init__(msg)
        self.cost = cost


class LLMLimit(LLMError):
    """A limit or overload refusal; `retry_after` is seconds until the reported reset, if any."""

    def __init__(self, msg: str, retry_after: float | None = None, cost: float = 0.0):
        super().__init__(msg, cost)
        self.retry_after = retry_after


def split_model(model: str) -> tuple[str, str]:
    if ":" not in model:
        raise LLMError(f"model must be 'claude:<name>' or 'ollama:<name>', got {model!r}")
    backend, name = model.split(":", 1)
    if backend not in ("claude", "ollama"):
        raise LLMError(f"unknown backend {backend!r}")
    return backend, name


def _reset_after(msg: str) -> float | None:
    """Seconds until the reset a limit message announces: a unix timestamp ("limit reached|<ts>") or a local
    clock time ("resets 10:20pm"); None when the message gives neither."""
    m = _RESET_TS.search(msg)
    if m:
        return max(0.0, int(m.group(1)) - time.time())
    m = _RESET_CLOCK.search(msg)
    if m:
        mon, day = m.group(1), m.group(2)
        hour, minute, ampm = int(m.group(3)) % 12, int(m.group(4) or 0), m.group(5).lower()
        hour += 12 if ampm == "pm" else 0
        now = time.localtime()
        if mon:  # a dated reset: that day this year, or next year if it has passed
            month = _MONTHS.index(mon.lower()[:3]) + 1
            target = time.mktime((now.tm_year, month, int(day), hour, minute, 0, 0, 0, -1))
            if target > time.time() + 300 * 86400:  # "Dec 31" read on Jan 1 is last year's, already past
                target = time.mktime((now.tm_year - 1, month, int(day), hour, minute, 0, 0, 0, -1))
            if target < time.time():  # a past date is retried soon, never read as next year (a stale or skewed clock)
                return 60.0
            return target - time.time()
        target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1))
        if target < time.time() - 3600:  # earlier today: the same time tomorrow
            target += 86400
        elif target < time.time():  # within the last hour: the reset has just passed (the CLI shows minutes), retry soon
            return 60.0
        return target - time.time()
    return None


def _limit_envelope(envelope: object) -> bool:
    """The CLI's structured limit refusal: an API status of 429 or 529, or an `api_error` naming a limit."""
    if not isinstance(envelope, dict):
        return False
    return envelope.get("api_error_status") in (429, 529) or "limit" in str(envelope.get("api_error") or "").lower()


def _raise_claude(msg: str, cost: float = 0.0) -> None:
    if _LIMIT.search(msg):
        raise LLMLimit(msg[-300:], _reset_after(msg), cost)
    raise LLMError(msg[-500:], cost)


# `claude -p` loads CLAUDE.md, memory and its file tools from the working directory it starts in. A labeller must not see
# this repository (its instructions, its notes, its other labels), so every call starts from an empty directory. Until
# 2026-09-29 the calls started from the repository root (see the DATASHEET's known limitations of the labelled data).
CLAUDE_CWD = Path(os.environ.get("LABEL_CLI_CWD", "/tmp/claude-empty-cwd"))


def claude_command(name: str, system: str | None = None) -> tuple[list[str], dict[str, str]]:
    """The `claude -p` command line and the extra environment for one call. With `system` (the lean route, D82) the
    call replaces Claude Code's own system prompt with `system`, offers no tools, and turns prompt caching off: a
    single-call-per-episode prompt is never read again, so a cache write (billed at twice the input rate) is pure loss.
    Without it the call is the default `claude -p` every earlier run used."""
    cmd = ["claude", "-p", "--model", name, "--output-format", "json"]
    env = dict(_claude_env)
    if system is not None:
        cmd += ["--tools", "", "--system-prompt", system]
        env["DISABLE_PROMPT_CACHING"] = "1"
    return cmd, env


def _claude(prompt: str, name: str, timeout: int, system: str | None = None) -> tuple[dict, float, dict]:
    """Returns (parsed JSON, cost in USD, the CLI's usage block: input/output/cache tokens as reported)."""
    CLAUDE_CWD.mkdir(parents=True, exist_ok=True)
    cmd, extra = claude_command(name, system)
    try:
        proc = subprocess.run(cmd,  # noqa: PLW1510
                              input=prompt, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, **extra}, cwd=CLAUDE_CWD)
    except subprocess.TimeoutExpired as e:  # a timed-out call is a failed call, retried like any other, never a crash
        raise LLMError(f"claude -p timed out after {timeout}s") from e
    try:
        envelope = json.loads(proc.stdout)
    except ValueError:
        envelope = None
    billed = float(envelope.get("total_cost_usd") or 0.0) if isinstance(envelope, dict) else 0.0
    if _limit_envelope(envelope):  # the structured refusal is read first, whatever its wording or exit code
        assert isinstance(envelope, dict)
        msg = str(envelope.get("result") or envelope.get("api_error") or "usage limit")
        raise LLMLimit(msg[-300:], _reset_after(msg), billed)
    if proc.returncode != 0:  # with an envelope, only its message is searched for limit words, never its usage counts
        said = str(envelope.get("result") or "") if isinstance(envelope, dict) else proc.stdout
        _raise_claude((proc.stderr + "\n" + said).strip(), billed)
    if not isinstance(envelope, dict):
        raise LLMError(f"claude -p printed no JSON envelope: {proc.stdout[:200]}")
    result = (envelope.get("result") or "").strip()
    cost = billed
    if envelope.get("is_error"):  # the structured limit check ran above; only the message is searched, never the usage
        if not result:
            raise LLMError(f"claude -p error: {envelope.get('api_error') or envelope.get('subtype') or 'no message'}", cost)
        _raise_claude(result, cost)
    try:
        parsed = json.loads(_FENCE.sub("", result))
    except json.JSONDecodeError:
        if system is not None and "{" in result:  # lean route: an object with stray text around it is still the answer
            try:
                parsed = json.loads(result[result.find("{"):result.rfind("}") + 1])
                return parsed, cost, envelope.get("usage") or {}
            except json.JSONDecodeError:
                pass
        # a short notice in place of JSON may be a limit, waited out; a malformed model reply (long, or an object) is a
        # plain retry: its text (summaries, line numbers such as 429) must never be read as a limit
        if len(result) <= 1000 and "{" not in _FENCE.sub("", result) and _LIMIT_NOTICE.search(result):
            raise LLMLimit(result[-300:], _reset_after(result), cost)
        raise LLMError(f"reply is not valid JSON ({len(result)} characters)", cost) from None
    return parsed, cost, envelope.get("usage") or {}


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
    """One labelling prompt: (parsed JSON, cost of the successful attempt). See `ask_json_full` for the per-call record."""
    parsed, _total, info = ask_json_full(prompt, model, schema, retries, timeout, route)
    return parsed, info["cost_last"]


def ask_json_full(prompt: str, model: str, schema: dict | None = None, retries: int = 2,
                  timeout: int = 1800, route: str | int | None = None,
                  system: str | None = None) -> tuple[dict, float, dict]:
    """One labelling prompt, with the full accounting: (parsed JSON, cost of every attempt summed, info) where info
    holds `attempts`, `cost_last` (the successful attempt), `cost_total` and `usage` (the CLI's token usage for the
    successful attempt, including cache reads, when the backend reports it). `route` (e.g. the call id) picks the Ollama server when several are configured. Malformed output is retried `retries` times immediately; a limit or
    overload refusal (claude backend) is waited out instead: sleep until the reported reset if the
    CLI gives one, otherwise exponential backoff from 30 s to 10 min, for up to LIMIT_MAX_WAIT in
    total, so a run pauses on a limit rather than failing every remaining pair in minutes. `system` (claude backend
    only) selects the lean call: see `claude_command`."""
    backend, name = split_model(model)
    if system is not None and backend != "claude":
        raise LLMError(f"a system prompt is only supported on the claude backend, not {backend!r}")
    last = ""
    attempts = 0
    waited = 0.0
    delay = 30.0
    spent = 0.0
    resets = 0
    while True:
        try:
            if backend == "claude":
                parsed, cost, usage = _claude(prompt, name, timeout, system)
            else:
                parsed, cost = _ollama(prompt, name, schema, timeout, url=ollama_url(route))
                usage = {}
            return parsed, spent + cost, {"attempts": attempts + 1, "cost_last": cost, "cost_total": spent + cost, "usage": usage}
        except LLMLimit as e:
            spent += getattr(e, "cost", 0.0)
            if e.retry_after is not None and (e.retry_after <= 300 or resets < RESET_MAX_WAITS):  # wait until it
                resets += e.retry_after > 300  # only long waits use up the allowance
                pause = min(e.retry_after, RESET_MAX_WAIT) + 15
                pause += min(60.0, 0.2 * pause) * random.random()  # jitter spreads the workers
            else:
                if waited >= LIMIT_MAX_WAIT:
                    raise LLMError(f"limit not cleared after {waited / 60:.0f} min: {e}", spent) from e
                if e.retry_after is not None:
                    pause = e.retry_after + 15
                else:
                    pause, delay = delay, min(delay * 2, 600)
                pause = min(pause, LIMIT_MAX_WAIT - waited) + min(60.0, 0.2 * pause) * random.random()
                waited += pause
            print(f"  [{model}] limit hit, waiting {pause / 60:.1f} min: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(pause)
        except (json.JSONDecodeError, TypeError, KeyError, LLMError, OSError) as e:
            spent += getattr(e, "cost", 0.0)
            last = f"{type(e).__name__}: {str(e)[:300]}"
            attempts += 1
            if attempts > retries:
                raise LLMError(last, spent) from e
