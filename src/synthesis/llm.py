"""Thin wrapper over `claude -p` for generation, returning parsed JSON and cost.

Every call is non-interactive (`claude -p --model <model> --output-format json`),
the prompt goes in on stdin, and the model is told to answer with a single JSON
object. Cost in USD comes back from the CLI and is summed by the caller.
"""

import json
import re
import subprocess

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


class LLMError(RuntimeError):
    pass


def ask_json(prompt: str, model: str, retries: int = 2, timeout: int = 300) -> tuple[dict, float]:
    last = ""
    for attempt in range(retries + 1):
        proc = subprocess.run(
            ["claude", "-p", "--model", model, "--output-format", "json"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            last = proc.stderr.strip()[-500:]
            continue
        try:
            envelope = json.loads(proc.stdout)
            text = _FENCE.sub("", (envelope.get("result") or "").strip())
            data = json.loads(text)
            return data, float(envelope.get("total_cost_usd") or 0.0)
        except (json.JSONDecodeError, TypeError) as e:
            last = f"unparseable result ({e}): {proc.stdout[:300]}"
    raise LLMError(last)
