"""One Ollama server per GPU, started on demand so a local labeller uses every GPU without any manual setup.

`ensure_servers()` looks at the GPUs (`nvidia-smi -L`), and for GPU i checks whether a server answers on
port 11435 + i; any that does not is started as a user process pinned to that GPU (CUDA_VISIBLE_DEVICES,
Vulkan disabled because it would otherwise see the other card too), reading the system service's model
store read-only, logs and pid files under /tmp/ollama-pinned. Servers are left running afterwards (an
idle server holds no GPU memory once its keep-alive lapses) and are reused by the next run. Returns the
server base URLs, which `llm.py` routes calls over, one call per server at a time.

Measured 2026-09-04 with qwen3.8: 47 labels/min on one server, 83 on two.

Overrides: OLLAMA_URLS (use these servers, start nothing), OLLAMA_BIN (default /usr/local/bin/ollama, the
service's binary; an older client binary cannot render qwen3.8's prompt), OLLAMA_MODELS_DIR (default the
system store), OLLAMA_PINNED_DIR (default /tmp/ollama-pinned).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_PORT = 11435
BIN = os.environ.get("OLLAMA_BIN", "/usr/local/bin/ollama")
MODELS = os.environ.get("OLLAMA_MODELS_DIR", "/usr/share/ollama/.ollama/models")
PINNED = Path(os.environ.get("OLLAMA_PINNED_DIR", "/tmp/ollama-pinned"))


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:
            json.load(r)
        return True
    except (OSError, ValueError):
        return False


def start(gpu: int, port: int) -> None:
    PINNED.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}", "CUDA_VISIBLE_DEVICES": str(gpu), "OLLAMA_VULKAN": "0",
           "OLLAMA_MODELS": MODELS, "OLLAMA_KEEP_ALIVE": "30m"}
    env.pop("OLLAMA_URLS", None)
    log = (PINNED / f"gpu{gpu}.log").open("a")
    proc = subprocess.Popen([BIN if Path(BIN).exists() else "ollama", "serve"], env=env, stdout=log, stderr=log,
                            start_new_session=True)  # survives this process; stopped with kill $(cat gpu<i>.pid)
    (PINNED / f"gpu{gpu}.pid").write_text(str(proc.pid))


def ensure_servers(quiet: bool = False) -> list[str]:
    """URLs of one running Ollama server per GPU, starting any that are missing. Falls back to the system
    service when there is no GPU to pin, and honours OLLAMA_URLS when set."""
    if os.environ.get("OLLAMA_URLS"):
        return [u.strip().rstrip("/") for u in os.environ["OLLAMA_URLS"].split(",") if u.strip()]
    n = gpu_count()
    if n == 0:
        return ["http://localhost:11434"]
    urls = []
    started = []
    for gpu in range(n):
        url = f"http://127.0.0.1:{BASE_PORT + gpu}"
        if not alive(url):
            start(gpu, BASE_PORT + gpu)
            started.append(url)
        urls.append(url)
    deadline = time.time() + 60
    for url in started:
        while not alive(url):
            if time.time() > deadline:
                raise RuntimeError(f"Ollama server {url} did not come up; see {PINNED}")
            time.sleep(1)
    if not quiet:
        print(f"ollama: {len(urls)} server(s), one per GPU: {', '.join(urls)}"
              + (f" (started {len(started)})" if started else " (all already running)"), file=sys.stderr, flush=True)
    return urls
