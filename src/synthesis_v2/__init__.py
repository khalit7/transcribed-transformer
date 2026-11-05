"""Labelled data v2: long podcast episodes (SPoRC) with a question bank written for them.

Kept apart from v1 (`src/synthesis`, `data/labelled_data/`) on purpose: v2 has its own episode pool, its own
question bank, its own ledger and splits under `data/labelled_data_v2/`, and its own experiment record, so the two
iterations can be read side by side without one's statistics bleeding into the other's. Generic pieces (the record
schema, the labeller backends, the prompt rendering) are imported from `src/synthesis` rather than copied.
"""

import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "labelled_data_v2"


def numbered(lines: list[str]) -> str:
    """The v2 transcript as every model sees it (owner's decision 2026-11-01, D76): `<n>. <speaker>:<utterance>`, 1-based.
    Records store lines in the shared schema's `<speaker>: <utterance>` form; only the prompt rendering differs from v1's
    `<n>: <speaker>: <utterance>` (`src.synthesis.label.numbered`)."""
    out = []
    for i, ln in enumerate(lines, 1):
        role, _, said = ln.partition(": ")
        out.append(f"{i}. {role}:{said}")
    return "\n".join(out)


def file_sha1(path: Path) -> str:
    """SHA-1 of a file, read in blocks (the v2 split files are several GB)."""
    import hashlib

    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Every parseable line of a JSONL file. A truncated final line (a crash mid-write) is skipped with a warning
    instead of raising, so a resumed run can carry on; `append_jsonl` repairs the file on the next write."""
    import json
    import sys

    if not path.exists():
        return []
    out = []
    lines = path.read_text().split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 or (i == len(lines) - 2 and lines[-1] == ""):
                print(f"warning: {path} ends in a truncated line; skipped", file=sys.stderr)
            else:
                raise
    return out


_APPEND_LOCK = threading.Lock()  # appends come from worker threads; the check-then-write below must not interleave


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record. A truncated final line (a crash mid-write) is cut back to the last newline first, so the
    fragment never stays in the file and two records never fuse."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK, path.open("a+b") as f:
        f.seek(0, 2)
        size = f.tell()
        if size > 0:
            f.seek(-1, 2)
            if f.read(1) != b"\n":  # search back from the end in blocks for the last newline (files reach gigabytes)
                pos, cut = size, 0
                while pos > 0:
                    start = max(0, pos - (1 << 20))
                    f.seek(start)
                    i = f.read(pos - start).rfind(b"\n")
                    if i >= 0:
                        cut = start + i + 1
                        break
                    pos = start
                f.truncate(cut)
        f.seek(0, 2)
        f.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
