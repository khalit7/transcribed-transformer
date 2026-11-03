"""Prompt templates as files, loaded with `string.Template`.

Every prompt a model sees lives beside the data it produces: `data/labelled_data/prompts/` for the v1 pipeline and the
transfer task, `data/labelled_data_v2/prompts/` for v2 (each folder has a README naming the files and their callers). The
code holds no prompt text. Placeholders are `$name`; a template file ends with one newline, which `text` removes, so a
rendered prompt is byte-for-byte what the string in the code used to be. The files are tracked in git and hashed by the
review gate (`.claude/skills/pre-run-review/scripts/gate.py`), since a changed prompt changes a run as much as code does.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "labelled_data" / "prompts"
V2 = ROOT / "data" / "labelled_data_v2" / "prompts"


@cache
def text(path: Path) -> str:
    """The template file's text without its single trailing newline."""
    return path.read_text(encoding="utf-8").removesuffix("\n")


@cache
def template(path: Path) -> Template:
    return Template(text(path))


def render(path: Path, /, **fields: object) -> str:
    """The template with every placeholder filled; a missing placeholder raises KeyError."""
    return template(path).substitute({k: str(v) for k, v in fields.items()})


def digest(*paths: Path) -> str:
    """A short hash of the given template files, for cache keys that must change when a prompt does."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(text(p).encode("utf-8"))  # the cached text, so the key and the rendered content come from one snapshot
    return h.hexdigest()[:12]
