"""Taskmaster-1 and Taskmaster-2 loader.

Track P. Licensed CC BY 4.0, stated directly in each release's README: *"made
available under the Creative Commons Attribution 4.0 License."* Attribution is
required wherever this data is used.

These are two-person **spoken** dialogues collected by Wizard of Oz, with
crowdsourced workers as the customer and trained call centre operators as the
assistant, then transcribed. That interaction shape is why this corpus is here:
it is the only permissively-licensed source of dyadic service dialogue at scale,
and it is the corpus that stops the Track P side of the licence comparison being
made entirely of meetings and written text.

## The trap this loader exists to avoid

Taskmaster-1 ships two files. ``woz-dialogs.json`` is the 5,507 two-person spoken
dialogues. ``self-dialogs.json`` is 7,708 dialogues **one person typed on both
sides** — written text, not speech, and not what this project models. They live
side by side in the same release and have the same record schema, so nothing
about loading the wrong one would look wrong downstream. It would simply put
typed prose into a speech corpus and quietly weaken every result that depends on
the input being disfluent.

So the self-dialogue file is never listed in :data:`FILES`, and
:func:`load_file` refuses a path that looks like it, rather than trusting the
caller to remember. Taskmaster-2 has no such split; all of it is spoken.

## Roles

Unlike AMI, this corpus supports the advisor/customer distinction directly:
``ASSISTANT`` is the service provider and ``USER`` is the customer. It is the
first corpus here that populates :class:`Role` with anything but ``UNKNOWN``.

## Known defects, handled explicitly

Taskmaster-1 contains 4 dialogues with no utterances at all, which are dropped,
and 55 utterances whose ``speaker`` field is an empty string despite carrying
real text (*"I'm sorry what was that?"*). Those keep their text and are labelled
``UNKNOWN``: dropping them would discard real speech and shift every line number
after them, and line numbers are what evidence labels refer to.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from tt.data.schema import Role, Track, Transcript, Turn

SOURCE = "taskmaster"
"""Corpus name. Matches the DATASHEET entry."""

_BASE = "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/master"

FILES: dict[str, str] = {
    # Taskmaster-1: the Wizard-of-Oz half only. self-dialogs.json is written
    # text and is deliberately absent; see the module docstring.
    "tm1-woz-dialogs.json": f"{_BASE}/TM-1-2019/woz-dialogs.json",
    # Taskmaster-2: entirely spoken, one file per domain.
    "tm2-flights.json": f"{_BASE}/TM-2-2020/data/flights.json",
    "tm2-food-ordering.json": f"{_BASE}/TM-2-2020/data/food-ordering.json",
    "tm2-hotels.json": f"{_BASE}/TM-2-2020/data/hotels.json",
    "tm2-movies.json": f"{_BASE}/TM-2-2020/data/movies.json",
    "tm2-music.json": f"{_BASE}/TM-2-2020/data/music.json",
    "tm2-restaurant-search.json": f"{_BASE}/TM-2-2020/data/restaurant-search.json",
    "tm2-sports.json": f"{_BASE}/TM-2-2020/data/sports.json",
}

CHECKSUMS: dict[str, str] = {
    "tm1-woz-dialogs.json": "cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f",
    "tm2-flights.json": "86b37b5ae25f530fd18ced78800d30c3b54f7b34bb208ecb51842718f04e760b",
    "tm2-food-ordering.json": "0a042e566a816a5d0abebe6f7e8cfd6abaa89729ffc42f433d327df7342b12f8",
    "tm2-hotels.json": "975b0242f1e37ea1ab94ccedd7e0d6ee5831599d5df1f16143e71110d6c6006a",
    "tm2-movies.json": "6f67c9a1f04abc111186e5bcfbe3050be01d0737fd6422901402715bc1f3dd0d",
    "tm2-music.json": "e5db60d6576fa010bef87a70a8b371d293d48cde8524c1d3ed7c3022f079d95d",
    "tm2-restaurant-search.json": (
        "fb9735f89e7ebc7c877f976da4c30391af6a6277991b597c0755564657ff8f47"
    ),
    "tm2-sports.json": "8191531bfa5a8426b1508c396ab9886a19c7c620b443c436ec10d8d4708d0eac",
}
"""Pinned: these are served from a branch head, so upstream can change under us."""

_ROLES: dict[str, Role] = {
    "USER": Role.CUSTOMER,
    "ASSISTANT": Role.ADVISOR,
}

_UNKNOWN_SPEAKER = "UNKNOWN"


def download(cache_dir: Path, *, verify: bool = True) -> list[Path]:
    """Fetch every spoken-dialogue file into ``cache_dir``. Idempotent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, url in FILES.items():
        path = cache_dir / name
        if not (path.exists() and (not verify or _sha256(path) == CHECKSUMS[name])):
            tmp = path.with_suffix(path.suffix + ".part")
            with urlopen(url) as response, tmp.open("wb") as handle:  # noqa: S310
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            digest = _sha256(tmp)
            if verify and digest != CHECKSUMS[name]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{name} checksum mismatch: expected {CHECKSUMS[name]}, got {digest}. "
                    "Upstream is served from a branch head and may have changed. Do not "
                    "proceed: utterance order drives line indices, and line indices are "
                    "what evidence labels refer to."
                )
            tmp.replace(path)
        paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load(cache_dir: Path, *, subsets: list[str] | None = None) -> Iterator[Transcript]:
    """Yield one :class:`Transcript` per spoken dialogue.

    Downloads on first use. ``subsets`` restricts to specific file stems, e.g.
    ``["tm2-flights.json"]``.
    """
    download(cache_dir)
    wanted = sorted(FILES) if subsets is None else sorted(subsets)
    for name in wanted:
        yield from load_file(cache_dir / name)


def load_file(path: Path) -> Iterator[Transcript]:
    """Parse one Taskmaster JSON file.

    Refuses the Taskmaster-1 self-dialogue file. That file has the same record
    schema as the spoken one, so loading it by mistake produces perfectly valid
    ``Transcript`` objects containing typed prose rather than speech, and nothing
    downstream would flag it.
    """
    if "self-dialog" in path.name.lower():
        raise ValueError(
            f"{path.name} is the Taskmaster-1 self-dialogue file: one person typing "
            "both sides, not transcribed speech. It is deliberately excluded from this "
            "corpus. Loading it would put written text into a speech corpus silently."
        )

    subset = path.stem
    with path.open() as handle:
        dialogues = json.load(handle)

    for dialogue in dialogues:
        transcript = _build(dialogue, subset)
        if transcript is not None:
            yield transcript


def _build(dialogue: dict[str, Any], subset: str) -> Transcript | None:
    """Convert one dialogue record, or ``None`` if it carries no usable turns."""
    turns: list[Turn] = []
    n_unknown_speaker = 0

    # Rebuilt from position rather than trusting the corpus 'index' field, so the
    # 0-based contiguity the schema requires holds even if upstream has a gap.
    for utterance in dialogue.get("utterances", []):
        text = (utterance.get("text") or "").strip()
        if not text:
            continue
        raw_speaker = (utterance.get("speaker") or "").strip().upper()
        if raw_speaker in _ROLES:
            speaker, role = raw_speaker, _ROLES[raw_speaker]
        else:
            speaker, role = _UNKNOWN_SPEAKER, Role.UNKNOWN
            n_unknown_speaker += 1
        turns.append(Turn(index=len(turns), speaker=speaker, role=role, text=text))

    if not turns:
        return None

    conversation_id = dialogue.get("conversation_id") or ""
    meta = {
        "subset": subset,
        "release": "tm1" if subset.startswith("tm1") else "tm2",
        "instruction_id": dialogue.get("instruction_id") or "",
    }
    if n_unknown_speaker:
        meta["n_unknown_speaker"] = str(n_unknown_speaker)

    return Transcript(
        id=f"{SOURCE}/{conversation_id}",
        source=SOURCE,
        track=Track.P,
        turns=turns,
        # Human transcription of recorded speech, not ASR output. So this is
        # usable as clean-speech reference, but it cannot stand in for ASR text
        # without the channel model applied.
        is_asr=False,
        asr_system=None,
        channel_version=None,
        meta=meta,
    )
