"""AMI Meeting Corpus loader.

Track P. Licensed CC BY 4.0, verified against the ``LICENCE.txt`` bundled inside
the annotation archive itself, which is the most authoritative statement
available: *"The AMI corpus and its annotations are released under the Creative
Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."*
Attribution is required wherever this data is used.

Only the manual annotations are needed here, not the audio: they are a 22MB zip
containing the orthographic transcription. Audio matters later, for fitting the
ASR channel model, and is deliberately not downloaded by this module.

## Why the parsing is not trivial

AMI is stored in NXT format, which separates words from turn boundaries and
splits both per speaker:

- ``words/{meeting}.{speaker}.words.xml`` is one speaker's token stream, in
  document order, each token carrying a ``nite:id``.
- ``segments/{meeting}.{speaker}.segments.xml`` gives that speaker's turns, each
  pointing at a *range* of word ids via an href like
  ``...words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)``.

So a meeting's turn sequence only exists once every speaker's segments are
resolved against that speaker's words and the result is merged in time order.
That merge is where line indices come from, and line indices are what evidence
labels refer to, so it has to be **deterministic**: segments are ordered by
``(start time, speaker, segment id)``, which is a total order even when two
speakers begin talking at the same timestamp.

## What is kept and what is dropped

Kept verbatim: every ``<w>``, including filled pauses (*um*, *hmm*, *mm-hmm*),
repairs, repetitions and truncated words (``trunc="true"``, e.g. *th-*). These
are the signal this project exists to model and are never cleaned up.

Dropped from the text: ``<vocalsound>``, ``<disfmarker>``, ``<gap>`` and
``<transformerror>``. These are *annotation markup about* the speech rather than
words anyone spoke, and no ASR system emits them, so injecting them would make
the text less like the target distribution rather than more.

Nothing is dropped silently. Each kind is counted and recorded on the
transcript's ``meta``, and the count is taken over the whole word file rather
than over the segments that survive: markup can sit between segments, or inside
a segment that holds nothing else and is therefore dropped, and a per-segment
tally would under-report the source in both cases.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

from tt.data.schema import Role, Track, Transcript, Turn

SOURCE = "ami"
"""Corpus name. Matches the DATASHEET entry."""

ARCHIVE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ARCHIVE_NAME = "ami_public_manual_1.6.2.zip"
ARCHIVE_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
"""Pinned so a re-cut upstream release fails loudly rather than shifting every line index."""

NITE_NS = "http://nite.sourceforge.net/"
_NITE_ID = f"{{{NITE_NS}}}id"
_NITE_CHILD = f"{{{NITE_NS}}}child"

# "ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)", or a single id.
_HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")

_NON_LEXICAL = ("vocalsound", "disfmarker", "gap", "transformerror")
"""Annotation markup, not spoken words. Excluded from text, counted into meta."""


@dataclass(frozen=True)
class _Segment:
    """One speaker's turn, before merging across speakers."""

    start: float
    speaker: str
    segment_id: str
    text: str

    @property
    def sort_key(self) -> tuple[float, str, str]:
        """Total order. Two speakers can share a start time, so the tie-breakers matter."""
        return (self.start, self.speaker, self.segment_id)


def download(cache_dir: Path) -> Path:
    """Fetch the annotation archive into ``cache_dir``, skipping if already valid.

    Idempotent and safe to re-run. A file that is present but whose checksum does
    not match is re-downloaded rather than trusted.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / ARCHIVE_NAME

    if archive.exists() and _sha256(archive) == ARCHIVE_SHA256:
        return archive

    tmp = archive.with_suffix(archive.suffix + ".part")
    with urlopen(ARCHIVE_URL) as response, tmp.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)

    digest = _sha256(tmp)
    if digest != ARCHIVE_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{ARCHIVE_NAME} checksum mismatch: expected {ARCHIVE_SHA256}, got {digest}. "
            "Upstream may have re-cut the release. Do not proceed: word ids drive line "
            "indices, and line indices are what evidence labels refer to."
        )
    tmp.replace(archive)
    return archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def meeting_ids(archive: Path) -> list[str]:
    """Every meeting id in the archive, sorted."""
    with zipfile.ZipFile(archive) as bundle:
        ids = {
            name.split("/")[1].split(".")[0]
            for name in bundle.namelist()
            if name.startswith("words/") and name.endswith(".words.xml")
        }
    return sorted(i for i in ids if i)


def load(
    cache_dir: Path,
    *,
    meetings: list[str] | None = None,
    min_turns: int = 1,
) -> Iterator[Transcript]:
    """Yield one :class:`Transcript` per AMI meeting, in meeting-id order.

    Downloads on first use. ``meetings`` restricts to specific ids, which is what
    quick inspections want; the default is the whole corpus.
    """
    yield from load_archive(download(cache_dir), meetings=meetings, min_turns=min_turns)


def load_archive(
    archive: Path,
    *,
    meetings: list[str] | None = None,
    min_turns: int = 1,
) -> Iterator[Transcript]:
    """Parse an already-downloaded archive.

    Split out from :func:`load` so the parsing, which is where the interesting
    failures live, is testable without touching the network.
    """
    wanted = meeting_ids(archive) if meetings is None else sorted(meetings)

    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        for meeting in wanted:
            transcript = _build_transcript(bundle, available, meeting)
            if transcript is not None and transcript.n_turns >= min_turns:
                yield transcript


def _build_transcript(
    bundle: zipfile.ZipFile, available: set[str], meeting: str
) -> Transcript | None:
    """Merge every speaker's segments for one meeting into a single ordered transcript."""
    segments: list[_Segment] = []
    speakers: set[str] = set()
    dropped: Counter[str] = Counter()

    for name in sorted(available):
        prefix, _, tail = name.partition("/")
        if prefix != "segments" or not tail.startswith(f"{meeting}."):
            continue
        speaker = tail.split(".")[1]
        words_name = f"words/{meeting}.{speaker}.words.xml"
        if words_name not in available:
            # A speaker with turn boundaries but no token stream contributes
            # nothing and must not shift the indices of speakers that do.
            continue
        speakers.add(speaker)
        speaker_segments, speaker_dropped = _parse_speaker(bundle, name, words_name, speaker)
        segments.extend(speaker_segments)
        dropped.update(speaker_dropped)

    if not segments:
        return None

    segments.sort(key=lambda s: s.sort_key)

    turns = [
        Turn(index=i, speaker=segment.speaker, role=Role.UNKNOWN, text=segment.text)
        for i, segment in enumerate(segments)
    ]

    meta = {
        "meeting_id": meeting,
        "n_speakers": str(len(speakers)),
        "annotation_release": ARCHIVE_NAME,
    }
    # Record what was excluded from the text so it is auditable rather than lost.
    meta.update({f"dropped_{kind}": str(count) for kind, count in sorted(dropped.items())})

    return Transcript(
        id=f"{SOURCE}/{meeting}",
        source=SOURCE,
        track=Track.P,
        turns=turns,
        is_asr=False,
        asr_system=None,
        channel_version=None,
        meta=meta,
    )


def _parse_speaker(
    bundle: zipfile.ZipFile, segments_name: str, words_name: str, speaker: str
) -> tuple[list[_Segment], Counter[str]]:
    """Resolve one speaker's segments against their word stream.

    Also returns a count of the non-lexical markup in the whole word file. It is
    counted at file level rather than per segment on purpose: markup can sit
    outside any segment, or inside a segment that turns out to hold nothing else
    and is dropped, and in both cases a per-segment tally would quietly
    under-report what the source actually contained.
    """
    order, elements = _parse_words(bundle, words_name)
    root = ElementTree.fromstring(bundle.read(segments_name))

    dropped = Counter(e.tag for e in elements if e.tag in _NON_LEXICAL)

    out: list[_Segment] = []
    for segment in root.iter("segment"):
        child = segment.find(_NITE_CHILD)
        if child is None:
            continue
        href = child.get("href")
        if href is None:
            continue
        match = _HREF_RE.search(href)
        if match is None:
            continue

        first, last = match.group(1), match.group(2) or match.group(1)
        if first not in order or last not in order:
            continue
        span = elements[order[first] : order[last] + 1]

        text = _render(span)
        if not text:
            # Segments containing only markup (a laugh, an untranscribed gap)
            # carry no words. Emitting them would create empty lines that
            # evidence could point at.
            continue

        out.append(
            _Segment(
                start=_as_float(segment.get("transcriber_start")),
                speaker=speaker,
                segment_id=segment.get(_NITE_ID) or "",
                text=text,
            )
        )
    return out, dropped


def _parse_words(
    bundle: zipfile.ZipFile, words_name: str
) -> tuple[dict[str, int], list[ElementTree.Element]]:
    """Return the word stream in document order, plus an id-to-position index."""
    root = ElementTree.fromstring(bundle.read(words_name))
    elements = list(root)
    order = {
        element.get(_NITE_ID, ""): position
        for position, element in enumerate(elements)
        if element.get(_NITE_ID)
    }
    return order, elements


def _render(span: list[ElementTree.Element]) -> str:
    """Join a word span into text, keeping disfluencies and dropping markup."""
    parts: list[str] = []

    for element in span:
        if element.tag != "w":
            continue
        token = (element.text or "").strip()
        if not token:
            continue
        if element.get("punc") == "true" and parts:
            # Punctuation is its own <w> and belongs to the preceding token.
            parts[-1] += token
        else:
            parts.append(token)

    return " ".join(parts)


def _as_float(value: str | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except ValueError:
        return 0.0
