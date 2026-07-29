"""AMI Meeting Corpus loader.

Track P. Licensed CC BY 4.0, verified against the ``LICENCE.txt`` bundled inside
the annotation archive itself, which is the most authoritative statement
available: *"The AMI corpus and its annotations are released under the Creative
Commons Attribution 4.0 International Public License agreement (CC BY 4.0)."*
Attribution is required wherever this data is used.

## Two transcript layers, and why the default is ASR

AMI ships the same speech transcribed twice, in two archives:

- ``manual`` (22MB) — human verbatim orthographic transcription.
- ``asr`` (68MB) — real recogniser output from ``ASR_AS_CTM_v1.0_feb07``, the
  AMI-ASR system circa February 2007.

**The default is ``asr``.** Under the data-tier rule in ``CLAUDE.md`` the human
layer is tier 3: it is a faithful record of speech, but a transcriber who quietly
repaired a false start deleted exactly the signal this project models. Reaching
for the manual layer because it looks cleaner inverts the point of the project.

The human layer is still needed, as the **reference side for fitting the ASR
channel model**: the same utterances transcribed both ways is what makes the
substitution, deletion and insertion distributions estimable at all. That pairing
is the single most valuable thing about this corpus.

Two consequences worth carrying. The 2007 recogniser has a far higher word error
rate than a modern one, so a channel fitted on it without severity calibration
will over-noise. And ASR output has **no punctuation and no annotated
disfluencies**, so the two layers do not tokenise to the same length; comparing
token counts across variants is meaningless.

Audio is not downloaded by this module and is not needed: the ASR side already
exists, which removes what would otherwise be a ~30GB download and a long
transcription run.

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
from typing import Literal
from urllib.request import urlopen
from xml.etree import ElementTree

from tt.data.schema import Role, Track, Transcript, Turn

SOURCE = "ami"
"""Corpus name. Matches the DATASHEET entry."""

Variant = Literal["asr", "manual"]
"""Which transcript layer to read. ``asr`` is the default and the tier 1 one."""

DEFAULT_VARIANT: Variant = "asr"
"""Tier 1 by default.

Reading ``manual`` because it looks cleaner is the exact mistake the data-tier
rule in CLAUDE.md exists to prevent: a transcriber who repaired a false start
removed the signal this project models.
"""

NITE_NS = "http://nite.sourceforge.net/"
_NITE_ID = f"{{{NITE_NS}}}id"
_NITE_CHILD = f"{{{NITE_NS}}}child"

# AMI uses two href formats, and both appear inside the ASR archive:
#
#   range form, one child per segment:
#     "ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"
#   bare form, several children per segment, one id each:
#     "ES2004c.A.words.xml#ES2004c.A.sil2289"
#
# Handling only the first silently drops every meeting written in the second,
# which is 43 of 169 in the ASR archive. Nothing errors; the meetings simply
# vanish, which is why the loader cross-checks its own counts.
_HREF_RANGE_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
_HREF_BARE_RE = re.compile(r"#([\w.\-]+)$")


@dataclass(frozen=True)
class _Layout:
    """Where one transcript layer lives, and how to read it.

    The two layers are the same NXT shape in different directories, with one real
    difference: manual segments carry ``transcriber_start``, ASR segments carry
    no timing of their own and the start has to come from the first element of
    the span. Everything else is shared.
    """

    archive_url: str
    archive_name: str
    sha256: str
    words_dir: str
    segments_dir: str
    non_lexical: tuple[str, ...]
    is_asr: bool
    asr_system: str | None


_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations"

LAYOUTS: dict[Variant, _Layout] = {
    "asr": _Layout(
        archive_url=f"{_BASE}/ami_public_auto_1.5.1.zip",
        archive_name="ami_public_auto_1.5.1.zip",
        sha256="eb7b8582acf53a5fa8ce2503fc079f193964ee144dc8da5700e973aa8f801439",
        words_dir="ASR/ASR_AS_CTM_v1.0_feb07",
        segments_dir="ASR/ASR_AS_CTM_v1.0_feb07",
        # ASR output has silences but none of the manual annotation markup, and
        # no punctuation tokens at all, which is part of why it is tier 1.
        non_lexical=("sil",),
        is_asr=True,
        asr_system="ami-asr-2007 (ASR_AS_CTM_v1.0_feb07)",
    ),
    "manual": _Layout(
        archive_url=f"{_BASE}/ami_public_manual_1.6.2.zip",
        archive_name="ami_public_manual_1.6.2.zip",
        sha256="b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d",
        words_dir="words",
        segments_dir="segments",
        non_lexical=("vocalsound", "disfmarker", "gap", "transformerror"),
        is_asr=False,
        asr_system=None,
    ),
}
"""Checksums pinned so a re-cut release fails loudly rather than shifting every line index."""


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


def download(cache_dir: Path, *, variant: Variant = DEFAULT_VARIANT) -> Path:
    """Fetch one layer's archive into ``cache_dir``, skipping if already valid.

    Idempotent and safe to re-run. A file that is present but whose checksum does
    not match is re-downloaded rather than trusted.
    """
    layout = LAYOUTS[variant]
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / layout.archive_name

    if archive.exists() and _sha256(archive) == layout.sha256:
        return archive

    tmp = archive.with_suffix(archive.suffix + ".part")
    with urlopen(layout.archive_url) as response, tmp.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)

    digest = _sha256(tmp)
    if digest != layout.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{layout.archive_name} checksum mismatch: expected {layout.sha256}, got "
            f"{digest}. Upstream may have re-cut the release. Do not proceed: word ids "
            "drive line indices, and line indices are what evidence labels refer to."
        )
    tmp.replace(archive)
    return archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def meeting_ids(archive: Path, *, variant: Variant = DEFAULT_VARIANT) -> list[str]:
    """Every meeting id in the archive, sorted."""
    layout = LAYOUTS[variant]
    prefix = f"{layout.words_dir}/"
    with zipfile.ZipFile(archive) as bundle:
        ids = {
            name[len(prefix) :].split(".")[0]
            for name in bundle.namelist()
            if name.startswith(prefix) and name.endswith(".words.xml")
        }
    return sorted(i for i in ids if i)


def load(
    cache_dir: Path,
    *,
    variant: Variant = DEFAULT_VARIANT,
    meetings: list[str] | None = None,
    min_turns: int = 1,
) -> Iterator[Transcript]:
    """Yield one :class:`Transcript` per AMI meeting, in meeting-id order.

    Downloads on first use. ``variant`` selects the transcript layer and defaults
    to ``asr``, which is the tier 1 one. ``meetings`` restricts to specific ids.
    """
    yield from load_archive(
        download(cache_dir, variant=variant),
        variant=variant,
        meetings=meetings,
        min_turns=min_turns,
    )


def load_archive(
    archive: Path,
    *,
    variant: Variant = DEFAULT_VARIANT,
    meetings: list[str] | None = None,
    min_turns: int = 1,
) -> Iterator[Transcript]:
    """Parse an already-downloaded archive.

    Split out from :func:`load` so the parsing, which is where the interesting
    failures live, is testable without touching the network.
    """
    layout = LAYOUTS[variant]
    wanted = meeting_ids(archive, variant=variant) if meetings is None else sorted(meetings)

    with zipfile.ZipFile(archive) as bundle:
        available = set(bundle.namelist())
        for meeting in wanted:
            transcript = _build_transcript(bundle, available, meeting, layout)
            if transcript is not None and transcript.n_turns >= min_turns:
                yield transcript


def _build_transcript(
    bundle: zipfile.ZipFile, available: set[str], meeting: str, layout: _Layout
) -> Transcript | None:
    """Merge every speaker's segments for one meeting into a single ordered transcript."""
    segments: list[_Segment] = []
    speakers: set[str] = set()
    dropped: Counter[str] = Counter()

    seg_prefix = f"{layout.segments_dir}/{meeting}."
    for name in sorted(available):
        if not name.startswith(seg_prefix) or not name.endswith(".segments.xml"):
            continue
        speaker = name[len(seg_prefix) :].split(".")[0]
        words_name = f"{layout.words_dir}/{meeting}.{speaker}.words.xml"
        if words_name not in available:
            # A speaker with turn boundaries but no token stream contributes
            # nothing and must not shift the indices of speakers that do.
            continue
        speakers.add(speaker)
        speaker_segments, speaker_dropped = _parse_speaker(
            bundle, name, words_name, speaker, layout
        )
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
        "annotation_release": layout.archive_name,
        "variant": "asr" if layout.is_asr else "manual",
    }
    # Record what was excluded from the text so it is auditable rather than lost.
    meta.update({f"dropped_{kind}": str(count) for kind, count in sorted(dropped.items())})

    return Transcript(
        id=f"{SOURCE}/{meeting}",
        source=SOURCE,
        track=Track.P,
        turns=turns,
        is_asr=layout.is_asr,
        asr_system=layout.asr_system,
        channel_version=None,
        meta=meta,
    )


def _parse_speaker(
    bundle: zipfile.ZipFile,
    segments_name: str,
    words_name: str,
    speaker: str,
    layout: _Layout,
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

    dropped = Counter(e.tag for e in elements if e.tag in layout.non_lexical)

    out: list[_Segment] = []
    for segment in root.iter("segment"):
        span = _resolve_span(segment, order, elements)
        if not span:
            continue

        text = _render(span)
        if not text:
            # Segments containing only markup (a laugh, an untranscribed gap)
            # carry no words. Emitting them would create empty lines that
            # evidence could point at.
            continue

        out.append(
            _Segment(
                start=_segment_start(segment, span),
                speaker=speaker,
                segment_id=segment.get(_NITE_ID) or "",
                text=text,
            )
        )
    return out, dropped


def _resolve_span(
    segment: ElementTree.Element,
    order: dict[str, int],
    elements: list[ElementTree.Element],
) -> list[ElementTree.Element]:
    """Collect every element this segment points at, across both href formats.

    A segment may carry one child naming a range, or several children each naming
    a single id. Reading only the first child handles the former and silently
    truncates the latter to one word.
    """
    span: list[ElementTree.Element] = []
    for child in segment.findall(_NITE_CHILD):
        href = child.get("href")
        if not href:
            continue

        ranged = _HREF_RANGE_RE.search(href)
        if ranged is not None:
            first = ranged.group(1)
            last = ranged.group(2) or first
            if first in order and last in order:
                span.extend(elements[order[first] : order[last] + 1])
            continue

        bare = _HREF_BARE_RE.search(href)
        if bare is not None and bare.group(1) in order:
            span.append(elements[order[bare.group(1)]])
    return span


def _segment_start(segment: ElementTree.Element, span: list[ElementTree.Element]) -> float:
    """When this turn began, which is what orders turns across speakers.

    Manual segments carry ``transcriber_start``. ASR segments carry no timing of
    their own, so it comes from the first timed element of the span instead.
    Falling back to 0.0 for a whole layer would collapse the merge order and
    produce a transcript where every speaker appears to talk at once, so this is
    worth getting right rather than defaulting.
    """
    stated = segment.get("transcriber_start")
    if stated is not None:
        return _as_float(stated)
    for element in span:
        start = element.get("starttime")
        if start is not None:
            return _as_float(start)
    return 0.0


@dataclass(frozen=True)
class TimedWord:
    """One recognised or transcribed word, with the timing that lets layers be aligned."""

    text: str
    start: float
    end: float


def speaker_ids(archive: Path, meeting: str, *, variant: Variant = DEFAULT_VARIANT) -> list[str]:
    """Speaker codes present for one meeting in this layer."""
    layout = LAYOUTS[variant]
    prefix = f"{layout.words_dir}/{meeting}."
    with zipfile.ZipFile(archive) as bundle:
        return sorted(
            {
                name[len(prefix) :].split(".")[0]
                for name in bundle.namelist()
                if name.startswith(prefix) and name.endswith(".words.xml")
            }
        )


def word_stream(
    archive: Path, meeting: str, speaker: str, *, variant: Variant = DEFAULT_VARIANT
) -> list[TimedWord]:
    """One speaker's words in document order, with timings, markup excluded.

    Deliberately below the turn abstraction. Fitting the ASR channel needs the
    reference and hypothesis **word** sequences for the same speech, and turns
    are segmented differently in the two layers (ASR produces roughly 1.75x as
    many), so aligning turn-to-turn would compare text that is not parallel.
    """
    layout = LAYOUTS[variant]
    name = f"{layout.words_dir}/{meeting}.{speaker}.words.xml"
    with zipfile.ZipFile(archive) as bundle:
        if name not in set(bundle.namelist()):
            return []
        root = ElementTree.fromstring(bundle.read(name))

    out: list[TimedWord] = []
    for element in root:
        if element.tag != "w":
            continue
        token = (element.text or "").strip()
        if not token:
            continue
        out.append(
            TimedWord(
                text=token,
                start=_as_float(element.get("starttime")),
                end=_as_float(element.get("endtime")),
            )
        )
    return out


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
