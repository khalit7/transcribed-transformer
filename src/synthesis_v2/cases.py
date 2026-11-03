"""Build v2 cases: a pool episode rendered one corpus turn per line with hosts labelled `host_N` and guests `guest_N`.

Rendering (owner's decision 2026-11-01, D75/D76): every turn exactly as SPoRC distributes it (`data/interim/sporc`), one
per line, with no preprocessing: same-speaker fragments are not merged, bracketed non-speech tokens are not removed,
repetition loops are not collapsed and long turns are not split. The only change to a line is its speaker: the hosts'
tags (every tag the v2 identifier named) become `host_1`, `host_2`, ... and the guests' tags `guest_1`, `guest_2`, ...,
each numbered in order of first appearance, so co-hosts and guests stay distinguishable; any other tag (an advert voice,
a clip, a fragment) stays verbatim. Records store `<speaker>: <utterance>` lines (the shared schema's form); every
model sees them through `src.synthesis_v2.numbered` as `<n>. <speaker>:<utterance>`. Line numbers, and so evidence, refer
to corpus turns.

Until 2026-11-01 the builder merged fragments, stripped non-speech tokens, collapsed loops and split long turns
(D66/D68); those rules survive only in `clean_turns`, which the speaker identifier still uses to choose the opening and
closing windows it reads (the identification is unchanged, so every episode in the pool was identified the same way).
The pool's text filters (`text_checks`) are applied again to the turns the case is built from; only episodes with at
least one identified host are built; the rest are skipped and counted. v1's builder (`src/synthesis/cases.py`) is
untouched.
"""

import json
import re
from collections.abc import Iterator
from collections.abc import Set as AbstractSet

from src.synthesis.schema import Case, Transcript, Variant
from src.synthesis_v2 import identify_hosts
from src.synthesis_v2.episodes import INTERIM, POOL, read_text, text_checks, turns_of

HOST = "host"  # label prefixes: host_1, host_2, ... and guest_1, guest_2, ... in order of first appearance
GUEST = "guest"
ORIGIN = "SPoRC Whisper + diarisation (real ASR); corpus turns verbatim, no merging or cleaning"
_NON_SPEECH = re.compile(r"[\[(][^\])]{1,40}[\])]")  # [Music], (laughs), [inaudible]: short bracketed tokens only
_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
MAX_WORDS = 150
LOOP_MIN_REPEATS = 3  # identical fragments in a row before a short repeat counts as a Whisper loop
LOOP_LONG_WORDS = 4   # a repeat this long is a loop at the second occurrence


def split_long(role: str, text: str, max_words: int = MAX_WORDS) -> list[tuple[str, str]]:
    """Rule 4: split a merged turn over max_words at sentence ends, else at the word count."""
    words = text.split()
    if len(words) <= max_words:
        return [(role, text)]
    out: list[tuple[str, str]] = []
    chunk: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sw = sentence.split()
        if chunk and len(chunk) + len(sw) > max_words:
            out.append((role, " ".join(chunk)))
            chunk = []
        while len(sw) > max_words:  # one unpunctuated run: cut at the word count
            if chunk:
                out.append((role, " ".join(chunk)))
                chunk = []
            out.append((role, " ".join(sw[:max_words])))
            sw = sw[max_words:]
        chunk += sw
    if chunk:
        out.append((role, " ".join(chunk)))
    return out


def clean_turns(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge same-speaker fragments, strip bracketed non-speech, collapse loops, split long turns. Not used for the
    rendering since 2026-11-01 (see the module docstring); the speaker identifier reads its windows from these turns."""
    merged: list[tuple[str, list[str]]] = []  # (role, fragments)
    repeats = 0
    for role, said in turns:
        text = _WS.sub(" ", _NON_SPEECH.sub(" ", said)).strip()
        if not text:
            continue
        if merged and merged[-1][0] == role:
            frags = merged[-1][1]
            if frags and text == frags[-1]:
                repeats += 1
                if repeats >= LOOP_MIN_REPEATS - 1 or len(text.split()) >= LOOP_LONG_WORDS:
                    continue  # a loop: no more copies are kept
            else:
                repeats = 0
            frags.append(text)
        else:
            merged.append((role, [text]))
            repeats = 0
    out: list[tuple[str, str]] = []
    for role, frags in merged:
        out += split_long(role, " ".join(frags))
    return out


def speaker_labels(turns: list[tuple[str, str]], hosts: AbstractSet[str], guests: AbstractSet[str] = frozenset()) -> dict[str, str]:
    """Each host tag -> `host_N` and each guest tag -> `guest_N`, numbered in order of first appearance in the transcript;
    a tag in both sets is a host; a tag that never speaks (identified from a dropped fragment) still gets a label. A
    verbatim tag that would read as a generated label (`host...`, `guest...`) is refused: VOCAB reads labels by prefix."""
    clash = sorted({r for r, _s in turns if r not in hosts and r not in guests and r.lower().startswith((HOST, GUEST))})
    if clash:
        raise ValueError(f"diarisation tags {clash} would read as host or guest labels")
    guests = set(guests) - set(hosts)
    out: dict[str, str] = {}
    counts = {HOST: 0, GUEST: 0}
    for role in [r for r, _s in turns] + sorted(hosts) + sorted(guests):
        if role in out:
            continue
        prefix = HOST if role in hosts else GUEST if role in guests else None
        if prefix is not None:
            counts[prefix] += 1
            out[role] = f"{prefix}_{counts[prefix]}"
    return out


def render(turns: list[tuple[str, str]], hosts: AbstractSet[str], guests: AbstractSet[str] = frozenset()) -> list[str]:
    labels = speaker_labels(turns, hosts, guests)
    return [f"{labels.get(role, role)}: {said}" for role, said in turns]


def build(doc_id: str, text: str, hosts: list[str], meta: dict, guests: list[str] | None = None) -> Case:
    turns = turns_of(text)
    reason, _stats = text_checks(turns)
    if reason is not None:  # the pool's filters, applied again to the turns the labeller will see
        raise ValueError(f"{doc_id}: {reason}")
    guests = [g for g in (guests or []) if g not in hosts]
    lines = render(turns, set(hosts), set(guests))
    speakers = list(dict.fromkeys(ln.partition(": ")[0] for ln in lines))
    return Case(id=f"sporc-{doc_id}", dataset="sporc", track="track-nc", source_id=f"sporc/episode/{doc_id}",
                transcript=Transcript(variants=[Variant(kind="messy", origin=ORIGIN, lines=lines)], speakers=speakers,
                                      role_source=f"hosts ({', '.join(hosts)}) and guests ({', '.join(guests) or 'none'}) "
                                                  "identified by the v2 identifier, rendered host_N / guest_N in order of "
                                                  "first appearance; other speakers keep their diarisation tags"),
                meta={**meta, "host_tags": hosts, "guest_tags": guests, "n_raw_turns": text.count("\n") + 1})


def pool_cases(doc_ids: list[str] | None = None, skipped: dict[str, int] | None = None) -> Iterator[Case]:
    """Cases for the pool episodes (or the given doc_ids) whose hosts are cached; skips the rest. `skipped`, if given,
    receives the counts of what was not built and why."""
    rows = {r["doc_id"]: r for r in (json.loads(line) for line in POOL.open())}
    cache = identify_hosts.load_cache(identify_hosts.CACHE)
    counts = skipped if skipped is not None else {}
    for doc_id in doc_ids if doc_ids is not None else list(rows):
        row = rows[doc_id]
        rec = cache.get(doc_id)
        if rec is None or not identify_hosts.current(rec):  # never identified, or by an earlier version without guests and kind
            counts["not identified"] = counts.get("not identified", 0) + 1
            continue
        if rec["kind"] != "conversation":
            counts[f"not a conversation ({rec['kind']})"] = counts.get(f"not a conversation ({rec['kind']})", 0) + 1
            continue
        if not rec["hosts"]:
            counts["no host"] = counts.get("no host", 0) + 1
            continue
        meta = {k: row[k] for k in ("podcast_id", "pod_title", "category", "stratum", "tokens", "n_speakers") if k in row}
        try:
            yield build(doc_id, read_text(INTERIM, row["offset"]), rec["hosts"], meta, rec.get("guests", []))
        except ValueError as e:
            counts["failed text checks"] = counts.get("failed text checks", 0) + 1
            print(f"skipped {e}", flush=True)


def skipped(doc_ids: list[str]) -> dict[str, int]:
    """How many of the given episodes cannot be built, and why."""
    cache = identify_hosts.load_cache(identify_hosts.CACHE)
    out = {"not_identified": 0, "no_host_found": 0, "not_a_conversation": 0, "buildable": 0}
    for d in doc_ids:
        rec = cache.get(d)
        if rec is None or not identify_hosts.current(rec):
            out["not_identified"] += 1
        elif rec["kind"] != "conversation":
            out["not_a_conversation"] += 1
        elif not rec["hosts"]:
            out["no_host_found"] += 1
        else:
            out["buildable"] += 1
    return out
