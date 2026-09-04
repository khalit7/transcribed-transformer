"""Base cases for the benchmark: one call rendered as line-aligned variants.

Sources (any conversation between different people qualifies):

- apptek (P): verbatim segments from the `diarization` config give the clean
  variant; the messy variant is REAL recogniser output, line-aligned by time:
  every word from our Whisper pass over the merged, telephone-degraded audio is
  assigned to the verbatim turn whose time span contains it, so line i is the
  same turn in both variants and one evidence key serves both.
- taskmaster (P, tier 3): clean from interim; messy = channel v2.2 applied to
  every line (synthetic, flagged as such in the variant origin).
- aci_bench (P): `challenge_data` dialogues as the clean variant; messy =
  channel v2.2 like Taskmaster (its real ASR layer is not line-aligned).
- sporc (NC): real diarised ASR from interim; single variant, kind "messy".

Every line is `<role>: text` with the role label exactly as the corpus records
it (Khalid, 2026-09-03: always the verbatim roles, never a remapping). AppTek:
`agent` / `customer`; Taskmaster: `assistant` / `user`; ACI-Bench: `doctor` /
`patient` (plus whatever other bracketed speaker the dialogue names). SPoRC has
diarised `SPEAKER_NN` tags only, so its HOST is identified by an LLM first
(identify_speakers.py, cached) and rendered as `host`; other speakers keep their
tags. A corpus with no role labels and no identification raises NoSpeakerRoles.
"""

import csv
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path

from src.channel.apply import Channel
from src.synthesis.schema import Case, Transcript, Variant

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
INTERIM = REPO_ROOT / "data" / "interim"
CHANNEL_ARTIFACT = REPO_ROOT / "data" / "channel" / "channel_v2_degraded.json"

_channel: Channel | None = None


def channel() -> Channel:
    global _channel
    if _channel is None:
        _channel = Channel.load(CHANNEL_ARTIFACT, burst=False)
    return _channel


def surface(text: str) -> str:
    """Crude surface restoration for channel output (lowercase, unpunctuated):
    sentence-initial capital, standalone `i`, terminal full stop. The channel's
    surface layer is not built yet (SYNTHSHEET section 4); this keeps noised
    lines from being trivially spottable among cased, punctuated ones."""
    text = text.strip()
    if not text:
        return text
    text = re.sub(r"\bi\b", "I", text)
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "."
    return text


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's hash() is salted per run)."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def noise_line(text: str, seed: int) -> str:
    noised = surface(channel().noise_text(text, seed))
    return noised or "<inaudible>"  # a short line deleted wholesale reads as an inaudible turn


class NoSpeakerRoles(RuntimeError):
    """The corpus records no speaker roles; how to handle role-less transcripts is undecided."""


def _render(turns: list[tuple[str, str]]) -> list[str]:
    return [f"{role}: {text}" for role, text in turns]


def _case(id: str, track: str, source_id: str, variants: list[Variant], role_source: str, meta: dict) -> Case:
    dataset = source_id.split("/")[0].replace("apptek_callcenter", "apptek")
    speakers = list(dict.fromkeys(ln.partition(": ")[0] for v in variants for ln in v.lines))
    return Case(id=id, dataset=dataset, track=track, source_id=source_id,
                transcript=Transcript(variants=variants, speakers=speakers, role_source=role_source), meta=meta)


# ---------------------------------------------------------------- AppTek

def apptek_cases() -> Iterator[Case]:
    hyp_root = INTERIM / "apptek_callcenter_selfasr" / "diarization-degraded"
    for meta in sorted((RAW / "apptek_callcenter" / "diarization").glob("*/metadata.jsonl")):
        locale = meta.parent.name
        for line in meta.open():
            row = json.loads(line)
            stem = Path(row["file_name"]).stem
            hyp_file = hyp_root / locale / f"{stem}.json"
            if not hyp_file.exists():
                continue
            segs = [s for s in row["segments"] if s.get("text", "").strip()]
            if len(segs) < 10:
                continue
            words = json.loads(hyp_file.read_text())["words"]
            spans = [(s["start"], s["end"]) for s in segs]
            buckets: list[list[str]] = [[] for _ in segs]
            j = 0
            for w in words:
                mid = (w["start"] + w["end"]) / 2
                while j < len(spans) - 1 and mid > spans[j][1] and mid >= spans[j + 1][0]:
                    j += 1
                # words in gaps go to the nearer neighbour's turn
                if mid < spans[j][0] and j > 0 and (spans[j][0] - mid) > (mid - spans[j - 1][1]):
                    buckets[j - 1].append(w["word"].strip())
                else:
                    buckets[j].append(w["word"].strip())
            if any(not s.get("role") for s in segs):
                raise NoSpeakerRoles(f"apptek {stem}: a segment has no role label")
            clean = _render([(s["role"], s["text"].strip()) for s in segs])
            messy = _render([(s["role"], " ".join(b) if b else "<inaudible>") for s, b in zip(segs, buckets)])
            yield _case(
                f"apptek-{stem}", "track-p", f"apptek_callcenter/diarization/{locale}/{stem}",
                [Variant(kind="clean", origin="AppTek verbatim segments", lines=clean),
                 Variant(kind="messy", lines=messy,
                         origin="faster-whisper large-v3 over telephone-degraded audio, time-aligned to the verbatim turns "
                                f"(interim/apptek_callcenter_selfasr/diarization-degraded/{locale}/{stem}.json)")],
                "AppTek segment role annotation (agent / customer)",
                {"locale": locale, "domain": row.get("domain", ""), "duration_s": row.get("duration")},
            )


# ---------------------------------------------------------------- interim-based sources

def _interim_turns(text: str) -> list[tuple[str, str]]:
    out = []
    for ln in text.split("\n"):
        tag, _, t = ln.partition(": ")
        out.append((tag, re.sub(r"^(>>\s*)+", "", t).strip()))  # SPoRC speaker-change markers
    return out


def taskmaster_cases() -> Iterator[Case]:
    for line in (INTERIM / "taskmaster" / "train.jsonl").open():
        d = json.loads(line)
        turns = _interim_turns(d["text"])
        if len(turns) < 8:
            continue
        # interim keeps Taskmaster's own ASSISTANT / USER labels as SPEAKER_00 / SPEAKER_01 (preprocessing.taskmaster);
        # undo that here so the verbatim roles are what is rendered
        role_of = {"SPEAKER_00": "assistant", "SPEAKER_01": "user"}
        unknown = {tag for tag, _ in turns} - set(role_of)
        if unknown:
            raise NoSpeakerRoles(f"taskmaster {d['doc_id']}: speaker labels without a corpus role: {sorted(unknown)}")
        clean = _render([(role_of[tag], text) for tag, text in turns])
        messy = [f"{ln.split(': ', 1)[0]}: {noise_line(ln.split(': ', 1)[1], seed=stable_seed(d['doc_id'], i))}"
                 for i, ln in enumerate(clean)]
        yield _case(
            f"taskmaster-{d['doc_id']}", "track-p", f"taskmaster/{d['meta'].get('subset', '')}/{d['doc_id']}",
            [Variant(kind="clean", origin="Taskmaster human transcription (partially repaired disfluencies)", lines=clean),
             Variant(kind="messy", origin="channel v2.2 noised + crude surface restoration (synthetic)", lines=messy)],
            "Taskmaster ASSISTANT / USER labels", {"subset": d["meta"].get("subset", "")},
        )


def aci_bench_cases() -> Iterator[Case]:
    csv.field_size_limit(10**8)
    root = RAW / "aci_bench" / "data" / "aci-bench" / "challenge_data"
    for path in sorted(root.glob("*.csv")):
        if path.name.endswith("_metadata.csv"):
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                turns = []
                for ln in r["dialogue"].split("\n"):
                    m = re.match(r"\[(doctor|patient|\w+)\]\s*(.*)", ln.strip(), re.IGNORECASE)
                    if m and m.group(2).strip():
                        turns.append((m.group(1).lower(), m.group(2).strip()))
                if len(turns) < 8:
                    continue
                doc_id = f"{path.stem}-{r.get('encounter_id') or r.get('ID')}"
                clean = _render(turns)
                messy = [f"{ln.split(': ', 1)[0]}: {noise_line(ln.split(': ', 1)[1], seed=stable_seed(doc_id, i))}"
                         for i, ln in enumerate(clean)]
                yield _case(
                    f"aci-{doc_id}", "track-p", f"aci_bench/challenge_data/{doc_id}",
                    [Variant(kind="clean", origin="ACI-Bench challenge_data dialogue (cleaned)", lines=clean),
                     Variant(kind="messy", origin="channel v2.2 noised + crude surface restoration (synthetic)", lines=messy)],
                    "ACI-Bench bracketed speaker labels ([doctor] / [patient])", {"split_file": path.stem},
                )


def sporc_episodes(max_lines: int = 160) -> Iterator[tuple[str, list[tuple[str, str]], dict]]:
    """Eligible SPoRC episodes in builder order: (doc_id, [(SPEAKER_NN, text)], meta)."""
    for line in (INTERIM / "sporc" / "train.jsonl").open():
        d = json.loads(line)
        if d["meta"].get("n_speakers", 0) not in (2, 3) or not (30 <= d["n_turns"] <= max_lines):
            continue
        yield d["doc_id"], _interim_turns(d["text"]), {"pod_title": d["meta"].get("pod_title", ""),
                                                        "category": d["meta"].get("category", "")}


def sporc_cases(max_lines: int = 160, identify_model: str | None = None, only: set[str] | None = None) -> Iterator[Case]:
    """SPoRC ships diarised `SPEAKER_NN` labels and no roles, so the host is identified first
    (identify_speakers.py, cached on disk; identified on demand here for any uncached episode
    with `identify_model`, default claude:sonnet; pass "" to forbid identification, in which case an
    uncached episode raises NoSpeakerRoles). The host's lines render as `host:`; the other speakers
    keep their diarisation tags. Episodes whose host could not be identified are skipped with a
    notice, never rendered role-less. `only` restricts to a set of case ids (sporc-<doc_id>) so a
    sampler can pick episodes by position in the raw stream and pay for identification only on
    those it uses."""
    from src.synthesis import identify_speakers as ids

    cache = ids.load_cache()
    model = ids.DEFAULT_MODEL if identify_model is None else identify_model
    for doc_id, turns, meta in sporc_episodes(max_lines):
        if only is not None and f"sporc-{doc_id}" not in only:
            continue
        rec = cache.get(doc_id)
        if rec is None:
            if not model:
                raise NoSpeakerRoles(f"sporc {doc_id}: no speaker roles and host identification is disabled")
            rec = ids.identify(doc_id, turns, model)
            ids.append(rec)
            cache[doc_id] = rec
        if rec["host"] is None:
            print(f"  sporc {doc_id}: no host identified ({rec['reason'][:80]}); skipped", flush=True)
            continue
        host = rec["host"]
        lines = [f"{'host' if t == host else t}: {x}" for t, x in turns]
        yield _case(
            f"sporc-{doc_id}", "track-nc", f"sporc/episode/{doc_id}",
            [Variant(kind="messy", origin="SPoRC Whisper + diarisation (real ASR)", lines=lines)],
            f"host identified by {rec['model']} (confidence {rec['confidence']}); other speakers keep diarisation tags",
            {**meta, "host_tag": host},
        )


# cheap id streams in the same order as the builder's raw stream, for samplers that must pick by
# position without building (and, for sporc, without paying to identify) every candidate
ID_STREAMS = {"sporc": lambda: (f"sporc-{d}" for d, _, _ in sporc_episodes())}

BUILDERS = {
    "apptek": apptek_cases,
    "taskmaster": taskmaster_cases,
    "aci_bench": aci_bench_cases,
    "sporc": sporc_cases,
}
