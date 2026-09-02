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

Speaker tags are SPEAKER_NN. Which tag the agent gets is decided per case by
`tag_policy`: "random" (seeded; the model must infer roles, as with real
diarisation) or "agent_first" (agent is always SPEAKER_00). Roles live only in
hidden metadata.
"""

import csv
import hashlib
import json
import random
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


def assign_tags(roles: list[str], policy: str, rng: random.Random) -> dict[str, str]:
    """roles -> SPEAKER_NN mapping. Returns tag -> role."""
    distinct = list(dict.fromkeys(roles))
    if policy == "agent_first":
        order = sorted(distinct, key=lambda r: {"agent": 0, "customer": 1}.get(r, 2))
    else:
        order = distinct[:]
        rng.shuffle(order)
    return {f"SPEAKER_{i:02d}": role for i, role in enumerate(order)}


def _render(turns: list[tuple[str, str]], tag_of_role: dict[str, str]) -> list[str]:
    return [f"{tag_of_role[role]}: {text}" for role, text in turns]


def _case(id: str, track: str, source_id: str, variants: list[Variant], tags: dict[str, str],
          policy: str, meta: dict) -> Case:
    return Case(id=id, track=track, source_id=source_id,
                transcript=Transcript(variants=variants, speaker_roles=tags, tag_policy=policy), meta=meta)


# ---------------------------------------------------------------- AppTek

def apptek_cases(policy: str, seed: int) -> Iterator[Case]:
    rng = random.Random(seed)
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
            roles = [s["role"] for s in segs]
            tags = assign_tags(roles, policy, rng)
            role_tag = {r: t for t, r in tags.items()}
            clean = _render([(s["role"], s["text"].strip()) for s in segs], role_tag)
            messy = _render(
                [(s["role"], " ".join(b) if b else "<inaudible>") for s, b in zip(segs, buckets)],
                role_tag,
            )
            yield _case(
                f"apptek-{stem}", "track-p", f"apptek_callcenter/diarization/{locale}/{stem}",
                [Variant(kind="clean", origin="AppTek verbatim segments", lines=clean),
                 Variant(kind="messy", lines=messy,
                         origin="faster-whisper large-v3 over telephone-degraded audio, time-aligned to the verbatim turns "
                                f"(interim/apptek_callcenter_selfasr/diarization-degraded/{locale}/{stem}.json)")],
                tags, policy, {"locale": locale, "domain": row.get("domain", ""), "duration_s": row.get("duration")},
            )


# ---------------------------------------------------------------- interim-based sources

def _interim_turns(text: str) -> list[tuple[str, str]]:
    out = []
    for ln in text.split("\n"):
        tag, _, t = ln.partition(": ")
        out.append((tag, re.sub(r"^(>>\s*)+", "", t).strip()))  # SPoRC speaker-change markers
    return out


def taskmaster_cases(policy: str, seed: int) -> Iterator[Case]:
    rng = random.Random(seed)
    for line in (INTERIM / "taskmaster" / "train.jsonl").open():
        d = json.loads(line)
        turns = _interim_turns(d["text"])
        if len(turns) < 8:
            continue
        role_of = {"SPEAKER_00": "agent", "SPEAKER_01": "customer", "SPEAKER_02": "other"}
        roles = [role_of.get(t, "other") for t, _ in turns]
        tags = assign_tags(roles, policy, rng)
        role_tag = {r: t for t, r in tags.items()}
        clean = _render([(r, t) for r, (_, t) in zip(roles, turns)], role_tag)
        messy = [f"{ln.split(': ', 1)[0]}: {noise_line(ln.split(': ', 1)[1], seed=stable_seed(d['doc_id'], i))}"
                 for i, ln in enumerate(clean)]
        yield _case(
            f"taskmaster-{d['doc_id']}", "track-p", f"taskmaster/{d['meta'].get('subset', '')}/{d['doc_id']}",
            [Variant(kind="clean", origin="Taskmaster human transcription (partially repaired disfluencies)", lines=clean),
             Variant(kind="messy", origin="channel v2.2 noised + crude surface restoration (synthetic)", lines=messy)],
            tags, policy, {"subset": d["meta"].get("subset", "")},
        )


def aci_bench_cases(policy: str, seed: int) -> Iterator[Case]:
    rng = random.Random(seed)
    csv.field_size_limit(10**8)
    root = RAW / "aci_bench" / "data" / "aci-bench" / "challenge_data"
    for path in sorted(root.glob("*.csv")):
        if path.name.endswith("_metadata.csv"):
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                turns = []
                for ln in r["dialogue"].split("\n"):
                    m = re.match(r"\[(doctor|patient|\w+)\]\s*(.*)", ln.strip(), re.I)
                    if m and m.group(2).strip():
                        role = {"doctor": "agent", "patient": "customer"}.get(m.group(1).lower(), "other")
                        turns.append((role, m.group(2).strip()))
                if len(turns) < 8:
                    continue
                tags = assign_tags([r_ for r_, _ in turns], policy, rng)
                role_tag = {r_: t for t, r_ in tags.items()}
                clean = _render(turns, role_tag)
                doc_id = f"{path.stem}-{r.get('encounter_id') or r.get('ID')}"
                messy = [f"{ln.split(': ', 1)[0]}: {noise_line(ln.split(': ', 1)[1], seed=stable_seed(doc_id, i))}"
                         for i, ln in enumerate(clean)]
                yield _case(
                    f"aci-{doc_id}", "track-p", f"aci_bench/challenge_data/{doc_id}",
                    [Variant(kind="clean", origin="ACI-Bench challenge_data dialogue (cleaned)", lines=clean),
                     Variant(kind="messy", origin="channel v2.2 noised + crude surface restoration (synthetic)", lines=messy)],
                    tags, policy, {"split_file": path.stem},
                )


def sporc_cases(policy: str, seed: int, max_lines: int = 160) -> Iterator[Case]:
    for line in (INTERIM / "sporc" / "train.jsonl").open():
        d = json.loads(line)
        if d["meta"].get("n_speakers", 0) not in (2, 3) or not (30 <= d["n_turns"] <= max_lines):
            continue
        turns = _interim_turns(d["text"])
        tags = {t: "other" for t in dict.fromkeys(t for t, _ in turns)}
        yield _case(
            f"sporc-{d['doc_id']}", "track-nc", f"sporc/episode/{d['doc_id']}",
            [Variant(kind="messy", origin="SPoRC Whisper + diarisation (real ASR)", lines=[f"{t}: {x}" for t, x in turns])],
            tags, policy, {"pod_title": d["meta"].get("pod_title", ""), "category": d["meta"].get("category", "")},
        )


BUILDERS = {
    "apptek": apptek_cases,
    "taskmaster": taskmaster_cases,
    "aci_bench": aci_bench_cases,
    "sporc": sporc_cases,
}
