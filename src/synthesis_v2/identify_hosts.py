"""Identify every host and every guest among the diarised speakers of a v2 pool episode.

SPoRC records `SPEAKER_NN` diarisation tags and no roles. Roles are always rendered verbatim from the corpus (D19), so
for podcasts the roles have to be established first: a labeller reads the opening and closing turns and names the tags
that are HOSTS (run the show: introduce it and the guests, ask the questions, read the ads, close) and the tags that are
GUESTS (people interviewed or taking part who do not run the show). Unlike v1 (`src/synthesis/identify_speakers.py`,
one host per episode), v2 returns both sets, because co-hosted shows are common and the owner wants co-hosts and guests
told apart in the rendering (`host_1`, `host_2`, `guest_1`, ...; D66, revised 2026-10-31). Tags that are neither (an
advert voice, a played clip, an announcer, a fragment) keep their diarisation tag. A cache record without a `guests`
key, or without `kind` or `same_person`, comes from an earlier version of this module and is identified again
(`current`). The reply also classes the recording
(`kind`: conversation, scripted, other); only conversations become cases, because a radio play or a private phone call
has no host and the bank's questions presume a show (the pilot's 120 episodes held two radio plays, one audio drama and
one private call). Where SPoRC's own
`host_speaker_labels` exist (about a fifth of the pool, from `sporc_meta.py`) they are stored beside Sonnet's answer as a
cross-check, and since D73 they are the answer where they are complete (next paragraph).

Results are cached in data/labelled_data_v2/speakers/sporc.jsonl, one line per episode (doc_id, kind, hosts, raw_hosts,
guests, raw_guests, same_person, confidence, reason, sporc_host_tags, sporc_guest_tags, agrees_with_sporc,
guests_agree_with_sporc, model, cost_usd, claude_account, timestamp). The identifier reads the merged turns
(`cases.clean_turns`), so its opening and closing windows cover far more speech than the raw fragments would, and every
tag that does not appear in those windows is shown with its first two turns, so no tag is named unseen. Tags that
diarisation seems to have split from one person are reported as `same_person` groups (counted, not yet applied to the
rendering: the rate on the pilot decides whether they are merged).

Where SPoRC's own metadata is complete (owner's decision 2026-10-31, D73), it is used instead of Sonnet: an episode whose
`host_speaker_labels` and `guest_speaker_labels` together cover every diarisation tag that speaks at least MIN_SHARE of
the words gets a record straight from the metadata (`source: sporc-metadata`, cost 0, kind conversation, no same-person
groups), and a metadata record replaces a Sonnet one for such an episode so the rule is applied uniformly, with one
exception: a Sonnet verdict that the recording is not a conversation (a radio play, a private call) is kept. On the in-cap
pool that is 419 of 6,000 episodes (2% threshold, shares computed per tag from the transcript); the pilot measured Sonnet
and SPoRC disagreeing on the host set in 10
of the 24 pilot episodes that carry SPoRC host labels, so the two sources are not interchangeable and a spot check of the
metadata records is owed before production. The v2 case builder renders only cached episodes
with at least one host and skips the rest; the skip count goes to the datasheet.

    uv run python -m src.synthesis_v2.identify_hosts --ids data/labelled_data_v2/pilot_ids.txt [--workers 16]
    uv run python -m src.synthesis_v2.identify_hosts --all                     # the whole pool
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import re
from pathlib import Path

from src import prompts
from src.synthesis import llm
from src.synthesis.llm import ask_json_full, set_claude_account
from src.synthesis_v2 import OUT_DIR, append_jsonl, read_jsonl
from src.synthesis_v2.episodes import INTERIM, POOL, read_text, turns_of
from src.synthesis_v2.sporc_meta import load_meta

DEFAULT_MODEL = "claude:sonnet"
CACHE = OUT_DIR / "speakers" / "sporc.jsonl"
CALLS = OUT_DIR / "speakers" / "sporc.calls.jsonl"  # one line per paid call: cost, attempts, token usage
KINDS = ("conversation", "scripted", "other")  # only a conversation is built into a case (cases.pool_cases)
PROMPT = prompts.V2 / "identify_speakers.txt"  # the template: kind, hosts, guests, same-person groups
HEAD, TAIL = 80, 20  # turns shown: hosts open and close the show; more than v1's 60/12 because the episodes are longer

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["conversation", "scripted", "other"]},
        "hosts": {"type": "array", "items": {"type": "string"}},
        "guests": {"type": "array", "items": {"type": "string"}},
        "same_person": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["kind", "hosts", "guests", "same_person", "confidence", "reason"],
}
FIELDS = ("kind", "guests", "same_person")  # a cache record lacking any of these is from an earlier version: redo it
MIN_SHARE = 0.02  # a tag speaking under this share of the words (fragments, idents) need not be covered by the metadata
METADATA = "sporc-metadata"  # the `source` of a record built from SPoRC's own speaker labels


def current(rec: dict | None) -> bool:
    """Whether a cache record was written by this version of the identifier (earlier versions lacked guests and kind)."""
    return rec is not None and all(k in rec for k in FIELDS)


def word_shares(turns: list[tuple[str, str]]) -> dict[str, float]:
    """Each tag's share of the words, from the raw turns (the pool row's `word_shares` is sorted by size and carries no
    tags, so it cannot be matched to `tags_in_text`)."""
    words: dict[str, int] = {}
    for role, said in turns:
        words[role] = words.get(role, 0) + len(said.split())
    total = sum(words.values()) or 1
    return {t: n / total for t, n in words.items()}


def from_metadata(doc_id: str, shares: dict[str, float], m: dict) -> dict | None:
    """A cache record from SPoRC's own speaker labels, when they cover every tag that speaks at least MIN_SHARE of the
    words (`shares`: tag -> share, from `word_shares`); None otherwise (the episode then goes to the model). Hosts and
    guests are the metadata's tags that occur in the transcript; a tag in both lists is a host."""
    hosts_meta, guests_meta = set(m.get("sporc_host_tags", [])), set(m.get("sporc_guest_tags", []))
    tags = list(shares)
    if not hosts_meta or not tags:
        return None
    needed = {t for t in tags if shares[t] >= MIN_SHARE}
    if not needed or not needed <= (hosts_meta | guests_meta):
        return None
    hosts = sorted(hosts_meta & set(tags))
    guests = sorted((guests_meta & set(tags)) - hosts_meta)
    if not hosts:
        return None
    return {
        "doc_id": doc_id, "kind": "conversation", "hosts": hosts, "raw_hosts": hosts, "guests": guests, "raw_guests": guests,
        "same_person": [], "confidence": None,
        "reason": f"SPoRC host_speaker_labels and guest_speaker_labels cover every tag with at least {MIN_SHARE:.0%} of the words",
        "sporc_host_tags": sorted(hosts_meta), "sporc_guest_tags": sorted(guests_meta - hosts_meta),
        "agrees_with_sporc": None, "guests_agree_with_sporc": None,  # a metadata record is not a cross-check of itself
        "source": METADATA, "model": METADATA, "cost_usd": 0.0, "claude_account": None,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def load_cache(path: Path = CACHE) -> dict[str, dict]:
    return {r["doc_id"]: r for r in read_jsonl(path)}


def normalise_hosts(raw, tags: set[str]) -> tuple[list[str], list[str]]:
    """The reply's host tags matched to the transcript's tags, case-insensitively; a lone string is one tag.
    Returns (matched tags, the raw values as strings)."""
    values: list[str] = []
    for v in ([raw] if isinstance(raw, str) else list(raw or [])):
        v = re.sub(r"\s*\([^)]*\)", "", str(v))  # a reply that echoes "(12 turns)" or adds "(host)"
        values.extend(x.strip("\"'.") for x in re.split(r"[,;\s]+", v.strip()) if x.strip("\"'."))
    by_lower = {t.lower(): t for t in tags}
    matched = sorted({by_lower[str(v).strip().lower()] for v in values if str(v).strip().lower() in by_lower})
    return matched, [str(v) for v in values]


def prompt(turns: list[tuple[str, str]]) -> str:
    """The identification prompt over the merged turns: the opening HEAD and closing TAIL turns, then the first two turns
    of every tag that appears in neither window, so that every tag the reply may name has been seen."""
    tags = list(dict.fromkeys(t for t, _ in turns))
    counts = {t: sum(1 for r, _ in turns if r == t) for t in tags}
    shown = turns if len(turns) <= HEAD + TAIL else turns[:HEAD] + [("...", "[middle of the episode omitted]")] + turns[-TAIL:]
    body = "\n".join(f"{t}: {x}" for t, x in shown)
    seen = {t for t, _ in shown}
    unseen = [t for t in tags if t not in seen]
    if unseen:
        extra = []
        for t in unseen:
            firsts = [(n, x) for n, (r, x) in enumerate(turns, 1) if r == t][:2]
            extra.extend(f"[turn {n} of {len(turns)}] {t}: {x}" for n, x in firsts)
        body += "\n\nTAGS THAT DO NOT APPEAR ABOVE, with their first two turns:\n" + "\n".join(extra)
    tag_list = ", ".join(f"{t} ({counts[t]} turns)" for t in tags)
    return prompts.render(PROMPT, tag_list=tag_list, body=body, tag_names=", ".join(tags))


NONE_WORDS = ("none", "n/a", "null", "", "and", "-", "no", "host", "hosts", "guest", "guests")  # connectives and "no host"


def unmatched(raw: list[str], matched: list[str], tags: set[str], field: str, cost: float) -> None:
    """Refuse a reply in which any named tag (other than a word for "none") matched no tag of the transcript: a typo
    such as SPEAKER_7 for SPEAKER_07 would otherwise silently drop a speaker from the hosts or the guests."""
    by_lower = {t.lower() for t in tags}
    missing = [v for v in raw if v.strip().lower() not in by_lower and v.strip().lower() not in NONE_WORDS]
    if missing:
        raise llm.LLMError(f"reply named {field} {missing} that match no tag in {sorted(tags)}", cost)


def normalise_groups(raw, tags: set[str]) -> list[list[str]]:
    """`same_person` groups matched to the transcript's tags; groups with fewer than two distinct matched tags are dropped."""
    groups = []
    for g in raw if isinstance(raw, list) else []:
        matched, _raw = normalise_hosts(g if isinstance(g, (list, str)) else [], tags)
        if len(matched) >= 2 and matched not in groups:
            groups.append(matched)
    return groups


def identify(doc_id: str, turns: list[tuple[str, str]], sporc_host_tags: list[str], model: str = DEFAULT_MODEL,
             sporc_guest_tags: list[str] | None = None) -> dict:
    """Identify the kind, hosts and guests of one episode from its merged turns; returns the cache record (not yet
    written). Raises LLMError when the reply named host or guest tags but none matched the transcript, or omitted the
    guests list (the caller then does not cache a false "no host" or "no guest")."""
    tags = {t for t, _ in turns}
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    try:
        resp, cost_total, info = ask_json_full(prompt(turns), model, SCHEMA)
    except Exception as e:  # the failed call's cost is logged before the error propagates
        append_jsonl(CALLS, {"doc_id": doc_id, "model": model, "cost_usd": round(getattr(e, "cost", 0.0), 5),
                             "failed": str(e)[:200], "timestamp": stamp})
        raise
    append_jsonl(CALLS, {"doc_id": doc_id, "model": model, "cost_usd": round(cost_total, 5), "attempts": info["attempts"],
                         "usage": info["usage"], "timestamp": stamp})
    if not isinstance(resp, dict):
        raise llm.LLMError(f"reply is not an object: {str(resp)[:100]}", cost_total)
    kind = str(resp.get("kind", "")).strip().lower()
    if kind not in KINDS:
        raise llm.LLMError(f"reply has no usable kind: {str(resp.get('kind'))[:50]}", cost_total)
    if "hosts" not in resp:
        raise llm.LLMError("reply has no hosts list", cost_total)
    if "guests" not in resp:
        raise llm.LLMError("reply has no guests list", cost_total)
    hosts, raw = normalise_hosts(resp.get("hosts"), tags)
    guests, raw_guests = normalise_hosts(resp.get("guests"), tags)
    if kind == "conversation":  # a non-conversation is cached whatever it calls its voices: nothing is rendered from it
        unmatched(raw, hosts, tags, "hosts", cost_total)
        unmatched(raw_guests, guests, tags, "guests", cost_total)
    guests = [g for g in guests if g not in hosts]  # a tag is never both
    same_person = normalise_groups(resp.get("same_person"), tags)
    sporc_guest_tags = sorted(set(sporc_guest_tags or []) - set(sporc_host_tags))  # SPoRC's own overlaps resolved as ours
    conf = resp.get("confidence")
    return {
        "doc_id": doc_id, "kind": kind, "hosts": hosts, "raw_hosts": raw, "guests": guests, "raw_guests": raw_guests,
        "same_person": same_person,
        "confidence": min(1.0, max(0.0, float(conf))) if isinstance(conf, (int, float)) else None,
        "reason": str(resp.get("reason", "")).strip()[:300], "sporc_host_tags": sporc_host_tags,
        "sporc_guest_tags": sporc_guest_tags,
        "agrees_with_sporc": (sorted(sporc_host_tags) == hosts) if sporc_host_tags else None,
        "guests_agree_with_sporc": (sorted(sporc_guest_tags) == guests) if sporc_guest_tags else None,
        "model": model, "cost_usd": round(cost_total, 5), "claude_account": llm.CLAUDE_ACCOUNT,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def append(rec: dict, path: Path = CACHE) -> None:
    append_jsonl(path, rec)


def pool_rows() -> dict[str, dict]:
    return {r["doc_id"]: r for r in (json.loads(line) for line in POOL.open())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ids", type=Path, help="file with one pool doc_id per line to identify")
    p.add_argument("--all", action="store_true", help="identify every pool episode")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--claude-account", "--claude_account", dest="claude_account", choices=["p", "w"], required=True,
                   help="which Claude account claude -p bills: p (personal) or w (work); recorded on every record")
    args = p.parse_args()
    if not args.ids and not args.all:
        p.error("give --ids <file> or --all")
    set_claude_account(args.claude_account)
    rows = pool_rows()
    wanted = list(rows) if args.all else [x.strip() for x in args.ids.read_text().split() if x.strip()]
    unknown = [d for d in wanted if d not in rows]
    if unknown:
        p.error(f"{len(unknown)} ids are not in the pool, e.g. {unknown[:3]}")
    cache = load_cache()
    meta = load_meta()
    from_meta, revoked = 0, 0  # complete SPoRC metadata first: it replaces a model record, so the rule is uniform over the pool
    for d in wanted:
        m = meta.get(d, {})
        cached = cache.get(d)
        if not m.get("sporc_host_tags") and (cached is None or cached.get("source") != METADATA):
            continue  # no metadata to apply, nothing to re-check
        if cached is not None and cached.get("source") != METADATA and current(cached) and cached["kind"] != "conversation":
            continue  # the model found a radio play or a private call: the metadata's labels do not make it a conversation
        rec = from_metadata(d, word_shares(turns_of(read_text(INTERIM, rows[d]["offset"]))), m)  # re-evaluated every run
        if rec is None:
            if cached is not None and cached.get("source") == METADATA and current(cached):
                revoked += 1  # the rule no longer accepts it: a marker without FIELDS goes on disk, so the record is not
                append({"doc_id": d, "source": METADATA, "revoked": True,  # current for any reader; the model redoes it below
                        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")})
                cache.pop(d)
            continue
        if cached is not None and cached.get("source") == METADATA and (cached["hosts"], cached["guests"]) == (rec["hosts"], rec["guests"]):
            continue  # unchanged
        append(rec)
        cache[d] = rec
        from_meta += 1
    todo = [d for d in wanted if not current(cache.get(d))]  # records from earlier versions, and revoked ones, are redone
    print(f"{len(wanted)} episodes requested, {from_meta} taken from SPoRC metadata this run ({revoked} metadata records no "
          f"longer accepted), {len(todo)} to identify with {args.model}", flush=True)
    spent = 0.0

    def work(doc_id: str) -> dict:
        from src.synthesis_v2.cases import (
            clean_turns,  # cases imports this module; imported here to avoid the cycle
        )

        turns = clean_turns(turns_of(read_text(INTERIM, rows[doc_id]["offset"])))  # merged turns: the identifier's windows (unchanged since the pool was identified)
        m = meta.get(doc_id, {})
        return identify(doc_id, turns, m.get("sporc_host_tags", []), args.model, m.get("sporc_guest_tags", []))

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, d): d for d in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            doc_id = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001  (a failed call is reported and skipped; re-running fills it)
                failed += 1
                print(f"[{i}/{len(todo)}] {doc_id}: failed ({str(e)[:100]})", flush=True)
                continue
            append(rec)
            spent += rec["cost_usd"]
            flag = "" if rec["agrees_with_sporc"] in (None, True) else "  [disagrees with SPoRC]"
            print(f"[{i}/{len(todo)}] {doc_id} hosts={rec['hosts']} guests={rec['guests']} conf={rec['confidence']} "
                  f"cost=${spent:.2f}{flag}", flush=True)
    done = {d: r for d, r in load_cache().items() if current(r)}
    n_hosts = [len(done[d]["hosts"]) for d in wanted if d in done]
    for d in wanted:  # the non-conversations, listed so the false negatives can be checked by hand
        if d in done and done[d]["kind"] != "conversation":
            print(f"not a conversation ({done[d]['kind']}): {d}: {done[d]['reason'][:160]}", flush=True)
    all_hosts = sum(1 for d in wanted if d in done and done[d]["hosts"]
                    and set(done[d]["hosts"]) >= set(rows[d].get("tags_in_text", done[d]["hosts"])))
    n_guests = [len(done[d].get("guests", [])) for d in wanted if d in done]
    kinds = {k: sum(1 for d in wanted if d in done and done[d].get("kind") == k) for k in KINDS}
    meta_records = sum(1 for d in wanted if d in done and done[d].get("source") == METADATA)
    split = sum(1 for d in wanted if d in done and done[d].get("same_person"))
    print(json.dumps({"identified": len(n_hosts), "no_host": sum(1 for n in n_hosts if n == 0), "kind": kinds,
                      "from_sporc_metadata": meta_records,
                      "episodes_with_same_person_groups": split,
                      "every_speaker_a_host": all_hosts, "with_guests": sum(1 for n in n_guests if n > 0),
                      "guest_tags_total": sum(n_guests),
                      "sporc_guest_agreements": sum(1 for d in wanted if d in done and done[d].get("guests_agree_with_sporc") is True),
                      "sporc_guest_disagreements": sum(1 for d in wanted if d in done and done[d].get("guests_agree_with_sporc") is False),
                      "one_host": sum(1 for n in n_hosts if n == 1), "co_hosted": sum(1 for n in n_hosts if n > 1),
                      "sporc_agreements": sum(1 for d in wanted if d in done and done[d]["agrees_with_sporc"] is True),
                      "sporc_disagreements": sum(1 for d in wanted if d in done and done[d]["agrees_with_sporc"] is False),
                      "failed_this_run": failed, "cost_usd_this_run": round(spent, 2)}))


if __name__ == "__main__":
    main()
