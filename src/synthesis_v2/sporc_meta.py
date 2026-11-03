"""Episode-level metadata for the v2 pool, pulled once from the raw SPoRC parquet files.

    uv run python -m src.synthesis_v2.sporc_meta

Writes data/labelled_data_v2/episode_meta.jsonl, one line per pool episode: title, description, SPoRC's own predicted
host and guest names and its `host_speaker_labels` (name -> diarisation tag, present for about a fifth of the pool).
The identifier stores SPoRC's host tags beside its own answer as a cross-check. The raw files are large (140 parquet parts), so this runs once per pool;
the output is regenerable and untracked.
"""

import ast
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.synthesis_v2 import OUT_DIR, REPO_ROOT

PARTS = REPO_ROOT / "data" / "raw" / "sporc" / "episodes"
POOL = OUT_DIR / "episodes.jsonl"
META = OUT_DIR / "episode_meta.jsonl"
COLUMNS = ["episode_id", "ep_title", "ep_description", "host_predicted_names", "guest_predicted_names",
           "host_speaker_labels", "guest_speaker_labels"]


def parse_labels(x) -> dict[str, str]:
    """SPoRC stores `host_speaker_labels` as a Python-literal string ({"Name": "SPEAKER_01"}) or an empty '{}'."""
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items()}
    if not x:
        return {}
    try:
        d = ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return {}
    return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}


def parse_names(x) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if not x:
        return []
    try:
        d = ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []
    return [str(v) for v in d] if isinstance(d, list) else []


def extract(doc_ids: set[str], parts: Path = PARTS) -> dict[str, dict]:
    want = pa.array(sorted(doc_ids))
    found: dict[str, dict] = {}
    for f in sorted(glob.glob(str(parts / "part-*.parquet"))):
        t = pq.read_table(f, columns=COLUMNS)
        for r in t.filter(pc.is_in(t["episode_id"], value_set=want)).to_pylist():
            found[r["episode_id"]] = {
                "doc_id": r["episode_id"], "ep_title": (r["ep_title"] or "").strip(),
                "ep_description": (r["ep_description"] or "").strip(),
                "host_predicted_names": parse_names(r["host_predicted_names"]),
                "guest_predicted_names": parse_names(r["guest_predicted_names"]),
                "sporc_host_tags": sorted(set(parse_labels(r["host_speaker_labels"]).values())),
                "sporc_guest_tags": sorted(set(parse_labels(r["guest_speaker_labels"]).values())),
            }
        if len(found) >= len(doc_ids):
            break
    return found


def load_meta(path: Path = META) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["doc_id"]: r for r in (json.loads(line) for line in path.open())}


def main() -> None:
    ids = {json.loads(line)["doc_id"] for line in POOL.open()}
    found = extract(ids)
    with META.open("w") as f:
        for doc_id in sorted(ids):
            if doc_id in found:
                f.write(json.dumps(found[doc_id], ensure_ascii=False) + "\n")
    with_host = sum(1 for r in found.values() if r["sporc_host_tags"])
    with_desc = sum(1 for r in found.values() if len(r["ep_description"]) >= 80)
    print(json.dumps({"pool": len(ids), "found": len(found), "with_sporc_host_tag": with_host,
                      "with_description_80_chars": with_desc}))


if __name__ == "__main__":
    main()
