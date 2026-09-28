#!/usr/bin/env python3
"""Review gate: hash the state that can affect a run and refuse a launch unless that hash was approved.

    gate.py hash                       print the current tree hash
    gate.py approve --note "..."       record the current hash as reviewed (appends to checkpoints/.reviews/approved.jsonl)
    gate.py check [queue-script ...]   exit 0 if the current hash is approved, else print what differs from the last approval and exit 1

The hash covers every file (tracked or not, .gitignore respected) under src/, configs/, tests/, plus pyproject.toml,
uv.lock and any extra paths given (queue scripts). Contents are hashed, not timestamps.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REVIEWS = ROOT / "checkpoints" / ".reviews" / "approved.jsonl"
ROOTS = ["src", "configs", "tests", "pyproject.toml", "uv.lock"]


def files(extra: list[str]) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *ROOTS], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    paths = sorted({ROOT / p for p in out if p and not p.endswith(".pyc") and "__pycache__" not in p})
    for e in extra:
        p = Path(e).resolve()
        if p.exists():
            paths.append(p)
    return paths


def digest(paths: list[Path]) -> tuple[str, dict[str, str]]:
    per: dict[str, str] = {}
    h = hashlib.sha256()
    for p in paths:
        if not p.is_file():
            continue
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        key = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)
        per[key] = d
        h.update(key.encode()); h.update(d.encode())
    return h.hexdigest(), per


def approvals() -> list[dict]:
    if not REVIEWS.exists():
        return []
    return [json.loads(l) for l in REVIEWS.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["hash", "approve", "check"])
    ap.add_argument("extra", nargs="*", help="queue scripts or other files that take part in the launch")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tree, per = digest(files(a.extra))
    if a.cmd == "hash":
        print(tree)
        return
    if a.cmd == "approve":
        if not a.note:
            sys.exit("approve needs --note describing the review (reviewers, findings, fixes)")
        REVIEWS.parent.mkdir(parents=True, exist_ok=True)
        with REVIEWS.open("a") as f:
            f.write(json.dumps({"hash": tree, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "note": a.note, "files": per}) + "\n")
        print(f"approved {tree[:12]} ({len(per)} files)")
        return
    ok = [r for r in approvals() if r["hash"] == tree]
    if ok:
        print(f"review gate: approved {tree[:12]} on {ok[-1]['time']}: {ok[-1]['note'][:120]}")
        return
    last = approvals()[-1] if approvals() else None
    print(f"review gate: REFUSED. Current state {tree[:12]} has no two-model review approval.", file=sys.stderr)
    if last:
        changed = sorted(k for k in set(per) | set(last["files"]) if per.get(k) != last["files"].get(k))
        print(f"  last approval {last['hash'][:12]} on {last['time']}; differing files: {', '.join(changed[:40])}", file=sys.stderr)
    print("  Run the pre-run-review skill (Codex gpt-6-astra + Opus), fix or reject the findings, then `gate.py approve --note ...`.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
