"""Download CourtListener oral-argument ASR transcripts (SURVEYSHEET tranche 1).

Source:  https://www.courtlistener.com/api/rest/v4/audio/
Licence: Free Law Project bulk data is released free of known copyright
         restrictions (public domain mark); federal oral arguments are
         government works. Track P.

Tier 1: the `stt_transcript` field is real ASR output (CourtListener runs an
open Whisper-family recogniser; `stt_source` records which). Flat text, no
speaker turns — diarisation would need the MP3s and a local pass, which is
deferred under the tier-1-first policy.

Resumable: the API cursor is persisted after every page. Anonymous access is
rate-limited, so pages are fetched politely; pass --court to extend beyond
SCOTUS later.

Output: data/raw/courtlistener/<court>/transcripts.jsonl (one record per line:
id, case_name, docket, date, duration, stt_source, download_url, text)
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API = "https://www.courtlistener.com/api/rest/v4/audio/"
FIELDS = (
    "id,case_name,docket,date_created,duration,stt_status,stt_source,"
    "download_url,stt_transcript"
)


def fetch(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "transcribed-transformer corpus fetch"}
    )
    while True:
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After") or 300)
                print(f"rate limited; sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            if e.code >= 500:
                print(f"server error {e.code}; sleeping 60s", flush=True)
                time.sleep(60)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"network error {e}; sleeping 60s", flush=True)
            time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--court", default="scotus")
    parser.add_argument("--sleep", type=float, default=10.0, help="seconds between pages")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "data" / "raw" / "courtlistener" / args.court
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transcripts.jsonl"
    cursor_path = out_dir / "cursor.txt"

    if cursor_path.exists():
        url = cursor_path.read_text().strip()
        mode = "a"
    else:
        params = {
            "docket__court": args.court,
            "stt_status": "1",
            "fields": FIELDS,
            "page_size": "20",
        }
        url = API + "?" + urllib.parse.urlencode(params)
        mode = "w"

    n = 0
    with open(out_path, mode) as out:
        while url:
            page = fetch(url)
            for r in page["results"]:
                rec = {
                    "id": r["id"],
                    "case_name": r.get("case_name", ""),
                    "docket": r.get("docket", ""),
                    "date_created": r.get("date_created", ""),
                    "duration": r.get("duration"),
                    "stt_source": r.get("stt_source"),
                    "download_url": r.get("download_url", ""),
                    "text": r.get("stt_transcript", ""),
                }
                out.write(json.dumps(rec) + "\n")
                n += 1
            out.flush()
            url = page.get("next") or ""
            cursor_path.write_text(url)
            print(f"{n} records", flush=True)
            if url:
                time.sleep(args.sleep)

    print(f"done: {n} new records -> {out_path}")


if __name__ == "__main__":
    main()
