"""CourtListener oral arguments -> data/interim/courtlistener/{train,val}.jsonl (Track P, tier 1).

Input: data/raw/courtlistener/<court>/transcripts.jsonl from scripts/download_courtlistener.py.

The `stt_transcript` field is flat recogniser output with no speaker turns, so
documents are written flat with has_speakers=False; inventing turn boundaries
would misrepresent the source. Arguments under 500 words are dropped (five in
ca1 were near-empty recordings). Val split: 2% of arguments by hash of id.

Usage: uv run python -m src.preprocessing.courtlistener
"""

import json

from src.preprocessing.common import RAW, Document, Writer

MIN_WORDS = 500
ASR_SYSTEM = "CourtListener stt_source=1 (Whisper-family, as distributed)"


def main() -> None:
    writer = Writer("courtlistener")
    dropped = 0
    for path in sorted((RAW / "courtlistener").glob("*/transcripts.jsonl")):
        court = path.parent.name
        for line in path.open():
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            n_words = len(text.split())
            if n_words < MIN_WORDS:
                dropped += 1
                continue
            doc_id = f"{court}-{r['id']}"
            writer.add(
                Document(
                    source="courtlistener", doc_id=doc_id, track="track-p", tier=1,
                    asr_system=ASR_SYSTEM, has_speakers=False, n_turns=0, n_words=n_words,
                    text=text,
                    meta={"court": court, "case_name": r.get("case_name", ""),
                          "duration_s": r.get("duration"), "mp3": r.get("download_url", "")},
                ),
                doc_id,
            )
    print(f"dropped under {MIN_WORDS} words: {dropped}")
    writer.close()


if __name__ == "__main__":
    main()
