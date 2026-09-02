"""CallCenterEN -> data/interim/callcenteren/{train,val}.jsonl (Track NC, tier 1).

Input: data/raw/callcenteren/*.zip, per-call JSON (AssemblyAI output: flat
`text`, word timings with speaker=null throughout, redacted PII policies).

Decisions:
- Flat text, has_speakers=False: no speaker labels exist in the release.
- Dedupe by transcript id (the JSON stem): two "(reupload)" zips overlap the
  originals, which is why 95,953 files exceed the paper's 91,706 calls.
- PII placeholders (`[PERSON_NAME]`, `[DATE]`...) are kept verbatim in the text
  but counted into meta so a mixture can down-weight or mask them; they are a
  redaction artefact, not speech, and must not be learned as PII patterns.
- Calls under 50 words are dropped (IVR fragments).

Usage: uv run python -m src.preprocessing.callcenteren
"""

import json
import re
import zipfile

from src.preprocessing.common import RAW, Document, Writer

MIN_WORDS = 50
PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
ASR_SYSTEM = "AssemblyAI (as distributed; paper arXiv:2507.02958)"


def main() -> None:
    writer = Writer("callcenteren")
    seen: set[str] = set()
    dupes = short = 0
    for zpath in sorted((RAW / "callcenteren").glob("*.zip")):
        domain = zpath.stem.split(")")[-1].strip()
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.endswith(".json") or name.startswith("__MACOSX"):
                    continue
                stem = name.rsplit("/", 1)[-1].replace("_transcript.json", "").replace(".json", "")
                if stem in seen:
                    dupes += 1
                    continue
                seen.add(stem)
                try:
                    r = json.loads(zf.read(name))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                text = (r.get("text") or "").strip()
                n_words = len(text.split())
                if n_words < MIN_WORDS:
                    short += 1
                    continue
                writer.add(
                    Document(
                        source="callcenteren", doc_id=stem, track="track-nc", tier=1,
                        asr_system=ASR_SYSTEM, has_speakers=False, n_turns=0, n_words=n_words,
                        text=text,
                        meta={"domain_zip": domain, "audio_duration_s": r.get("audio_duration"),
                              "confidence": r.get("confidence"),
                              "n_pii_placeholders": len(PLACEHOLDER.findall(text))},
                    ),
                    stem,
                )
    print(f"duplicates skipped: {dupes}; under {MIN_WORDS} words: {short}")
    writer.close()


if __name__ == "__main__":
    main()
