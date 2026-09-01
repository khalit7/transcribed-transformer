"""Download the AppTek Call-Center Dialogues corpus (SURVEYSHEET tranche 1, item 1).

Source: https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues
Licence: CC BY-SA 4.0 (dataset card metadata; Track P, SA flag). Ungated.

Layout in the source repo:
  test/<locale>/         split-channel WAV (one speaker per file) + metadata.jsonl
  diarization/<locale>/  merged mono WAV (one per call) + metadata.jsonl

Downloads resume natively via huggingface_hub. Revision is pinned so the raw
copy is reproducible.

Output: data/raw/apptek_callcenter/
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "apptek_callcenter"
REPO_ID = "apptek-com/apptek_callcenter_dialogues"
REVISION = "b98967d9946f7f59f58d08624a2a00fe98fe0219"  # 2026-08-25 state, checked 2026-08-31


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        choices=["metadata", "audio", "all"],
        default="metadata",
        help="metadata: transcripts + docs only (small). audio: WAVs too.",
    )
    args = parser.parse_args()

    patterns = ["*.jsonl", "README.md", "*.py", ".gitattributes"]
    if args.what in ("audio", "all"):
        patterns.append("*.wav")

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=RAW_DIR,
        allow_patterns=patterns,
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
