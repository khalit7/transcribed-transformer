"""Download MeetingBank transcripts (city-council meetings, diarised ASR).

Source:  https://huggingface.co/datasets/huuuyeah/meetingbank (transcripts +
         summaries; audio lives in huuuyeah/MeetingBank_Audio and is NOT
         fetched here under the tier-1-first policy).
Licence: CC BY-NC-SA 4.0 (Track NC).
Tier 1:  transcripts are Speechmatics ASR output with speaker segments.

Output: data/raw/meetingbank/
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "meetingbank"
REPO_ID = "huuuyeah/meetingbank"


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=RAW_DIR,
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
