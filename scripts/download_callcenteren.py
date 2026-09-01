"""Download CallCenterEN (AIxBlock 92k call-center transcripts).

Source:  https://huggingface.co/datasets/AIxBlock/92k-real-world-call-center-scripts-english
Licence: CC BY-NC-4.0 (card tag; Track NC). Ungated (verified 2026-08-31,
         contrary to the survey's expectation).
Tier 1:  transcripts are AssemblyAI ASR output (paper arXiv:2507.02958);
         audio is withheld by the publisher, so there is no tier-2 path.

Output: data/raw/callcenteren/
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "callcenteren"
REPO_ID = "AIxBlock/92k-real-world-call-center-scripts-english"


def main() -> None:
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=RAW_DIR,
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
