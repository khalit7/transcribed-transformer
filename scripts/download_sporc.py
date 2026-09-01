"""Download SPoRC text layers (SURVEYSHEET remaining-tier-1 rank 3).

Source:  https://huggingface.co/datasets/blitt/SPoRC (gated `auto`; Khalid's
         HF account accepted the terms 2026-09-01)
Licence: research/non-commercial gate -> Track NC.
Tier 1:  Whisper transcripts of ~1.1M podcast episodes with diarised speaker
         turns (the `turns/` parquet layer, 185M+ turns).

Fetches text only: turns/ (13.5 GiB) and episodes/ (14.9 GiB) parquet plus
manifest/READMEs. The acoustics/ and bulk metadata layers are skipped.

Output: data/raw/sporc/
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "sporc"


def main() -> None:
    path = snapshot_download(
        repo_id="blitt/SPoRC",
        repo_type="dataset",
        local_dir=RAW_DIR,
        allow_patterns=[
            "turns/*",
            "episodes/*",
            "manifest.json",
            "README.md",
            "metadata/*.md",
        ],
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
