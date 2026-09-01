#!/usr/bin/env bash
# Download the ICSI meeting corpus annotations (SURVEYSHEET tranche 1, item 5).
#
# Source:  https://groups.inf.ed.ac.uk/ami/icsi/
# Licence: CC BY 4.0 — https://groups.inf.ed.ac.uk/ami/icsi/license.shtml:
#          "The ICSI corpus and its annotations are released under the Creative
#          Commons Attribution 4.0 license agreement (also called CC BY 4.0)."
#
# Audio (headset mix WAV per meeting, ~120MB x 74 meetings) is fetched
# separately when the self-ASR pass runs; this script gets the transcripts.
#
# Output: data/raw/icsi/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw/icsi"
mkdir -p "$RAW_DIR"

BASE="https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations"
for f in ICSI_core_NXT.zip ICSI_original_transcripts.zip; do
    curl -L --retry 3 -C - -o "$RAW_DIR/$f" "$BASE/$f"
done
sha256sum "$RAW_DIR"/*.zip | tee "$RAW_DIR/checksums.sha256"
du -sh "$RAW_DIR"
