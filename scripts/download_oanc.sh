#!/usr/bin/env bash
# Download the Open American National Corpus (SURVEYSHEET tranche 1, item 4).
# The spoken sections (Switchboard telephone conversations, Charlotte narratives,
# face-to-face) are what this project wants; they ship inside the full archive.
#
# Source:  https://anc.org/data/oanc/download/
# Licence: ANC terms on that page: "freely available for download and use for
#          research and development, including commercial development" (Track P).
#          anc.org's TLS certificate is expired (checked 2026-08-31), so the
#          licence page is archived alongside the download and the archive's
#          sha256 is recorded in the datasheet.
#
# Output: data/raw/oanc/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw/oanc"
mkdir -p "$RAW_DIR"

# -k: anc.org serves an expired certificate; integrity is recorded via sha256 below.
curl -k -L --retry 3 -C - -o "$RAW_DIR/LICENSE_PAGE.html" "https://anc.org/data/oanc/download/"
curl -k -L --retry 3 -C - -o "$RAW_DIR/OANC-1.0.1-UTF8.zip" "https://www.anc.org/OANC/OANC-1.0.1-UTF8.zip"

sha256sum "$RAW_DIR/OANC-1.0.1-UTF8.zip" | tee "$RAW_DIR/OANC-1.0.1-UTF8.zip.sha256"
du -sh "$RAW_DIR"
