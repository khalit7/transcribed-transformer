#!/usr/bin/env bash
# Download PriMock57 (SURVEYSHEET tranche 1, item 2).
#
# Source:  https://github.com/babylonhealth/primock57
# Licence: CC BY 4.0 (LICENSE.md in the repo; Track P). Audio ships via Git-LFS.
#
# Output: data/raw/primock57/  (the clone; .git kept for provenance/pinning)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw/primock57"

if [ ! -d "$RAW_DIR/.git" ]; then
    git clone https://github.com/babylonhealth/primock57.git "$RAW_DIR"
fi
cd "$RAW_DIR"
git lfs install --local
git lfs pull
git rev-parse HEAD
du -sh .
