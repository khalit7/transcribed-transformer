#!/usr/bin/env bash
# Download ACI-Bench (SURVEYSHEET tranche 1, item 3).
#
# Source:  https://github.com/microsoft/clinical_visit_note_summarization_corpus
# Licence: CC BY 4.0 (LICENSE file in the repo; Track P).
#
# Output: data/raw/aci_bench/  (the clone; .git kept for provenance/pinning)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw/aci_bench"

if [ ! -d "$RAW_DIR/.git" ]; then
    git clone https://github.com/microsoft/clinical_visit_note_summarization_corpus.git "$RAW_DIR"
fi
cd "$RAW_DIR"
git rev-parse HEAD
du -sh .
