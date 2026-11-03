#!/usr/bin/env bash
# Production labelling of labelled data v2 (D82): every approved bank question asked of every selected episode by Opus
# through the lean `claude -p` route (one call per episode, shared definitions printed once, one-line system prompt, no
# tools, prompt caching off), billed to the Claude subscription account given. The review gate is checked before each
# pass, over the code and also this run's inputs (the episode list, the pool rows and the speaker cache), so an input
# edited mid-run stops the next pass; a refused gate stops the script (re-approve, then rerun it). Resumable:
# label_episodes skips every (episode, question) already in the output, so the second pass retries only the labels that
# failed, and rerunning the script after any stop picks up where it ended. A usage limit (5-hour or weekly) is waited
# out inside the labeller (src/synthesis/llm.py), until the reset the CLI reports, however far off; a refusal without a
# reset time is backed off for up to 6 h and then fails its episode, which a later pass or rerun retries. One launch at a time: a lock on the output file refuses a second launch (two launches
# would label every episode twice).
#
#   nohup scripts/label_v2.sh p 2 >> checkpoints/label_v2.log 2>&1 &     # account p, 2 workers
set -euo pipefail
cd "$(dirname "$0")/.."
ACCOUNT=${1:?account p or w}
WORKERS=${2:-2}
SELF=scripts/label_v2.sh
GATE="python3 .claude/skills/pre-run-review/scripts/gate.py"
IDS=data/labelled_data_v2/selected_ids.txt
INPUTS="$IDS data/labelled_data_v2/episodes.jsonl data/labelled_data_v2/speakers/sporc.jsonl"
OUT=data/labelled_data_v2/labelled_data.jsonl
log() { echo "[$(date '+%F %T')] $*"; }
trap 'log "stopped with exit code $? (rerun the script to resume)"' ERR

exec 9>"$OUT.lock"
flock -n 9 || { log "another labelling launch holds $OUT.lock; stopping"; exit 1; }

for pass in 1 2; do
    # shellcheck disable=SC2086
    $GATE check "$SELF" $INPUTS >/dev/null || { log "review gate refused before pass $pass; stopping (re-approve, then rerun)"; exit 1; }
    log "pass $pass: labelling $(wc -l < "$IDS") episodes into $OUT (account $ACCOUNT, $WORKERS workers)"
    uv run --no-sync python -m src.synthesis_v2.label_episodes claude:opus --lean --ids "$IDS" --seed 0 \
        --out "$OUT" --claude-account "$ACCOUNT" --workers "$WORKERS"
done
log "done: $( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0) of $(( $(wc -l < "$IDS") * 44 )) labels in $OUT (rerun the script to retry any missing)"
