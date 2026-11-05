#!/usr/bin/env bash
# The transfer-grid queue (experiments/2026-10-28-transfer-grid/): trains, generates and scores transfer runs in order, and
# scores the existing compliance checkpoints zero-shot on the transfer benchmarks. Every GPU stage is preceded by the
# review gate, which refuses unless the current tree (src/, configs/, tests/, pyproject.toml, uv.lock and this script)
# is an approved hash; a stage that fails aborts the queue; nothing is scored unless every benchmark pair was generated
# exactly once. Resumable: a run whose metrics.json exists is skipped.
#
#   scripts/queue_transfer.sh run configs/transfer/ecthr/pilot-e1-gemma3-1b-pt-ecthr.yaml [more configs ...]
#   scripts/queue_transfer.sh zeroshot ecthr scotus quality      # the nine compliance checkpoints on each benchmark
#
# Generation renders the prompt the run trained on and cuts documents from the tail so prompt plus answer budget fit the
# 16k cap: transfer-trained runs get 64 answer tokens (--truncate-to 16320; training kept prompts to 16384 minus the
# 17-token target, so a cut prompt at generation is at most 47 tokens shorter than at training), the compliance
# checkpoints scored zero-shot get the compliance budget of 512 (they write an evidence list and a summary; 64 would cut
# most of them off and score the cap, not the model). Decoders generate through vLLM in its own environment; native
# encoder-decoders and stitches through their training-environment backends. Scoring also slices each benchmark by
# whether the full prompt exceeded the cap, so the uncut records can be read on their own.
set -euo pipefail
cd "$(dirname "$0")/.."
GATE="python3 .claude/skills/pre-run-review/scripts/gate.py"
SELF=scripts/queue_transfer.sh
GEN=scripts/make_transfer_configs.py  # hashed with the queue: the configs it writes are hashed too, but its arm table decides what runs
LOG=checkpoints/queue_transfer.log
CAP=16384
MAX_TOKENS=64        # transfer-trained runs: targets are 14 to 17 tokens
ZS_MAX_TOKENS=512    # the compliance checkpoints write an evidence list and a summary (p50 over 80 tokens); the compliance budget
# expected benchmark build per dataset: records and unseen_q records (the ECtHR build is task A with Articles 8 and 10
# held out; a rebuild with other arguments would silently change the cell, so the queue refuses to run on it)
declare -A EXPECT_RECORDS=([ecthr]=10000 [scotus]=1400 [quality]=2086)
declare -A EXPECT_UNSEEN=([ecthr]=2000 [scotus]=0 [quality]=2086)
# the questions that make up the unseen cell ("*" = every question is unseen, as on QuALITY)
declare -A EXPECT_UNSEEN_QIDS=([ecthr]="ecthr-a-10 ecthr-a-8" [scotus]="" [quality]="*")
# the compliance checkpoints the zero-shot stage scores (their training run directories)
ARMS="e1-gemma3-270m e1-gemma3-1b-pt e1-gemma3-4b-pt e4-t5gemma2-270m-270m e4-t5gemma2-1b-1b e4mm-270m-enc-1b-dec e4mm-270m-enc-4b-dec e4mm-1b-enc-270m-dec e4mm-4b-enc-270m-dec"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
gate() { $GATE check "$SELF" "$GEN" >/dev/null || { log "review gate refused before $1; stopping"; $GATE check "$SELF" "$GEN" | tee -a "$LOG"; exit 1; }; }

# name arch base encoder_base wandb_project, read through the config loader so defaults apply
cfg_fields() { uv run --no-sync python -c "
import sys; from pathlib import Path; from src.train.config import load_config
c = load_config(Path(sys.argv[1])); print(c.name, c.model.arch, c.model.base, c.model.encoder_base or '-', c.wandb_project)" "$1"; }

# the generation command for a checkpoint directory, by architecture, into GEN_CMD. The answer budget sizes the prompt
# cut: prompt + answer never exceeds the 16k the run trained at. A 4B donor on either side of a stitch needs the small
# batch budget (the 4B decoder's cross-attention cache overflowed the 160k budget on the compliance benchmark).
gen_cmd() {  # $1 model_dir $2 arch $3 base $4 encoder_base $5 split $6 out $7 max_tokens
    local trunc common
    trunc=$((CAP - $7))  # separate statements: bash expands every word of one `local` before any assignment
    common="$5 $6 --variants clean --max-tokens $7 --truncate-to $trunc"
    case "$2" in
        decoder) GEN_CMD=".venv-vllm/bin/python -m src.train.generate $1 $common --backend vllm --max-model-len $CAP" ;;
        hf_encdec) GEN_CMD="uv run --no-sync python -m src.train.generate $1 $common --backend hf_encdec" ;;
        stitched)
            local bt=160000; [[ "$3" == *4b* || "$4" == *4b* ]] && bt=40000
            GEN_CMD="uv run --no-sync python -m src.train.generate $1 $common --backend stitched --batch-tokens $bt" ;;
        soft)  # E5-soft-tokens: the soft-token stitch into a Gemma 3 decoder; base is the adapted SoftStitched export, its name carries the encoder size
            local bt=160000; [[ "$3" == *4b* || "$4" == *4b* ]] && bt=40000
            GEN_CMD="uv run --no-sync python -m src.train.generate $1 $common --backend soft --batch-tokens $bt" ;;
        *) log "no generation backend for arch $2 ($1); stopping"; exit 1 ;;
    esac
}

# the split on disk is the build the record describes: record count, unseen count and which questions are unseen
check_build() {  # $1 dataset $2 split
    uv run --no-sync python -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
unseen = sorted({r['question']['id'] for r in rows if r['cell'] == 'unseen_q'})
n_unseen = sum(r['cell'] == 'unseen_q' for r in rows)
want = sys.argv[4].split()
qids_ok = True if want == ['*'] else unseen == sorted(want)
print(f'{len(rows)} records, {n_unseen} unseen_q over {len(unseen)} questions (expected {sys.argv[2]}, {sys.argv[3]}, {want})')
sys.exit(0 if (len(rows), n_unseen) == (int(sys.argv[2]), int(sys.argv[3])) and qids_ok else 1)" \
        "$2" "${EXPECT_RECORDS[$1]}" "${EXPECT_UNSEEN[$1]}" "${EXPECT_UNSEEN_QIDS[$1]}"
}

# a step-* checkpoint about to be resumed was written by this very config (name, data, recipe), not by an earlier one
resume_matches() {  # $1 config $2 run_dir
    uv run --no-sync python -c "
import json, sys
from pathlib import Path
from src.train.config import load_config
steps = sorted(Path(sys.argv[2]).glob('step-*'), key=lambda p: int(p.name.split('-')[1]))
if not steps: sys.exit(0)
saved = json.loads((steps[-1] / 'state.json').read_text())['config']
now = load_config(Path(sys.argv[1])).model_dump(mode='json')
diff = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
print(f'{steps[-1]}: config ' + ('matches' if not diff else 'differs in ' + ', '.join(diff)))
sys.exit(1 if diff else 0)" "$1" "$2"
}

# every (id, variant) of the split present exactly once in the outputs
complete() {  # $1 split $2 outputs
    uv run --no-sync python -c "
import json, sys, collections
ids = [json.loads(l)['id'] for l in open(sys.argv[1])]
got = collections.Counter((json.loads(l)['id'], json.loads(l)['variant']) for l in open(sys.argv[2]))
missing = [i for i in ids if (i, 'clean') not in got]; dup = [k for k, n in got.items() if n > 1]
print(f'{len(ids)} pairs, {len(got)} outputs, {len(missing)} missing, {len(dup)} duplicated')
sys.exit(1 if missing or dup else 0)" "$1" "$2"
}

stage() {  # $1 label, rest: the command
    local label=$1; shift
    log "START $label"
    if "$@" >> "$LOG.$(echo "$label" | tr ' /' '__')" 2>&1; then log "END $label rc=0"; else local rc=$?; log "END $label rc=$rc; queue aborted"; exit $rc; fi
}

score() {  # $1 run_dir $2 dataset $3 split $4 wandb_project $5 prefix ("" for a run's own benchmark) $6 max_tokens
    local eval_dir gen prefix_arg=()
    if [[ -z "$5" ]]; then eval_dir=$1/eval; else eval_dir=$1/eval/$5; prefix_arg=(--wandb-prefix "$5/bench/"); fi
    gen=$eval_dir/gen_benchmark.jsonl
    gate "score $1 $2"
    stage "complete $gen" complete "$3" "$gen"
    stage "score $1 $2" uv run --no-sync python -m src.train.evaluate "$3" "$gen" "$eval_dir/benchmark" \
        --cap-tokens $((CAP - $6)) --tokenizer "$1/final" \
        --wandb-run "$(cat "$1/wandb_id")" --wandb-project "$4" "${prefix_arg[@]}"
}

# a run's final export is complete: train_config.json is the last file every export path writes (a crash mid-export
# leaves config.json or stitched.json beside partial weights, but never this file)
trained() { [[ -f $1/final/train_config.json ]]; }

# vLLM loads a Gemma 3 4B export as Gemma3ForConditionalGeneration and wants the base model's processor files beside it
# (preprocessor_config.json, processor_config.json); the trainer's export writes the model and tokenizer only (the
# compliance 4B export carried copies identical to the base's). Copy whatever the base model's cached snapshot has and
# the export lacks; model weights and tokenizer are never touched.
ensure_processor_files() {  # $1 final_dir $2 base model id
    uv run --no-sync python -c "
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
final, base = Path(sys.argv[1]), sys.argv[2]
names = ['preprocessor_config.json', 'processor_config.json', 'special_tokens_map.json', 'added_tokens.json']
try:
    snap = Path(snapshot_download(base, local_files_only=True, allow_patterns=names))
except Exception as e:  # a local checkpoint as base, or nothing cached: nothing to copy
    print(f'no cached snapshot for {base}: {e}'); sys.exit(0)
for n in names:
    if (snap / n).exists() and not (final / n).exists():
        shutil.copy(snap / n, final / n); print(f'copied {n} from {base}')
" "$1" "$2"
}

run() {  # $1 config
    local cfg=$1 name arch base enc proj ds
    read -r name arch base enc proj < <(cfg_fields "$cfg") || { log "cannot read config $cfg; stopping"; exit 1; }
    ds=$(basename "$(dirname "$cfg")")
    local split=data/labelled_data/$ds/benchmark.jsonl
    stage "check build $ds" check_build "$ds" "$split"
    # scored and logged: the evaluator writes wandb.json only after the numbers reached the run
    if [[ -f checkpoints/$name/eval/benchmark/wandb.json ]]; then log "skip $name: already scored and logged"; return; fi
    if ! trained "checkpoints/$name"; then
        gate "train $name"
        # expandable_segments: the CUDA caching allocator grows segments instead of fragmenting them. The 4B FSDP arms run
        # within 0.5 GB of the 32 GB cards; e1-gemma3-4b-pt-ecthr went out of memory in backward right after a resume
        # (2026-10-29 15:51, 118 MiB free) on a stretch of steps it had passed before the interruption, so the margin is
        # fragmentation, not the batch. A memory-layout setting only; it does not change the arithmetic.
        local alloc=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        stage "smoke $name" env WANDB_MODE=disabled $alloc uv run --no-sync torchrun --nproc_per_node 2 -m src.train.train "$cfg" --smoke 3
        # --resume picks up the latest step-* checkpoint of an interrupted attempt and is a no-op on a fresh run; the
        # checkpoint must have been written by this config (a changed config means delete the run directory, not resume)
        stage "resume check $name" resume_matches "$cfg" "checkpoints/$name"
        stage "train $name" env WANDB_MODE=online $alloc uv run --no-sync torchrun --nproc_per_node 2 -m src.train.train "$cfg" --resume
    else log "train $name: final export exists, not retraining"; fi
    if [[ ! -f checkpoints/$name/eval/gen_benchmark.jsonl ]] || ! complete "$split" checkpoints/$name/eval/gen_benchmark.jsonl >/dev/null 2>&1; then
        gate "gen $name"
        [[ "$arch" == decoder ]] && stage "processor files $name" ensure_processor_files "checkpoints/$name/final" "$base"
        gen_cmd "checkpoints/$name/final" "$arch" "$base" "$enc" "$split" "checkpoints/$name/eval/gen_benchmark.jsonl" "$MAX_TOKENS"
        # shellcheck disable=SC2086
        stage "gen $name" $GEN_CMD
    fi
    score "checkpoints/$name" "$ds" "$split" "$proj" "" "$MAX_TOKENS"
}

zeroshot() {  # datasets...
    for ds in "$@"; do
        local split=data/labelled_data/$ds/benchmark.jsonl
        stage "check build $ds" check_build "$ds" "$split"
        for arm in $ARMS; do
            local dir=checkpoints/$arm cfg
            trained "$dir" || { log "zeroshot: no final export for $arm; skipped"; continue; }
            if [[ -f $dir/eval/zeroshot-$ds/benchmark/wandb.json ]]; then log "skip zeroshot $arm $ds: scored and logged"; continue; fi
            cfg=$(grep -l "^name: $arm$" configs/e1/*.yaml configs/e4/*.yaml | head -1 || true)
            [[ -n "$cfg" ]] || { log "zeroshot: no compliance config named $arm; stopping"; exit 1; }
            read -r _ arch base enc proj < <(cfg_fields "$cfg") || { log "cannot read config $cfg; stopping"; exit 1; }
            gate "zeroshot gen $arm $ds"
            gen_cmd "$dir/final" "$arch" "$base" "$enc" "$split" "$dir/eval/zeroshot-$ds/gen_benchmark.jsonl" "$ZS_MAX_TOKENS"
            # shellcheck disable=SC2086
            stage "zeroshot gen $arm $ds" $GEN_CMD
            score "$dir" "$ds" "$split" "$proj" "zeroshot-$ds" "$ZS_MAX_TOKENS"
        done
    done
}

mkdir -p checkpoints
case "${1:-}" in
    run) shift; for c in "$@"; do run "$c"; done; log "QUEUE DONE ($#)" ;;
    zeroshot) shift; zeroshot "$@"; log "ZEROSHOT DONE" ;;
    *) echo "usage: $SELF run <config.yaml>... | zeroshot <dataset>..."; exit 2 ;;
esac
