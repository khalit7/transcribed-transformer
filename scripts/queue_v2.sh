#!/usr/bin/env bash
# The labelled-data-v2 grid queue (experiments/2026-10-31-v2-long-podcast/): trains, generates and scores the v2 arms in
# order. Every GPU stage, and the scoring step, is preceded by the review gate, which refuses unless the current tree
# (src/, configs/, tests/, the prompt folders, pyproject.toml, uv.lock, this script and the config generator) is an
# approved hash; a stage that fails aborts the queue; nothing is scored unless every benchmark pair was generated exactly
# once. Resumable: a run already scored and logged is skipped, a trained run is not retrained, an interrupted run resumes
# from its last checkpoint if that was written by the same config.
#
#   scripts/queue_v2.sh run configs/v2/e1-gemma3-270m-v2.yaml   # arm 1 alone: the pre-registered budget check
#   echo keep > checkpoints/queue_v2.budget                    # the decision on arm 1 (keep | raise), written by hand
#   scripts/queue_v2.sh all                                     # every arm, in the order scripts/make_v2_configs.py gives
#
# `all` refuses until the budget decision is written, and a finished arm is only skipped or reused if its exported
# training config equals the current one: a budget raised after arm 1 makes the queue stop at arm 1 rather than report an
# arm trained on 8 questions per episode beside arms trained on 16.
#
# Generation renders the prompt the run trained on (src.train.data.record_prompt, the v2 template and line format) and
# never cuts a transcript: evidence is a set of line numbers, so a cut could remove the lines the answer rests on. The
# longest benchmark prompt is 15,428 Gemma 3 tokens and the longest answer in the data 211 (measured 2026-11-05), so a
# 512-token answer budget fits the 16,384 cap without cutting; the evaluator flags any prompt over 16,384 - 512 anyway.
# Decoders generate through vLLM in its own environment; native encoder-decoders and stitches through their
# training-environment backends.
set -euo pipefail
cd "$(dirname "$0")/.."
GATE="python3 .claude/skills/pre-run-review/scripts/gate.py"
SELF=scripts/queue_v2.sh
GEN=scripts/make_v2_configs.py  # hashed with the queue: its ORDER decides what runs and in which order
LOG=checkpoints/queue_v2.log
CAP=16384
MAX_TOKENS=512
VARIANT=messy  # v2 records carry the one variant they were labelled on
SPLIT=data/labelled_data_v2/benchmark.jsonl
EXPECT_RECORDS=52800
EXPECT_UNSEEN=14400
PROBES=data/labelled_data_v2/probes/report.json  # the shortcut probes, run before any arm (src.synthesis_v2.probes)
PROBES_PIN=checkpoints/queue_v2.probes_sha1        # every arm is scored against the same probe report

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
gate() { $GATE check "$SELF" "$GEN" >/dev/null || { log "review gate refused before $1; stopping"; $GATE check "$SELF" "$GEN" | tee -a "$LOG"; exit 1; }; }

# name arch base encoder_base wandb_project, read through the config loader so defaults apply
cfg_fields() { uv run --no-sync python -c "
import sys; from pathlib import Path; from src.train.config import load_config
c = load_config(Path(sys.argv[1])); print(c.name, c.model.arch, c.model.base, c.model.encoder_base or '-', c.wandb_project)" "$1"; }

# the generation command for a checkpoint directory, by architecture, into GEN_CMD. A 4B donor on either side of a
# stitch needs the small batch budget (the 4B decoder's cross-attention cache overflowed the 160k budget on compliance).
gen_cmd() {  # $1 model_dir $2 arch $3 base $4 encoder_base $5 out
    local common="$1 $SPLIT $5 --variants $VARIANT --max-tokens $MAX_TOKENS"
    case "$2" in
        decoder) GEN_CMD=".venv-vllm/bin/python -m src.train.generate $common --backend vllm --max-model-len $CAP" ;;
        hf_encdec) GEN_CMD="uv run --no-sync python -m src.train.generate $common --backend hf_encdec" ;;
        stitched|soft)
            local bt=160000; [[ "$3" == *4b* || "$4" == *4b* ]] && bt=40000
            GEN_CMD="uv run --no-sync python -m src.train.generate $common --backend $2 --batch-tokens $bt" ;;
        *) log "no generation backend for arch $2 ($1); stopping"; exit 1 ;;
    esac
}

# the benchmark on disk is the build the record describes: its hashes match benchmark_summary.json (the split, the
# question file and the three split files), its size is the frozen one, and its unseen cell is exactly the held-out bank
check_build() {
    uv run --no-sync python -c "
import json, sys
from pathlib import Path
from src.train.data import check_questions
split = Path(sys.argv[1])
check_questions(split.parent)
held = sorted(json.loads(l)['id'] for l in open(split.parent / 'questions.jsonl') if json.loads(l).get('held_out'))
n = n_unseen = 0; unseen = set()
for line in open(split):
    r = json.loads(line); n += 1
    if r['cell'] == 'unseen_q':
        n_unseen += 1; unseen.add(r['question']['id'])
print(f'{n} records, {n_unseen} unseen_q over {len(unseen)} questions (expected {sys.argv[2]}, {sys.argv[3]}, the {len(held)} held out)')
sys.exit(0 if (n, n_unseen) == (int(sys.argv[2]), int(sys.argv[3])) and sorted(unseen) == held else 1)" \
        "$SPLIT" "$EXPECT_RECORDS" "$EXPECT_UNSEEN"
}

# a step-* checkpoint about to be resumed was written by this very config (name, data, recipe), not by an earlier one
resume_matches() {  # $1 config $2 run_dir
    uv run --no-sync python -c "
import json, sys
from pathlib import Path
from src.train.config import load_config
import re
steps = sorted((p for p in Path(sys.argv[2]).glob('step-*') if re.fullmatch(r'step-\d+', p.name)),
               key=lambda p: int(p.name.split('-')[1]))  # a step-N.tmp left by an interrupted save is not a checkpoint
if not steps: sys.exit(0)
saved = json.loads((steps[-1] / 'state.json').read_text())['config']
now = load_config(Path(sys.argv[1])).model_dump(mode='json')
diff = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
print(f'{steps[-1]}: config ' + ('matches' if not diff else 'differs in ' + ', '.join(diff)))
sys.exit(1 if diff else 0)" "$1" "$2"
}

# every (id, variant) of the split present exactly once in the outputs
complete() {  # $1 outputs
    uv run --no-sync python -c "
import json, sys, collections
ids = [json.loads(l)['id'] for l in open(sys.argv[1])]
got = collections.Counter((json.loads(l)['id'], json.loads(l)['variant']) for l in open(sys.argv[2]))
missing = [i for i in ids if (i, sys.argv[3]) not in got]; dup = [k for k, n in got.items() if n > 1]
print(f'{len(ids)} pairs, {len(got)} outputs, {len(missing)} missing, {len(dup)} duplicated')
sys.exit(1 if missing or dup else 0)" "$SPLIT" "$1" "$VARIANT"
}

stage() {  # $1 label, rest: the command
    local label=$1; shift
    log "START $label"
    if "$@" >> "$LOG.$(echo "$label" | tr ' /' '__')" 2>&1; then log "END $label rc=0"; else local rc=$?; log "END $label rc=$rc; queue aborted"; exit $rc; fi
}

# a run's final export is complete: train_config.json is the last file every export path writes
trained() { [[ -f $1/final/train_config.json ]]; }

# the finished export was trained by this very config: a changed config (a raised budget, a regenerated recipe) must
# not be skipped as done or reused for generation; delete the run directory to retrain it under the new config
final_matches() {  # $1 config $2 run_dir
    uv run --no-sync python -c "
import json, sys
from pathlib import Path
from src.train.config import load_config
saved = json.loads((Path(sys.argv[2]) / 'final' / 'train_config.json').read_text())
now = load_config(Path(sys.argv[1])).model_dump(mode='json')
diff = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
print(f'{sys.argv[2]}/final: config ' + ('matches' if not diff else 'differs in ' + ', '.join(diff)))
sys.exit(1 if diff else 0)" "$1" "$2"
}

# the probe report is the one the first scored arm used: re-running the probes between arms would give arms different
# no_shortcut pair sets
pin_probes() {
    local sha; sha=$(sha1sum "$PROBES" | cut -d' ' -f1)
    if [[ ! -f $PROBES_PIN ]]; then echo "$sha" > "$PROBES_PIN"; log "pinned probe report $sha"; return; fi
    [[ "$(cat "$PROBES_PIN")" == "$sha" ]] || { log "probe report changed since $PROBES_PIN was written; stopping"; exit 1; }
}

# the budget decision agrees with the configs: keep = 8 questions per episode, raise = 16, in every config of the order
budget_matches() {  # $1 decision
    uv run --no-sync python -c "
import sys; from pathlib import Path; from src.train.config import load_config
want = {'keep': 8, 'raise': 16}[sys.argv[1]]
got = {p: load_config(Path(p)).data.questions_per_episode for p in sys.argv[2:]}
bad = {p: k for p, k in got.items() if k != want}
print(f'budget {sys.argv[1]}: {len(got) - len(bad)} of {len(got)} configs at {want} questions per episode')
sys.exit(1 if bad else 0)" "$@"
}

# what a generation was made from: the export (its training config and the name, size and modification time of every
# file in final/, so a re-export under the same config differs) and the benchmark's recorded hash. Outputs are reused
# only when all are unchanged, so a retrained arm or a rebuilt benchmark is never scored on stale generations.
gen_stamp() {  # $1 run_dir
    uv run --no-sync python -c "
import hashlib, json, sys
from pathlib import Path
final = Path(sys.argv[1]) / 'final'
h = hashlib.sha1((final / 'train_config.json').read_bytes())
for p in sorted(final.rglob('*')):
    if p.is_file():
        st = p.stat(); h.update(f'{p.relative_to(final)}|{st.st_size}|{st.st_mtime_ns}'.encode())
h.update(json.loads(Path('data/labelled_data_v2/benchmark_summary.json').read_text())['files_sha1']['benchmark.jsonl'].encode())
print(h.hexdigest())" "$1"
}

# a generation killed mid-write can leave a partial last line, which generate.py's resume cannot parse: drop it
repair_tail() {  # $1 outputs
    uv run --no-sync python -c "
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); data = p.read_bytes()
if not data or data.endswith(b'\\n'):
    sys.exit(0)
head, _, last = data.rpartition(b'\\n')
try:
    json.loads(last); p.write_bytes(data + b'\\n')
except json.JSONDecodeError:
    p.write_bytes(head + b'\\n' if head else b''); print(f'{p}: dropped a partial last line')" "$1"
}

# vLLM loads a Gemma 3 4B export as Gemma3ForConditionalGeneration and wants the base model's processor files beside it;
# the trainer's export writes the model and tokenizer only. Copy whatever the base model's cached snapshot has and the
# export lacks; model weights and tokenizer are never touched. A local checkpoint as base has nothing to copy.
ensure_processor_files() {  # $1 final_dir $2 base model id
    uv run --no-sync python -c "
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
final, base = Path(sys.argv[1]), sys.argv[2]
names = ['preprocessor_config.json', 'processor_config.json', 'special_tokens_map.json', 'added_tokens.json']
try:
    snap = Path(snapshot_download(base, local_files_only=True, allow_patterns=names))
except Exception as e:
    print(f'no cached snapshot for {base}: {e}'); sys.exit(0)
for n in names:
    if (snap / n).exists() and not (final / n).exists():
        shutil.copy(snap / n, final / n); print(f'copied {n} from {base}')
" "$1" "$2"
}

run() {  # $1 config
    local cfg=$1 name arch base enc proj dir
    read -r name arch base enc proj < <(cfg_fields "$cfg") || { log "cannot read config $cfg; stopping"; exit 1; }
    dir=checkpoints/$name
    if trained "$dir"; then
        stage "final config $name" final_matches "$cfg" "$dir"  # aborts the queue if the export came from another config
        # complete the export before anything stamps it (idempotent; see below)
        [[ "$arch" == decoder ]] && stage "processor files $name" ensure_processor_files "$dir/final" "$base"
    fi
    # scored and logged from outputs of this export and benchmark, under the pinned probe report: score.stamp is written
    # only after the evaluator succeeded (and so after wandb.json), and names exactly what was scored
    if trained "$dir" && [[ -f $dir/eval/benchmark/score.stamp && -f $PROBES ]] \
        && [[ "$(cat "$dir/eval/benchmark/score.stamp")" == "$(gen_stamp "$dir") $(sha1sum "$PROBES" | cut -d' ' -f1)" ]]; then
        log "skip $name: already scored and logged"; return
    fi
    if ! trained "$dir"; then
        gate "train $name"
        # expandable_segments: the CUDA caching allocator grows segments instead of fragmenting them; the 4B FSDP arms run
        # within 0.5 GB of the 32 GB cards. A memory-layout setting only; it does not change the arithmetic.
        local alloc=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # the smoke trains on the longest examples (15.6k tokens): peak memory, which the first records in file order (all
        # in the 0-4k band) would never reach
        stage "smoke $name" env WANDB_MODE=disabled $alloc uv run --no-sync torchrun --nproc_per_node 2 -m src.train.train "$cfg" --smoke 3 --smoke-longest
        stage "resume check $name" resume_matches "$cfg" "$dir"
        stage "train $name" env WANDB_MODE=online $alloc uv run --no-sync torchrun --nproc_per_node 2 -m src.train.train "$cfg" --resume
    else log "train $name: final export exists, not retraining"; fi
    [[ -s $dir/wandb_id ]] || { log "$name: no wandb_id in $dir, so the scores could not reach the run; stopping"; exit 1; }
    # the export is complete before it is stamped: vLLM's processor files are copied into final/ first, so the stamp
    # (which hashes every file there) does not change on the next invocation and trigger a needless regeneration
    [[ "$arch" == decoder ]] && stage "processor files $name" ensure_processor_files "$dir/final" "$base"
    local gen=$dir/eval/gen_benchmark.jsonl stamp
    stamp=$(gen_stamp "$dir")
    # the stamp is written before generation starts, so an interrupted generation of this export resumes; outputs with
    # another stamp (or none) came from another export or benchmark and are discarded
    if [[ -f $gen && "$(cat "$gen.stamp" 2>/dev/null || true)" != "$stamp" ]]; then
        log "gen $name: outputs were made from another export or benchmark; regenerating"
        rm -f "$gen" "$dir/eval/benchmark/score.stamp" "$dir/eval/benchmark/wandb.json"
    fi
    [[ -f $gen ]] && stage "repair $name" repair_tail "$gen"
    if [[ ! -f $gen ]] || ! complete "$gen" >/dev/null 2>&1; then
        gate "gen $name"
        gen_cmd "$dir/final" "$arch" "$base" "$enc" "$gen"
        mkdir -p "$(dirname "$gen")"; echo "$stamp" > "$gen.stamp"
        # shellcheck disable=SC2086
        stage "gen $name" $GEN_CMD
    fi
    gate "score $name"
    pin_probes
    stage "complete $name" complete "$gen"
    rm -f "$dir/eval/benchmark/score.stamp"
    stage "score $name" env WANDB_MODE=online uv run --no-sync python -m src.train.evaluate "$SPLIT" "$gen" "$dir/eval/benchmark" \
        --cap-tokens $((CAP - MAX_TOKENS)) --tokenizer "$dir/final" --probes "$PROBES" \
        --wandb-run "$(cat "$dir/wandb_id")" --wandb-project "$proj"
    echo "$stamp $(sha1sum "$PROBES" | cut -d' ' -f1)" > "$dir/eval/benchmark/score.stamp"
}

# the pre-registered pause: arm 1 scored, then the budget decision (keep | raise) written after its score
budget_pause_passed() {
    local arm1=checkpoints/e1-gemma3-270m-v2/eval/benchmark/score.stamp decision
    decision=$(cat checkpoints/queue_v2.budget 2>/dev/null || true)
    if [[ "$decision" != keep && "$decision" != raise ]]; then
        log "no budget decision in checkpoints/queue_v2.budget (keep | raise); run arm 1 alone first"; exit 1
    fi
    if [[ ! -f $arm1 || ! checkpoints/queue_v2.budget -nt $arm1 ]]; then
        log "the budget decision must be written after arm 1 is scored ($arm1); run arm 1 alone first"; exit 1
    fi
}

mkdir -p checkpoints
case "${1:-}" in
    all) shift
         [[ -f $PROBES ]] || { log "all: no shortcut-probe report at $PROBES; run python -m src.synthesis_v2.probes first"; exit 1; }
         decision=$(cat checkpoints/queue_v2.budget 2>/dev/null || true)
         if [[ "$decision" != keep && "$decision" != raise ]]; then
             log "all: no budget decision in checkpoints/queue_v2.budget (keep | raise); run arm 1 alone first"; exit 1
         fi
         budget_pause_passed
         stage "check build" check_build
         ORDER_OUT=$(uv run --no-sync python "$GEN" order) || { log "cannot read the run order; stopping"; exit 1; }
         mapfile -t CONFIGS <<< "$ORDER_OUT"
         [[ ${#CONFIGS[@]} -eq 16 && -n "${CONFIGS[0]}" ]] || { log "expected 16 configs in the run order, got ${#CONFIGS[@]}; stopping"; exit 1; }
         stage "budget $decision" budget_matches "$decision" "${CONFIGS[@]}"
         log "budget decision: $decision"
         for c in "${CONFIGS[@]}"; do run "$c"; done; log "QUEUE DONE (${#CONFIGS[@]})" ;;
    run) shift; [[ -f $PROBES ]] || { log "run: no shortcut-probe report at $PROBES; run python -m src.synthesis_v2.probes first"; exit 1; }
         for c in "$@"; do  # only arm 1 may run before the pre-registered pause; every other arm waits for the decision
             if [[ "$c" != configs/v2/e1-gemma3-270m-v2.yaml ]]; then budget_pause_passed; fi
         done
         if [[ -f checkpoints/queue_v2.budget ]]; then  # once decided, a single arm follows the decision too
             stage "budget" budget_matches "$(cat checkpoints/queue_v2.budget)" "$@"
         fi
         stage "check build" check_build; for c in "$@"; do run "$c"; done; log "QUEUE DONE ($#)" ;;
    *) echo "usage: $SELF all | run <config.yaml>..."; exit 2 ;;
esac
