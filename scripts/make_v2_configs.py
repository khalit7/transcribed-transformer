"""Write the labelled-data-v2 grid configs (configs/v2/<run>.yaml) from the compliance configs they copy.

The v2 grid (experiments/2026-10-31-v2-long-podcast/) runs the capacity arms on long podcast episodes, four length bands
from one corpus, one question bank and one labeller, so the effect of length is read within each arm. The recipe and
model block are the compliance config's unchanged; a v2 run differs from its compliance counterpart only in the data
(paths, `questions_per_episode`, `question_seed`), the epoch count, `val_every` and, for a stitched arm, the stitch
(refitted on the v2 training prompts, STITCH_SUFFIX). Configs are generated rather than
hand-copied so the arms cannot drift apart.

The grid runs in three stages by decoder size, in the order of ORDER (owner-approved 2026-11-05, one seed). Since
2026-11-05 it holds E1 and E4 only (owner's decision): the E1++ controls and the E5-soft-tokens arms are in PARKED while
E5-merged-attention is built.

- stage 1: the decoder-only Gemma 3 270M (E1), the native and stitched encoder-decoders with the 270m / 1b / 4b encoders
  in front of the 270M decoder (E4), and the 1B and 4B decoder-only controls, so the 1b- and 4b-encoder arms have a
  decoder of the same total size beside them;
- stage 2: the 1B-decoder arms (E4 native, and the 270m encoder in front of the 1B decoder);
- stage 3: the 270m encoder with the 4B decoder, read against the 4B control from the first stage.

Arms that do not fit two 32 GB cards (native 4b-4b and the arms marked for 80 GB cards) are not generated.

    uv run python scripts/make_v2_configs.py all       # write every config
    uv run python scripts/make_v2_configs.py order     # print the config paths in run order (the queue reads this)
    uv run python scripts/make_v2_configs.py count     # the number of arms in the order (the queue checks it)

Steps per epoch come from the record counts on disk, so the split files must be built first.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path("data/labelled_data_v2")
# run name -> the compliance config whose recipe and model block it copies
# (stage, run name, the compliance config whose recipe and model block it copies), in run order. E1 and E4 only since
# 2026-11-05 (owner's decision): the E1++ controls and the E5-soft-tokens arms are parked (PARKED) while E5-merged-attention
# is built, and run later from the same generator.
ORDER: list[tuple[str, str, str]] = [
    ("1", "e1-gemma3-270m", "configs/e1/gemma3-270m.yaml"),
    ("1", "e1-gemma3-1b-pt", "configs/e1/gemma3-1b-pt.yaml"),
    ("1", "e1-gemma3-4b-pt", "configs/e1/gemma3-4b-pt.yaml"),
    ("1", "e4-t5gemma2-270m-270m", "configs/e4/t5gemma2-270m-270m.yaml"),
    ("1", "e4mm-1b-enc-270m-dec", "configs/e4/mm-1b-enc-270m-dec.yaml"),
    ("1", "e4mm-4b-enc-270m-dec", "configs/e4/mm-4b-enc-270m-dec.yaml"),
    ("2", "e4-t5gemma2-1b-1b", "configs/e4/t5gemma2-1b-1b.yaml"),
    ("2", "e4mm-270m-enc-1b-dec", "configs/e4/mm-270m-enc-1b-dec.yaml"),
    ("3", "e4mm-270m-enc-4b-dec", "configs/e4/mm-270m-enc-4b-dec.yaml"),
]
# Parked arms, in the order they would run; not generated while parked. E5-soft-tokens++ and its E1++ control start from
# the adaptation-stage exports (checkpoints/adapt-continuation-corpora-*/final).
PARKED: list[tuple[str, str, str]] = [
    ("1", "e1pp-gemma3-270m", "configs/e1/gemma3-270m-plusplus.yaml"),
    ("1", "e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-270m", "configs/e5-soft-tokens/t5gemma2-270m-enc-gemma3-270m-plusplus.yaml"),
    ("1", "e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-270m", "configs/e5-soft-tokens/t5gemma2-1b-enc-gemma3-270m-plusplus.yaml"),
    ("1", "e5-soft-tokens-pp-t5gemma2-4b-enc-gemma3-270m", "configs/e5-soft-tokens/t5gemma2-4b-enc-gemma3-270m-plusplus.yaml"),
    ("2", "e1pp-gemma3-1b-pt", "configs/e1/gemma3-1b-pt-plusplus.yaml"),
    ("2", "e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-1b", "configs/e5-soft-tokens/t5gemma2-270m-enc-gemma3-1b-plusplus.yaml"),
    ("2", "e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-1b", "configs/e5-soft-tokens/t5gemma2-1b-enc-gemma3-1b-plusplus.yaml"),
]
# Compute cap: each training episode contributes this many of its 32 trained questions, a seeded fixed choice
# (src.train.data.question_subset), every episode kept. Raised from 8 to 20 by the owner on 2026-11-05 after arm 1 at 8
# (36,800 examples, 1,150 steps) scored below the majority floor on accuracy with 11.5% runaway outputs: 20 x 4,600 =
# 92,000 examples per epoch, about the 91,502 of the v1 compliance runs (2,875 steps of 32 against v1's 2,858).
QUESTIONS_PER_EPISODE = 20
QUESTION_SEED = 0
EPOCHS = 1
VALS_PER_EPOCH = 4
# The E4-mix-and-match stitches are refitted on labelled data v2 (owner's decision 2026-11-05): the same fit
# (src.train.fit_stitch, 3.0M fit tokens, ridge 0.01) on the v2 training prompts, one per episode, up to 16,384 tokens,
# written beside the compliance fit as <pair>-v2.pt. The compliance configs keep the compliance stitch.
STITCH_SUFFIX = "-v2"


def n_examples() -> int:
    """Training examples per epoch: the episodes in train.jsonl times QUESTIONS_PER_EPISODE (every episode carries all 32
    trained questions, which the split guarantees; none is dropped for length, measured 2026-11-05)."""
    episodes = set()
    with (ROOT / DATA / "train.jsonl").open() as f:
        for line in f:
            episodes.add(line[line.index('"id"'):][:80].split('"')[3].split("::")[0])
    return len(episodes) * QUESTIONS_PER_EPISODE


def val_every(steps: int) -> int:
    """The divisor of the steps per epoch closest to VALS_PER_EPOCH val points per epoch (more on a tie)."""
    return min((d for d in range(1, steps + 1) if steps % d == 0), key=lambda d: (abs(steps / d - VALS_PER_EPOCH), d))


def config_path(arm: str) -> Path:
    return ROOT / "configs" / "v2" / f"{arm}-v2.yaml"


def make(stage: str, arm: str, src: str, steps: int) -> Path:
    base = yaml.safe_load((ROOT / src).read_text())
    name = f"{arm}-v2"
    cfg = dict(base)
    cfg["name"] = name
    cfg["data"] = {**base["data"], "train": str(DATA / "train.jsonl"), "val": str(DATA / "val.jsonl"), "overlong": "drop",
                   "questions_per_episode": QUESTIONS_PER_EPISODE, "question_seed": QUESTION_SEED}
    if base["model"].get("stitch"):
        old = Path(base["model"]["stitch"])
        new = old.with_name(old.stem + STITCH_SUFFIX + old.suffix)
        if not (ROOT / new).is_file():
            sys.exit(f"{arm}: the v2 stitch {new} is missing; fit it first (src.train.fit_stitch --one-per-transcript)")
        cfg["model"] = {**base["model"], "stitch": str(new)}
    cfg["epochs"] = EPOCHS
    cfg["val_every"] = val_every(steps)
    # compliance comparison tags (pair-*, grid-*, control-*, size-control) are dropped so a wandb table filtered on them
    # never picks up a v2 run; the model-describing tags stay
    kept = [t for t in base.get("tags", []) if not (t.startswith(("pair-", "grid-", "control-")) or t == "size-control")]
    questions = hashlib.sha1((ROOT / DATA / "questions.jsonl").read_bytes()).hexdigest()[:8]
    cfg["tags"] = sorted({*kept, "labelled-v2", f"v2-stage-{stage}", f"v2-questions-{questions}",
                          f"v2-q{QUESTIONS_PER_EPISODE}"})
    head = (f"# {name}: an arm of the labelled-data-v2 grid, stage {stage} of 3. Generated by scripts/make_v2_configs.py\n"
            f"# from {src} (recipe and model block unchanged); differs from it in the data (v2 split, {QUESTIONS_PER_EPISODE}\n"
            f"# questions per episode, seed {QUESTION_SEED}), epochs ({EPOCHS}) and val_every ({cfg['val_every']}: {steps} steps per epoch).\n")
    if cfg["model"].get("stitch") != base["model"].get("stitch"):
        head = head.replace("(recipe and model block unchanged)", "(recipe and model block, see below)")
        head += f"# Recipe unchanged; model block unchanged but its stitch, the v2 refit {cfg['model']['stitch']}.\n"
    out = config_path(arm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["all", "order", "count"])
    a = ap.parse_args()
    if a.what == "count":
        print(len(ORDER))
        return
    if a.what == "order":
        for _, arm, _ in ORDER:
            print(config_path(arm).relative_to(ROOT))
        return
    n = n_examples()
    for stage, arm, src in ORDER:
        batch = yaml.safe_load((ROOT / src).read_text()).get("batch_sequences", 32)
        print(make(stage, arm, src, n // batch))


if __name__ == "__main__":
    main()
