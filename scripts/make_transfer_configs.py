"""Write the transfer-grid configs (configs/transfer/<dataset>/<run>.yaml) from the compliance configs they copy.

The transfer test (experiments/2026-10-28-transfer-grid/) re-runs the capacity grid on ECtHR with the compliance
recipe unchanged except for what the datasets force: the data paths, the over-long policy (documents are cut
from the tail to the 16k cap instead of dropped), the epoch count (chosen by the dataset's pilot) and
`val_every` (a divisor of the steps per epoch, so the val loss is read at every epoch boundary). Everything else, model
block included, is the compliance config's, so a transfer run differs from its compliance counterpart in data and budget
only. The configs are generated rather than hand-copied so the sixteen arms of a dataset cannot drift apart.

    uv run python scripts/make_transfer_configs.py pilots          # the 1B-decoder epoch pilot
    uv run python scripts/make_transfer_configs.py grid ecthr=2    # the 16 arms, at the epoch count the pilot chose

Steps per epoch are computed from the record counts on disk (batch 32), so the splits must be built first.
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = ("ecthr",)
# run name -> the compliance config whose recipe and model block it copies
ARMS = {
    "e1-gemma3-270m": "configs/e1/gemma3-270m.yaml",
    "e1-gemma3-1b-pt": "configs/e1/gemma3-1b-pt.yaml",
    "e1-gemma3-4b-pt": "configs/e1/gemma3-4b-pt.yaml",
    "e4-t5gemma2-270m-270m": "configs/e4/t5gemma2-270m-270m.yaml",
    "e4-t5gemma2-1b-1b": "configs/e4/t5gemma2-1b-1b.yaml",
    "e4mm-270m-enc-1b-dec": "configs/e4/mm-270m-enc-1b-dec.yaml",
    "e4mm-270m-enc-4b-dec": "configs/e4/mm-270m-enc-4b-dec.yaml",
    "e4mm-1b-enc-270m-dec": "configs/e4/mm-1b-enc-270m-dec.yaml",
    "e4mm-4b-enc-270m-dec": "configs/e4/mm-4b-enc-270m-dec.yaml",
    # E5-soft-tokens with the compliance adaptation stage: the soft-token stitches and their decoder-only controls, every one starting
    # from the compliance stage's export (checkpoints/adapt-continuation-corpora-*/final), so both sides of each pair went
    # through the identical stage and the comparison is matched as it was on compliance
    "e1pp-gemma3-1b-pt": "configs/e1/gemma3-1b-pt-plusplus.yaml",
    "e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-1b": "configs/e5-soft-tokens/t5gemma2-270m-enc-gemma3-1b-plusplus.yaml",
    "e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-1b": "configs/e5-soft-tokens/t5gemma2-1b-enc-gemma3-1b-plusplus.yaml",
    "e1pp-gemma3-270m": "configs/e1/gemma3-270m-plusplus.yaml",
    "e5-soft-tokens-pp-t5gemma2-270m-enc-gemma3-270m": "configs/e5-soft-tokens/t5gemma2-270m-enc-gemma3-270m-plusplus.yaml",
    "e5-soft-tokens-pp-t5gemma2-1b-enc-gemma3-270m": "configs/e5-soft-tokens/t5gemma2-1b-enc-gemma3-270m-plusplus.yaml",
    "e5-soft-tokens-pp-t5gemma2-4b-enc-gemma3-270m": "configs/e5-soft-tokens/t5gemma2-4b-enc-gemma3-270m-plusplus.yaml",
}
PILOT_ARM = "e1-gemma3-1b-pt"
PILOT_EPOCHS = {"ecthr": 2}  # the top of the candidate range; the val-loss curve picks within it
VALS_PER_EPOCH = 4


def n_records(dataset: str, split: str) -> int:
    with open(ROOT / "data" / "labelled_data" / dataset / f"{split}.jsonl") as f:
        return sum(1 for _ in f)


def val_every(dataset: str, batch: int) -> int:
    """The divisor of the steps per epoch that gives closest to VALS_PER_EPOCH val points per epoch (more on a tie), so
    every epoch boundary is a val point and the pilot's curve can be read per epoch."""
    steps = n_records(dataset, "train") // batch
    return min((d for d in range(1, steps + 1) if steps % d == 0), key=lambda d: (abs(steps / d - VALS_PER_EPOCH), d))


def make(arm: str, dataset: str, epochs: float, pilot: bool) -> Path:
    base = yaml.safe_load((ROOT / ARMS[arm]).read_text())
    name = f"{'pilot-' if pilot else ''}{arm}-{dataset}"
    cfg = dict(base)
    cfg["name"] = name
    cfg["data"] = {**base["data"], "train": f"data/labelled_data/{dataset}/train.jsonl",
                   "val": f"data/labelled_data/{dataset}/val.jsonl", "overlong": "truncate"}
    cfg["epochs"] = epochs
    cfg["val_every"] = val_every(dataset, base.get("batch_sequences", 32))
    # the compliance tags that name a compliance comparison (pair-*, grid-*, control-*, size-control) are dropped so a
    # wandb table filtered on them never picks up a transfer run; the model-describing tags stay
    kept = [t for t in base.get("tags", []) if not (t.startswith(("pair-", "grid-", "control-")) or t == "size-control")]
    cfg["tags"] = sorted({*kept, "transfer-grid", f"transfer-{dataset}", *(["pilot"] if pilot else [])})
    if pilot:  # constant learning rate after warm-up, so the epoch at which the val loss bottoms out is not an artefact of
        # where the cosine happened to be; the arms then train for that count under the compliance cosine
        cfg["optim"] = {**base["optim"], "min_lr_ratio": 1.0}
    role = (f"the epoch pilot for {dataset}: trains {epochs} epochs at a constant learning rate after warm-up, the val-loss curve picks the count every arm then uses"
            if pilot else f"one of the sixteen arms of the {dataset} transfer grid (nine E1/E4, seven E1++/E5-soft-tokens++)")
    head = (f"# {name}: {role}. Generated by scripts/make_transfer_configs.py from {ARMS[arm]} (recipe and model block\n"
            f"# unchanged); differs from it in the data paths, the over-long policy (truncate, data.fit_prompt), epochs and\n"
            f"# val_every ({cfg['val_every']}: {n_records(dataset, 'train') // base.get('batch_sequences', 32)} steps per epoch).\n")
    out = ROOT / "configs" / "transfer" / dataset / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["pilots", "grid"])
    ap.add_argument("epochs", nargs="*", help="grid: dataset=epochs for every dataset to write, e.g. ecthr=1")
    a = ap.parse_args()
    if a.what == "pilots":
        for ds in DATASETS:
            print(make(PILOT_ARM, ds, PILOT_EPOCHS[ds], pilot=True))
        return
    pairs = [kv.split("=") for kv in a.epochs]
    chosen = dict(pairs)
    unknown = set(chosen) - set(DATASETS)
    if unknown or not chosen or len(pairs) != len(chosen):
        ap.error(f"grid needs one dataset=epochs pair per dataset to write, over {DATASETS}; got {a.epochs}")
    for ds, ep in chosen.items():
        for arm in ARMS:
            print(make(arm, ds, float(ep) if "." in ep else int(ep), pilot=False))


if __name__ == "__main__":
    main()
