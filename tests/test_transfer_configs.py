"""The transfer-grid configs (configs/transfer/) are generated copies of the compliance configs: check that every one
loads, points at its dataset's splits, truncates over-long documents, and differs from its compliance source only where
the generator says it does."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.config import load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((ROOT / "configs" / "transfer").glob("*/*.yaml"))
GENERATED_FIELDS = {"name", "data", "epochs", "val_every", "tags", "optim"}


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_transfer_config_is_its_compliance_source_plus_the_dataset(path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_transfer_configs import ARMS

    cfg = load_config(path)
    dataset = path.parent.name
    assert cfg.name == path.stem
    arm = cfg.name.removeprefix("pilot-").removesuffix(f"-{dataset}")
    assert arm in ARMS, cfg.name
    assert cfg.data.train == Path(f"data/labelled_data/{dataset}/train.jsonl")
    assert cfg.data.val == Path(f"data/labelled_data/{dataset}/val.jsonl")
    assert cfg.data.overlong == "truncate" and cfg.data.max_seq_len == 16384
    assert {"transfer-grid", f"transfer-{dataset}"} <= set(cfg.tags)
    assert ("pilot" in cfg.tags) == cfg.name.startswith("pilot-")
    base = yaml.safe_load((ROOT / ARMS[arm]).read_text())
    ours = yaml.safe_load(path.read_text())
    for key in set(base) | set(ours):
        if key not in GENERATED_FIELDS:
            assert ours.get(key) == base.get(key), (cfg.name, key)
    assert ours["data"] == {**base["data"], "train": str(cfg.data.train), "val": str(cfg.data.val), "overlong": "truncate"}
    if cfg.name.startswith("pilot-"):
        assert ours["optim"] == {**base["optim"], "min_lr_ratio": 1.0}
    else:
        assert ours["optim"] == base["optim"]
    dropped = {t for t in base["tags"] if t.startswith(("pair-", "grid-", "control-")) or t == "size-control"}
    assert set(base["tags"]) - dropped <= set(ours["tags"]) and not dropped & set(ours["tags"])


@pytest.mark.skipif(not (ROOT / "data" / "labelled_data" / "ecthr" / "train.jsonl").exists(), reason="splits not built")
@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_val_every_divides_the_steps_per_epoch(path):
    cfg = load_config(path)
    with cfg.data.train.open() as f:
        n = sum(1 for _ in f)
    steps = n // cfg.batch_sequences
    assert steps % cfg.val_every == 0 and 1 <= steps // cfg.val_every <= 8, (cfg.name, steps, cfg.val_every)
