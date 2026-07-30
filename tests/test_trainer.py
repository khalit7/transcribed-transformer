"""Training loop, checkpointing and resume.

The property under test is that a resumed run continues *identically*. Restoring
weights but not the optimiser, the step count or the RNG produces a run that looks
fine and is quietly not the run it claims to be, which is the failure worth
spending tests on.
"""

import json
from pathlib import Path

import pytest
import torch

from tt.data.schema import Track
from tt.training.config import (
    Arm,
    CheckpointConfig,
    ModelConfig,
    Objective,
    OptimConfig,
    RunConfig,
    ScheduleConfig,
    WandbConfig,
)
from tt.training.trainer import (
    TrainState,
    latest_checkpoint,
    load_checkpoint,
    lr_at,
    mlm_batch,
    prune_checkpoints,
    save_checkpoint,
)


def _cfg(**kw: object) -> RunConfig:
    base: dict[str, object] = {
        "name": "test-run",
        "arm": Arm.E,
        "track": Track.P,
        "model": ModelConfig(name_or_path="tiny", mask_token_id=50284, vocab_size=1000),
        "wandb": WandbConfig(project="tt-scratch", mode="disabled"),
        "optim": OptimConfig(lr=1e-3),
        "steps": 100,
    }
    base.update(kw)
    return RunConfig(**base)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


# --- learning rate ----------------------------------------------------------


def test_warmup_ramps_then_decays() -> None:
    cfg = _cfg(schedule=ScheduleConfig(kind="cosine", warmup_steps=10), steps=100)
    assert lr_at(0, cfg) == pytest.approx(1e-4)  # first step of 10
    assert lr_at(9, cfg) == pytest.approx(1e-3)  # peak at end of warmup
    assert lr_at(50, cfg) < 1e-3
    assert lr_at(99, cfg) < lr_at(50, cfg)


def test_decay_stops_at_the_floor_not_zero() -> None:
    """A schedule reaching zero stops learning before the token budget is spent."""
    cfg = _cfg(schedule=ScheduleConfig(kind="cosine", warmup_steps=0, min_lr_ratio=0.1))
    assert lr_at(cfg.steps - 1, cfg) >= 1e-3 * 0.1 * 0.999


def test_constant_schedule_is_flat() -> None:
    cfg = _cfg(schedule=ScheduleConfig(kind="constant", warmup_steps=0))
    assert lr_at(0, cfg) == lr_at(99, cfg) == pytest.approx(1e-3)


def test_lr_is_a_pure_function_of_step() -> None:
    """Resume recomputes the LR rather than restoring scheduler state.

    That only works if the schedule depends on nothing but the step, so a
    resumed run lands on exactly the value an uninterrupted one would.
    """
    cfg = _cfg(schedule=ScheduleConfig(warmup_steps=5))
    assert [lr_at(i, cfg) for i in range(20)] == [lr_at(i, cfg) for i in range(20)]


# --- objectives -------------------------------------------------------------


def test_mlm_scores_only_selected_positions() -> None:
    cfg = _cfg(mlm_probability=0.3)
    ids = torch.randint(0, 100, (2, 512))
    batch = mlm_batch(ids, cfg)
    scored = (batch["labels"] != -100).float().mean().item()
    assert 0.15 < scored < 0.45, "roughly the configured rate"


def test_mlm_actually_corrupts_the_input() -> None:
    """The bug this guards against, and it is silent.

    Selecting positions but leaving the tokens in place hands the model the
    answer in its own input. It learns to copy, the loss collapses toward zero,
    and it reads as fast convergence. A multi-day run would finish having learned
    nothing. This was measured at 0.015 MLM loss after 10 steps before the fix.
    """
    cfg = _cfg(mlm_probability=0.5)
    ids = torch.full((4, 512), 7, dtype=torch.long)
    batch = mlm_batch(ids, cfg)
    assert not torch.equal(batch["input_ids"], ids), "input must differ from the target"

    scored = batch["labels"] != -100
    changed = batch["input_ids"] != ids
    # Corruption happens only where the model is being scored.
    assert not (changed & ~scored).any()
    # ~90% of scored positions are corrupted (80% mask + 10% random).
    assert changed.sum().item() / max(1, scored.sum().item()) > 0.75


def test_mlm_uses_the_mask_token_for_most_corruptions() -> None:
    cfg = _cfg(mlm_probability=1.0)
    ids = torch.full((8, 256), 7, dtype=torch.long)
    inputs = mlm_batch(ids, cfg)["input_ids"]
    mask_id = cfg.model.mask_token_id
    assert mask_id is not None
    as_mask = (inputs == mask_id).float().mean().item()
    assert 0.7 < as_mask < 0.9, "roughly the 80% mask share"
    untouched = (inputs == 7).float().mean().item()
    assert 0.03 < untouched < 0.2, "the ~10% left alone, so masks are not the only input seen"


def test_mlm_refuses_without_a_mask_token() -> None:
    """Failing loudly beats training an identity task for days."""
    cfg = _cfg(model=ModelConfig(name_or_path="tiny"))
    with pytest.raises(ValueError, match="mask_token_id"):
        mlm_batch(torch.randint(0, 100, (1, 8)), cfg)


def test_unselected_positions_carry_no_loss() -> None:
    batch = mlm_batch(torch.randint(0, 100, (1, 256)), _cfg(mlm_probability=0.0))
    assert (batch["labels"] == -100).all()


# --- checkpointing ----------------------------------------------------------


def test_checkpoint_round_trip_restores_weights_and_step(tmp_path: Path) -> None:
    cfg = _cfg()
    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take a step so weights and optimiser state are non-trivial.
    model.fc(torch.randn(2, 4)).sum().backward()
    opt.step()

    saved = save_checkpoint(
        tmp_path, model=model, optimizer=opt, state=TrainState(step=42, tokens_seen=1234), cfg=cfg
    )
    before = model.fc.weight.detach().clone()

    restored = _Tiny()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert not torch.equal(restored.fc.weight, before), "fresh model differs"

    state = load_checkpoint(saved, model=restored, optimizer=restored_opt)
    assert state.step == 42
    assert state.tokens_seen == 1234
    assert torch.equal(restored.fc.weight, before)
    assert restored_opt.state_dict()["state"], "optimiser moments restored, not just weights"


def test_latest_checkpoint_picks_the_most_advanced(tmp_path: Path) -> None:
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    for step in (5, 40, 1000):
        save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=step), cfg=cfg)
    found = latest_checkpoint(tmp_path)
    assert found is not None and found.name == "step-00001000"


def test_zero_padded_names_sort_numerically(tmp_path: Path) -> None:
    """Without padding, 'step-9' sorts after 'step-1000' and resume goes backwards."""
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    for step in (9, 1000):
        save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=step), cfg=cfg)
    assert sorted(p.name for p in tmp_path.glob("step-*"))[-1] == "step-00001000"


def test_incomplete_checkpoint_is_not_offered(tmp_path: Path) -> None:
    """A checkpoint interrupted mid-write must not look loadable."""
    (tmp_path / "step-00000099").mkdir(parents=True)
    (tmp_path / "step-00000099" / "model.pt.tmp").write_bytes(b"partial")
    assert latest_checkpoint(tmp_path) is None


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path / "never-written") is None


def test_pruning_keeps_only_the_most_recent(tmp_path: Path) -> None:
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    for step in range(1, 6):
        save_checkpoint(
            tmp_path, model=model, optimizer=opt, state=TrainState(step=step * 10), cfg=cfg
        )
    prune_checkpoints(tmp_path, keep_last=2)
    remaining = sorted(p.name for p in tmp_path.glob("step-*"))
    assert remaining == ["step-00000040", "step-00000050"]


def test_keep_all_when_keep_last_is_zero(tmp_path: Path) -> None:
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    for step in (1, 2, 3):
        save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=step), cfg=cfg)
    prune_checkpoints(tmp_path, keep_last=0)
    assert len(list(tmp_path.glob("step-*"))) == 3


def test_resolved_config_is_written_beside_the_weights(tmp_path: Path) -> None:
    """A checkpoint has to be interpretable without this repository's working tree."""
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    saved = save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=1), cfg=cfg)
    written = json.loads((saved / "config.json").read_text())
    assert written["name"] == "test-run"
    assert written["objective"] == Objective.MLM.value


def test_rng_state_is_saved_per_rank(tmp_path: Path) -> None:
    """Restoring weights but not RNG changes the masking pattern and data order."""
    cfg, model = _cfg(), _Tiny()
    opt = torch.optim.AdamW(model.parameters())
    saved = save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=1), cfg=cfg)
    assert (saved / "rng-rank0.pt").exists()

    torch.manual_seed(1234)
    expected = torch.randn(3)
    torch.manual_seed(1234)
    save_checkpoint(tmp_path, model=model, optimizer=opt, state=TrainState(step=2), cfg=cfg)
    torch.randn(99)  # perturb the stream
    load_checkpoint(tmp_path / "step-00000002", model=model, optimizer=None)
    assert torch.allclose(torch.randn(3), expected), "RNG stream continues where it left off"


def test_resume_disabled_starts_from_zero() -> None:
    assert _cfg(checkpoint=CheckpointConfig(resume=False)).checkpoint.resume is False
