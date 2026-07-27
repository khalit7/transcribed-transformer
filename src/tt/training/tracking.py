"""wandb integration.

Every training and evaluation run logs here. There is no second logging path,
so anything worth knowing about a run must go through this module.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any

import torch

from tt.training import hardware
from tt.training.config import RunConfig


class ThroughputMeter:
    """Measures tokens/s and MFU over a window of steps.

    Warmup steps are excluded. The first steps of any run include CUDA graph
    capture, autotuning and allocator growth, and including them understates
    steady-state throughput by enough to matter when the number is being used
    to plan multi-day runs.
    """

    def __init__(self, *, flops_per_token: float, peak_tflops: float | None, warmup: int = 5):
        self.flops_per_token = flops_per_token
        self.peak_tflops = peak_tflops
        self.warmup = warmup
        self._steps = 0
        self._tokens = 0
        self._start: float | None = None

    def step(self, n_tokens: int) -> None:
        self._steps += 1
        if self._steps <= self.warmup:
            return
        if self._start is None:
            torch.cuda.synchronize()
            self._start = time.perf_counter()
            return
        self._tokens += n_tokens

    def measure(self) -> dict[str, float]:
        """Current throughput. Empty while still in warmup."""
        if self._start is None or self._tokens == 0:
            return {}
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        tps = self._tokens / elapsed
        out = {"perf/tokens_per_second": tps, "perf/elapsed_s": elapsed}
        util = hardware.mfu(
            flops_per_token=self.flops_per_token,
            tokens_per_second=tps,
            peak_tflops=self.peak_tflops,
        )
        if util is not None:
            out["perf/mfu"] = util
        return out


def memory_metrics() -> dict[str, float]:
    """Peak memory on this rank's own device.

    Each process only sees its own allocations, so querying every device from
    rank 0 reports zero for the others and makes an idle GPU indistinguishable
    from a busy one. Report the local device and label it with the rank.
    """
    local = hardware.local_rank()
    return {
        "mem/peak_gib": torch.cuda.max_memory_allocated(local) / 2**30,
        "mem/reserved_gib": torch.cuda.max_memory_reserved(local) / 2**30,
    }


class Run:
    """A wandb run with the project's mandatory conventions applied.

    Only rank 0 logs. Other ranks get a no-op so training code does not need to
    branch on rank at every call site.
    """

    def __init__(self, cfg: RunConfig, *, extra_config: dict[str, Any] | None = None):
        self.cfg = cfg
        self._active = hardware.rank() == 0
        self._run: Any = None
        self._extra = extra_config or {}

    def __enter__(self) -> Run:
        if not self._active:
            return self
        import wandb

        config = dict(self.cfg.resolved())
        config.update(self._extra)
        config["hardware"] = [
            {
                "index": d.index,
                "name": d.name,
                "capability": f"sm_{d.capability[0]}{d.capability[1]}",
                "memory_gib": round(d.total_memory_gib, 1),
                "peak_bf16_tflops": d.peak_bf16_tflops,
            }
            for d in (hardware.device_info(i) for i in range(torch.cuda.device_count()))
        ]
        config["world_size"] = hardware.world_size()

        self._run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.name,
            tags=self.cfg.tags,
            notes=self.cfg.wandb.notes,
            mode=self.cfg.wandb.mode,
            config=config,
        )
        return self

    def log(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        if self._active and self._run is not None:
            self._run.log(metrics, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        if self._active and self._run is not None:
            self._run.summary.update(metrics)

    @property
    def url(self) -> str | None:
        return None if self._run is None else str(self._run.url)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._active and self._run is not None:
            self._run.finish(exit_code=0 if exc_type is None else 1)
