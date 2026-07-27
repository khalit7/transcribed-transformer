"""Hardware capability checks and FLOP accounting.

The point of this module is that compute claims in this project are measured
rather than assumed. ``model_flops_per_token`` is the denominator behind every
MFU number reported, so its assumptions are written down here and can be argued
with, rather than being buried in a training loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Dense bf16 tensor-core throughput with fp32 accumulate, which is what PyTorch
# matmuls actually do.
#
# Two wrong numbers are easy to reach for here, and both distort MFU badly:
#   - vendor "AI TOPS" headline figures assume fp4 with sparsity, roughly 16x
#     higher, which would crush MFU towards zero;
#   - the FP32 CUDA-core rate (104.8 TFLOP/s on a 5090), which is half the bf16
#     tensor rate and would inflate MFU by 2x.
#
# The values below are the vendor dense bf16 spec, which is the conventional
# MFU denominator and keeps numbers comparable with published results. A local
# 8192-cube matmul microbenchmark on this machine measured ~232 TFLOP/s per
# card, above spec because of boost clocks; see `python -m tt.training.hardware`
# to re-measure. If a device is missing here, MFU is reported as None rather
# than guessed.
PEAK_BF16_TFLOPS: dict[str, float] = {
    "NVIDIA GeForce RTX 5090": 209.5,
}


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    capability: tuple[int, int]
    total_memory_gib: float
    peak_bf16_tflops: float | None

    @property
    def is_blackwell(self) -> bool:
        return self.capability[0] >= 12


def device_info(index: int = 0) -> DeviceInfo:
    props = torch.cuda.get_device_properties(index)
    name = torch.cuda.get_device_name(index)
    return DeviceInfo(
        index=index,
        name=name,
        capability=torch.cuda.get_device_capability(index),
        total_memory_gib=props.total_memory / 2**30,
        peak_bf16_tflops=PEAK_BF16_TFLOPS.get(name),
    )


def check_environment() -> list[str]:
    """Return a list of problems with the training environment. Empty means good.

    Checked explicitly because every one of these fails in a way that is either
    silent or misleading: a missing sm_120 kernel surfaces as a cryptic runtime
    error deep in a training step, and an unavailable flash backend surfaces as
    an out-of-memory at long sequence length rather than as a capability error.
    """
    problems: list[str] = []

    if not torch.cuda.is_available():
        return ["CUDA is not available"]

    arch_list = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        sm = f"sm_{cap[0]}{cap[1]}"
        if sm not in arch_list:
            problems.append(
                f"device {i} is {sm} but this torch build only has {arch_list}. "
                f"Blackwell needs a CUDA 12.8+ build."
            )

    if not torch.cuda.is_bf16_supported():
        problems.append("bf16 is not supported; this project trains in bf16 throughout")

    # The flash SDPA backend is what makes attention memory O(n) instead of
    # O(n^2), which is the whole basis of the 32k-context plan.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(1, 4, 4096, 64, device="cuda:0", dtype=torch.bfloat16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        problems.append(f"flash SDPA backend unavailable: {exc}")

    return problems


def model_flops_per_token(
    *,
    n_params: int,
    n_layers: int,
    hidden: int,
    seq_len: int,
    global_every: int | None = None,
    local_window: int | None = None,
) -> float:
    """Training FLOPs per token, following the PaLM convention.

    ``6 * N`` covers forward and backward through the parameters. The second
    term is attention, which the parameter count does not include and which
    grows with sequence length.

    ModernBERT-style models alternate global attention every ``global_every``
    layers with local sliding-window attention elsewhere. Ignoring that and
    charging every layer full quadratic attention overstates the work done, and
    therefore overstates MFU, by a large margin at long sequence length. Pass
    ``global_every`` and ``local_window`` to account for it; omit them for a
    dense model.
    """
    param_flops = 6.0 * n_params

    if global_every is None or local_window is None:
        attn_span = n_layers * seq_len
    else:
        n_global = n_layers // global_every
        n_local = n_layers - n_global
        attn_span = n_global * seq_len + n_local * min(local_window, seq_len)

    attn_flops = 12.0 * hidden * attn_span
    return param_flops + attn_flops


def mfu(
    *, flops_per_token: float, tokens_per_second: float, peak_tflops: float | None
) -> float | None:
    """Model FLOPs utilisation as a fraction of peak. None if peak is unknown."""
    if peak_tflops is None:
        return None
    return (flops_per_token * tokens_per_second) / (peak_tflops * 1e12)


def measure_peak_bf16_tflops(index: int = 0, *, n: int = 8192, iters: int = 50) -> float:
    """Measure achievable bf16 matmul throughput, as a check on the table above.

    A large square matmul is the friendliest possible workload, so this is an
    upper bound on what any real training step can reach, not a target.
    """
    import time

    dev = f"cuda:{index}"
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(index)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize(index)
    return (2 * n**3 * iters) / (time.perf_counter() - t0) / 1e12


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _report() -> None:
    """`python -m tt.training.hardware`: environment check and measured ceilings."""
    problems = check_environment()
    print("environment:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print(f"  ! {p}")
    print(
        f"torch {torch.__version__} | cuda {torch.version.cuda} | arch {torch.cuda.get_arch_list()}"
    )
    for i in range(torch.cuda.device_count()):
        d = device_info(i)
        measured = measure_peak_bf16_tflops(i)
        print(
            f"  [{i}] {d.name} sm_{d.capability[0]}{d.capability[1]} "
            f"{d.total_memory_gib:.1f} GiB | table {d.peak_bf16_tflops} TFLOP/s | "
            f"measured {measured:.1f} TFLOP/s"
        )


if __name__ == "__main__":
    _report()
