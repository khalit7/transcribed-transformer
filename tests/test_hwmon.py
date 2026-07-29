"""Hardware monitoring.

The monitor runs alongside training, so the property that matters most is that it
cannot take a run down. The rest is report correctness: a throttled run has to be
called out, because a throughput number taken from one does not generalise.
"""

from pathlib import Path

from tt.training.hwmon import (
    CpuSample,
    GpuSample,
    HardwareMonitor,
    Snapshot,
    _parse_float,
    sample_cpu,
)


def _gpu(
    index: int = 0,
    *,
    util: float = 95.0,
    temp: float = 60.0,
    clock: float = 2800.0,
    max_clock: float = 3105.0,
    throttle: int = 0,
    limit: float = 600.0,
) -> GpuSample:
    return GpuSample(
        index=index,
        name="NVIDIA GeForce RTX 5090",
        util_pct=util,
        memory_used_mib=20000.0,
        temperature_c=temp,
        power_w=500.0,
        power_limit_w=limit,
        sm_clock_mhz=clock,
        sm_clock_max_mhz=max_clock,
        fan_pct=55.0,
        throttle_bits=throttle,
    )


def _cpu() -> CpuSample:
    return CpuSample(
        util_pct=30.0, ram_used_gib=40.0, ram_total_gib=128.0, load_1m=8.0, temperature_c=65.0
    )


def _monitor(gpu_lists: list[list[GpuSample]], name: str = "test-run") -> HardwareMonitor:
    m = HardwareMonitor(run_name=name)
    m._started_at = 1000.0
    m._ended_at = 1120.0
    m.snapshots = [Snapshot(at=1000.0 + i, gpus=g, cpu=_cpu()) for i, g in enumerate(gpu_lists)]
    return m


def test_parse_float_handles_unavailable_fields() -> None:
    """nvidia-smi writes [N/A] rather than a number for unsupported fields."""
    assert _parse_float("95") == 95.0
    assert _parse_float("500.25 W") == 500.25
    assert _parse_float("[N/A]") != _parse_float("[N/A]")  # NaN
    assert _parse_float("[Not Supported]") != _parse_float("[Not Supported]")


def test_healthy_run_reports_no_limiting() -> None:
    m = _monitor([[_gpu()] for _ in range(10)])
    report = m.report()
    assert "No thermal or power limiting detected" in report
    assert "### GPU" in report and "### CPU and host" in report


def test_throttling_is_called_out_with_its_reason() -> None:
    """The headline signal. A throughput number from a throttled run does not generalise."""
    samples = [[_gpu(throttle=0)] for _ in range(5)] + [[_gpu(throttle=0x20)] for _ in range(5)]
    report = _monitor(samples).report()
    assert "### Warnings" in report
    assert "THERMALLY limited in 50% of samples" in report
    assert "sw thermal slowdown" in report
    assert "cooling problem" in report


def test_power_cap_is_distinguished_from_thermal_throttling() -> None:
    """A card at its power cap under full load is working as designed.

    Reporting that with the same alarm as a thermal fault would be wrong in both
    directions: it overstates the problem, and it wrongly says the number cannot
    be extrapolated when a stable power ceiling is exactly what does extrapolate.
    """
    report = _monitor([[_gpu(throttle=0x4)] for _ in range(10)]).report()
    assert "power cap" in report
    assert "expected" in report
    assert "cooling problem" not in report
    assert "THERMALLY" not in report


def test_idle_bit_is_not_counted_as_throttling() -> None:
    """The bug this guards against: an idle card sets GpuIdle (0x1) on every sample.

    Counting the raw bitmask reported 75% throttling on a healthy 20-second run,
    which would have made every entry in the report cry wolf and be ignored.
    """
    assert not _gpu(throttle=0x1).throttled, "GpuIdle is benign"
    assert not _gpu(throttle=0x2).throttled, "ApplicationsClocksSetting is benign"
    assert _gpu(throttle=0x4).throttled, "SwPowerCap is real"
    assert _gpu(throttle=0x40).throttled, "HwThermalSlowdown is real"
    report = _monitor([[_gpu(throttle=0x1)] for _ in range(10)]).report()
    assert "No thermal or power limiting detected" in report


def test_sustained_clock_below_max_is_flagged() -> None:
    """Heat-soak shows up as a clock ceiling long before it shows up anywhere else."""
    report = _monitor([[_gpu(clock=2400.0, max_clock=3105.0)] for _ in range(10)]).report()
    assert "sustained 2400 MHz" in report
    assert "thermal or power ceiling" in report


def test_idle_samples_do_not_drag_down_the_sustained_clock() -> None:
    """A card idles at ~200 MHz; averaging that in would fake a throttling report."""
    samples = [[_gpu(util=0.0, clock=195.0)] for _ in range(5)]
    samples += [[_gpu(util=99.0, clock=3000.0)] for _ in range(5)]
    report = _monitor(samples).report()
    assert "thermal or power ceiling" not in report


def test_asymmetric_power_limits_are_reported() -> None:
    """These two cards really do differ, and under DDP the slower one caps the run."""
    report = _monitor([[_gpu(0, limit=600.0), _gpu(1, limit=575.0)] for _ in range(5)]).report()
    assert "different power limits" in report
    assert "600W" in report and "575W" in report


def test_hot_gpu_is_reported() -> None:
    assert "peaked at 88 °C" in _monitor([[_gpu(temp=88.0)] for _ in range(5)]).report()


def test_empty_run_does_not_crash() -> None:
    m = HardwareMonitor(run_name="nothing")
    m._started_at = 1.0
    m._ended_at = 2.0
    assert "No samples captured" in m.report()


def test_append_creates_then_appends(tmp_path: Path) -> None:
    path = tmp_path / "hardware_performance.md"
    _monitor([[_gpu()] for _ in range(3)], name="run-one").append_report(path)
    assert path.read_text().startswith("# Hardware performance")
    assert "## run-one" in path.read_text()

    _monitor([[_gpu()] for _ in range(3)], name="run-two").append_report(path)
    text = path.read_text()
    assert text.count("# Hardware performance") == 1, "header written once only"
    assert "## run-one" in text and "## run-two" in text


def test_context_manager_samples_and_stops() -> None:
    """Real sampling against the actual machine, briefly."""
    with HardwareMonitor(run_name="smoke", interval_s=0.1) as m:
        time_slept = 0.0
        while len(m.snapshots) < 2 and time_slept < 5.0:
            import time as _t

            _t.sleep(0.1)
            time_slept += 0.1
    assert len(m.snapshots) >= 1
    assert m.duration_s > 0
    assert m._thread is not None and not m._thread.is_alive(), "thread must stop on exit"


def test_sample_cpu_returns_real_values() -> None:
    cpu = sample_cpu()
    assert cpu.ram_total_gib > 0
    assert 0 <= cpu.util_pct <= 100 * 64
