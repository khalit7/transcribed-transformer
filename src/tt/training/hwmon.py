"""Hardware sampling during a run, reported to ``hardware_performance.md``.

wandb already records per-step memory and throughput. This exists for the thing
wandb does not show: whether the machine was *able* to run at full speed, and for
how long.

Two RTX 5090s in one desktop case draw over a kilowatt between them and heat-soak
over hours. A run that starts at 2.9 GHz and settles at 2.4 GHz has lost 17% of
its throughput to thermals, and nothing in a loss curve reveals that. On a
training job measured in days, on a machine that is also somebody's desktop, that
is the difference between an estimate holding and an estimate being wrong by a
day. So the report leads with **throttling and sustained clocks**, not with
averages.

The two cards are not identical: they report different power limits, 600W and
575W. Asymmetric limits produce asymmetric throughput, and under DDP the whole
step waits for the slower rank, so per-device figures are kept separate rather
than averaged into one number that would hide it.

Sampling is done in a background thread via ``nvidia-smi``, which costs a
subprocess every few seconds and is irrelevant next to a training step. Nothing
here is on the critical path, and a monitor failure must never take a run down
with it.
"""

from __future__ import annotations

import contextlib
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psutil

_GPU_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.max.sm",
    "fan.speed",
    "clocks_throttle_reasons.active",
)

DEFAULT_REPORT = Path("hardware_performance.md")

# nvidia-smi's throttle bitmask mixes benign states with real limiting.
# GpuIdle and ApplicationsClocksSetting are *not* problems: an idle card sets
# GpuIdle on every sample, so counting the raw bitmask reports ~100% throttling
# on any run with startup or teardown in it, which makes the whole report noise.
THROTTLE_REASONS: dict[int, str] = {
    0x0000000004: "sw power cap",
    0x0000000008: "hw slowdown",
    0x0000000010: "sync boost",
    0x0000000020: "sw thermal slowdown",
    0x0000000040: "hw thermal slowdown",
    0x0000000080: "hw power brake",
}
"""Bits that mean the card was genuinely held back. Idle and clock-setting bits excluded."""


_REAL_THROTTLE_MASK = sum(THROTTLE_REASONS)


@dataclass
class GpuSample:
    index: int
    name: str
    util_pct: float
    memory_used_mib: float
    temperature_c: float
    power_w: float
    power_limit_w: float
    sm_clock_mhz: float
    sm_clock_max_mhz: float
    fan_pct: float
    throttle_bits: int

    @property
    def throttled(self) -> bool:
        """Whether the card was genuinely held back, ignoring benign idle bits."""
        return bool(self.throttle_bits & _REAL_THROTTLE_MASK)

    @property
    def throttle_names(self) -> list[str]:
        """Human-readable reasons this sample was limited."""
        return [name for bit, name in THROTTLE_REASONS.items() if self.throttle_bits & bit]


@dataclass
class CpuSample:
    util_pct: float
    ram_used_gib: float
    ram_total_gib: float
    load_1m: float
    temperature_c: float | None


@dataclass
class Snapshot:
    at: float
    gpus: list[GpuSample]
    cpu: CpuSample


def _cpu_temperature() -> float | None:
    """CPU package temperature, or ``None`` where the platform does not expose one."""
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    for key in ("k10temp", "coretemp", "zenpower"):
        for entry in temps.get(key, []):
            if entry.label in ("Tctl", "Tdie", "Package id 0", ""):
                return float(entry.current)
        if temps.get(key):
            return float(temps[key][0].current)
    return None


def _parse_float(value: str) -> float:
    """nvidia-smi writes '[N/A]' and '[Not Supported]' for unavailable fields."""
    try:
        return float(value.strip().split()[0])
    except (ValueError, IndexError):
        return float("nan")


def sample_gpus() -> list[GpuSample]:
    """One reading per visible GPU. Returns empty on any failure, never raises."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except Exception:
        return []

    samples: list[GpuSample] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_FIELDS):
            continue
        try:
            throttle = int(parts[10], 16) if parts[10].startswith("0x") else 0
        except ValueError:
            throttle = 0
        samples.append(
            GpuSample(
                index=int(_parse_float(parts[0])),
                name=parts[1],
                util_pct=_parse_float(parts[2]),
                memory_used_mib=_parse_float(parts[3]),
                temperature_c=_parse_float(parts[4]),
                power_w=_parse_float(parts[5]),
                power_limit_w=_parse_float(parts[6]),
                sm_clock_mhz=_parse_float(parts[7]),
                sm_clock_max_mhz=_parse_float(parts[8]),
                fan_pct=_parse_float(parts[9]),
                throttle_bits=throttle,
            )
        )
    return samples


def sample_cpu() -> CpuSample:
    memory = psutil.virtual_memory()
    return CpuSample(
        util_pct=psutil.cpu_percent(interval=None),
        ram_used_gib=memory.used / 2**30,
        ram_total_gib=memory.total / 2**30,
        load_1m=psutil.getloadavg()[0],
        temperature_c=_cpu_temperature(),
    )


@dataclass
class HardwareMonitor:
    """Sample hardware in the background for the duration of a run.

    Used as a context manager::

        with HardwareMonitor("arm-e-phase1") as monitor:
            train(...)
        monitor.append_report()
    """

    run_name: str
    interval_s: float = 5.0
    note: str = ""
    snapshots: list[Snapshot] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _started_at: float = 0.0
    _ended_at: float = 0.0

    def __enter__(self) -> HardwareMonitor:
        self._started_at = time.time()
        psutil.cpu_percent(interval=None)  # prime the delta-based reading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
        self._ended_at = time.time()

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Monitoring must never take a training run down with it.
            with contextlib.suppress(Exception):
                self.snapshots.append(
                    Snapshot(at=time.time(), gpus=sample_gpus(), cpu=sample_cpu())
                )
            self._stop.wait(self.interval_s)

    @property
    def duration_s(self) -> float:
        end = self._ended_at or time.time()
        return end - self._started_at

    def report(self) -> str:
        """Render one markdown section for this run."""
        started = datetime.fromtimestamp(self._started_at, tz=UTC)
        lines = [
            f"## {self.run_name}",
            "",
            f"- **Started**: {started.isoformat(timespec='seconds')}",
            f"- **Duration**: {self.duration_s / 60:.1f} min",
            f"- **Samples**: {len(self.snapshots)} at {self.interval_s:g}s intervals",
        ]
        if self.note:
            lines.append(f"- **Note**: {self.note}")
        lines.append("")

        if not self.snapshots:
            lines += ["_No samples captured._", ""]
            return "\n".join(lines)

        lines += self._gpu_table()
        lines += self._cpu_table()
        lines += self._warnings()
        return "\n".join(lines)

    def _gpu_table(self) -> list[str]:
        by_index: dict[int, list[GpuSample]] = {}
        for snap in self.snapshots:
            for gpu in snap.gpus:
                by_index.setdefault(gpu.index, []).append(gpu)
        if not by_index:
            return ["_No GPU samples._", ""]

        out = [
            "### GPU",
            "",
            "| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) "
            "| Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |",
            "|---|---|---|---|---|---|---:|---:|---:|",
        ]
        for index in sorted(by_index):
            s = by_index[index]
            util = [x.util_pct for x in s]
            mem = [x.memory_used_mib / 1024 for x in s]
            temp = [x.temperature_c for x in s]
            power = [x.power_w for x in s]
            clock = [x.sm_clock_mhz for x in s]
            fan = [x.fan_pct for x in s]
            max_clock = max((x.sm_clock_max_mhz for x in s), default=0.0)
            # Only count clocks while the card is actually busy: an idle card
            # drops to ~200 MHz and would drag the sustained figure down.
            busy = [x.sm_clock_mhz for x in s if x.util_pct > 50]
            pct_of_max = (statistics.mean(busy) / max_clock * 100) if busy and max_clock else 0.0
            throttled = sum(x.throttled for x in s) / len(s) * 100
            out.append(
                f"| {index} ({s[0].name}, {s[0].power_limit_w:.0f}W cap) "
                f"| {statistics.mean(util):.0f} / {max(util):.0f} "
                f"| {statistics.mean(mem):.1f} / {max(mem):.1f} "
                f"| {statistics.mean(temp):.0f} / {max(temp):.0f} "
                f"| {statistics.mean(power):.0f} / {max(power):.0f} "
                f"| {statistics.mean(clock):.0f} / {min(clock):.0f} "
                f"| {pct_of_max:.0f}% | {statistics.mean(fan):.0f} | {throttled:.0f}% |"
            )
        out.append("")
        return out

    def _cpu_table(self) -> list[str]:
        cpus = [s.cpu for s in self.snapshots]
        util = [c.util_pct for c in cpus]
        ram = [c.ram_used_gib for c in cpus]
        load = [c.load_1m for c in cpus]
        temps = [c.temperature_c for c in cpus if c.temperature_c is not None]
        out = [
            "### CPU and host",
            "",
            "| Metric | Mean | Max |",
            "|---|---:|---:|",
            f"| Utilisation % | {statistics.mean(util):.0f} | {max(util):.0f} |",
            f"| RAM GiB (of {cpus[0].ram_total_gib:.0f}) "
            f"| {statistics.mean(ram):.1f} | {max(ram):.1f} |",
            f"| Load average (1m) | {statistics.mean(load):.1f} | {max(load):.1f} |",
        ]
        if temps:
            out.append(f"| CPU temp °C | {statistics.mean(temps):.0f} | {max(temps):.0f} |")
        out.append("")
        return out

    def _warnings(self) -> list[str]:
        """Anything that would make a throughput number misleading if unstated."""
        notes: list[str] = []
        by_index: dict[int, list[GpuSample]] = {}
        for snap in self.snapshots:
            for gpu in snap.gpus:
                by_index.setdefault(gpu.index, []).append(gpu)

        for index, s in sorted(by_index.items()):
            throttled = sum(x.throttled for x in s) / len(s)
            if throttled > 0.05:
                reasons = sorted({r for x in s for r in x.throttle_names})
                thermal = any("thermal" in r or r == "hw slowdown" for r in reasons)
                if thermal:
                    notes.append(
                        f"- **GPU {index} was THERMALLY limited in {throttled:.0%} of "
                        f"samples** ({', '.join(reasons)}). This is a cooling problem, not a "
                        "steady state: throughput here understates what the hardware does "
                        "when cool, and it will get worse as a long run heat-soaks. Do not "
                        "extrapolate a multi-day estimate from this run."
                    )
                else:
                    notes.append(
                        f"- GPU {index} hit its **power cap** in {throttled:.0%} of samples "
                        f"({', '.join(reasons)}). At full load this is expected and is the "
                        "card working as designed, so it *is* a fair basis for extrapolation "
                        "— it simply means the ceiling here is watts rather than the kernel. "
                        "Raising the cap, if the PSU and cooling allow, is the lever."
                    )
            busy = [x for x in s if x.util_pct > 50]
            if busy:
                max_clock = max(x.sm_clock_max_mhz for x in s)
                sustained = statistics.mean(x.sm_clock_mhz for x in busy)
                if max_clock and sustained < 0.9 * max_clock:
                    notes.append(
                        f"- GPU {index} sustained {sustained:.0f} MHz against a "
                        f"{max_clock:.0f} MHz maximum ({sustained / max_clock:.0%}), "
                        "which is a thermal or power ceiling rather than a workload one."
                    )
                hot = max(x.temperature_c for x in busy)
                if hot >= 83:
                    notes.append(f"- GPU {index} peaked at {hot:.0f} °C.")

        cpu_temps = [s.cpu.temperature_c for s in self.snapshots if s.cpu.temperature_c]
        if cpu_temps and max(cpu_temps) >= 85:
            notes.append(
                f"- CPU peaked at {max(cpu_temps):.0f} °C. On AMD parts this reading is "
                "Tctl, which carries an offset and runs hotter than the true junction "
                "temperature, so treat it as a trend to watch rather than a fault."
            )

        limits = {x.power_limit_w for s in by_index.values() for x in s}
        if len(limits) > 1:
            notes.append(
                "- **The cards have different power limits** ("
                + ", ".join(f"{x:.0f}W" for x in sorted(limits))
                + "). "
                "Under DDP every step waits for the slower rank, so this caps the "
                "whole run, not just one device."
            )

        if not notes:
            return ["_No thermal or power limiting detected._", ""]
        return ["### Warnings", "", *notes, ""]

    def append_report(self, path: Path = DEFAULT_REPORT) -> Path:
        """Append this run's section to the report, creating it if absent."""
        path = Path(path)
        if not path.exists():
            path.write_text(_HEADER)
        with path.open("a") as handle:
            handle.write("\n" + self.report())
        return path


_HEADER = """# Hardware performance

Sampled during runs by `tt.training.hwmon`. Every number here comes from an actual
run; nothing is estimated.

This exists for what a loss curve cannot show: whether the machine sustained full
speed or quietly lost it to heat. Two RTX 5090s in one desktop case heat-soak over
hours, and a run that settles 15% below peak clocks has lost 15% of its throughput
without anything else looking wrong. **Read the Warnings section of each run first**
— a throughput figure taken from a throttled run will not hold over days.
"""
