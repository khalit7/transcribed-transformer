# Hardware performance

Sampled during runs by `tt.training.hwmon`. Every number here comes from an actual
run; nothing is estimated.

This exists for what a loss curve cannot show: whether the machine sustained full
speed or quietly lost it to heat. Two RTX 5090s in one desktop case heat-soak over
hours, and a run that settles 15% below peak clocks has lost 15% of its throughput
without anything else looking wrong. **Read the Warnings section of each run first**
— a throughput figure taken from a throttled run will not hold over days.

## p0-throughput-modernbert-large-seq8192 @ 8192 tokens

- **Started**: 2026-07-29T18:19:19+00:00
- **Duration**: 3.0 min
- **Samples**: 36 at 5s intervals
- **Note**: answerdotai/ModernBERT-large, 396M params, micro batch 1, world size 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 97 / 100 | 21.3 / 21.8 | 82 / 91 | 576 / 603 | 2845 / 2542 | 92% | 90 | 94% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 100 / 100 | 20.8 / 21.3 | 85 / 90 | 560 / 579 | 2775 / 2737 | 89% | 82 | 94% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 7 | 8 |
| RAM GiB (of 60) | 14.7 | 14.9 |
| Load average (1m) | 1.7 | 2.0 |
| CPU temp °C | 95 | 96 |

### Warnings

- GPU 0 hit its **power cap** in 94% of samples (sw power cap). At full load this is expected and is the card working as designed, so it *is* a fair basis for extrapolation — it simply means the ceiling here is watts rather than the kernel. Raising the cap, if the PSU and cooling allow, is the lever.
- GPU 0 peaked at 91 °C.
- GPU 1 hit its **power cap** in 94% of samples (sw power cap). At full load this is expected and is the card working as designed, so it *is* a fair basis for extrapolation — it simply means the ceiling here is watts rather than the kernel. Raising the cap, if the PSU and cooling allow, is the lever.
- GPU 1 sustained 2775 MHz against a 3105 MHz maximum (89%), which is a thermal or power ceiling rather than a workload one.
- GPU 1 peaked at 90 °C.
- CPU peaked at 96 °C. On AMD parts this reading is Tctl, which carries an offset and runs hotter than the true junction temperature, so treat it as a trend to watch rather than a fault.
- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## resume-check (train)

- **Started**: 2026-07-29T18:35:19+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 6 / 6 | 1.7 / 1.7 | 48 / 48 | 36 / 36 | 195 / 195 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 19 / 19 | 1.7 / 1.7 | 50 / 50 | 49 / 49 | 1710 / 1710 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 13.4 | 13.4 |
| Load average (1m) | 0.1 | 0.1 |
| CPU temp °C | 72 | 72 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## resume-check (train)

- **Started**: 2026-07-29T18:35:21+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 0 / 0 | 3.8 / 3.8 | 50 / 50 | 94 / 94 | 2955 / 2955 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 100 / 100 | 3.8 / 3.8 | 55 / 55 | 113 / 113 | 2865 / 2865 | 92% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 15.6 | 15.6 |
| Load average (1m) | 0.4 | 0.4 |
| CPU temp °C | 85 | 85 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.
