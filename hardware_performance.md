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

## pipeline-real (train)

- **Started**: 2026-07-30T08:32:20+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 16 / 16 | 1.7 / 1.7 | 33 / 33 | 38 / 38 | 2542 / 2542 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 6 / 6 | 1.7 / 1.7 | 36 / 36 | 58 / 58 | 2460 / 2460 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 9.1 | 9.1 |
| Load average (1m) | 1.7 | 1.7 |
| CPU temp °C | 65 | 65 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:32:36+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 1

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 0 / 0 | 0.9 / 0.9 | 32 / 32 | 23 / 23 | 195 / 195 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 0 / 0 | 0.0 / 0.0 | 35 / 35 | 19 / 19 | 195 / 195 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 7.0 | 7.0 |
| Load average (1m) | 1.6 | 1.6 |
| CPU temp °C | 61 | 61 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:32:38+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 1

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 10 / 10 | 4.5 / 4.5 | 37 / 37 | 147 / 147 | 2940 / 2940 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 0 / 0 | 0.0 / 0.0 | 35 / 35 | 21 / 21 | 195 / 195 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 8.2 | 8.2 |
| Load average (1m) | 1.5 | 1.5 |
| CPU temp °C | 61 | 61 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:34:29+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 1

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 0 / 0 | 0.9 / 0.9 | 33 / 33 | 16 / 16 | 195 / 195 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 0 / 0 | 0.0 / 0.0 | 35 / 35 | 17 / 17 | 195 / 195 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 7.2 | 7.2 |
| Load average (1m) | 0.6 | 0.6 |
| CPU temp °C | 62 | 62 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:34:31+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 1

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 7 / 7 | 4.5 / 4.5 | 37 / 37 | 147 / 147 | 2940 / 2940 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 0 / 0 | 0.0 / 0.0 | 35 / 35 | 20 / 20 | 195 / 195 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 8.6 | 8.6 |
| Load average (1m) | 0.6 | 0.6 |
| CPU temp °C | 61 | 61 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:34:53+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 16 / 16 | 1.7 / 1.7 | 35 / 35 | 33 / 33 | 2640 / 2640 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 6 / 6 | 1.7 / 1.7 | 37 / 37 | 46 / 46 | 2437 / 2437 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 9.1 | 9.1 |
| Load average (1m) | 0.7 | 0.7 |
| CPU temp °C | 66 | 66 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:35:18+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 14 / 14 | 1.7 / 1.7 | 35 / 35 | 44 / 44 | 2632 / 2632 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 6 / 6 | 1.7 / 1.7 | 38 / 38 | 61 / 61 | 2437 / 2437 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 9.1 | 9.1 |
| Load average (1m) | 0.7 | 0.7 |
| CPU temp °C | 66 | 66 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:35:20+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 14 / 14 | 4.6 / 4.6 | 37 / 37 | 150 / 150 | 2970 / 2970 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 100 / 100 | 5.0 / 5.0 | 44 / 44 | 166 / 166 | 2872 / 2872 | 92% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 11.7 | 11.7 |
| Load average (1m) | 0.7 | 0.7 |
| CPU temp °C | 70 | 70 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:42:19+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 10 / 10 | 1.7 / 1.7 | 38 / 38 | 32 / 32 | 2670 / 2670 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 12 / 12 | 1.7 / 1.7 | 44 / 44 | 52 / 52 | 2460 / 2460 | 0% | 31 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 9.1 | 9.1 |
| Load average (1m) | 1.0 | 1.0 |
| CPU temp °C | 73 | 73 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:42:21+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 8 / 8 | 4.6 / 4.6 | 39 / 39 | 160 / 160 | 2970 / 2970 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 100 / 100 | 5.0 / 5.0 | 51 / 51 | 180 / 180 | 2872 / 2872 | 92% | 31 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 11.5 | 11.5 |
| Load average (1m) | 1.0 | 1.0 |
| CPU temp °C | 76 | 76 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:52:55+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 0 / 0 | 1.7 / 1.7 | 40 / 40 | 32 / 32 | 990 / 990 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 2 / 2 | 1.7 / 1.7 | 44 / 44 | 51 / 51 | 2017 / 2017 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 9.1 | 9.1 |
| Load average (1m) | 0.7 | 0.7 |
| CPU temp °C | 75 | 75 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.

## pipeline-real (train)

- **Started**: 2026-07-30T08:52:57+00:00
- **Duration**: 0.0 min
- **Samples**: 1 at 5s intervals
- **Note**: answerdotai/ModernBERT-base, mlm, seq 512, world 2

### GPU

| GPU | Util % (mean/max) | Mem GiB (mean/peak) | Temp °C (mean/max) | Power W (mean/max) | SM MHz (mean/min) | % of max clock | Fan % | Throttled |
|---|---|---|---|---|---|---:|---:|---:|
| 0 (NVIDIA GeForce RTX 5090, 600W cap) | 16 / 16 | 5.0 / 5.0 | 42 / 42 | 160 / 160 | 2962 / 2962 | 0% | 0 | 0% |
| 1 (NVIDIA GeForce RTX 5090, 575W cap) | 14 / 14 | 4.6 / 4.6 | 52 / 52 | 180 / 180 | 2865 / 2865 | 0% | 0 | 0% |

### CPU and host

| Metric | Mean | Max |
|---|---:|---:|
| Utilisation % | 0 | 0 |
| RAM GiB (of 60) | 12.1 | 12.1 |
| Load average (1m) | 0.7 | 0.7 |
| CPU temp °C | 78 | 78 |

### Warnings

- **The cards have different power limits** (575W, 600W). Under DDP every step waits for the slower rank, so this caps the whole run, not just one device.
