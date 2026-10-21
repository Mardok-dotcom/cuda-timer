# CI performance gate

`cuda_timer::compare()` / `compare_runs()` and the `perf_gate` CLI turn
per-iteration samples into a pass/fail decision, so a kernel that got slower
fails CI before it reaches main.

## Why the median

One slow iteration (throttling, a scheduler hiccup, another process on the GPU)
inflates the mean but does not move the median. The gate decides on the median
and reports p90 for diagnosis. This is the difference between "the GPU hiccuped
once" and "the kernel is 8% slower".

## Workflow

1. Produce a baseline CSV plus its environment sidecar and commit both:

```cpp
#include <cuda_timer/benchmark.cuh>
#include <cuda_timer/environment.cuh>
#include <cuda_timer/export.cuh>

cuda_timer::BenchmarkConfig cfg;
cfg.warmup_runs = 10;
cfg.measure_runs = 200;
cfg.store_samples = true;               // keeps every sample
auto stats = cuda_timer::benchmark(launch_kernel, cfg);
cuda_timer::write_samples("bench/baseline.csv", "my_kernel", stats);
cuda_timer::write_environment("bench/baseline.meta.json");
```

2. In CI, produce `current.csv` and `current.meta.json` the same way.
3. Compare:

```bash
perf_gate --baseline bench/baseline.csv --current current.csv \
          --threshold 0.05 --json perf-report.json
```

4. The exit code decides the job:

| Exit code | Meaning |
|-----------|---------|
| `0` | Pass or improvement |
| `1` | Regression beyond `--threshold` |
| `2` | Suspect (environment mismatch, too few/noisy samples, unstable runs) or error |

## Environment fingerprint

A baseline measured on another GPU, compute capability or CUDA version is not
comparable. `write_environment()` records:

```json
{
  "device_name": "NVIDIA GeForce RTX 2060 SUPER",
  "compute_capability": "7.5",
  "driver_cuda_version": "13.3",
  "runtime_cuda_version": "12.1",
  "device_count": 1
}
```

`perf_gate` auto-detects the sidecar by replacing `.csv` with `.meta.json`
(override with `--baseline-meta` / `--current-meta`). Behaviour:

| Situation | Verdict |
|-----------|---------|
| Both sidecars, all fields match | normal comparison, `"match": true` |
| Both sidecars, any field differs | `suspect` + exit 2, reason names the field |
| Mismatch with `--allow-env-mismatch` | warning on stderr, computed verdict kept |
| Only one sidecar present | warning, comparison not verified (does not fail) |

Device count is ignored: a machine can gain or lose a GPU without invalidating a
measurement of a specific device.

## Multiple runs

Repeat `--baseline` and `--current` to feed several runs per side:

```bash
perf_gate --baseline b1.csv --baseline b2.csv --baseline b3.csv \
          --current c1.csv --current c2.csv --current c3.csv
```

Each run is reduced to its median, the gate decides on the median of the run
medians, and descriptive statistics are computed on the pooled samples.
`--max-run-spread` (default `0.02`) turns the verdict into `suspect` when the
run medians spread more than 2%: it means the machine was not stable between
runs, and a pass/fail answer would be luck.

With a single run per side the spread is zero, so the default never triggers.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--threshold <f>` | `0.05` | Max relative median regression that still passes |
| `--min-samples <n>` | `20` | Below this many pooled samples, the verdict is `suspect` |
| `--max-cv <f>` | `0` (off) | Reject runs whose coefficient of variation exceeds this |
| `--max-run-spread <f>` | `0.02` | Max spread between run medians |
| `--baseline-meta <f>` | auto | Environment sidecar for the baseline |
| `--current-meta <f>` | auto | Environment sidecar for the current run |
| `--allow-env-mismatch` | off | Do not fail on environment mismatch |
| `--json <file>` | — | Also write the JSON report to a file |
| `--quiet` | off | Do not print the JSON report to stdout |

## Programmatic use

```cpp
#include <cuda_timer/compare.cuh>

// Single run per side
auto result = cuda_timer::compare_files("baseline.csv", "current.csv");

// Several runs per side
auto multi = cuda_timer::compare_runs({baseline_a, baseline_b},
                                      {current_a, current_b});

if (result.verdict == cuda_timer::Verdict::Regression) {
    // fail the build
}
std::cout << cuda_timer::to_json(result);
```

`compare.cuh` is host-only: it does not include any CUDA header, so the CLI runs
anywhere, including a container that only holds the CSV files.

## GitHub Actions (self-hosted GPU runner)

Copy [`docs/examples/perf-gate.yml`](examples/perf-gate.yml) into
`.github/workflows/`. Hosted GitHub runners have no GPU: this job needs a
self-hosted runner with the same GPU model as the baseline.

## What makes a comparison valid

- **Same hardware, driver, CUDA version and clocks.** The fingerprint enforces
  this when sidecars are present; keep them in the repository.
- **Warm up.** Keep `warmup_runs >= 10`; the first iterations run at low clocks.
- **Enough samples.** Default `min_samples = 20`; 100-500 is better.
- **Several runs.** Three runs per side catch machine-level instability that a
  single run cannot see.
- **Quiet machine.** If the GPU is shared, set `--max-cv` and let the gate answer
  `suspect` instead of guessing.

## Timing floor

`measure_event_overhead()` (see `examples/overhead_floor.cu`) reports the
smallest interval the timing mechanism can resolve: back-to-back CUDA events
with no work in between.

```bash
$ ./build/overhead_floor
{"runs": 200, "min_ms": 0.002048, "median_ms": 0.003424, "mean_ms": 0.00398512}
```

On that device (RTX 2060 SUPER) any difference below ~0.003 ms is measurement
noise, whatever the gate says. Report the floor next to your numbers.

## Scope

- The gate compares samples. It does not capture CUDA Graphs and does not do
  per-node analysis: for that, use Nsight Systems/Compute. Timing a
  `cudaGraphLaunch` still works, because a launch is ordinary stream work.
- `compare.cuh` reads the CSV format written by `write_samples()`
  (`label,index,value`). Labels containing commas are fine: only the last field
  is parsed, with the classic "C" locale.
