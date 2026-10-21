# cuda-timer

[![CI](https://github.com/Mardok-dotcom/cuda-timer/actions/workflows/ci.yml/badge.svg)](https://github.com/Mardok-dotcom/cuda-timer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue)
![Version](https://img.shields.io/badge/version-1.1.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus&logoColor=white)

Header-only C++/CUDA library to measure GPU kernels, keep every sample, and fail CI when a kernel regresses.

`cuda-timer` is what you reach for when you want numbers *inside your own application*, not a separate benchmarking framework: RAII CUDA events, warmup + statistics, per-iteration CSV export, and a median-based performance gate that refuses to give a verdict when the comparison is not valid. 11 standalone headers, no link-time NVTX dependency.

- **RAII `Timer`** — `cudaEvent_t` wrapper with `unique_ptr` + custom deleter, move-only, multi-device aware
- **`benchmark()` harness** — warmup + measurement loop, `BenchmarkConfig`-driven, NVTX-annotated
- **Welford `Stats`** — online mean/variance/stddev/min/max + optional median and raw samples
- **CSV / JSON loggers** — thread-safe, RFC 4180 escaping, streaming JSON array, classic (C) locale for every number
- **Raw sample export** — `write_samples()` keeps every iteration for histograms and multi-modal analysis
- **CI performance gate** — `compare.cuh` + the `perf_gate` CLI turn two sample CSVs into a `pass`/`regression`/`suspect` verdict with exit codes. Hardware fingerprint and run-to-run spread make an invalid comparison `suspect` instead of silently wrong
- **NVTX runtime-loaded** — `LoadLibrary`/`dlopen`, silent no-op when `nvToolsExt` is absent

## When to use it

| You want | Use |
|----------|-----|
| Numbers inside your own app, no framework to adopt | **cuda-timer** |
| A CI gate on kernel latency, with a real exit code | **cuda-timer** (`perf_gate`) |
| Deep per-kernel counters, timeline, graph per-node analysis | [Nsight Systems / Compute](https://developer.nvidia.com/nsight-systems) |
| A parameter-sweep microbenchmark framework (grid/block axes, throttle detection, CUPTI metrics) | [nvbench](https://github.com/NVIDIA/nvbench) |
| A few lines around one `cudaEvent` pair | the CUDA runtime itself |

`cuda-timer` does not try to replace Nsight or nvbench. It fills the gap between "ten lines of `cudaEvent`" and "adopt a benchmarking framework": a small, honest measurement layer plus a gate.

## Prerequisites

- CUDA Toolkit 11.8+
- CMake 3.18+
- C++17 compiler: Visual Studio 2019+ / GCC 9+ / Clang 11+
- NVIDIA GPU with compute capability 7.0+ (CI compiles for sm_70 PTX, JIT-forward-compatible with sm_80/sm_90)
- (Optional) Nsight Systems / Nsight Compute to visualize NVTX ranges

## Quickstart

```bash
git clone https://github.com/Mardok-dotcom/cuda-timer.git
cd cuda-timer
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release

./build/basic_usage          # one-shot timing + CSV export
./build/benchmark_harness    # warmup + statistics + CSV/JSON
./build/overhead_floor       # smallest interval this GPU can resolve
./build/environment_info     # hardware fingerprint sidecar
```

On Windows with the Ninja generator the binaries are in `build\`; with the Visual Studio generator they are in `build\Release\`.

With tests (GoogleTest via FetchContent):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_TIMER_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
# Visual Studio generator: add -C Release to ctest
```

## Minimal example

```cpp
#include <cuda_timer/benchmark.cuh>
#include <cuda_timer/export.cuh>
#include <cuda_timer/timer.cuh>

// One-shot timing on a specific stream
cuda_timer::Timer timer(stream);
timer.start();
saxpy<<<grid, block, 0, stream>>>(n, a, x, y);
const float ms = timer.stop();            // synchronizes (Blocking semantics)

// Warmup + statistics + every raw sample
cuda_timer::BenchmarkConfig cfg;
cfg.warmup_runs = 10;
cfg.measure_runs = 100;
cfg.store_samples = true;                 // required for median() and export
auto stats = cuda_timer::benchmark([&] {
    saxpy<<<grid, block, 0, stream>>>(n, a, x, y);
}, cfg);

cuda_timer::write_samples("saxpy.csv", "saxpy", stats);
printf("mean %.3f ms  median %.3f ms\n", stats.mean(), stats.median());
```

`Timer::stop()` synchronizes; `Timer::elapsed()` queries the same events without synchronizing, for `NonBlocking` flows after `cudaEventQuery` reports completion.

## Integration

### Option A: `add_subdirectory`

```cmake
add_subdirectory(third_party/cuda-timer)
target_link_libraries(my_app PRIVATE cuda_timer::cuda_timer)
# Header-only INTERFACE target: include path, C++17 and NVTX_ENABLED come
# with it. NVTX itself is runtime-loaded: no NVTX link library required.
# Disable our examples/tools/tests in your build (optional):
#   set(CUDA_TIMER_BUILD_EXAMPLES OFF)
#   set(CUDA_TIMER_BUILD_TOOLS OFF)
#   set(CUDA_TIMER_BUILD_TESTS OFF)
```

Or include the headers directly:

```cpp
#include <cuda_timer/timer.cuh>
#include <cuda_timer/benchmark.cuh>
#include <cuda_timer/stats.cuh>
#include <cuda_timer/export.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/compare.cuh>
#include <cuda_timer/environment.cuh>
#include <cuda_timer/nvtx.cuh>
```

Include path: `cuda-timer/include`.

### Option B: `FetchContent`

```cmake
include(FetchContent)
FetchContent_Declare(
  cuda_timer
  GIT_REPOSITORY https://github.com/Mardok-dotcom/cuda-timer.git
  GIT_TAG        v1.1.0
)
FetchContent_MakeAvailable(cuda_timer)
target_link_libraries(my_app PRIVATE cuda_timer::cuda_timer)
```

### Option C: installed package

```cmake
find_package(cuda_timer 1.1.0 REQUIRED)
target_link_libraries(my_app PRIVATE cuda_timer::cuda_timer)
```

## API reference

### `types.cuh` — shared types

| Symbol | Kind | Description |
|--------|------|-------------|
| `TimingRecord` | struct | Label, `duration_ms`, `device_id`, `stream_id`, `timestamp_ns`, grid/block dims, `kernel_name` |
| `SyncMode` | enum | `Blocking` (sync per iteration) / `NonBlocking` (event polling) |
| `BenchmarkConfig` | struct | `warmup_runs`, `measure_runs`, `stream`, `use_nvtx`, `sync_mode`, `store_samples` |
| `CudaError` | class | `std::runtime_error` wrapper; `code()` returns the raw `cudaError_t` |
| `check_cuda_error(err, file, line)` | function | Throws `CudaError` unless `err == cudaSuccess` |
| `CUDA_TIMER_CHECK(err)` | macro | Calls `check_cuda_error` with `__FILE__` / `__LINE__` |

### `timer.cuh` — RAII GPU timer

| Symbol | Description |
|--------|-------------|
| `Timer(stream = 0, flags = cudaEventDefault)` | Creates two events, captures the current device |
| `Timer::start()` | `cudaSetDevice` + `cudaEventRecord` start event; throws `CudaError` |
| `Timer::stop()` | Records stop, synchronizes, returns ms; throws `CudaError` |
| `Timer::elapsed() const` | `cudaEventElapsedTime` without synchronizing; throws `CudaError` |
| `Timer::event_start()` / `event_stop()` | Raw `cudaEvent_t` handles (`nullptr` if moved-from) |
| `Timer::device_id()` / `stream()` | Device captured at construction / stream used |
| `detail::make_event(flags)` | `EventPtr` (`unique_ptr` + `EventDeleter`) factory |

### `stats.cuh` — online statistics

| Symbol | Description |
|--------|-------------|
| `Stats::add(x)` | Welford update of mean/variance/min/max; stores the sample only if reserved |
| `Stats::reserve(n)` | Pre-allocates the sample buffer; required for `median()` and `samples()` |
| `Stats::count()` / `mean()` / `variance()` / `stddev()` / `min()` / `max()` | Summary values |
| `Stats::median() const` | Median via `nth_element`; `0.0` if no samples stored |
| `Stats::samples() const` | Raw stored samples (empty unless `reserve()` was called) |

### `benchmark.cuh` — benchmark harness

| Symbol | Description |
|--------|-------------|
| `benchmark(kernel, cfg = {})` | Warmup loop + timed loop; returns `Stats` |
| `benchmark_kernel(launch, cfg, grid, block, name = nullptr)` | Adapts a `(grid, block)` launch callable to `benchmark()` |

`BenchmarkConfig` defaults: `warmup_runs = 10`, `measure_runs = 100`, `stream = 0`, `use_nvtx = false`, `sync_mode = SyncMode::Blocking`, `store_samples = false`.

### `csv_logger.cuh` / `json_logger.cuh` — exporters

| Symbol | Description |
|--------|-------------|
| `CsvLogger(filepath, mode = std::ios::trunc)` | Opens with the classic locale (`std::ios::app` appends); throws if unreachable |
| `CsvLogger::log(record)` | Thread-safe row write; header on first call (or first append to an empty file) |
| `CsvLogger::flush()` / `close()` | Flush / flush + close (call before reading the file on Windows) |
| `JsonLogger(filepath, pretty = true)` | Opens (truncated) and writes `[` |
| `JsonLogger::log(record)` | Thread-safe streaming append; throws if called after `close()` |
| `JsonLogger::flush()` | Pushes bytes; never writes `]`, safe to interleave with `log()` |
| `JsonLogger::close()` | Writes `]`, flush + close (also called by the destructor) |

Numbers are always formatted with `std::locale::classic()`, so a decimal comma installed by the host program cannot shift CSV columns or produce invalid JSON. Files are truncated on open unless `std::ios::app` is passed; JSON is always truncated, because two arrays cannot be concatenated into one document.

### `export.cuh` — raw sample export

| Symbol | Description |
|--------|-------------|
| `write_samples(path, label, samples, mode = std::ios::trunc)` | One CSV row per sample (`label,index,value`) at full double precision; `std::ios::app` appends without duplicating the header |
| `write_samples(path, label, stats, mode = std::ios::trunc)` | Convenience overload exporting `Stats::samples()` |

### `compare.cuh` — CI performance gate (host-only, no CUDA headers)

| Symbol | Description |
|--------|-------------|
| `Verdict` / `to_string(Verdict)` | `Pass`, `Regression`, `Improvement`, `Suspect` |
| `summarize(samples)` | `count`, `mean`, `stddev`, `min`, `max`, `median`, `p90`, `p99` |
| `compare(baseline, current, opts = {})` | Median-based verdict + relative deltas + human-readable reason |
| `compare_runs(baseline_runs, current_runs, opts = {})` | Median of per-run medians + run-to-run spread check |
| `compare_files(baseline_csv, current_csv, opts = {})` | Same, reading two `write_samples()` CSVs |
| `read_samples_csv(path)` | Locale-independent reader for `label,index,value` files |
| `Environment` / `read_environment(path)` | Hardware identity read from a `.meta.json` sidecar |
| `environment_compatible(a, b, reason)` | Compares device name, compute capability, driver and runtime CUDA versions |
| `to_json(result)` | Stable JSON report (deltas, runs, spread, environment), classic locale |

### `environment.cuh` — hardware fingerprint

| Symbol | Description |
|--------|-------------|
| `capture_environment(device = 0)` | Device name, compute capability, driver API version, runtime version, device count |
| `write_environment(path, device = 0)` | Writes the JSON sidecar consumed by `perf_gate` |

### `overhead.cuh` — timing floor

| Symbol | Description |
|--------|-------------|
| `measure_event_overhead(runs = 100, stream = 0)` | `OverheadFloor{min_ms, median_ms, mean_ms}` of back-to-back event pairs: the smallest interval the timer can resolve |

### `nvtx.cuh` — NVTX annotation

| Symbol | Description |
|--------|-------------|
| `nvtx::push_range(name, color = 0xFF0000FF)` | `nvtxRangePushEx`; no-op when the runtime DLL/`.so` is missing |
| `nvtx::pop_range()` | `nvtxRangePop`; no-op when NVTX is unavailable |
| `nvtx::ScopedRange(name, color)` | RAII push on construct / pop on destroy; move-only |

## CI performance gate

A slower kernel should fail CI, not be discovered in production. The gate compares two `write_samples()` CSVs and decides on the **median**, so a single throttling hiccup cannot fail a job on its own:

```bash
./build/perf_gate --baseline bench/baseline.csv --current current.csv \
                  --threshold 0.05 --json perf-report.json
```

Exit codes: `0` pass/improvement, `1` regression beyond `--threshold`, `2` suspect or error.

Two things make the verdict honest rather than merely numeric:

- **Hardware fingerprint.** `write_environment("current.meta.json")` records device name, compute capability and CUDA versions. `perf_gate` auto-detects the sidecar (`baseline.csv` → `baseline.meta.json`) and reports `suspect` when baseline and current do not come from the same environment.
- **Run-to-run spread.** Pass `--baseline` / `--current` more than once to compare several runs. A spread between run medians above `--max-run-spread` (default 2%) is reported as `suspect`, because the machine was not stable between runs.

```cpp
cuda_timer::BenchmarkConfig cfg;
cfg.store_samples = true;                  // keep every iteration
auto stats = cuda_timer::benchmark(launch, cfg);
cuda_timer::write_samples("current.csv", "my_kernel", stats);
cuda_timer::write_environment("current.meta.json");
```

Full guide, options, GitHub Actions template and noise rules: [`docs/perf-gate.md`](docs/perf-gate.md).

## Example benchmark

SAXPY (`n = 1<<22`, `grid = 16384`, `block = 256`), `warmup_runs = 10`, `measure_runs = 100`, `SyncMode::Blocking`, RTX 2060 SUPER, CUDA 12.1, Release:

| Metric | ms |
|--------|----|
| mean | 0.131 |
| stddev | 0.004 |
| min | 0.127 |
| max | 0.148 |
| median | 0.130 |
| samples | 100 |

```bash
./build/benchmark_harness
```

Writes `benchmark_results.csv` + `benchmark_results.json`. Results vary with GPU, clocks, driver and background load: always report the GPU model, driver and CUDA version next to the numbers.

## Performance tips

- **Warm up first.** Keep `warmup_runs >= 10` so clocks boost and caches settle before measuring.
- **`NonBlocking` for microbenchmarks.** Polls with `cudaEventQuery` instead of `cudaDeviceSynchronize()` per iteration, removing device-wide sync overhead on sub-millisecond kernels.
- **`reserve()` before `median()`.** Set `store_samples = true` (which calls `stats.reserve(measure_runs)`); otherwise `median()` returns `0.0` and memory stays flat.
- **NVTX costs ~zero when absent.** If `nvToolsExt64_1.dll` / `libnvToolsExt.so` is not found, `push_range` / `pop_range` / `ScopedRange` return immediately.
- **One `Timer` per stream/device.** Events are stream-ordered; `Timer` re-selects its construction device on every call, so multi-stream and multi-GPU code stays correct.
- **Export the distribution, not just the summary.** Mean and median hide multi-modal behaviour (clock- vs memory-bound regimes) that a histogram shows immediately.
- **Know your floor.** Run `overhead_floor` on the target machine: differences below that value are measurement noise.

## Quality

- **8 test suites / 116 GoogleTest cases**, wired into `ctest`.
- Host-only suites always run; device suites exit with code 77 without a GPU and are reported as `skipped` instead of failing.
- CI matrix: Ubuntu and Windows, CUDA 11.8 and 12.6, `-Wall -Wextra -Wpedantic -Werror` / `/W4 /WX`.
- Sanitizer report on RTX 2060 SUPER (CUDA 12.1): memcheck 0 errors, racecheck 0 hazards — [`docs/sanitizer-v1.0.0.md`](docs/sanitizer-v1.0.0.md).

## Limitations

- **No CUDA Graph support (by design).** cuda-timer does not capture graphs and does not do per-node analysis; use [Nsight Systems](https://developer.nvidia.com/nsight-systems) / Nsight Compute for that. Timing a graph *launch* needs no special support: `cudaGraphLaunch()` is ordinary stream work, so `Timer` / `benchmark()` bracket it like any kernel (covered by `tests/test_benchmark.cu`).
- **No clock/throttle sampling.** NVML is out of scope, so the fingerprint does not include clocks. Run-to-run spread catches instability between runs, not a uniform clock drop within a run.
- **The fingerprint is opt-in.** If the harness does not call `write_environment()`, `perf_gate` compares the CSVs without verifying the environment (it prints a warning). The gate is honest only when you feed it the sidecar.
- **One kernel per comparison.** There is no multi-kernel manifest yet; invoke `perf_gate` once per kernel.
- **NVTX visibility needs the runtime.** Ranges compile everywhere, but Nsight only shows them when `nvToolsExt64_1.dll` (Windows, ships with the driver/Nsight) or `libnvToolsExt.so` (Linux) is loadable at runtime.
- **Clock = GPU timestamps.** Durations come from `cudaEventElapsedTime` (GPU timestamps, ns resolution); host launch overhead is not included.

## Repository structure

```text
cuda-timer/
├── include/cuda_timer/
│   ├── types.cuh
│   ├── timer.cuh
│   ├── nvtx.cuh
│   ├── stats.cuh
│   ├── csv_logger.cuh
│   ├── json_logger.cuh
│   ├── export.cuh
│   ├── compare.cuh
│   ├── environment.cuh
│   ├── overhead.cuh
│   └── benchmark.cuh
├── examples/
│   ├── basic_usage.cu
│   ├── multi_stream.cu
│   ├── multi_device.cu
│   ├── benchmark_harness.cu
│   ├── overhead_floor.cu
│   └── environment_info.cu
├── tools/
│   └── perf_gate.cpp
├── tests/
│   ├── test_timer.cu
│   ├── test_stats.cu
│   ├── test_logger.cu
│   ├── test_compare.cpp
│   ├── test_overhead.cu
│   ├── test_environment.cu
│   ├── test_benchmark.cu
│   ├── test_types.cu
│   ├── test_main.cpp
│   └── test_main_device.cpp
├── docs/
│   ├── perf-gate.md
│   ├── examples/perf-gate.yml
│   └── sanitizer-v1.0.0.md
├── cmake/cuda_timer-config.cmake.in
├── CMakeLists.txt
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
