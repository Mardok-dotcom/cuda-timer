# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Warning flags for `.cu` files on GCC/Clang now go through `-Xcompiler`. `nvcc` has its own `-Werror` option and parsed the following flag as its value, so any Linux build of the examples/tests died with `nvcc fatal : Value '-Wno-unused-parameter' is not defined for option 'Werror'`. Found by CI (the local toolchain is MSVC, where the flags already used `-Xcompiler`).

### CI

- `windows-2022` runners ship a Visual Studio 2022 toolset (MSVC 14.44) newer than the one CUDA 11.8/12.1 support, so `nvcc` aborted the compiler-id check with `#error: -- unsupported Microsoft Visual Studio version!`. The workflow now sets `NVCC_PREPEND_FLAGS=--allow-unsupported-compiler`.

## [1.1] - 2026-10-21

### Added

- Exported `cuda_timer::cuda_timer` INTERFACE target (include path, C++17, `NVTX_ENABLED`) for `add_subdirectory`, `FetchContent` and `find_package` consumers, with real `install(EXPORT)` set.
- Raw sample export in `export.cuh`: `cuda_timer::write_samples()` writes per-iteration samples to CSV at full double precision, plus a `Stats::samples()` accessor. Enables distribution analysis (histograms, multi-modal detection) that mean/median hide. +10 GoogleTest cases.
- CI performance gate: `compare.cuh` (host-only) compares sample CSVs on the median and p90, classifies the result as `pass`/`regression`/`improvement`/`suspect`, and emits stable JSON; `compare_runs()` adds multi-run comparison with a run-to-run spread check. `environment.cuh` writes a hardware fingerprint sidecar (device name, compute capability, CUDA driver/runtime versions) and `perf_gate` refuses to decide when baseline and current come from different environments. `overhead.cuh` adds `measure_event_overhead()`, the smallest interval the event timer can resolve. The `perf_gate` CLI maps the verdict to exit codes `0`/`1`/`2`. The suite is now 8 suites / 116 GoogleTest cases.
- `std::ios::openmode` parameter on `CsvLogger` and `write_samples()` (default `std::ios::trunc`): `std::ios::app` accumulates several runs in one file and writes the header only when the file is new or empty.
- The test suite splits host-only suites (`test_logger`, `test_stats`, `test_types`) from device suites (`test_timer`, `test_benchmark`): the former run everywhere, the latter exit with code 77 without a CUDA device and ctest reports them as `skipped` (`SKIP_RETURN_CODE`). No `continue-on-error` in the CI runtime job, so the verdict is real. README now shows the actual CI status badge.
- `Timer::start()`, `Timer::stop()` and `Timer::elapsed()` now call `CUDA_TIMER_CHECK` on every CUDA API call and are no longer `noexcept`: CUDA errors surface as `CudaError` instead of failing silently.

### Changed

- CUDA Graph support is explicitly out of scope (not planned for a future release): for graph capture and per-node analysis use Nsight Systems/Compute. Timing a graph launch already works through `Timer`/`benchmark()` because a launch is ordinary stream work (covered by `tests/test_benchmark.cu`).
- CSV escaping factored into `detail::escape_csv()` (`types.cuh`), shared by `CsvLogger` and `write_samples()`; it now also quotes fields containing CR/LF (previously only comma/quote).
- `JsonLogger` now streams: `[` on construction, one record per `log()`, `]` only in `close()` (also from the destructor). `flush()` pushes bytes and never finalizes the array.

### Fixed

- A decimal comma coming from the host program's global locale (`setlocale`, `std::locale::global`) could split CSV columns and, worse, produce invalid JSON: numbers are now written with `std::locale::classic()` in `CsvLogger`, `JsonLogger` and `write_samples()`.
- `JsonLogger::log()` after a `flush()` appended records after the already-emitted `]`, silently producing invalid JSON. Records are now streamed inside a single array; `log()` after `close()` throws.
- Logger destructors are `noexcept` and swallow exceptions from `flush()` (the internal mutex lock can throw), instead of risking `std::terminate` and losing the whole measurement.
- Move operations on `CsvLogger`/`JsonLogger` are explicitly deleted and documented as such: they were already implicitly deleted by the `std::mutex` member while the code claimed the types were movable.
- Truncation on open was undocumented: now documented on `CsvLogger`, `JsonLogger` and `write_samples()`, including how to append.

## [1.0.0] - 2026-10-06

### Added

- Header-only library (7 headers in `include/cuda_timer/`): `types.cuh` (`TimingRecord`, `BenchmarkConfig`, `SyncMode`, `CudaError`), `timer.cuh` (RAII `Timer`), `nvtx.cuh` (runtime-loaded ranges), `stats.cuh` (Welford `Stats`), `csv_logger.cuh` (`CsvLogger`), `json_logger.cuh` (`JsonLogger`), `benchmark.cuh` (`benchmark`, `benchmark_kernel`).
- 4 examples: `basic_usage.cu`, `multi_stream.cu`, `multi_device.cu`, `benchmark_harness.cu`.
- 5 test suites / 61 GoogleTest cases: `test_timer` (9), `test_stats` (11), `test_logger` (16), `test_benchmark` (12), `test_types` (13), all wired into `ctest`.
- CMake install rules + `cuda_timer-config.cmake` package files and CPack ZIP/TGZ packaging.
- GoogleTest via `FetchContent` fallback (`v1.14.0`) when system GTest is missing.
- CI matrix (Ubuntu/Windows, CUDA 11.8/12.x, `CMAKE_CUDA_ARCHITECTURES=70` PTX) with examples build and best-effort `ctest` (hosted runners have no GPU; the compile job is the merge gate). No `compute-sanitizer` step and no coverage gate in CI.
- Documentation: `README.md` (badges, quickstart, integration, API reference, performance tips, limitations), `CONTRIBUTING.md`, MIT `LICENSE`.
- Sanitizer-clean on RTX 2060 SUPER (`docs/sanitizer-v1.0.0.md`): compute-sanitizer 2023.1.0 memcheck 0 errors (`test_timer`, `test_types`, `test_benchmark`, `basic_usage`); racecheck 0 hazards (single-stream + NonBlocking stream/event path).
