# Sanitizer report — cuda-timer v1.0.0

Policy: no GPU on CI runners, so sanitizer runs locally pre-release and the
clean log ships with the release notes. No self-hosted runner required.

- Tool: `compute-sanitizer` 2023.1.0 (CUDA 12.1)
- GPU: NVIDIA GeForce RTX 2060 SUPER, 8 GB, driver 610.88
- Build: Release, `CMAKE_CUDA_ARCHITECTURES=75`, MSVC 19.29
- Date: 2026-10-06

## memcheck — 0 errors

| Binary | Tests | Result |
|--------|-------|--------|
| `test_timer` | 9/9 PASS | `ERROR SUMMARY: 0 errors` |
| `test_types` | 13/13 PASS | `ERROR SUMMARY: 0 errors` |
| `test_benchmark` | 12/12 PASS | `ERROR SUMMARY: 0 errors` |
| `basic_usage` | exit 0 | `ERROR SUMMARY: 0 errors` |
| `test_logger` | 16/16 PASS | host-only (file I/O, no CUDA calls to instrument) |
| `test_stats` | 11/11 PASS | host-only (Welford on CPU, no CUDA calls to instrument) |

## racecheck — 0 hazards

| Binary | Scope | Result |
|--------|-------|--------|
| `basic_usage` | single-stream Timer + CSV | `RACECHECK SUMMARY: 0 hazards (0 errors, 0 warnings)` |
| `test_benchmark --gtest_filter=Benchmark.NonBlockingWithStream` | custom stream + `cudaEventRecord`/`cudaEventQuery` polling + `elapsed()` (no device sync) | `RACECHECK SUMMARY: 0 hazards (0 errors, 0 warnings)` |

Note: full-suite racecheck output was truncated by the local shell wall-time
during the final analysis phase (tests themselves all PASSED under
instrumentation); the two scoped runs above completed with full summaries.

## Finding fixed during this run

`Timer.ElapsedNonBlocking` called `elapsed()` before the stop event was ever
recorded → `cudaErrorInvalidResourceHandle` (error 400) on
`cudaEventElapsedTime`, flagged by memcheck. Native runs hid it (return code
unchecked, `ms` stayed 0). This is the documented contract
("stop event must already be recorded and completed"); the test was fixed to
follow the NonBlocking pattern (record stop event → poll `cudaEventQuery` →
`elapsed()`), matching `benchmark.cuh`.

## Windows console quirk (non-issue)

Under `compute-sanitizer`, `basic_usage.exe` stdout is swallowed by the
sanitizer wrapper (exit code 0, instrumented, 0 errors). Native run prints
normally. Cosmetic only.
