# Contributing to cuda-timer

Thanks for your interest in contributing! This document covers how to build, test, follow the code style, and submit a pull request.

## Build and Test

Requirements: CUDA 11.8+, CMake 3.18+, C++17 compiler (VS 2019+ / GCC 9+ / Clang 11+).

Configure with the Ninja generator (or Visual Studio 2019+ on Windows):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native -DCUDA_TIMER_BUILD_TESTS=ON
# 'native' needs CMake 3.24+; on CMake 3.18-3.23 use -DCMAKE_CUDA_ARCHITECTURES=70 instead.
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows (Visual Studio 2019 or later):

```powershell
cmake -B build -G "Visual Studio 16 2019" -A x64 -DCUDA_TIMER_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Options: `-DCUDA_TIMER_NVTX=ON/OFF`, `-DCUDA_TIMER_BUILD_EXAMPLES=ON/OFF`, `-DCUDA_TIMER_BUILD_TESTS=ON/OFF`.

## Code Style

- **C++17 only.** No C++20 features; host code may use `string_view`; keep device code minimal.
- **RAII everywhere.** Acquire in constructors, release in destructors; `Timer` events use `unique_ptr` + custom deleter.
- **`noexcept` discipline.** `Timer::start/stop/elapsed`, `Stats::add`, and all NVTX calls are `noexcept`; throwing paths (`Timer` ctor, loggers, `CUDA_TIMER_CHECK`) use `CudaError` / `std::runtime_error`.
- **Doxygen comments.** Every public class, method, and function gets `///` / `/** */` docs with `@param` / `@return` and a `@code` example where useful.
- **Header-only rules.** One module per `.cuh`, `inline` everything, `#pragma once`, only `<cuda_runtime.h>` + STL includes (`<memory>`, `<string_view>`, `<vector>`, `<fstream>`, `<chrono>`, `<algorithm>`, `<limits>`, `<cstdint>`, `<stdexcept>`).
- **Warnings are errors.** GCC/Clang: `-Wall -Wextra -Wpedantic -Werror` (plus `-Wno-unused-parameter -Wno-missing-field-initializers`); MSVC: `/W4 /WX` (plus `/wd4100 /wd4244 /wd4267 /wd4505`) — see `cuda_timer_apply_warnings()` in `CMakeLists.txt` for the exact set.

## Pull Request Checklist

- [ ] Zero warnings with `-Werror` (GCC/Clang) / `/WX` (MSVC).
- [ ] All tests green: `ctest --output-on-failure` passes on your machine.
- [ ] New behavior covered by a test in `tests/` (`test_timer`, `test_stats`, `test_logger`, `test_benchmark`, `test_types`).
- [ ] Docs updated: Doxygen comments, `README.md` API tables, and `CHANGELOG.md` `[Unreleased]` entry.
- [ ] Headers remain standalone and header-only (no `.cpp`, no new link dependency — NVTX stays runtime-loaded).

Report issues with GPU model, driver/CUDA version, OS/compiler, CMake flags, and a minimal reproducer.
