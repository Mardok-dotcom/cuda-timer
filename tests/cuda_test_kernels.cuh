#pragma once

// Device-side helpers for the CUDA test suites.
//
// GoogleTest is never included from a CUDA translation unit. nvcc's device
// pass cannot parse gtest's portability layer on a glibc host: the POSIX
// declarations it relies on (`rmdir`, `isatty`, `chdir`, `read`, `write`,
// `close`, `struct stat`) are not visible there, and the build dies inside
// gtest-port.h with "identifier is undefined" / "incomplete class type". The
// suites are therefore plain host C++ (.cpp) and only the kernels live here,
// behind host-callable launchers.

#include <cuda_runtime.h>

namespace cuda_test {

// Launches an empty kernel with the given configuration. The Timer and
// Benchmark suites only need "some stream work" to bracket with events.
void launch_dummy(dim3 grid = dim3(1), dim3 block = dim3(1), cudaStream_t stream = 0);

}  // namespace cuda_test
