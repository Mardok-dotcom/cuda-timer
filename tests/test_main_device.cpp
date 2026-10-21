#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <iostream>

// Device test main: the suite in this binary creates CUDA events, streams and
// kernel launches, so it needs a physical device. Without one we exit with 77
// (= ctest SKIP_RETURN_CODE), which makes the verdict explicit ("skipped")
// instead of a misleading mix of passing host-only tests and failing device
// ones. Host-only suites use tests/test_main.cpp and always run.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cout << "[ SKIPPED ] no CUDA device available (cudaGetDeviceCount -> "
                  << static_cast<int>(status) << ")\n";
        return 77;
    }

    return RUN_ALL_TESTS();
}
