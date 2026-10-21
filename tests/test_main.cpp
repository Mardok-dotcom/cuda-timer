#include <gtest/gtest.h>

// Host-only test main: this binary runs everywhere, including GPU-less CI
// runners, because its suite (loggers, export, Stats, types) never touches a
// CUDA device. Suites that do need a device (test_timer, test_benchmark) are
// built with tests/test_main_device.cpp instead, which exits 77 when no device
// is present.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
