#include <cuda_timer/overhead.cuh>

#include <gtest/gtest.h>

#include <stdexcept>

TEST(Overhead, EventPairFloorIsMeasurable) {
    const auto floor = cuda_timer::measure_event_overhead(50);
    EXPECT_EQ(floor.runs, 50u);
    EXPECT_GE(floor.min_ms, 0.0);
    EXPECT_LE(floor.min_ms, floor.median_ms);
    EXPECT_GE(floor.mean_ms, floor.min_ms);
    EXPECT_LT(floor.median_ms, 1.0);
}

TEST(Overhead, ZeroRunsThrows) {
    EXPECT_THROW(cuda_timer::measure_event_overhead(0), std::invalid_argument);
}
