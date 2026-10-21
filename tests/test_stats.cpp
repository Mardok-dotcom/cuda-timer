#include <cuda_timer/stats.cuh>
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>

TEST(Stats, EmptyStats) {
    cuda_timer::Stats s;
    EXPECT_EQ(s.count(), 0);
    EXPECT_DOUBLE_EQ(s.mean(), 0.0);
    EXPECT_DOUBLE_EQ(s.stddev(), 0.0);
    EXPECT_DOUBLE_EQ(s.min(), std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(s.max(), -std::numeric_limits<double>::infinity());
}

TEST(Stats, SingleSample) {
    cuda_timer::Stats s;
    s.add(5.0);
    EXPECT_EQ(s.count(), 1);
    EXPECT_DOUBLE_EQ(s.mean(), 5.0);
    EXPECT_DOUBLE_EQ(s.stddev(), 0.0);
    EXPECT_DOUBLE_EQ(s.min(), 5.0);
    EXPECT_DOUBLE_EQ(s.max(), 5.0);
}

TEST(Stats, WelfordAlgorithm) {
    cuda_timer::Stats s;
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (double x : data) s.add(x);

    double expected_mean = 3.0;
    double expected_var = 2.5; // sample variance

    EXPECT_DOUBLE_EQ(s.mean(), expected_mean);
    EXPECT_NEAR(s.variance(), expected_var, 1e-10);
    EXPECT_NEAR(s.stddev(), std::sqrt(expected_var), 1e-10);
    EXPECT_DOUBLE_EQ(s.min(), 1.0);
    EXPECT_DOUBLE_EQ(s.max(), 5.0);
}

TEST(Stats, ReserveAvoidsRealloc) {
    cuda_timer::Stats s;
    s.reserve(1000);
    EXPECT_GE(s.samples_.capacity(), 1000);

    for (int i = 0; i < 1000; ++i) s.add(i);
    EXPECT_EQ(s.count(), 1000);
    EXPECT_GE(s.samples_.capacity(), 1000);
}

TEST(Stats, MedianOdd) {
    cuda_timer::Stats s;
    s.reserve(5);
    s.add(3.0); s.add(1.0); s.add(2.0);
    EXPECT_DOUBLE_EQ(s.median(), 2.0);
}

TEST(Stats, MedianEven) {
    cuda_timer::Stats s;
    s.reserve(4);
    s.add(4.0); s.add(1.0); s.add(3.0); s.add(2.0);
    EXPECT_DOUBLE_EQ(s.median(), 2.5);
}

TEST(Stats, NoSamplesEmptyMedian) {
    cuda_timer::Stats s;
    EXPECT_DOUBLE_EQ(s.median(), 0.0);
}

TEST(Stats, LargeNumbersNoOverflow) {
    cuda_timer::Stats s;
    s.add(1e15); s.add(1e15 + 1); s.add(1e15 + 2);
    EXPECT_NEAR(s.mean(), 1e15 + 1, 1.0);
}

TEST(Stats, TwoSamplesVariance) {
    cuda_timer::Stats s;
    s.add(1.0);
    s.add(3.0);
    // mean = 2, variance = ((1-2)^2 + (3-2)^2) / (2-1) = 2
    EXPECT_DOUBLE_EQ(s.mean(), 2.0);
    EXPECT_DOUBLE_EQ(s.variance(), 2.0);
    EXPECT_DOUBLE_EQ(s.stddev(), std::sqrt(2.0));
}

TEST(Stats, ReserveZeroDoesNotAllocate) {
    cuda_timer::Stats s;
    s.reserve(0);
    s.add(1.0);
    EXPECT_EQ(s.count(), 1);
    EXPECT_DOUBLE_EQ(s.mean(), 1.0);
    EXPECT_TRUE(s.samples_.empty());
}

TEST(Stats, MinMaxUpdate) {
    cuda_timer::Stats s;
    s.add(10.0);
    s.add(5.0);
    s.add(15.0);
    s.add(2.0);
    EXPECT_DOUBLE_EQ(s.min(), 2.0);
    EXPECT_DOUBLE_EQ(s.max(), 15.0);
}

TEST(Stats, SamplesAccessorEmptyWithoutReserve) {
    cuda_timer::Stats s;
    s.add(1.0);
    EXPECT_TRUE(s.samples().empty());
}

TEST(Stats, SamplesAccessorReturnsStoredValues) {
    cuda_timer::Stats s;
    s.reserve(3);
    s.add(1.0);
    s.add(2.0);
    s.add(3.0);
    ASSERT_EQ(s.samples().size(), 3u);
    EXPECT_DOUBLE_EQ(s.samples()[0], 1.0);
    EXPECT_DOUBLE_EQ(s.samples()[2], 3.0);
}