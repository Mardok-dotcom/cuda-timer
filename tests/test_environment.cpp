#include <cuda_timer/environment.cuh>

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

TEST(Environment, CapturesDeviceInfo) {
    const auto environment = cuda_timer::capture_environment(0);
    EXPECT_FALSE(environment.device_name.empty());
    EXPECT_FALSE(environment.compute_capability.empty());
    EXPECT_FALSE(environment.driver_cuda_version.empty());
    EXPECT_FALSE(environment.runtime_cuda_version.empty());
    EXPECT_GE(environment.device_count, 1);
}

TEST(Environment, WriteAndReadRoundTrip) {
    const std::string path = "test_environment_roundtrip.meta.json";
    cuda_timer::write_environment(path, 0);
    const auto loaded = cuda_timer::read_environment(path);
    std::filesystem::remove(path);

    const auto live = cuda_timer::capture_environment(0);
    EXPECT_EQ(loaded.device_name, live.device_name);
    EXPECT_EQ(loaded.compute_capability, live.compute_capability);
    EXPECT_EQ(loaded.driver_cuda_version, live.driver_cuda_version);
    EXPECT_EQ(loaded.runtime_cuda_version, live.runtime_cuda_version);
    EXPECT_EQ(loaded.device_count, live.device_count);
}

TEST(Environment, CompatibleWithItself) {
    const auto environment = cuda_timer::capture_environment(0);
    std::string reason;
    EXPECT_TRUE(cuda_timer::environment_compatible(environment, environment, &reason));
    EXPECT_TRUE(reason.empty());
}

TEST(Environment, MismatchReported) {
    auto a = cuda_timer::capture_environment(0);
    auto b = a;
    b.device_name = "Different GPU";

    std::string reason;
    EXPECT_FALSE(cuda_timer::environment_compatible(a, b, &reason));
    EXPECT_NE(reason.find("device_name"), std::string::npos);
}

TEST(Environment, FormatsCudaVersion) {
    EXPECT_EQ(cuda_timer::detail::format_cuda_version(12010), "12.1");
    EXPECT_EQ(cuda_timer::detail::format_cuda_version(11080), "11.8");
    EXPECT_EQ(cuda_timer::detail::format_cuda_version(0), "unknown");
}
