#include <cuda_timer/timer.cuh>
#include <cuda_timer/types.cuh>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <memory>

TEST(Timer, ConstructorCreatesEvents) {
    cuda_timer::Timer timer;
    EXPECT_NE(timer.event_start(), nullptr);
    EXPECT_NE(timer.event_stop(), nullptr);
}

TEST(Timer, MoveSemantics) {
    cuda_timer::Timer timer1;
    auto start1 = timer1.event_start();
    auto stop1 = timer1.event_stop();

    cuda_timer::Timer timer2 = std::move(timer1);
    EXPECT_EQ(timer2.event_start(), start1);
    EXPECT_EQ(timer2.event_stop(), stop1);
    EXPECT_EQ(timer1.event_start(), nullptr);
    EXPECT_EQ(timer1.event_stop(), nullptr);
}

TEST(Timer, MoveAssignment) {
    cuda_timer::Timer timer1;
    auto start1 = timer1.event_start();

    cuda_timer::Timer timer2;
    timer2 = std::move(timer1);
    EXPECT_EQ(timer2.event_start(), start1);
    EXPECT_EQ(timer1.event_start(), nullptr);
}

TEST(Timer, DeviceIdCapturedAtConstruction) {
    int current = 0;
    cudaGetDevice(&current);
    cuda_timer::Timer timer;
    EXPECT_EQ(timer.device_id(), current);
}

TEST(Timer, StartStopReturnsPositiveTime) {
    cuda_timer::Timer timer;
    timer.start();
    cudaDeviceSynchronize();
    float ms = timer.stop();
    EXPECT_GT(ms, 0.0f);
}

__global__ void timer_dummy_kernel() {}

TEST(Timer, ElapsedNonBlocking) {
    cuda_timer::Timer timer;
    timer.start();
    timer_dummy_kernel<<<1, 1>>>();
    // NonBlocking pattern: record stop event, poll for completion,
    // then query elapsed WITHOUT synchronizing (stop event must be
    // recorded first, otherwise cudaEventElapsedTime is invalid).
    CUDA_TIMER_CHECK(cudaEventRecord(timer.event_stop(), 0));
    cudaError_t err;
    do {
        err = cudaEventQuery(timer.event_stop());
    } while (err == cudaErrorNotReady);
    CUDA_TIMER_CHECK(err);
    float ms1 = timer.elapsed();
    float ms2 = timer.stop();
    EXPECT_GE(ms1, 0.0f);
    EXPECT_GE(ms2, ms1);
}

TEST(Timer, ExceptionSafetyConstructorThrows) {
    EXPECT_NO_THROW({
        cuda_timer::Timer timer;
    });
}

TEST(Timer, StreamParameter) {
    cudaStream_t stream = 0;
    cuda_timer::Timer timer(stream);
    EXPECT_EQ(timer.stream(), stream);
}

TEST(Timer, CustomFlags) {
    cuda_timer::Timer timer(0, cudaEventBlockingSync);
    EXPECT_NE(timer.event_start(), nullptr);
    EXPECT_NE(timer.event_stop(), nullptr);
}