#include <cuda_timer/types.cuh>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

TEST(Types, TimingRecordDefaults) {
    cuda_timer::TimingRecord r;
    EXPECT_EQ(r.label, "");
    EXPECT_FLOAT_EQ(r.duration_ms, 0.0f);
    EXPECT_EQ(r.device_id, 0);
    EXPECT_EQ(r.stream_id, 0);
    EXPECT_EQ(r.timestamp_ns, 0);
    EXPECT_EQ(r.grid_x, 1);
    EXPECT_EQ(r.grid_y, 1);
    EXPECT_EQ(r.grid_z, 1);
    EXPECT_EQ(r.block_x, 1);
    EXPECT_EQ(r.block_y, 1);
    EXPECT_EQ(r.block_z, 1);
    EXPECT_EQ(r.kernel_name, "");
}

TEST(Types, TimingRecordCustomValues) {
    cuda_timer::TimingRecord r;
    r.label = "custom";
    r.duration_ms = 1.5f;
    r.device_id = 2;
    r.stream_id = 3;
    r.timestamp_ns = 123456789;
    r.grid_x = 4; r.grid_y = 5; r.grid_z = 6;
    r.block_x = 7; r.block_y = 8; r.block_z = 9;
    r.kernel_name = "my_kernel";

    EXPECT_EQ(r.label, "custom");
    EXPECT_FLOAT_EQ(r.duration_ms, 1.5f);
    EXPECT_EQ(r.device_id, 2);
    EXPECT_EQ(r.stream_id, 3);
    EXPECT_EQ(r.timestamp_ns, 123456789);
    EXPECT_EQ(r.grid_x, 4);
    EXPECT_EQ(r.grid_y, 5);
    EXPECT_EQ(r.grid_z, 6);
    EXPECT_EQ(r.block_x, 7);
    EXPECT_EQ(r.block_y, 8);
    EXPECT_EQ(r.block_z, 9);
    EXPECT_EQ(r.kernel_name, "my_kernel");
}

TEST(Types, BenchmarkConfigDefaults) {
    cuda_timer::BenchmarkConfig cfg;
    EXPECT_EQ(cfg.warmup_runs, 10);
    EXPECT_EQ(cfg.measure_runs, 100);
    EXPECT_EQ(cfg.stream, nullptr);
    EXPECT_FALSE(cfg.use_nvtx);
    EXPECT_EQ(cfg.sync_mode, cuda_timer::SyncMode::Blocking);
    EXPECT_FALSE(cfg.store_samples);
}

TEST(Types, BenchmarkConfigCustomValues) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "no CUDA device available";
    }

    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 5;
    cfg.measure_runs = 50;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cfg.stream = stream;
    cfg.use_nvtx = true;
    cfg.sync_mode = cuda_timer::SyncMode::NonBlocking;
    cfg.store_samples = true;

    EXPECT_EQ(cfg.warmup_runs, 5);
    EXPECT_EQ(cfg.measure_runs, 50);
    EXPECT_EQ(cfg.stream, stream);
    EXPECT_TRUE(cfg.use_nvtx);
    EXPECT_EQ(cfg.sync_mode, cuda_timer::SyncMode::NonBlocking);
    EXPECT_TRUE(cfg.store_samples);

    cudaStreamDestroy(stream);
}

TEST(Types, SyncModeValues) {
    EXPECT_EQ(static_cast<int>(cuda_timer::SyncMode::Blocking), 0);
    EXPECT_EQ(static_cast<int>(cuda_timer::SyncMode::NonBlocking), 1);
}

TEST(Types, CudaErrorContainsCode) {
    cudaError_t err = cudaErrorInvalidValue;
    cuda_timer::CudaError e(err, __FILE__, __LINE__);
    EXPECT_EQ(e.code(), err);
    EXPECT_TRUE(std::string(e.what()).find("invalid") != std::string::npos);
    EXPECT_TRUE(std::string(e.what()).find(std::to_string(static_cast<int>(err))) != std::string::npos);
}

TEST(Types, CudaErrorSuccessNoThrow) {
    EXPECT_NO_THROW({
        cuda_timer::check_cuda_error(cudaSuccess, __FILE__, __LINE__);
    });
}

TEST(Types, CudaErrorInvalidValueThrows) {
    EXPECT_THROW({
        cuda_timer::check_cuda_error(cudaErrorInvalidValue, __FILE__, __LINE__);
    }, cuda_timer::CudaError);
}

TEST(Types, CudaErrorOutOfMemoryThrows) {
    EXPECT_THROW({
        cuda_timer::check_cuda_error(cudaErrorMemoryAllocation, __FILE__, __LINE__);
    }, cuda_timer::CudaError);
}

TEST(Types, CudaErrorInheritsRuntimeError) {
    cuda_timer::CudaError e(cudaErrorInvalidValue, __FILE__, __LINE__);
    const std::runtime_error* base = dynamic_cast<const std::runtime_error*>(&e);
    EXPECT_NE(base, nullptr);
}

TEST(Types, CudaErrorWhatContainsFileAndLine) {
    const char* file = "test_file.cu";
    int line = 42;
    cuda_timer::CudaError e(cudaErrorInvalidValue, file, line);
    std::string msg = e.what();
    EXPECT_TRUE(msg.find(file) != std::string::npos);
    EXPECT_TRUE(msg.find(std::to_string(line)) != std::string::npos);
}

TEST(Types, CUDA_TIMER_CHECKMacroSuccess) {
    EXPECT_NO_THROW({
        CUDA_TIMER_CHECK(cudaSuccess);
    });
}

TEST(Types, CUDA_TIMER_CHECKMacroFailure) {
    EXPECT_THROW({
        CUDA_TIMER_CHECK(cudaErrorInvalidValue);
    }, cuda_timer::CudaError);
}