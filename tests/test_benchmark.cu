#include <cuda_timer/benchmark.cuh>
#include <cuda_timer/timer.cuh>
#include <gtest/gtest.h>
#include <cuda_runtime.h>

__global__ void dummy_kernel() {}

TEST(Benchmark, BlockingModeRuns) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 2;
    cfg.measure_runs = 5;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 5);
    EXPECT_GT(stats.mean(), 0.0);
}

TEST(Benchmark, NonBlockingModeRuns) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 2;
    cfg.measure_runs = 5;
    cfg.sync_mode = cuda_timer::SyncMode::NonBlocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 5);
    EXPECT_GT(stats.mean(), 0.0);
}

TEST(Benchmark, StoreSamplesEnablesMedian) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 2;
    cfg.measure_runs = 10;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;
    cfg.store_samples = true;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 10);
    EXPECT_GT(stats.median(), 0.0);
}

TEST(Benchmark, NVTXDoesNotCrash) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 2;
    cfg.use_nvtx = true;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 2);
}

TEST(BenchmarkKernelHelper, WorksWithGridBlock) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 3;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark_kernel(
        [](dim3 g, dim3 b) { dummy_kernel<<<g,b>>>(); },
        cfg, dim3(1), dim3(256)
    );

    EXPECT_EQ(stats.count(), 3);
}

TEST(Benchmark, DefaultConfig) {
    cuda_timer::BenchmarkConfig cfg;
    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 100); // default measure_runs
    EXPECT_GT(stats.mean(), 0.0);
}

TEST(Benchmark, CustomStream) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 3;
    cfg.stream = stream;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1,0,0>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 3);
    cudaStreamDestroy(stream);
}

TEST(Benchmark, NonBlockingWithStream) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 3;
    cfg.stream = stream;
    cfg.sync_mode = cuda_timer::SyncMode::NonBlocking;

    auto stats = cuda_timer::benchmark([stream]() {
        dummy_kernel<<<1,1,0,stream>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 3);
    cudaStreamDestroy(stream);
}

TEST(Benchmark, ZeroMeasureRuns) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 0;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 0);
    EXPECT_DOUBLE_EQ(stats.mean(), 0.0);
}

TEST(Benchmark, ZeroWarmupRuns) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 0;
    cfg.measure_runs = 3;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 3);
    EXPECT_GT(stats.mean(), 0.0);
}

TEST(BenchmarkKernelHelper, KernelNameParameter) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 2;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;

    auto stats = cuda_timer::benchmark_kernel(
        [](dim3 g, dim3 b) { dummy_kernel<<<g,b>>>(); },
        cfg, dim3(2), dim3(128), "my_kernel"
    );

    EXPECT_EQ(stats.count(), 2);
}

TEST(Benchmark, StatsContainCorrectFields) {
    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 1;
    cfg.measure_runs = 10;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;
    cfg.store_samples = true;

    auto stats = cuda_timer::benchmark([]() {
        dummy_kernel<<<1,1>>>();
    }, cfg);

    EXPECT_EQ(stats.count(), 10);
    EXPECT_GT(stats.mean(), 0.0);
    EXPECT_GE(stats.max(), stats.mean());
    EXPECT_LE(stats.min(), stats.mean());
    EXPECT_GE(stats.median(), 0.0);
    EXPECT_GE(stats.stddev(), 0.0);
}

TEST(Benchmark, BracketsGraphLaunch) {
    // A cudaGraphLaunch is ordinary stream work, so benchmark() brackets it
    // with no graph-specific API: this is how a replayed graph gets timed.
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const size_t n_bytes = 4096;
    void* d_buf = nullptr;
    ASSERT_EQ(cudaMalloc(&d_buf, n_bytes), cudaSuccess);

    cudaGraph_t graph;
    ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);

    cudaMemsetParams mp{};
    mp.dst = d_buf;
    mp.pitch = 0;
    mp.value = 1;
    mp.elementSize = 1;
    mp.width = n_bytes;
    mp.height = 1;

    cudaGraphNode_t node = nullptr;
    ASSERT_EQ(cudaGraphAddMemsetNode(&node, graph, nullptr, 0, &mp), cudaSuccess);

    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiateWithFlags(&exec, graph, 0), cudaSuccess);

    cuda_timer::BenchmarkConfig cfg;
    cfg.warmup_runs = 2;
    cfg.measure_runs = 5;
    cfg.stream = stream;
    cfg.sync_mode = cuda_timer::SyncMode::Blocking;
    cfg.store_samples = true;

    auto stats = cuda_timer::benchmark([&]() {
        cudaGraphLaunch(exec, stream);
    }, cfg);

    EXPECT_EQ(stats.count(), 5);
    EXPECT_GE(stats.mean(), 0.0);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}