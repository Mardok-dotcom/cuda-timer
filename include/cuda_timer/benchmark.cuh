#pragma once

#include "types.cuh"
#include "timer.cuh"
#include "stats.cuh"
#include "nvtx.cuh"
#include <functional>
#include <chrono>

namespace cuda_timer {

/**
 * @brief Benchmarks a callable kernel with configurable warmup, measurement runs, and sync mode.
 *
 * Performs warmup iterations (discarded), then timed measurement iterations.
 * Supports both Blocking (cudaDeviceSynchronize) and NonBlocking (event polling) modes.
 * Optionally emits NVTX ranges for profiling.
 *
 * @tparam Fn Callable type (lambda, function pointer, functor) with signature void()
 * @param kernel Callable to benchmark
 * @param cfg BenchmarkConfig with warmup_runs, measure_runs, sync_mode, etc.
 * @return Stats containing timing statistics
 *
 * Example:
 * @code
 * cuda_timer::BenchmarkConfig cfg;
 * cfg.warmup_runs = 5;
 * cfg.measure_runs = 100;
 * cfg.sync_mode = cuda_timer::SyncMode::Blocking;
 * cfg.store_samples = true;
 * cfg.use_nvtx = true;
 *
 * auto stats = cuda_timer::benchmark([&]() {
 *     my_kernel<<<grid, block, 0, stream>>>(args...);
 * }, cfg);
 * @endcode
 */
template <typename Fn>
Stats benchmark(Fn&& kernel, const BenchmarkConfig& cfg = {}) {
    // Warmup runs (not measured)
    for (size_t i = 0; i < cfg.warmup_runs; ++i) {
        if (cfg.use_nvtx) nvtx::push_range("warmup");
        kernel();
        if (cfg.use_nvtx) nvtx::pop_range();
        if (cfg.sync_mode == SyncMode::Blocking) {
            CUDA_TIMER_CHECK(cudaDeviceSynchronize());
        }
    }
    if (cfg.sync_mode == SyncMode::Blocking) {
        CUDA_TIMER_CHECK(cudaDeviceSynchronize());
    }

    // Measurement runs
    Stats stats;
    if (cfg.store_samples) stats.reserve(cfg.measure_runs);
    Timer timer(cfg.stream);

    for (size_t i = 0; i < cfg.measure_runs; ++i) {
        if (cfg.use_nvtx) nvtx::push_range("measure");

        timer.start();
        kernel();

        if (cfg.sync_mode == SyncMode::Blocking) {
            stats.add(timer.stop());
        } else {
            // NonBlocking: record stop event and poll for completion
            cudaSetDevice(timer.device_id());
            cudaEventRecord(timer.event_stop(), cfg.stream);
            cudaError_t err;
            do {
                err = cudaEventQuery(timer.event_stop());
            } while (err == cudaErrorNotReady);
            CUDA_TIMER_CHECK(err);
            float ms = 0.0f;
            CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, timer.event_start(), timer.event_stop()));
            stats.add(ms);
        }

        if (cfg.use_nvtx) nvtx::pop_range();
    }
    return stats;
}

/**
 * @brief Benchmark helper for kernel launches with explicit grid/block dimensions.
 *
 * Wraps a kernel launch callable (taking grid, block) into a void() callable
 * for use with benchmark().
 *
 * @tparam Fn Callable with signature void(dim3 grid, dim3 block)
 * @param launch Kernel launch callable
 * @param cfg BenchmarkConfig
 * @param grid Grid dimensions
 * @param block Block dimensions
 * @param kernel_name Optional kernel name for logging (stored in TimingRecord)
 * @return Stats containing timing statistics
 *
 * Example:
 * @code
 * auto stats = cuda_timer::benchmark_kernel(
 *     [&](dim3 grid, dim3 block) {
 *         my_kernel<<<grid, block, 0, stream>>>(args...);
 *     },
 *     cfg, grid, block, "my_kernel"
 * );
 * @endcode
 */
template <typename Fn>
Stats benchmark_kernel(Fn&& launch, const BenchmarkConfig& cfg,
                       dim3 grid, dim3 block, const char* kernel_name = nullptr) {
    auto wrapped = [&]() { launch(grid, block); };
    (void)kernel_name;  // Currently unused, reserved for future TimingRecord integration
    return benchmark(wrapped, cfg);
}

} // namespace cuda_timer