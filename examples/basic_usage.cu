#include <cuda_timer/timer.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/nvtx.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>

/**
 * @brief Simple vector addition kernel.
 * Each thread computes one element: c[idx] = a[idx] + b[idx].
 */
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // Host data initialization
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N);
    float *d_a, *d_b, *d_c;

    // Device memory allocation
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Host-to-Device copies
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // CSV logger for timing results
    cuda_timer::CsvLogger logger("basic_usage_timing.csv");

    // NVTX range for H2D copies (already completed above, shown for demonstration)
    {
        cuda_timer::nvtx::ScopedRange range("H2D copies");
        // Data already copied before this scope
    }

    // Time the kernel execution
    {
        cuda_timer::Timer timer;  // Default stream, default flags
        cuda_timer::nvtx::ScopedRange range("vector_add kernel");

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        timer.start();
        vector_add<<<grid, block>>>(d_a, d_b, d_c, N);

        // Stop timer and get elapsed time in milliseconds
        float ms = timer.stop();

        // Log timing record to CSV
        // Note: stream_id is 0 for default stream
        cuda_timer::TimingRecord record;
        record.label = "vector_add";
        record.duration_ms = ms;
        record.device_id = 0;
        record.stream_id = 0;
        record.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record.grid_x = grid.x; record.grid_y = grid.y; record.grid_z = grid.z;
        record.block_x = block.x; record.block_y = block.y; record.block_z = block.z;
        record.kernel_name = "vector_add";

        logger.log(record);

        std::cout << "vector_add: " << ms << " ms\n";
    }

    // Device-to-Host copy for verification
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(h_c[i] - 3.0f) > 1e-5) {
            ok = false;
            break;
        }
    }
    std::cout << "Result: " << (ok ? "PASS" : "FAIL") << "\n";

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return ok ? 0 : 1;
}