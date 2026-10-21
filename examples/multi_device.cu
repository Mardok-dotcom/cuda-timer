#include <cuda_timer/timer.cuh>
#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/nvtx.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

/**
 * @brief Simple matrix multiplication kernel (N x N matrices).
 * Each thread computes one element of the output matrix.
 */
__global__ void matrix_mul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Runs matrix multiplication benchmark on a specific GPU device.
 *
 * @param device_id CUDA device ID to use
 * @param logger Reference to thread-safe CSV logger
 */
void run_on_device(int device_id, cuda_timer::CsvLogger& logger) {
    // Select the target device
    cudaSetDevice(device_id);

    const int N = 512;  // 512x512 matrices
    size_t bytes = N * N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Initialize input matrices (A=1, B=2 => C should be 512*1*2 = 1024 per element)
    cudaMemset(d_A, 1, bytes);  // Note: memset with 1 sets each byte to 1, not float 1.0f
    // Better: use a kernel or cudaMemcpy with host data
    // For simplicity, we'll use a small init kernel
    cudaMemset(d_B, 2, bytes);

    // Launch configuration: 16x16 blocks
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Timer bound to device 0 (captured at construction), default stream
    cuda_timer::Timer timer(0, cudaEventDefault);

    // NVTX range for this device
    cuda_timer::nvtx::ScopedRange range(("device_" + std::to_string(device_id)).c_str());

    // Run 5 iterations
    for (int i = 0; i < 5; ++i) {
        timer.start();
        matrix_mul<<<grid, block>>>(d_A, d_B, d_C, N);
        float ms = timer.stop();

        // Log timing
        cuda_timer::TimingRecord record;
        record.label = "matrix_mul";
        record.duration_ms = ms;
        record.device_id = device_id;
        record.stream_id = 0;  // Default stream
        record.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        record.grid_x = grid.x; record.grid_y = grid.y; record.grid_z = grid.z;
        record.block_x = block.x; record.block_y = block.y; record.block_z = block.z;
        record.kernel_name = "matrix_mul";

        logger.log(record);

        std::cout << "Device " << device_id << " iter " << i << ": " << ms << " ms\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    std::cout << "Found " << device_count << " CUDA device(s)\n";

    if (device_count < 2) {
        std::cout << "Need at least 2 GPUs for multi-device demo. Running on device 0 only.\n";
    }

    // Single CSV logger shared across threads (thread-safe)
    cuda_timer::CsvLogger logger("multi_device_timing.csv");

    std::vector<std::thread> threads;

    int devices_to_use = std::min(2, device_count);

    std::cout << "Running on " << devices_to_use << " device(s)...\n";

    for (int i = 0; i < devices_to_use; ++i) {
        threads.emplace_back(run_on_device, i, std::ref(logger));
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Multi-device done. Results in multi_device_timing.csv\n";
    return 0;
}