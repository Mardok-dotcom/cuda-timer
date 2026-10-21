#pragma once

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace cuda_timer {

/**
 * @brief Record containing timing information for a kernel execution.
 *
 * Used by loggers (CSV/JSON) to export timing data.
 */
struct TimingRecord {
    std::string label;           ///< User-provided label for the timing
    float duration_ms = 0.0f;    ///< Measured duration in milliseconds
    int device_id = 0;           ///< CUDA device ID where kernel executed
    int stream_id = 0;           ///< Stream ID (0 for default stream)
    uint64_t timestamp_ns = 0;   ///< Host timestamp (steady_clock) when measurement started
    int grid_x = 1, grid_y = 1, grid_z = 1;  ///< Grid dimensions
    int block_x = 1, block_y = 1, block_z = 1; ///< Block dimensions
    std::string kernel_name;     ///< Kernel name (if available)
};

/**
 * @brief Synchronization mode for benchmarking.
 *
 * - Blocking: Uses cudaDeviceSynchronize() after each kernel launch
 * - NonBlocking: Uses cudaEventRecord + cudaEventQuery polling loop
 */
enum class SyncMode { Blocking, NonBlocking };

/**
 * @brief Configuration for benchmark runs.
 */
struct BenchmarkConfig {
    size_t warmup_runs = 10;           ///< Number of warmup iterations (discarded)
    size_t measure_runs = 100;         ///< Number of measurement iterations
    cudaStream_t stream = 0;           ///< CUDA stream to use (0 = default)
    bool use_nvtx = false;             ///< Whether to emit NVTX ranges
    SyncMode sync_mode = SyncMode::Blocking; ///< Synchronization mode
    bool store_samples = false;        ///< Whether to store individual samples for median
};

/**
 * @brief Exception type for CUDA API errors.
 *
 * Inherits from std::runtime_error and stores the raw cudaError_t code.
 */
class CudaError : public std::runtime_error {
    cudaError_t code_;
public:
    /**
     * @brief Construct from CUDA error code, source file, and line number.
     */
    explicit CudaError(cudaError_t err, const char* file, int line)
        : std::runtime_error(std::string("CUDA error ") + std::to_string(static_cast<int>(err))
                           + " at " + file + ":" + std::to_string(line)
                           + ": " + cudaGetErrorString(err))
        , code_(err) {}

    /**
     * @brief Returns the raw CUDA error code.
     */
    cudaError_t code() const noexcept { return code_; }
};

/**
 * @brief Checks a CUDA error code and throws CudaError if not success.
 *
 * @param err CUDA error code to check
 * @param file Source file (automatically filled by macro)
 * @param line Source line (automatically filled by macro)
 */
inline void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw CudaError(err, file, line);
    }
}

namespace detail {

/**
 * @brief Escapes a field for RFC 4180 CSV output.
 *
 * Wraps the value in double quotes (and doubles any embedded double quote)
 * when it contains a comma, double quote, CR or LF; otherwise returns it
 * unchanged. Shared by the CSV exporters so escaping stays consistent.
 *
 * @param s Raw field value
 * @return CSV-safe field value
 */
inline std::string escape_csv(std::string_view s) {
    if (s.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string(s);
    }
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

} // namespace detail

/**
 * @brief Macro to check CUDA errors with automatic file/line capture.
 *
 * Usage: CUDA_TIMER_CHECK(cudaMalloc(...));
 */
#define CUDA_TIMER_CHECK(err) cuda_timer::check_cuda_error(err, __FILE__, __LINE__)

} // namespace cuda_timer