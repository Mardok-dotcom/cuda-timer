#pragma once

#include "compare.cuh"
#include "types.cuh"

#include <fstream>
#include <locale>
#include <stdexcept>
#include <string>

namespace cuda_timer {
namespace detail {

/**
 * @brief Formats an integer CUDA version (e.g. 12010) as "12.1".
 */
inline std::string format_cuda_version(int version) {
    if (version <= 0) return "unknown";
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

} // namespace detail

/**
 * @brief Captures the hardware/runtime identity of a measurement.
 *
 * Uses only the CUDA runtime (no NVML): device name, compute capability,
 * CUDA driver API version and CUDA runtime version. Two series measured with
 * different values are not comparable.
 *
 * @param device_id Device to describe (default: 0)
 * @throws CudaError on CUDA failure
 */
inline Environment capture_environment(int device_id = 0) {
    int device_count = 0;
    CUDA_TIMER_CHECK(cudaGetDeviceCount(&device_count));

    cudaDeviceProp properties{};
    CUDA_TIMER_CHECK(cudaGetDeviceProperties(&properties, device_id));

    int driver_version = 0;
    int runtime_version = 0;
    CUDA_TIMER_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_TIMER_CHECK(cudaRuntimeGetVersion(&runtime_version));

    Environment environment;
    environment.device_name = properties.name;
    environment.compute_capability = std::to_string(properties.major) + "." +
                                     std::to_string(properties.minor);
    environment.driver_cuda_version = detail::format_cuda_version(driver_version);
    environment.runtime_cuda_version = detail::format_cuda_version(runtime_version);
    environment.device_count = device_count;
    return environment;
}

/**
 * @brief Writes the environment sidecar (`*.meta.json`) for a samples file.
 *
 * Typical use, next to write_samples():
 * @code
 * cuda_timer::write_samples("current.csv", "my_kernel", stats);
 * cuda_timer::write_environment("current.meta.json");
 * @endcode
 *
 * The perf_gate CLI auto-detects the sidecar by replacing the `.csv`
 * extension with `.meta.json`.
 *
 * @param path      Output path
 * @param device_id Device to describe (default: 0)
 * @throws std::runtime_error if the file cannot be written
 * @throws CudaError on CUDA failure
 */
inline void write_environment(const std::string& path, int device_id = 0) {
    const Environment environment = capture_environment(device_id);
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open environment file: " + path);
    }
    file.imbue(std::locale::classic());
    file << environment_json(environment);
    if (!file) {
        throw std::runtime_error("Failed while writing environment file: " + path);
    }
}

} // namespace cuda_timer
