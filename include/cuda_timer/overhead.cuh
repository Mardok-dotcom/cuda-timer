#pragma once

#include "timer.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cuda_timer {

/**
 * @brief Measured floor of the timing mechanism itself.
 *
 * Back-to-back CUDA events with no work in between show how small a duration
 * can be measured before it is indistinguishable from measurement overhead.
 * Treat values at or below `median_ms` as "not measurable" on this device.
 */
struct OverheadFloor {
    std::size_t runs = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;  ///< Upper middle element for even run counts
    double mean_ms = 0.0;
};

/**
 * @brief Measures the event-pair overhead by timing empty start/stop pairs.
 *
 * Each iteration records the start event, records the stop event and waits for
 * it, so the measured interval is the cost of the two events themselves, not
 * the cost of any kernel.
 *
 * @param runs   Number of start/stop pairs (>= 1)
 * @param stream Stream to record on (default: 0)
 * @return Floor statistics in milliseconds
 * @throws std::invalid_argument if runs == 0
 * @throws CudaError on CUDA failure
 */
inline OverheadFloor measure_event_overhead(std::size_t runs = 100, cudaStream_t stream = 0) {
    if (runs == 0) {
        throw std::invalid_argument("measure_event_overhead: runs must be >= 1");
    }

    Timer timer(stream);
    std::vector<double> samples;
    samples.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i) {
        timer.start();
        samples.push_back(timer.stop());
    }

    std::sort(samples.begin(), samples.end());

    OverheadFloor floor;
    floor.runs = runs;
    floor.min_ms = samples.front();
    floor.median_ms = samples[samples.size() / 2];
    double sum = 0.0;
    for (double value : samples) sum += value;
    floor.mean_ms = sum / static_cast<double>(samples.size());
    return floor;
}

} // namespace cuda_timer
