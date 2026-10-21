#pragma once

#include "types.cuh"
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

namespace cuda_timer {

/**
 * @brief Online statistics using Welford's algorithm.
 *
 * Maintains running mean, variance, min, max, and optionally stores samples
 * for median calculation. Uses numerically stable Welford algorithm for
 * incremental variance computation.
 *
 * Example:
 * @code
 * cuda_timer::Stats stats;
 * stats.reserve(100);  // Pre-allocate for performance
 * for (int i = 0; i < 100; ++i) {
 *     stats.add(measure_kernel());
 * }
 * std::cout << "Mean: " << stats.mean() << " ms\n";
 * std::cout << "StdDev: " << stats.stddev() << " ms\n";
 * std::cout << "Median: " << stats.median() << " ms\n";
 * @endcode
 */
struct Stats {
    size_t count_ = 0;                            ///< Number of samples added
    double mean_ = 0.0;                           ///< Running mean
    double m2_ = 0.0;                             ///< Sum of squared differences (for variance)
    double min_ = std::numeric_limits<double>::infinity();  ///< Minimum value
    double max_ = -std::numeric_limits<double>::infinity(); ///< Maximum value
    std::vector<double> samples_;                 ///< Stored samples (if reserve() called)

    /**
     * @brief Adds a sample using Welford's online algorithm.
     *
     * Updates mean, variance, min, max in O(1) time.
     * If samples vector has reserved capacity, stores the value.
     *
     * @param x Value to add
     */
    void add(double x) noexcept {
        ++count_;
        double delta = x - mean_;
        mean_ += delta / count_;
        double delta2 = x - mean_;
        m2_ += delta * delta2;
        if (x < min_) min_ = x;
        if (x > max_) max_ = x;
        if (!samples_.empty() || samples_.capacity() > 0) {
            samples_.push_back(x);
        }
    }

    /**
     * @brief Pre-allocates space for samples to avoid reallocations.
     *
     * Call before adding samples when you know the number of iterations.
     * If not called, samples are not stored (saving memory), and median()
     * will return 0.0.
     *
     * @param n Number of samples to reserve space for
     */
    void reserve(size_t n) { samples_.reserve(n); }

    /**
     * @brief Returns the number of samples added.
     * @return Sample count
     */
    size_t count() const noexcept { return count_; }

    /**
     * @brief Returns the stored raw samples.
     *
     * Empty unless reserve() was called before adding samples (e.g. via
     * BenchmarkConfig::store_samples = true). Use with write_samples() to
     * export the full distribution instead of just summary statistics.
     *
     * @return Reference to the stored samples
     */
    const std::vector<double>& samples() const noexcept { return samples_; }

    /**
     * @brief Returns the running mean.
     * @return Mean value, or 0.0 if no samples added
     */
    double mean() const noexcept { return mean_; }

    /**
     * @brief Returns the sample variance (Bessel's correction).
     * @return Variance, or 0.0 if fewer than 2 samples
     */
    double variance() const noexcept { return count_ > 1 ? m2_ / (count_ - 1) : 0.0; }

    /**
     * @brief Returns the sample standard deviation.
     * @return Standard deviation, or 0.0 if fewer than 2 samples
     */
    double stddev() const noexcept { return std::sqrt(variance()); }

    /**
     * @brief Returns the minimum value seen.
     * @return Minimum, or +infinity if no samples added
     */
    double min() const noexcept { return min_; }

    /**
     * @brief Returns the maximum value seen.
     * @return Maximum, or -infinity if no samples added
     */
    double max() const noexcept { return max_; }

    /**
     * @brief Returns the median of stored samples.
     *
     * Requires that reserve() was called before adding samples.
     * Makes a copy to avoid mutating internal state.
     * For even count, returns average of two middle elements.
     *
     * @return Median value, or 0.0 if no samples stored
     */
    double median() const {
        if (samples_.empty()) return 0.0;
        auto s = samples_;  // Copy to avoid mutating internal state
        const size_t mid = s.size() / 2;
        std::nth_element(s.begin(), s.begin() + mid, s.end());
        if (s.size() % 2 == 0) {
            auto max_left = *std::max_element(s.begin(), s.begin() + mid);
            return (s[mid] + max_left) / 2.0;
        }
        return s[mid];
    }
};

} // namespace cuda_timer