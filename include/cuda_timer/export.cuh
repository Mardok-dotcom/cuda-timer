#pragma once

#include "types.cuh"
#include "stats.cuh"

#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace cuda_timer {

/**
 * @brief Writes raw samples to a CSV file, one row per sample.
 *
 * Unlike the summary statistics in Stats (mean/median/stddev), this keeps
 * every individual sample, so consumers can build histograms and detect
 * multi-modal timing distributions that a single number would hide.
 *
 * Output format (values at full double precision, always written with the
 * classic "C" locale so a decimal comma can never split the columns):
 * @code
 * label,index,value
 * saxpy,0,0.13270000000000001
 * saxpy,1,0.13310000000000001
 * @endcode
 *
 * The file is truncated by default. Pass `std::ios::app` to accumulate
 * several runs in one file; the header is then written only if the file is
 * new or empty.
 *
 * @param path    Output CSV file path
 * @param label   Series label written to every row (CSV-escaped)
 * @param samples Samples to export
 * @param mode    Open mode (default `std::ios::trunc`; `std::ios::app` appends)
 * @throws std::runtime_error if the file cannot be opened
 */
inline void write_samples(std::string_view path,
                          std::string_view label,
                          const std::vector<double>& samples,
                          std::ios::openmode mode = std::ios::trunc) {
    std::ofstream file{std::string(path), mode | std::ios::out};
    if (!file) {
        throw std::runtime_error("Cannot open samples file: " + std::string(path));
    }
    file.imbue(std::locale::classic());
    file.precision(std::numeric_limits<double>::max_digits10);

    bool write_header = true;
    if ((mode & std::ios::app) != 0) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(std::string(path), ec);
        write_header = ec || size == 0;
    }

    const std::string escaped = detail::escape_csv(label);
    if (write_header) {
        file << "label,index,value\n";
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        file << escaped << ',' << i << ',' << samples[i] << '\n';
    }
}

/**
 * @brief Convenience overload exporting the samples stored in a Stats object.
 *
 * Requires Stats::reserve() to have been called before adding samples
 * (e.g. BenchmarkConfig::store_samples = true); otherwise the file contains
 * only the header.
 *
 * @param path  Output CSV file path
 * @param label Series label written to every row
 * @param stats Stats whose samples should be exported
 * @param mode  Open mode (default `std::ios::trunc`; `std::ios::app` appends)
 * @throws std::runtime_error if the file cannot be opened
 */
inline void write_samples(std::string_view path,
                          std::string_view label,
                          const Stats& stats,
                          std::ios::openmode mode = std::ios::trunc) {
    write_samples(path, label, stats.samples(), mode);
}

} // namespace cuda_timer
