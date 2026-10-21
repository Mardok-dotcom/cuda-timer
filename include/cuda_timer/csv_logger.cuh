#pragma once

#include "types.cuh"
#include <filesystem>
#include <fstream>
#include <locale>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace cuda_timer {

/**
 * @brief Thread-safe CSV logger compliant with RFC 4180.
 *
 * Writes timing records as CSV rows with RFC 4180 escaping (comma, double
 * quote, CR and LF). Numbers are always written with the classic "C" locale,
 * so a decimal comma coming from the host program's global locale can never
 * split a value across columns.
 *
 * The file is opened with `std::ios::trunc` by default: previous content is
 * discarded. Pass `std::ios::app` to append to an existing file; the header
 * is then written only when the file is new or empty.
 *
 * Example:
 * @code
 * cuda_timer::CsvLogger logger("timings.csv");                    // truncate
 * cuda_timer::CsvLogger appender("timings.csv", std::ios::app);   // append
 * logger.log(record);
 * logger.close();  // flush + close (do it before reading the file on Windows)
 * @endcode
 */
class CsvLogger {
    std::ofstream file_;
    std::mutex mutex_;
    bool header_written_ = false;

public:
    /**
     * @brief Constructs CSV logger and opens the file.
     *
     * @param filepath Path to CSV output file
     * @param mode Open mode (default `std::ios::trunc`). Use `std::ios::app`
     *             to append; the header is then emitted only if the file is
     *             new or empty.
     * @throws std::runtime_error if the file cannot be opened
     */
    explicit CsvLogger(std::string_view filepath,
                       std::ios::openmode mode = std::ios::trunc)
        : file_(std::string(filepath), mode | std::ios::out) {
        if (!file_) {
            throw std::runtime_error("Cannot open CSV file: " + std::string(filepath));
        }
        file_.imbue(std::locale::classic());
        if ((mode & std::ios::app) != 0) {
            std::error_code ec;
            const auto size = std::filesystem::file_size(std::string(filepath), ec);
            header_written_ = !ec && size > 0;
        }
    }

    /// Flushes on destruction; never throws (a throwing destructor would
    /// terminate the process and lose the whole measurement).
    ~CsvLogger() noexcept {
        try {
            flush();
        } catch (...) {
        }
    }

    // Non-copyable, non-movable: owns a stream and a mutex.
    CsvLogger(const CsvLogger&) = delete;
    CsvLogger& operator=(const CsvLogger&) = delete;
    CsvLogger(CsvLogger&&) = delete;
    CsvLogger& operator=(CsvLogger&&) = delete;

    /**
     * @brief Logs a timing record as a CSV row.
     *
     * Thread-safe. Writes the header on first call (unless appending to a
     * non-empty file).
     *
     * @param record TimingRecord to log
     */
    void log(const TimingRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!header_written_) {
            file_ << "label,duration_ms,device_id,stream_id,timestamp_ns,"
                  << "grid_x,grid_y,grid_z,block_x,block_y,block_z,kernel_name\n";
            header_written_ = true;
        }
        file_ << detail::escape_csv(record.label) << ','
              << record.duration_ms << ','
              << record.device_id << ','
              << record.stream_id << ','
              << record.timestamp_ns << ','
              << record.grid_x << ',' << record.grid_y << ',' << record.grid_z << ','
              << record.block_x << ',' << record.block_y << ',' << record.block_z << ','
              << detail::escape_csv(record.kernel_name) << '\n';
    }

    /**
     * @brief Flushes output buffer to disk.
     */
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.flush();
    }

    /**
     * @brief Flushes and closes the output file.
     *
     * Required on Windows before reading or deleting the file, as open
     * files cannot be removed. Safe to call multiple times.
     */
    void close() {
        flush();
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.close();
    }
};

} // namespace cuda_timer
