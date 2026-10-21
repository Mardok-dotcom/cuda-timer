#pragma once

#include "types.cuh"
#include <cstdio>
#include <fstream>
#include <locale>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cuda_timer {

/**
 * @brief Thread-safe streaming JSON array logger.
 *
 * Records are written as they are logged: the array is opened with `[` at
 * construction, records are appended (comma-separated) on log(), and the
 * closing `]` is written by close() — also invoked by the destructor.
 *
 * This is what makes flush() safe to call at any time, any number of times:
 * flush() only pushes buffered bytes to disk, it never finalizes the array.
 * Consequence to be aware of: the file holds a complete, valid JSON array
 * only after close(). While the logger is open it contains a valid array
 * *prefix*, which survives a crash and can be salvaged by appending `]`.
 *
 * Numbers are written with the classic "C" locale, so a decimal comma coming
 * from the host program's global locale cannot produce invalid JSON.
 *
 * The file is always truncated on open and cannot be appended to (two arrays
 * cannot be concatenated into one valid document). Use CsvLogger with
 * `std::ios::app` when accumulating runs in a single file matters.
 *
 * Example:
 * @code
 * cuda_timer::JsonLogger logger("timings.json", true);  // pretty
 * logger.log(record);
 * logger.flush();   // safe, repeatable, never corrupts the document
 * logger.log(record2);
 * logger.close();   // writes the closing ']'
 * @endcode
 */
class JsonLogger {
    std::ofstream file_;
    std::mutex mutex_;
    bool pretty_ = true;
    bool first_record_ = true;
    bool closed_ = false;

    /**
     * @brief Escapes a string for JSON.
     * Handles quotes, backslashes, and control characters.
     */
    static std::string escape_json(const std::string& s) {
        std::string escaped;
        escaped.reserve(s.size() + 4);
        for (char c : s) {
            switch (c) {
                case '"':  escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b";  break;
                case '\f': escaped += "\\f";  break;
                case '\n': escaped += "\\n";  break;
                case '\r': escaped += "\\r";  break;
                case '\t': escaped += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        escaped += buf;
                    } else {
                        escaped += c;
                    }
            }
        }
        return escaped;
    }

public:
    /**
     * @brief Constructs JSON logger, opens the file and starts the array.
     *
     * The file is opened with `std::ios::trunc`: previous content is
     * discarded.
     *
     * @param filepath Path to JSON output file
     * @param pretty If true, writes pretty-printed JSON with indentation
     * @throws std::runtime_error if file cannot be opened
     */
    explicit JsonLogger(std::string_view filepath, bool pretty = true)
        : file_(std::string(filepath), std::ios::trunc | std::ios::out),
          pretty_(pretty) {
        if (!file_) {
            throw std::runtime_error("Cannot open JSON file: " + std::string(filepath));
        }
        file_.imbue(std::locale::classic());
        file_ << '[';
    }

    /// Closes the array and flushes on destruction; never throws (a throwing
    /// destructor would terminate the process and lose the whole run).
    ~JsonLogger() noexcept {
        try {
            close();
        } catch (...) {
        }
    }

    // Non-copyable, non-movable: owns a stream and a mutex.
    JsonLogger(const JsonLogger&) = delete;
    JsonLogger& operator=(const JsonLogger&) = delete;
    JsonLogger(JsonLogger&&) = delete;
    JsonLogger& operator=(JsonLogger&&) = delete;

    /**
     * @brief Logs a timing record, appending it to the JSON array.
     *
     * Thread-safe. The record is written immediately (not buffered), so a
     * crash loses at most the record in flight.
     *
     * @param record TimingRecord to log
     * @throws std::runtime_error if called after close()
     */
    void log(const TimingRecord& record) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            throw std::runtime_error("JsonLogger::log() called after close()");
        }
        if (!first_record_) file_ << ',';
        if (pretty_) file_ << "\n  ";
        file_ << "{"
              << "\"label\":\"" << escape_json(record.label) << "\","
              << "\"duration_ms\":" << record.duration_ms << ","
              << "\"device_id\":" << record.device_id << ","
              << "\"stream_id\":" << record.stream_id << ","
              << "\"timestamp_ns\":" << record.timestamp_ns << ","
              << "\"grid\":[" << record.grid_x << "," << record.grid_y << ","
              << record.grid_z << "],"
              << "\"block\":[" << record.block_x << "," << record.block_y << ","
              << record.block_z << "],"
              << "\"kernel_name\":\"" << escape_json(record.kernel_name) << "\""
              << "}";
        first_record_ = false;
    }

    /**
     * @brief Pushes buffered bytes to disk.
     *
     * Thread-safe. Never finalizes the array and never writes a `]`, so it
     * can be called periodically and interleaved with log() calls.
     */
    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) file_.flush();
    }

    /**
     * @brief Writes the closing `]`, flushes, and closes the file.
     *
     * Required on Windows before reading or deleting the file, as open files
     * cannot be removed. Safe to call multiple times.
     */
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        if (file_.is_open()) {
            if (pretty_ && !first_record_) file_ << '\n';
            file_ << ']';
            file_.flush();
            file_.close();
        }
        closed_ = true;
    }
};

} // namespace cuda_timer
