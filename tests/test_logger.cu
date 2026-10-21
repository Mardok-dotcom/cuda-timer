#include <cuda_timer/csv_logger.cuh>
#include <cuda_timer/json_logger.cuh>
#include <cuda_timer/export.cuh>
#include <cuda_timer/types.cuh>
#include <gtest/gtest.h>
#include <fstream>
#include <iterator>
#include <locale>
#include <thread>
#include <type_traits>
#include <vector>
#include <string>
#include <filesystem>

// Contract checks for the logger types: owning a stream and a mutex makes
// them non-movable, and their destructors must never throw.
static_assert(!std::is_copy_constructible_v<cuda_timer::CsvLogger>);
static_assert(!std::is_move_constructible_v<cuda_timer::CsvLogger>,
              "CsvLogger is intentionally non-movable (owns stream + mutex)");
static_assert(!std::is_copy_constructible_v<cuda_timer::JsonLogger>);
static_assert(!std::is_move_constructible_v<cuda_timer::JsonLogger>,
              "JsonLogger is intentionally non-movable (owns stream + mutex)");
static_assert(std::is_nothrow_destructible_v<cuda_timer::CsvLogger>);
static_assert(std::is_nothrow_destructible_v<cuda_timer::JsonLogger>);

namespace {

// numpunct facet with a comma decimal point: simulates an it/fr/de global
// locale deterministically, without depending on installed system locales.
// The facet is handed to std::locale with `new`: the locale takes ownership
// (reference-counted facet set) and releases it when the restored global
// locale drops the last reference, so there is no leak here.
class CommaDecimalPunct : public std::numpunct<char> {
protected:
    char do_decimal_point() const override { return ','; }
};

// Installs a global locale and restores the previous one on scope exit.
class GlobalLocaleGuard {
    std::locale previous_;
public:
    explicit GlobalLocaleGuard(const std::locale& loc)
        : previous_(std::locale::global(loc)) {}
    ~GlobalLocaleGuard() { std::locale::global(previous_); }
    GlobalLocaleGuard(const GlobalLocaleGuard&) = delete;
    GlobalLocaleGuard& operator=(const GlobalLocaleGuard&) = delete;
};

std::string read_whole_file(const std::string& path) {
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// Exact JSON object produced for the canonical record used by the format tests
// (default stream precision prints 1.0f as "1"). Comparing whole documents
// pins down separators, indentation and the single top-level closing bracket.
const char* kCanonicalJson =
    "{\"label\":\"test\",\"duration_ms\":1,\"device_id\":0,\"stream_id\":0,"
    "\"timestamp_ns\":0,\"grid\":[1,1,1],\"block\":[1,1,1],"
    "\"kernel_name\":\"kernel\"}";

} // namespace

TEST(CsvLogger, BasicWrite) {
    const std::string filename = "test_basic.csv";
    cuda_timer::CsvLogger logger(filename);
    cuda_timer::TimingRecord r{"test", 1.5f, 0, 0, 12345, 1,1,1, 256,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line); // header
    EXPECT_TRUE(line.find("label") != std::string::npos);
    std::getline(f, line);
    EXPECT_TRUE(line.find("test") != std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, EscapeComma) {
    const std::string filename = "test_comma.csv";
    cuda_timer::CsvLogger logger(filename);
    cuda_timer::TimingRecord r{"label,with,commas", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string header, data;
    std::getline(f, header);
    std::getline(f, data);
    EXPECT_TRUE(data.find("\"label,with,commas\"") != std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, EscapeQuotes) {
    const std::string filename = "test_quotes.csv";
    cuda_timer::CsvLogger logger(filename);
    cuda_timer::TimingRecord r{"label\"with\"quotes", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string header, data;
    std::getline(f, header);
    std::getline(f, data);
    EXPECT_TRUE(data.find("\"label\"\"with\"\"quotes\"") != std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, EscapeBothCommaAndQuotes) {
    const std::string filename = "test_both.csv";
    cuda_timer::CsvLogger logger(filename);
    cuda_timer::TimingRecord r{"label,\"with\"quotes", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string header, data;
    std::getline(f, header);
    std::getline(f, data);
    EXPECT_TRUE(data.find("\"label,\"\"with\"\"quotes\"") != std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, NoEscapeNeeded) {
    const std::string filename = "test_noescape.csv";
    cuda_timer::CsvLogger logger(filename);
    cuda_timer::TimingRecord r{"simple_label", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string header, data;
    std::getline(f, header);
    std::getline(f, data);
    EXPECT_TRUE(data.find("simple_label") != std::string::npos);
    EXPECT_TRUE(data.find("\"simple_label\"") == std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, MultipleRecords) {
    const std::string filename = "test_multi.csv";
    cuda_timer::CsvLogger logger(filename);
    for (int i = 0; i < 5; ++i) {
        cuda_timer::TimingRecord r{"record_" + std::to_string(i), float(i), 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
    }
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line); // header
    int count = 0;
    while (std::getline(f, line)) ++count;
    EXPECT_EQ(count, 5);
    f.close();
    std::filesystem::remove(filename);
}

TEST(JsonLogger, BasicWrite) {
    const std::string filename = "test_basic.json";
    cuda_timer::JsonLogger logger(filename, false);
    cuda_timer::TimingRecord r{"test", 1.5f, 0, 0, 12345, 1,1,1, 256,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("\"label\":\"test\"") != std::string::npos);
    EXPECT_TRUE(content.front() == '[');
    EXPECT_TRUE(content.back() == ']');
    f.close();
    std::filesystem::remove(filename);
}

TEST(JsonLogger, PrettyPrint) {
    const std::string filename = "test_pretty.json";
    cuda_timer::JsonLogger logger(filename, true);
    cuda_timer::TimingRecord r{"test", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);
    EXPECT_EQ(content, std::string("[\n  ") + kCanonicalJson + "\n]");
}

TEST(JsonLogger, Minified) {
    const std::string filename = "test_minified.json";
    cuda_timer::JsonLogger logger(filename, false);
    cuda_timer::TimingRecord r{"test", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);
    EXPECT_EQ(content, std::string("[") + kCanonicalJson + "]");
    EXPECT_EQ(content.find('\n'), std::string::npos);
}

TEST(JsonLogger, EmptyLoggerClosesEmptyArray) {
    const std::string filename = "test_empty.json";
    {
        cuda_timer::JsonLogger logger(filename, true);
        logger.flush();
    }
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);
    EXPECT_EQ(content, "[]");
}

TEST(JsonLogger, EscapeJson) {
    const std::string filename = "test_escape.json";
    cuda_timer::JsonLogger logger(filename, false);
    cuda_timer::TimingRecord r{"label\"with\\backslash\nnewline", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("\\\"") != std::string::npos);
    EXPECT_TRUE(content.find("\\\\") != std::string::npos);
    EXPECT_TRUE(content.find("\\n") != std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(JsonLogger, MultipleRecords) {
    const std::string filename = "test_multi.json";
    cuda_timer::JsonLogger logger(filename, false);
    for (int i = 0; i < 3; ++i) {
        cuda_timer::TimingRecord r{"record_" + std::to_string(i), float(i), 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
    }
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int label_count = 0;
    size_t pos = 0;
    while ((pos = content.find("\"label\":\"record_", pos)) != std::string::npos) {
        ++label_count;
        pos += 1;
    }
    EXPECT_EQ(label_count, 3);
    f.close();
    std::filesystem::remove(filename);
}

TEST(JsonLogger, DoubleFlushProducesValidJson) {
    const std::string filename = "test_double_flush.json";
    cuda_timer::JsonLogger logger(filename, false);
    cuda_timer::TimingRecord r{"test", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    logger.log(r);
    logger.flush();
    logger.close();
    logger.flush();
    logger.close(); // Second flush should not corrupt

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.front() == '[');
    EXPECT_TRUE(content.back() == ']');
    f.close();
    std::filesystem::remove(filename);
}

TEST(Logger, ThreadSafe) {
    const std::string filename = "test_thread.csv";
    cuda_timer::CsvLogger logger(filename);
    const int threads = 4;
    const int per_thread = 100;

    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&logger, t, per_thread]() {
            for (int i = 0; i < per_thread; ++i) {
                cuda_timer::TimingRecord r{"thread_" + std::to_string(t),
                                          float(i), t, 0, 0, 1,1,1, 1,1,1, "kernel"};
                logger.log(r);
            }
        });
    }
    for (auto& th : workers) th.join();
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    int lines = 0;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) ++lines;
    EXPECT_EQ(lines, threads * per_thread);
    f.close();
    std::filesystem::remove(filename);
}

TEST(Logger, JsonThreadSafe) {
    const std::string filename = "test_thread.json";
    cuda_timer::JsonLogger logger(filename, false);
    const int threads = 4;
    const int per_thread = 50;

    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&logger, t, per_thread]() {
            for (int i = 0; i < per_thread; ++i) {
                cuda_timer::TimingRecord r{"thread_" + std::to_string(t),
                                          float(i), t, 0, 0, 1,1,1, 1,1,1, "kernel"};
                logger.log(r);
            }
        });
    }
    for (auto& th : workers) th.join();
    logger.flush();
    logger.close();

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    int total_records = 0;
    size_t pos = 0;
    while ((pos = content.find("\"label\":\"thread_", pos)) != std::string::npos) {
        ++total_records;
        pos += 1;
    }
    EXPECT_EQ(total_records, threads * per_thread);
    f.close();
    std::filesystem::remove(filename);
}

TEST(CsvLogger, FileOpenFailureThrows) {
    const std::string bad_path = "/invalid/path/that/does/not/exist/test.csv";
    EXPECT_THROW({
        cuda_timer::CsvLogger logger(bad_path);
    }, std::runtime_error);
}

TEST(JsonLogger, FileOpenFailureThrows) {
    const std::string bad_path = "/invalid/path/that/does/not/exist/test.json";
    EXPECT_THROW({
        cuda_timer::JsonLogger logger(bad_path);
    }, std::runtime_error);
}

// ---------------------------------------------------------------------------
// write_samples() — raw sample export (export.cuh)
// ---------------------------------------------------------------------------

TEST(WriteSamples, WritesHeaderAndOneRowPerSample) {
    const std::string filename = "test_samples.csv";
    cuda_timer::write_samples(filename, "saxpy", std::vector<double>{0.5, 1.25, 2.0});

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line);
    EXPECT_EQ(line, "label,index,value");
    int rows = 0;
    while (std::getline(f, line)) {
        ++rows;
        EXPECT_NE(line.find("saxpy"), std::string::npos);
    }
    EXPECT_EQ(rows, 3);
    f.close();
    std::filesystem::remove(filename);
}

TEST(WriteSamples, ValuesRoundTripAtFullPrecision) {
    const std::string filename = "test_samples_rt.csv";
    const double value = 0.123456789012345;
    cuda_timer::write_samples(filename, "k", std::vector<double>{value});

    std::ifstream f(filename);
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    const std::string field = row.substr(row.rfind(',') + 1);
    EXPECT_DOUBLE_EQ(std::stod(field), value);
    f.close();
    std::filesystem::remove(filename);
}

TEST(WriteSamples, EscapesLabelWithComma) {
    const std::string filename = "test_samples_esc.csv";
    cuda_timer::write_samples(filename, "a,b", std::vector<double>{1.0});

    std::ifstream f(filename);
    std::string header, row;
    std::getline(f, header);
    std::getline(f, row);
    EXPECT_NE(row.find("\"a,b\""), std::string::npos);
    f.close();
    std::filesystem::remove(filename);
}

TEST(WriteSamples, EscapesLabelWithNewline) {
    const std::string filename = "test_samples_nl.csv";
    cuda_timer::write_samples(filename, "a\nb", std::vector<double>{1.0});

    std::ifstream f(filename);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    std::filesystem::remove(filename);

    // A label containing a newline must be quoted (RFC 4180);
    // escape_csv (types.cuh) now quotes CR/LF too, shared with CsvLogger.
    EXPECT_NE(content.find("\"a\nb\",0,1"), std::string::npos);
}

TEST(WriteSamples, EmptySamplesWritesHeaderOnly) {
    const std::string filename = "test_samples_empty.csv";
    cuda_timer::write_samples(filename, "k", std::vector<double>{});

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line);
    EXPECT_EQ(line, "label,index,value");
    int extra = 0;
    while (std::getline(f, line)) ++extra;
    EXPECT_EQ(extra, 0);
    f.close();
    std::filesystem::remove(filename);
}

TEST(WriteSamples, FromStatsOverloadWritesStoredSamples) {
    const std::string filename = "test_samples_stats.csv";
    cuda_timer::Stats stats;
    stats.reserve(2);
    stats.add(1.0);
    stats.add(2.0);
    cuda_timer::write_samples(filename, "k", stats);

    std::ifstream f(filename);
    std::string line;
    std::getline(f, line);  // header
    int rows = 0;
    while (std::getline(f, line)) ++rows;
    EXPECT_EQ(rows, 2);
    f.close();
    std::filesystem::remove(filename);
}

TEST(WriteSamples, BadPathThrows) {
    EXPECT_THROW(
        cuda_timer::write_samples("/invalid/path/that/does/not/exist/samples.csv",
                                  "k", std::vector<double>{1.0}),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// Regression tests for the review findings
// ---------------------------------------------------------------------------

// #1 — a decimal comma in the host's global locale must never split a value
// across CSV columns.
TEST(CsvLogger, LocaleIndependentDecimalPoint) {
    const std::string filename = "test_locale.csv";
    {
        GlobalLocaleGuard guard(std::locale(std::locale::classic(), new CommaDecimalPunct));
        cuda_timer::CsvLogger logger(filename);
        cuda_timer::TimingRecord r{"k", 0.5f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
        logger.close();
    }
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);

    ASSERT_GE(content.size(), 2u);
    const size_t row_start = content.find('\n') + 1;  // skip header
    const size_t c1 = content.find(',', row_start);
    const size_t c2 = content.find(',', c1 + 1);
    ASSERT_NE(c1, std::string::npos);
    ASSERT_NE(c2, std::string::npos);
    EXPECT_EQ(content.substr(c1 + 1, c2 - c1 - 1), "0.5");
}

TEST(WriteSamples, LocaleIndependentDecimalPoint) {
    const std::string filename = "test_samples_locale.csv";
    {
        GlobalLocaleGuard guard(std::locale(std::locale::classic(), new CommaDecimalPunct));
        cuda_timer::write_samples(filename, "k", std::vector<double>{0.5});
    }
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);

    EXPECT_NE(content.find(",0,0.5"), std::string::npos);
    EXPECT_EQ(content.find(",0,0,5"), std::string::npos);
}

// The same locale hole made JSON invalid: duration_ms was written with the
// global locale's decimal separator.
TEST(JsonLogger, LocaleIndependentDecimalPoint) {
    const std::string filename = "test_locale.json";
    {
        GlobalLocaleGuard guard(std::locale(std::locale::classic(), new CommaDecimalPunct));
        cuda_timer::JsonLogger logger(filename, false);
        cuda_timer::TimingRecord r{"k", 0.5f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
        logger.close();
    }
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);

    EXPECT_NE(content.find("\"duration_ms\":0.5"), std::string::npos);
}

// #3 — appending must keep previous rows and must not duplicate the header.
TEST(CsvLogger, AppendKeepsExistingRowsAndSingleHeader) {
    const std::string filename = "test_append.csv";
    {
        cuda_timer::CsvLogger logger(filename);
        cuda_timer::TimingRecord r{"first", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
        logger.close();
    }
    {
        cuda_timer::CsvLogger logger(filename, std::ios::app);
        cuda_timer::TimingRecord r{"second", 2.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
        logger.close();
    }

    std::ifstream f(filename);
    std::string line;
    int headers = 0;
    int rows = 0;
    while (std::getline(f, line)) {
        if (line.rfind("label,duration_ms", 0) == 0) {
            ++headers;
        } else {
            ++rows;
        }
    }
    f.close();
    std::filesystem::remove(filename);
    EXPECT_EQ(headers, 1);
    EXPECT_EQ(rows, 2);
}

TEST(WriteSamples, AppendKeepsExistingRowsAndSingleHeader) {
    const std::string filename = "test_samples_append.csv";
    cuda_timer::write_samples(filename, "k", std::vector<double>{1.0});
    cuda_timer::write_samples(filename, "k2", std::vector<double>{2.0}, std::ios::app);

    std::ifstream f(filename);
    std::string line;
    int headers = 0;
    int rows = 0;
    while (std::getline(f, line)) {
        if (line == "label,index,value") {
            ++headers;
        } else {
            ++rows;
        }
    }
    f.close();
    std::filesystem::remove(filename);
    EXPECT_EQ(headers, 1);
    EXPECT_EQ(rows, 2);
}

// #6 — a periodic flush() must not finalize the array; further records have
// to stay inside it.
TEST(JsonLogger, LogAfterFlushKeepsValidJson) {
    const std::string filename = "test_json_flush_log.json";
    cuda_timer::TimingRecord a{"a", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    cuda_timer::TimingRecord b{"b", 2.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    {
        cuda_timer::JsonLogger logger(filename, false);
        logger.log(a);
        logger.flush();  // periodic flush, must not close the array
        logger.log(b);
        logger.close();
    }
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);

    EXPECT_EQ(content.front(), '[');
    EXPECT_EQ(content.back(), ']');
    EXPECT_NE(content.find("\"label\":\"a\""), std::string::npos);
    EXPECT_NE(content.find("\"label\":\"b\""), std::string::npos);
    EXPECT_EQ(content.find("]["), std::string::npos);
}

TEST(JsonLogger, LogAfterCloseThrows) {
    const std::string filename = "test_json_log_after_close.json";
    cuda_timer::TimingRecord r{"a", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
    cuda_timer::JsonLogger logger(filename, false);
    logger.log(r);
    logger.close();
    EXPECT_THROW(logger.log(r), std::runtime_error);
    std::filesystem::remove(filename);
}

TEST(JsonLogger, DestructorClosesArray) {
    const std::string filename = "test_json_dtor.json";
    {
        cuda_timer::JsonLogger logger(filename, false);
        cuda_timer::TimingRecord r{"a", 1.0f, 0, 0, 0, 1,1,1, 1,1,1, "kernel"};
        logger.log(r);
    }  // no explicit close(): the destructor must emit ']'
    const std::string content = read_whole_file(filename);
    std::filesystem::remove(filename);

    EXPECT_EQ(content.front(), '[');
    EXPECT_EQ(content.back(), ']');
}