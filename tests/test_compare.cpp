#include <cuda_timer/compare.cuh>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <locale>
#include <string>
#include <vector>

namespace {

std::vector<double> make_series(double base, std::size_t count, double spread) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        values.push_back(base + static_cast<double>(i % 5) * spread);
    }
    return values;
}

void write_csv(const std::string& path, double base, std::size_t count, double spread) {
    std::ofstream out(path);
    out.imbue(std::locale::classic());
    out << "label,index,value\n";
    for (std::size_t i = 0; i < count; ++i) {
        out << "kernel," << i << ',' << (base + static_cast<double>(i % 5) * spread) << '\n';
    }
}

// numpunct facet with a comma decimal point: simulates an it/fr/de global
// locale. The locale owns the facet and releases it on restoration.
class CommaDecimalPunct : public std::numpunct<char> {
protected:
    char do_decimal_point() const override { return ','; }
};

class GlobalLocaleGuard {
    std::locale previous_;
public:
    explicit GlobalLocaleGuard(const std::locale& loc)
        : previous_(std::locale::global(loc)) {}
    ~GlobalLocaleGuard() { std::locale::global(previous_); }
    GlobalLocaleGuard(const GlobalLocaleGuard&) = delete;
    GlobalLocaleGuard& operator=(const GlobalLocaleGuard&) = delete;
};

} // namespace

TEST(Compare, SummarizeBasic) {
    const auto summary = cuda_timer::summarize({1.0, 2.0, 3.0, 4.0, 5.0});
    EXPECT_EQ(summary.count, 5u);
    EXPECT_DOUBLE_EQ(summary.mean, 3.0);
    EXPECT_DOUBLE_EQ(summary.median, 3.0);
    EXPECT_DOUBLE_EQ(summary.min, 1.0);
    EXPECT_DOUBLE_EQ(summary.max, 5.0);
    EXPECT_NEAR(summary.stddev, std::sqrt(2.5), 1e-12);
}

TEST(Compare, SummarizePercentiles) {
    std::vector<double> values;
    for (int i = 1; i <= 10; ++i) values.push_back(static_cast<double>(i));

    const auto summary = cuda_timer::summarize(values);
    EXPECT_DOUBLE_EQ(summary.median, 5.5);
    EXPECT_NEAR(summary.p90, 9.1, 1e-12);
    EXPECT_NEAR(summary.p99, 9.91, 1e-12);
}

TEST(Compare, PassWithinThreshold) {
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.002),
                                            make_series(1.02, 40, 0.002));
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Pass);
    EXPECT_GT(result.median_delta, 0.0);
    EXPECT_LT(result.median_delta, 0.05);
}

TEST(Compare, RegressionDetected) {
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.002),
                                            make_series(1.10, 40, 0.002));
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Regression);
    EXPECT_GT(result.median_delta, 0.05);
    EXPECT_EQ(std::string(cuda_timer::to_string(result.verdict)), "regression");
}

TEST(Compare, ImprovementDetected) {
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.002),
                                            make_series(0.90, 40, 0.002));
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Improvement);
    EXPECT_LT(result.median_delta, -0.05);
}

TEST(Compare, SuspectInsufficientSamples) {
    const auto result = cuda_timer::compare(make_series(1.0, 10, 0.002),
                                            make_series(1.1, 10, 0.002));
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("insufficient samples"), std::string::npos);
}

TEST(Compare, SuspectNonPositiveBaseline) {
    const std::vector<double> baseline(40, 0.0);
    const auto result = cuda_timer::compare(baseline, make_series(1.0, 40, 0.002));
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("not positive"), std::string::npos);
}

TEST(Compare, SuspectHighVariation) {
    cuda_timer::ComparisonOptions opts;
    opts.max_cv = 0.1;
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.002),
                                            make_series(1.0, 40, 1.0),
                                            opts);
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("coefficient of variation"), std::string::npos);
}

TEST(Compare, ReadSamplesCsvSkipsHeader) {
    const std::string path = "test_compare_read.csv";
    {
        std::ofstream out(path);
        out << "label,index,value\n";
        out << "a,0,1.5\n";
        out << "a,1,2.5\n";
    }
    const auto samples = cuda_timer::read_samples_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(samples.size(), 2u);
    EXPECT_DOUBLE_EQ(samples[0], 1.5);
    EXPECT_DOUBLE_EQ(samples[1], 2.5);
}

TEST(Compare, ReadSamplesCsvHandlesCrlfAndBlankLines) {
    const std::string path = "test_compare_crlf.csv";
    {
        std::ofstream out(path);
        out << "label,index,value\r\n";
        out << "\r\n";
        out << "a,0,1.25\r\n";
    }
    const auto samples = cuda_timer::read_samples_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(samples.size(), 1u);
    EXPECT_DOUBLE_EQ(samples[0], 1.25);
}

TEST(Compare, ReadSamplesCsvAcceptsLabelWithComma) {
    const std::string path = "test_compare_quoted.csv";
    {
        std::ofstream out(path);
        out << "label,index,value\n";
        out << "\"a,b\",0,3.5\n";
    }
    const auto samples = cuda_timer::read_samples_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(samples.size(), 1u);
    EXPECT_DOUBLE_EQ(samples[0], 3.5);
}

TEST(Compare, ReadSamplesCsvRejectsMalformedRow) {
    const std::string path = "test_compare_bad.csv";
    {
        std::ofstream out(path);
        out << "label,index,value\n";
        out << "no-comma-here\n";
    }
    EXPECT_THROW(cuda_timer::read_samples_csv(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Compare, ReadSamplesCsvRejectsEmptyFile) {
    const std::string path = "test_compare_empty.csv";
    {
        std::ofstream out(path);
        out << "label,index,value\n";
    }
    EXPECT_THROW(cuda_timer::read_samples_csv(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Compare, CompareFilesEndToEnd) {
    const std::string baseline_path = "test_compare_baseline.csv";
    const std::string current_path = "test_compare_current.csv";
    write_csv(baseline_path, 1.0, 40, 0.002);
    write_csv(current_path, 1.10, 40, 0.002);

    const auto result = cuda_timer::compare_files(baseline_path, current_path);
    std::filesystem::remove(baseline_path);
    std::filesystem::remove(current_path);

    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Regression);
    EXPECT_EQ(result.baseline.count, 40u);
    EXPECT_EQ(result.current.count, 40u);
}

TEST(Compare, JsonContainsVerdictAndDeltas) {
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.002),
                                            make_series(1.10, 40, 0.002));
    const std::string json = cuda_timer::to_json(result);

    EXPECT_NE(json.find("\"verdict\": \"regression\""), std::string::npos);
    EXPECT_NE(json.find("\"reason\": \"median regression above threshold\""), std::string::npos);
    EXPECT_NE(json.find("\"median_delta_pct\": "), std::string::npos);
    EXPECT_NE(json.find("\"baseline\": {"), std::string::npos);
    EXPECT_NE(json.find("\"current\": {"), std::string::npos);
}

TEST(Compare, JsonEscapeHandlesControlCharacters) {
    const std::string escaped = cuda_timer::detail::json_escape(std::string("a\bb\fc\x01"));
    EXPECT_EQ(escaped, "a\\bb\\fc\\u0001");
}

TEST(Compare, LocaleIndependentParsingAndJson) {
    GlobalLocaleGuard guard(std::locale(std::locale::classic(), new CommaDecimalPunct));

    const std::string path = "test_compare_locale.csv";
    {
        std::ofstream out(path);
        out.imbue(std::locale::classic());
        out << "label,index,value\n";
        out << "k,0,1.5\n";
    }
    const auto samples = cuda_timer::read_samples_csv(path);
    std::filesystem::remove(path);

    ASSERT_EQ(samples.size(), 1u);
    EXPECT_DOUBLE_EQ(samples[0], 1.5);

    const auto result = cuda_timer::compare(make_series(1.0, 30, 0.0),
                                            make_series(1.1, 30, 0.0));
    const std::string json = cuda_timer::to_json(result);
    EXPECT_NE(json.find("\"median_delta_pct\": 10.000000"), std::string::npos);
    EXPECT_EQ(json.find("10,000000"), std::string::npos);
}

TEST(Compare, CompareRunsSingleRunMatchesCompare) {
    const auto single = cuda_timer::compare_runs({make_series(1.0, 40, 0.002)},
                                                 {make_series(1.1, 40, 0.002)});
    EXPECT_EQ(single.verdict, cuda_timer::Verdict::Regression);
    EXPECT_EQ(single.baseline_runs, 1u);
    EXPECT_EQ(single.current_runs, 1u);
    EXPECT_DOUBLE_EQ(single.baseline_run_spread, 0.0);
    EXPECT_DOUBLE_EQ(single.current_run_spread, 0.0);
}

TEST(Compare, CompareRunsAveragesMultipleRuns) {
    const std::vector<std::vector<double>> baseline{
        make_series(1.000, 40, 0.002),
        make_series(1.002, 40, 0.002),
        make_series(0.998, 40, 0.002)};
    const std::vector<std::vector<double>> current{
        make_series(1.100, 40, 0.002),
        make_series(1.102, 40, 0.002),
        make_series(1.098, 40, 0.002)};

    const auto result = cuda_timer::compare_runs(baseline, current);
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Regression);
    EXPECT_EQ(result.baseline_runs, 3u);
    EXPECT_EQ(result.current_runs, 3u);
    EXPECT_LT(result.baseline_run_spread, 0.02);
    EXPECT_LT(result.current_run_spread, 0.02);
}

TEST(Compare, CompareRunsDetectsSpread) {
    const std::vector<std::vector<double>> baseline{make_series(1.0, 40, 0.0)};
    const std::vector<std::vector<double>> current{
        make_series(1.00, 40, 0.0),
        make_series(1.05, 40, 0.0),
        make_series(1.10, 40, 0.0)};

    const auto result = cuda_timer::compare_runs(baseline, current);
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("run-to-run"), std::string::npos);
}

TEST(Compare, CompareRunsRejectsEmptyRun) {
    const auto result = cuda_timer::compare_runs({std::vector<double>{}},
                                                 {make_series(1.0, 40, 0.0)});
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("empty baseline run"), std::string::npos);
}

TEST(Compare, ReadEnvironmentParsesJson) {
    const std::string path = "test_environment_parse.meta.json";
    {
        std::ofstream out(path);
        out.imbue(std::locale::classic());
        out << "{\n"
            << "  \"device_name\": \"NVIDIA GeForce RTX 2060 SUPER\",\n"
            << "  \"compute_capability\": \"7.5\",\n"
            << "  \"driver_cuda_version\": \"12.1\",\n"
            << "  \"runtime_cuda_version\": \"12.1\",\n"
            << "  \"device_count\": 1\n"
            << "}\n";
    }
    const auto env = cuda_timer::read_environment(path);
    std::filesystem::remove(path);

    EXPECT_EQ(env.device_name, "NVIDIA GeForce RTX 2060 SUPER");
    EXPECT_EQ(env.compute_capability, "7.5");
    EXPECT_EQ(env.driver_cuda_version, "12.1");
    EXPECT_EQ(env.runtime_cuda_version, "12.1");
    EXPECT_EQ(env.device_count, 1);
}

TEST(Compare, EnvironmentCompatibleDetectsMismatch) {
    cuda_timer::Environment a;
    a.device_name = "GPU A";
    a.compute_capability = "7.5";
    a.driver_cuda_version = "12.1";
    a.runtime_cuda_version = "12.1";
    a.device_count = 1;

    std::string reason;
    EXPECT_TRUE(cuda_timer::environment_compatible(a, a, &reason));
    EXPECT_TRUE(reason.empty());

    auto b = a;
    b.runtime_cuda_version = "12.6";
    EXPECT_FALSE(cuda_timer::environment_compatible(a, b, &reason));
    EXPECT_NE(reason.find("runtime_cuda_version"), std::string::npos);
}

TEST(Compare, JsonIncludesRunAndEnvironmentFields) {
    auto result = cuda_timer::compare_runs({make_series(1.0, 40, 0.0)},
                                           {make_series(1.1, 40, 0.0)});
    result.has_environment = true;
    result.environment_match = true;
    result.baseline_environment.device_name = "GPU A";
    result.current_environment = result.baseline_environment;

    const std::string json = cuda_timer::to_json(result);
    EXPECT_NE(json.find("\"runs\": {\"baseline\": 1, \"current\": 1}"), std::string::npos);
    EXPECT_NE(json.find("\"environment\": {\"present\": true, \"match\": true"), std::string::npos);
    EXPECT_NE(json.find("\"device_name\": \"GPU A\""), std::string::npos);
}

TEST(Compare, CompareRunsAppliesMinSamplesToPooled) {
    cuda_timer::ComparisonOptions opts;
    opts.min_samples = 100;
    const auto result = cuda_timer::compare_runs({make_series(1.0, 40, 0.0)},
                                                 {make_series(1.1, 40, 0.0)}, opts);
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("insufficient samples"), std::string::npos);
}

TEST(Compare, CompareRunsAppliesMaxCvToPooled) {
    cuda_timer::ComparisonOptions opts;
    opts.max_cv = 0.1;
    const auto result = cuda_timer::compare_runs({make_series(1.0, 40, 0.002)},
                                                 {make_series(1.0, 40, 1.0)}, opts);
    EXPECT_EQ(result.verdict, cuda_timer::Verdict::Suspect);
    EXPECT_NE(result.reason.find("coefficient of variation"), std::string::npos);
}

TEST(Compare, ReadEnvironmentRejectsMissingField) {
    const std::string path = "test_environment_missing.meta.json";
    {
        std::ofstream out(path);
        out << "{\n  \"device_name\": \"GPU\"\n}\n";
    }
    EXPECT_THROW(cuda_timer::read_environment(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Compare, ReadEnvironmentRejectsNonIntegerDeviceCount) {
    const std::string path = "test_environment_bad_count.meta.json";
    {
        std::ofstream out(path);
        out << "{\n"
            << "  \"device_name\": \"GPU\",\n"
            << "  \"compute_capability\": \"7.5\",\n"
            << "  \"driver_cuda_version\": \"12.1\",\n"
            << "  \"runtime_cuda_version\": \"12.1\",\n"
            << "  \"device_count\": 1.5\n"
            << "}\n";
    }
    EXPECT_THROW(cuda_timer::read_environment(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Compare, JsonWithoutEnvironmentReportsAbsent) {
    const auto result = cuda_timer::compare(make_series(1.0, 40, 0.0),
                                            make_series(1.0, 40, 0.0));
    const std::string json = cuda_timer::to_json(result);
    EXPECT_NE(json.find("\"environment\": {\"present\": false, \"match\": false}"),
              std::string::npos);
}
