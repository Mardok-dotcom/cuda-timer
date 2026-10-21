#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_timer {

/**
 * @brief Outcome of a baseline-vs-current performance comparison.
 */
enum class Verdict {
    Pass,        ///< Difference within the configured threshold
    Regression,  ///< Current is slower than baseline beyond the threshold
    Improvement, ///< Current is faster than baseline beyond the threshold
    Suspect      ///< Not enough samples, non-positive baseline, noisy data, or
                 ///< incompatible test environments
};

/**
 * @brief Returns the stable lowercase name of a verdict (for logs/JSON).
 */
inline const char* to_string(Verdict verdict) noexcept {
    switch (verdict) {
        case Verdict::Pass:        return "pass";
        case Verdict::Regression:  return "regression";
        case Verdict::Improvement: return "improvement";
        case Verdict::Suspect:     return "suspect";
    }
    return "unknown";
}

/**
 * @brief Summary statistics of one series of samples (milliseconds).
 */
struct SeriesSummary {
    std::size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

/**
 * @brief Hardware/runtime identity of a measurement.
 *
 * Read from the `.meta.json` sidecar written by environment.cuh. Two series
 * measured on different devices, compute capabilities or CUDA runtimes are not
 * comparable, so the gate reports `suspect` instead of a meaningless verdict.
 */
struct Environment {
    std::string device_name;
    std::string compute_capability;
    std::string driver_cuda_version;
    std::string runtime_cuda_version;
    int device_count = 0;
};

/**
 * @brief Tuning knobs for compare() / compare_runs().
 */
struct ComparisonOptions {
    /// Maximum relative regression on the median that still counts as Pass.
    /// The comparison uses the median, not the mean: it is far less sensitive
    /// to a single slow outlier.
    double threshold = 0.05;
    /// Below this many samples the verdict is Suspect (noise, not a result).
    std::size_t min_samples = 20;
    /// If > 0, a current series with coefficient of variation above this value
    /// is reported as Suspect (the run was too noisy to draw a conclusion).
    double max_cv = 0.0;
    /// If > 0 and several runs are supplied, a spread between run medians
    /// above this relative value is reported as Suspect. Applies to
    /// compare_runs() only; a single run has zero spread.
    double max_run_spread = 0.02;
};

/**
 * @brief Result of a comparison: summaries, deltas, verdict and environment.
 */
struct ComparisonResult {
    SeriesSummary baseline;
    SeriesSummary current;
    double median_delta = 0.0;  ///< Relative delta on the median (0.05 = +5%)
    double p90_delta = 0.0;     ///< Relative delta on the 90th percentile
    Verdict verdict = Verdict::Suspect;
    std::string reason;

    /// Run-to-run information (1 run when compare() is used directly).
    std::size_t baseline_runs = 1;
    std::size_t current_runs = 1;
    double baseline_run_spread = 0.0;
    double current_run_spread = 0.0;

    /// Environment metadata, filled by the CLI when both sidecars are present.
    bool has_environment = false;
    bool environment_match = false;
    Environment baseline_environment;
    Environment current_environment;
};

/**
 * @brief Computes count/mean/stddev/min/max/median/p90/p99 of a series.
 */
inline SeriesSummary summarize(const std::vector<double>& samples);

namespace detail {

/**
 * @brief Locale-independent string-to-double conversion.
 *
 * std::stod uses the global locale: with an Italian/French/German global
 * locale installed it rejects "0.5". This helper always parses the classic
 * "C" format, matching what write_samples() emits.
 */
inline double parse_double(const std::string& text) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    double value = 0.0;
    stream >> value;
    if (stream.fail()) {
        throw std::runtime_error("invalid number: '" + text + "'");
    }
    stream >> std::ws;
    if (!stream.eof()) {
        throw std::runtime_error("trailing characters after number: '" + text + "'");
    }
    return value;
}

inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted.front();
    const double position = p * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const double fraction = position - static_cast<double>(lower);
    const std::size_t upper = (lower + 1 < sorted.size()) ? lower + 1 : lower;
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

inline std::string json_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size() + 4);
    for (unsigned char c : text) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b";  break;
            case '\f': escaped += "\\f";  break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                                  static_cast<unsigned int>(c));
                    escaped += buffer;
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    return escaped;
}

inline std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

/**
 * @brief Extracts a flat JSON field (string or number) from a known schema.
 *
 * Only used for the `.meta.json` sidecar written by environment.cuh: it is not
 * a general-purpose JSON parser. Throws if the field is missing or malformed.
 */
inline std::string json_field(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing field in environment file: " + key);
    }
    const auto colon = text.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        throw std::runtime_error("malformed field in environment file: " + key);
    }

    std::size_t pos = colon + 1;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;

    if (pos < text.size() && text[pos] == '"') {
        ++pos;
        std::string value;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (c == '\\' && pos < text.size()) {
                value += text[pos++];
                continue;
            }
            if (c == '"') return value;
            value += c;
        }
        throw std::runtime_error("unterminated string field in environment file: " + key);
    }

    const auto end = text.find_first_of(",\n}", pos);
    const std::string raw = (end == std::string::npos) ? text.substr(pos)
                                                       : text.substr(pos, end - pos);
    return trim(raw);
}

inline std::string summary_json(const SeriesSummary& s) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6);
    out << "{\"count\": " << s.count
        << ", \"mean_ms\": " << s.mean
        << ", \"median_ms\": " << s.median
        << ", \"p90_ms\": " << s.p90
        << ", \"p99_ms\": " << s.p99
        << ", \"min_ms\": " << s.min
        << ", \"max_ms\": " << s.max
        << ", \"stddev_ms\": " << s.stddev << "}";
    return out.str();
}

inline std::string environment_inline_json(const Environment& e) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"device_name\": \"" << json_escape(e.device_name) << "\""
        << ", \"compute_capability\": \"" << json_escape(e.compute_capability) << "\""
        << ", \"driver_cuda_version\": \"" << json_escape(e.driver_cuda_version) << "\""
        << ", \"runtime_cuda_version\": \"" << json_escape(e.runtime_cuda_version) << "\""
        << ", \"device_count\": " << e.device_count << "}";
    return out.str();
}

inline std::vector<double> run_medians(const std::vector<std::vector<double>>& runs) {
    std::vector<double> medians;
    medians.reserve(runs.size());
    for (const auto& run : runs) {
        medians.push_back(summarize(run).median);
    }
    return medians;
}

inline std::vector<double> pool_runs(const std::vector<std::vector<double>>& runs) {
    std::vector<double> pooled;
    for (const auto& run : runs) {
        pooled.insert(pooled.end(), run.begin(), run.end());
    }
    return pooled;
}

inline double run_spread(const std::vector<double>& medians) {
    if (medians.size() < 2) return 0.0;
    const auto bounds = std::minmax_element(medians.begin(), medians.end());
    const double center = summarize(medians).median;
    if (!(center > 0.0)) return 0.0;
    return (*bounds.second - *bounds.first) / center;
}

} // namespace detail

/**
 * @brief Computes count/mean/stddev/min/max/median/p90/p99 of a series.
 *
 * Median and percentiles use linear interpolation between the two closest
 * order statistics, which keeps them stable for both odd and even counts.
 */
inline SeriesSummary summarize(const std::vector<double>& samples) {
    SeriesSummary summary;
    if (samples.empty()) return summary;

    summary.count = samples.size();

    double sum = 0.0;
    for (double value : samples) sum += value;
    summary.mean = sum / static_cast<double>(samples.size());

    double m2 = 0.0;
    for (double value : samples) {
        const double delta = value - summary.mean;
        m2 += delta * delta;
    }
    summary.stddev = (samples.size() > 1)
        ? std::sqrt(m2 / static_cast<double>(samples.size() - 1))
        : 0.0;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    summary.min = sorted.front();
    summary.max = sorted.back();
    summary.median = detail::percentile(sorted, 0.50);
    summary.p90 = detail::percentile(sorted, 0.90);
    summary.p99 = detail::percentile(sorted, 0.99);
    return summary;
}

/**
 * @brief Compares two series of samples and classifies the difference.
 *
 * The median is the decision statistic: a single slow iteration (throttling,
 * scheduler hiccup) must not fail a gate on its own. p90 is reported for
 * diagnostics.
 *
 * @param baseline Samples of the reference run
 * @param current  Samples of the run under test
 * @param opts     Thresholds and minimum sample count
 * @return Summaries, relative deltas, verdict and a human-readable reason
 */
inline ComparisonResult compare(const std::vector<double>& baseline,
                                const std::vector<double>& current,
                                const ComparisonOptions& opts = {}) {
    ComparisonResult result;
    result.baseline = summarize(baseline);
    result.current = summarize(current);

    if (result.baseline.count < opts.min_samples || result.current.count < opts.min_samples) {
        result.verdict = Verdict::Suspect;
        result.reason = "insufficient samples (min_samples=" +
                        std::to_string(opts.min_samples) + ")";
        return result;
    }
    if (!(result.baseline.median > 0.0)) {
        result.verdict = Verdict::Suspect;
        result.reason = "baseline median is not positive";
        return result;
    }

    result.median_delta = (result.current.median - result.baseline.median) / result.baseline.median;
    result.p90_delta = (result.baseline.p90 > 0.0)
        ? (result.current.p90 - result.baseline.p90) / result.baseline.p90
        : 0.0;

    const double cv = (result.current.mean > 0.0)
        ? result.current.stddev / result.current.mean
        : 0.0;
    if (opts.max_cv > 0.0 && cv > opts.max_cv) {
        result.verdict = Verdict::Suspect;
        result.reason = "current coefficient of variation too high";
        return result;
    }

    if (result.median_delta > opts.threshold) {
        result.verdict = Verdict::Regression;
        result.reason = "median regression above threshold";
    } else if (result.median_delta < -opts.threshold) {
        result.verdict = Verdict::Improvement;
        result.reason = "median improvement above threshold";
    } else {
        result.verdict = Verdict::Pass;
        result.reason = "within threshold";
    }
    return result;
}

/**
 * @brief Compares two sets of runs (each run is one measured series).
 *
 * Decision statistic: the median of the per-run medians. Descriptive
 * statistics (mean, p90, stddev) are computed on the pooled samples of all
 * runs. When more than one run per side is provided, the spread between run
 * medians is checked against `opts.max_run_spread`; a larger spread means the
 * machine was not stable between runs and the verdict is Suspect.
 *
 * @param baseline_runs One series of samples per reference run
 * @param current_runs  One series of samples per run under test
 * @param opts          Thresholds and minimum sample count
 */
inline ComparisonResult compare_runs(
    const std::vector<std::vector<double>>& baseline_runs,
    const std::vector<std::vector<double>>& current_runs,
    const ComparisonOptions& opts = {}) {
    ComparisonResult result;

    if (baseline_runs.empty() || current_runs.empty()) {
        result.verdict = Verdict::Suspect;
        result.reason = "no runs provided";
        return result;
    }
    for (const auto& run : baseline_runs) {
        if (run.empty()) {
            result.verdict = Verdict::Suspect;
            result.reason = "empty baseline run";
            return result;
        }
    }
    for (const auto& run : current_runs) {
        if (run.empty()) {
            result.verdict = Verdict::Suspect;
            result.reason = "empty current run";
            return result;
        }
    }

    const auto baseline_medians = detail::run_medians(baseline_runs);
    const auto current_medians = detail::run_medians(current_runs);

    // Decide on the per-run medians; min_samples/max_cv are re-applied below
    // on the pooled samples, which is what they are documented to describe.
    ComparisonOptions inner = opts;
    inner.min_samples = 1;
    inner.max_cv = 0.0;
    inner.max_run_spread = 0.0;
    result = compare(baseline_medians, current_medians, inner);

    result.baseline = summarize(detail::pool_runs(baseline_runs));
    result.current = summarize(detail::pool_runs(current_runs));
    result.baseline_runs = baseline_runs.size();
    result.current_runs = current_runs.size();
    result.baseline_run_spread = detail::run_spread(baseline_medians);
    result.current_run_spread = detail::run_spread(current_medians);
    result.p90_delta = (result.baseline.p90 > 0.0)
        ? (result.current.p90 - result.baseline.p90) / result.baseline.p90
        : 0.0;

    if (result.baseline.count < opts.min_samples || result.current.count < opts.min_samples) {
        result.verdict = Verdict::Suspect;
        result.reason = "insufficient samples (min_samples=" +
                        std::to_string(opts.min_samples) + ")";
        return result;
    }
    if (!(result.baseline.median > 0.0)) {
        result.verdict = Verdict::Suspect;
        result.reason = "baseline median is not positive";
        return result;
    }
    if (opts.max_cv > 0.0) {
        const double cv = (result.current.mean > 0.0)
            ? result.current.stddev / result.current.mean
            : 0.0;
        if (cv > opts.max_cv) {
            result.verdict = Verdict::Suspect;
            result.reason = "current coefficient of variation too high";
            return result;
        }
    }
    const double worst_spread = std::max(result.baseline_run_spread,
                                         result.current_run_spread);
    if (opts.max_run_spread > 0.0 && worst_spread > opts.max_run_spread) {
        result.verdict = Verdict::Suspect;
        result.reason = "run-to-run variance too high";
        return result;
    }
    return result;
}

/**
 * @brief Reads a samples CSV as produced by write_samples().
 *
 * Expected format: `label,index,value` header (optional) followed by one row
 * per sample. Only the last comma-separated field is read, so labels
 * containing commas (quoted per RFC 4180) are harmless. Parsing uses the
 * classic "C" locale, so a decimal comma installed globally by the host
 * program cannot break it.
 *
 * @param path Path to the CSV file
 * @throws std::runtime_error if the file is missing, empty or malformed
 */
inline std::vector<double> read_samples_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open samples file: " + path);
    }

    std::vector<double> samples;
    std::string line;
    std::size_t line_number = 0;
    bool first_content_line = true;

    while (std::getline(file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (first_content_line && line.rfind("label,", 0) == 0) {
            first_content_line = false;
            continue;
        }
        first_content_line = false;

        const auto comma = line.rfind(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("malformed row at line " + std::to_string(line_number) +
                                     " in " + path);
        }
        try {
            samples.push_back(detail::parse_double(line.substr(comma + 1)));
        } catch (const std::exception& e) {
            throw std::runtime_error("line " + std::to_string(line_number) + " in " + path +
                                     ": " + e.what());
        }
    }

    if (samples.empty()) {
        throw std::runtime_error("no samples in " + path);
    }
    return samples;
}

/**
 * @brief Convenience overload comparing two samples CSV files.
 */
inline ComparisonResult compare_files(const std::string& baseline_csv,
                                      const std::string& current_csv,
                                      const ComparisonOptions& opts = {}) {
    return compare(read_samples_csv(baseline_csv), read_samples_csv(current_csv), opts);
}

/**
 * @brief Reads the environment sidecar written by write_environment().
 *
 * @throws std::runtime_error if the file is missing or a required field is not
 *         present
 */
inline Environment read_environment(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open environment file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    Environment env;
    env.device_name = detail::json_field(text, "device_name");
    env.compute_capability = detail::json_field(text, "compute_capability");
    env.driver_cuda_version = detail::json_field(text, "driver_cuda_version");
    env.runtime_cuda_version = detail::json_field(text, "runtime_cuda_version");
    const double device_count = detail::parse_double(
        detail::json_field(text, "device_count"));
    if (device_count < 0.0 || device_count != std::floor(device_count)) {
        throw std::runtime_error("invalid device_count in environment file: " + path);
    }
    env.device_count = static_cast<int>(device_count);
    return env;
}

/**
 * @brief Serializes an environment as a multi-line JSON document.
 */
inline std::string environment_json(const Environment& env) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\n"
        << "  \"device_name\": \"" << detail::json_escape(env.device_name) << "\",\n"
        << "  \"compute_capability\": \"" << detail::json_escape(env.compute_capability) << "\",\n"
        << "  \"driver_cuda_version\": \"" << detail::json_escape(env.driver_cuda_version) << "\",\n"
        << "  \"runtime_cuda_version\": \"" << detail::json_escape(env.runtime_cuda_version) << "\",\n"
        << "  \"device_count\": " << env.device_count << "\n"
        << "}\n";
    return out.str();
}

/**
 * @brief Decides whether two environments are comparable.
 *
 * Device count is intentionally ignored: a machine can gain or lose a GPU
 * without invalidating a measurement of a specific device. Name, compute
 * capability and CUDA versions must match.
 *
 * @param a First environment
 * @param b Second environment
 * @param reason Optional out-parameter, set to the first differing field
 * @return true when the two environments are comparable
 */
inline bool environment_compatible(const Environment& a, const Environment& b,
                                   std::string* reason = nullptr) {
    auto mismatch = [reason](const char* field) {
        if (reason) *reason = std::string("environment mismatch on ") + field;
        return false;
    };
    if (a.device_name != b.device_name) return mismatch("device_name");
    if (a.compute_capability != b.compute_capability) return mismatch("compute_capability");
    if (a.driver_cuda_version != b.driver_cuda_version) return mismatch("driver_cuda_version");
    if (a.runtime_cuda_version != b.runtime_cuda_version) return mismatch("runtime_cuda_version");
    if (reason) reason->clear();
    return true;
}

/**
 * @brief Serializes a comparison result as stable, human-readable JSON.
 *
 * Numbers are always written with the classic "C" locale, so the document is
 * valid JSON regardless of the host program's global locale.
 */
inline std::string to_json(const ComparisonResult& result) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(6);
    out << "{\n"
        << "  \"verdict\": \"" << to_string(result.verdict) << "\",\n"
        << "  \"reason\": \"" << detail::json_escape(result.reason) << "\",\n"
        << "  \"median_delta_pct\": " << result.median_delta * 100.0 << ",\n"
        << "  \"p90_delta_pct\": " << result.p90_delta * 100.0 << ",\n"
        << "  \"runs\": {\"baseline\": " << result.baseline_runs
        << ", \"current\": " << result.current_runs << "},\n"
        << "  \"run_spread_pct\": {\"baseline\": " << result.baseline_run_spread * 100.0
        << ", \"current\": " << result.current_run_spread * 100.0 << "},\n"
        << "  \"baseline\": " << detail::summary_json(result.baseline) << ",\n"
        << "  \"current\": " << detail::summary_json(result.current) << ",\n"
        << "  \"environment\": {"
        << "\"present\": " << (result.has_environment ? "true" : "false")
        << ", \"match\": " << (result.environment_match ? "true" : "false");
    if (result.has_environment) {
        out << ", \"baseline\": " << detail::environment_inline_json(result.baseline_environment)
            << ", \"current\": " << detail::environment_inline_json(result.current_environment);
    }
    out << "}\n"
        << "}\n";
    return out.str();
}

} // namespace cuda_timer
