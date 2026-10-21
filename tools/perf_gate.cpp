#include <cuda_timer/compare.cuh>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* program) {
    std::cout
        << "usage: " << program << " --baseline <file.csv> --current <file.csv> [options]\n"
        << "\n"
        << "Compare samples CSV files (format: label,index,value) and decide whether\n"
        << "the current run regressed against the baseline. Repeat --baseline and\n"
        << "repeat --current to compare several runs per side.\n"
        << "\n"
        << "Options:\n"
        << "  --threshold <f>        max relative median regression (default 0.05)\n"
        << "  --min-samples <n>      min pooled samples per side (default 20)\n"
        << "  --max-cv <f>           max coefficient of variation, 0 disables (default 0)\n"
        << "  --max-run-spread <f>   max spread between run medians (default 0.02)\n"
        << "  --baseline-meta <f>    environment sidecar for the baseline\n"
        << "  --current-meta <f>     environment sidecar for the current run\n"
        << "  --allow-env-mismatch   do not fail when the environments differ\n"
        << "  --json <file>          also write the JSON report to a file\n"
        << "  --quiet                do not print the JSON report to stdout\n"
        << "  --help, -h             show this help\n"
        << "\n"
        << "Environment sidecars default to <samples>.meta.json (written by\n"
        << "cuda_timer::write_environment). When both exist and differ, the verdict\n"
        << "is 'suspect' unless --allow-env-mismatch is given.\n"
        << "\n"
        << "Exit codes: 0 pass/improvement, 1 regression, 2 suspect or error.\n";
}

bool parse_size(const std::string& text, std::size_t& out) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    unsigned long long value = 0;
    stream >> value;
    if (stream.fail()) return false;
    stream >> std::ws;
    if (!stream.eof()) return false;
    out = static_cast<std::size_t>(value);
    return true;
}

std::string sidecar_path(const std::string& csv_path) {
    std::filesystem::path path(csv_path);
    path.replace_extension(".meta.json");
    return path.string();
}

bool file_exists(const std::string& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    return exists && !error;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> baseline_paths;
    std::vector<std::string> current_paths;
    std::string baseline_meta;
    std::string current_meta;
    std::string json_path;
    bool quiet = false;
    bool allow_env_mismatch = false;
    cuda_timer::ComparisonOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto take_value = [&](std::string& target) -> bool {
            if (i + 1 >= argc) return false;
            target = argv[++i];
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--baseline") {
            std::string value;
            if (!take_value(value)) {
                std::cerr << "missing value for --baseline\n";
                return 2;
            }
            baseline_paths.push_back(value);
        } else if (arg == "--current") {
            std::string value;
            if (!take_value(value)) {
                std::cerr << "missing value for --current\n";
                return 2;
            }
            current_paths.push_back(value);
        } else if (arg == "--baseline-meta") {
            if (!take_value(baseline_meta)) {
                std::cerr << "missing value for --baseline-meta\n";
                return 2;
            }
        } else if (arg == "--current-meta") {
            if (!take_value(current_meta)) {
                std::cerr << "missing value for --current-meta\n";
                return 2;
            }
        } else if (arg == "--json") {
            if (!take_value(json_path)) {
                std::cerr << "missing value for --json\n";
                return 2;
            }
        } else if (arg == "--threshold") {
            std::string value;
            if (!take_value(value)) {
                std::cerr << "missing value for --threshold\n";
                return 2;
            }
            try {
                options.threshold = cuda_timer::detail::parse_double(value);
            } catch (const std::exception& e) {
                std::cerr << "invalid --threshold: " << e.what() << "\n";
                return 2;
            }
        } else if (arg == "--min-samples") {
            std::string value;
            if (!take_value(value) || !parse_size(value, options.min_samples) ||
                options.min_samples == 0) {
                std::cerr << "invalid --min-samples (expected a positive integer)\n";
                return 2;
            }
        } else if (arg == "--max-cv") {
            std::string value;
            if (!take_value(value)) {
                std::cerr << "missing value for --max-cv\n";
                return 2;
            }
            try {
                options.max_cv = cuda_timer::detail::parse_double(value);
            } catch (const std::exception& e) {
                std::cerr << "invalid --max-cv: " << e.what() << "\n";
                return 2;
            }
        } else if (arg == "--max-run-spread") {
            std::string value;
            if (!take_value(value)) {
                std::cerr << "missing value for --max-run-spread\n";
                return 2;
            }
            try {
                options.max_run_spread = cuda_timer::detail::parse_double(value);
            } catch (const std::exception& e) {
                std::cerr << "invalid --max-run-spread: " << e.what() << "\n";
                return 2;
            }
        } else if (arg == "--allow-env-mismatch") {
            allow_env_mismatch = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (baseline_paths.empty() || current_paths.empty()) {
        std::cerr << "at least one --baseline and one --current are required\n\n";
        print_usage(argv[0]);
        return 2;
    }
    if (options.threshold < 0.0 || !std::isfinite(options.threshold)) {
        std::cerr << "--threshold must be a finite value >= 0\n";
        return 2;
    }
    if (options.max_cv < 0.0 || !std::isfinite(options.max_cv)) {
        std::cerr << "--max-cv must be a finite value >= 0\n";
        return 2;
    }
    if (options.max_run_spread < 0.0 || !std::isfinite(options.max_run_spread)) {
        std::cerr << "--max-run-spread must be a finite value >= 0\n";
        return 2;
    }

    std::vector<std::vector<double>> baseline_runs;
    std::vector<std::vector<double>> current_runs;
    try {
        for (const auto& path : baseline_paths) {
            baseline_runs.push_back(cuda_timer::read_samples_csv(path));
        }
        for (const auto& path : current_paths) {
            current_runs.push_back(cuda_timer::read_samples_csv(path));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    cuda_timer::ComparisonResult result =
        cuda_timer::compare_runs(baseline_runs, current_runs, options);

    // Environment sidecars: explicit flag wins, otherwise probe next to the CSV.
    if (baseline_meta.empty()) baseline_meta = sidecar_path(baseline_paths.front());
    if (current_meta.empty()) current_meta = sidecar_path(current_paths.front());

    const bool has_baseline_meta = file_exists(baseline_meta);
    const bool has_current_meta = file_exists(current_meta);

    if (has_baseline_meta && has_current_meta) {
        try {
            result.baseline_environment = cuda_timer::read_environment(baseline_meta);
            result.current_environment = cuda_timer::read_environment(current_meta);
            result.has_environment = true;

            std::string reason;
            result.environment_match = cuda_timer::environment_compatible(
                result.baseline_environment, result.current_environment, &reason);

            if (!result.environment_match) {
                if (allow_env_mismatch) {
                    std::cerr << "warning: " << reason
                              << " (ignored: --allow-env-mismatch)\n";
                } else {
                    result.verdict = cuda_timer::Verdict::Suspect;
                    result.reason = reason;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "warning: cannot read environment sidecar: " << e.what() << "\n";
        }
    } else if (has_baseline_meta || has_current_meta) {
        std::cerr << "warning: environment sidecar found on one side only; "
                     "comparison not verified\n";
    }

    const std::string report = cuda_timer::to_json(result);
    if (!quiet) {
        std::cout << report;
    }
    if (!json_path.empty()) {
        std::ofstream out(json_path);
        if (!out) {
            std::cerr << "error: cannot write " << json_path << "\n";
            return 2;
        }
        out << report;
        if (!out) {
            std::cerr << "error: failed while writing " << json_path << "\n";
            return 2;
        }
    }

    switch (result.verdict) {
        case cuda_timer::Verdict::Pass:
        case cuda_timer::Verdict::Improvement:
            return 0;
        case cuda_timer::Verdict::Regression:
            return 1;
        case cuda_timer::Verdict::Suspect:
            return 2;
    }
    return 2;
}
