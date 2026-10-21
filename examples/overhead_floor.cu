#include <cuda_timer/overhead.cuh>

#include <exception>
#include <iostream>
#include <locale>

int main() {
    std::cout.imbue(std::locale::classic());

    try {
        const auto floor = cuda_timer::measure_event_overhead(200);
        std::cout << "{\"runs\": " << floor.runs
                  << ", \"min_ms\": " << floor.min_ms
                  << ", \"median_ms\": " << floor.median_ms
                  << ", \"mean_ms\": " << floor.mean_ms << "}\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
