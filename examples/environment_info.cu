#include <cuda_timer/environment.cuh>

#include <exception>
#include <iostream>
#include <locale>

int main() {
    std::cout.imbue(std::locale::classic());

    try {
        const auto environment = cuda_timer::capture_environment(0);
        std::cout << cuda_timer::environment_json(environment);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
