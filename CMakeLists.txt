cmake_minimum_required(VERSION 3.18)

project(cuda_timer
    LANGUAGES CUDA CXX
    VERSION 1.1.0
    DESCRIPTION "Header-only CUDA timing/profiling library for GPU kernels"
    HOMEPAGE_URL "https://github.com/Mardok-dotcom/cuda-timer"
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CUDA_TIMER_NVTX "Enable NVTX (NVIDIA Tools Extension) support" ON)
option(CUDA_TIMER_BUILD_EXAMPLES "Build example executables" ON)
option(CUDA_TIMER_BUILD_TESTS "Build unit tests (requires GoogleTest)" OFF)
option(CUDA_TIMER_BUILD_TOOLS "Build command-line tools (perf_gate)" ON)

# ---------------------------------------------------------------------------
# CUDA / C++ Standard
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Find CUDAToolkit for NVTX (cross-platform)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Warning flags per compiler
# ---------------------------------------------------------------------------
function(cuda_timer_apply_warnings target)
    # Host C++: direct flags (CXX language only, never passed raw to nvcc)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
            -Wall -Wextra -Wpedantic -Werror
            -Wno-unused-parameter
            -Wno-missing-field-initializers
        >
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
            /W4 /WX
            /wd4100   # unreferenced formal parameter
            /wd4244   # conversion from 'double' to 'float', possible loss of data
            /wd4267   # conversion from 'size_t' to 'int', possible loss of data
        >
    )
    # CUDA (.cu via nvcc): host-compiler flags MUST go through -Xcompiler.
    # nvcc has its own -Werror option and would swallow the next flag as its
    # value, failing with:
    #   nvcc fatal : Value '-Wno-unused-parameter' is not defined for option 'Werror'
    # MSVC-style flags have the same problem ("/W4" is read as an input file:
    # "A single input file is required for a non-link phase").
    # -Wpedantic is deliberately absent: nvcc's own generated code
    # (cudafe1.cpp) uses line directives GCC calls a "GCC extension", so
    # -Wpedantic -Werror fails on every .cu.
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
            SHELL:-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Werror
            SHELL:-Xcompiler=-Wno-unused-parameter -Xcompiler=-Wno-missing-field-initializers
        >
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:
            SHELL:-Xcompiler=/W4 -Xcompiler=/WX -Xcompiler=/wd4100 -Xcompiler=/wd4244 -Xcompiler=/wd4267 -Xcompiler=/wd4505
        >
    )
endfunction()

# ---------------------------------------------------------------------------
# Enable separable compilation for all CUDA targets globally
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# INTERFACE target (header-only library)
#
# Consumers use it as:
#   add_subdirectory(cuda-timer) / FetchContent -> cuda_timer::cuda_timer
#   find_package(cuda_timer)                      -> cuda_timer::cuda_timer
# ---------------------------------------------------------------------------
add_library(cuda_timer INTERFACE)
add_library(cuda_timer::cuda_timer ALIAS cuda_timer)

target_include_directories(cuda_timer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cuda_timer INTERFACE cxx_std_17)

# NVTX ranges are compiled in by default; the NVTX runtime itself is loaded
# dynamically (see include/cuda_timer/nvtx.cuh), so no link libraries needed.
target_compile_definitions(cuda_timer INTERFACE
    $<$<BOOL:${CUDA_TIMER_NVTX}>:NVTX_ENABLED>
)

# ---------------------------------------------------------------------------
# Install headers + export set for find_package(cuda_timer)
# ---------------------------------------------------------------------------
install(DIRECTORY include/cuda_timer
    DESTINATION include
    FILES_MATCHING PATTERN "*.cuh"
)

install(TARGETS cuda_timer
    EXPORT cuda_timer-targets
)

install(EXPORT cuda_timer-targets
    FILE cuda_timer-targets.cmake
    NAMESPACE cuda_timer::
    DESTINATION lib/cmake/cuda_timer
)

# ---------------------------------------------------------------------------
# NVTX support (optional)
# ---------------------------------------------------------------------------
if(CUDA_TIMER_NVTX)
    add_compile_definitions(NVTX_ENABLED)
    # NOTE: NVTX is used header-only + loaded dynamically at runtime
    # (see include/cuda_timer/nvtx.cuh). The CUDA toolkit does not ship an
    # NVTX import library, so nothing is linked here on purpose.
endif()

# ---------------------------------------------------------------------------
# Helper function to create examples with common settings
# ---------------------------------------------------------------------------
function(cuda_timer_add_example name source)
    if(NOT CUDA_TIMER_BUILD_EXAMPLES)
        return()
    endif()
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE
        cuda_timer::cuda_timer
        CUDA::cudart
    )
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    cuda_timer_apply_warnings(${name})
    if(CUDA_TIMER_NVTX)
        target_compile_definitions(${name} PRIVATE NVTX_ENABLED)
        # No NVTX link libraries: loaded dynamically at runtime (see nvtx.cuh).
    endif()
    install(TARGETS ${name} DESTINATION bin)
endfunction()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(CUDA_TIMER_BUILD_EXAMPLES)
    cuda_timer_add_example(basic_usage        examples/basic_usage.cu)
    cuda_timer_add_example(multi_stream       examples/multi_stream.cu)
    cuda_timer_add_example(multi_device       examples/multi_device.cu)
    cuda_timer_add_example(benchmark_harness  examples/benchmark_harness.cu)
    cuda_timer_add_example(overhead_floor     examples/overhead_floor.cu)
    cuda_timer_add_example(environment_info   examples/environment_info.cu)
endif()

# ---------------------------------------------------------------------------
# Command-line tools (host-only: they never call CUDA APIs)
# ---------------------------------------------------------------------------
if(CUDA_TIMER_BUILD_TOOLS)
    add_executable(perf_gate tools/perf_gate.cpp)
    target_link_libraries(perf_gate PRIVATE cuda_timer::cuda_timer)
    set_target_properties(perf_gate PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    cuda_timer_apply_warnings(perf_gate)
    install(TARGETS perf_gate DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Testing (GoogleTest)
# ---------------------------------------------------------------------------
if(CUDA_TIMER_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Fallback: fetch GoogleTest from source (requires internet at configure time)
        if(POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/v1.14.0.zip
        )
        # For Windows: prevent overriding parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        # Keep `cmake --install` clean: don't install googletest alongside us
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        # FetchContent provides gtest/gtest_main; normalize to FindGTest names
        if(NOT TARGET GTest::GTest)
            add_library(GTest::GTest ALIAS gtest)
        endif()
        if(NOT TARGET GTest::Main)
            add_library(GTest::Main ALIAS gtest_main)
        endif()
    endif()

    function(cuda_timer_configure_test name source main_file)
        add_executable(${name} ${source} ${main_file})
        target_link_libraries(${name} PRIVATE
            GTest::GTest
            cuda_timer::cuda_timer
            CUDA::cudart
        )
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
        cuda_timer_apply_warnings(${name})
        if(CUDA_TIMER_NVTX)
            target_compile_definitions(${name} PRIVATE NVTX_ENABLED)
            # No NVTX link libraries: loaded dynamically at runtime (see nvtx.cuh).
        endif()
        add_test(NAME ${name} COMMAND ${name})
        # Exit code 77 from tests/test_main_device.cpp means "no CUDA device":
        # report it as skipped, not as a failure (and not as a silent pass).
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    # Host-only suite (loggers, export, Stats, types): its main never checks for
    # a device, so the suite runs everywhere, including GPU-less CI runners.
    function(cuda_timer_add_test name source)
        cuda_timer_configure_test(${name} ${source} ${PROJECT_SOURCE_DIR}/tests/test_main.cpp)
    endfunction()

    # Suite that needs a real GPU: its main exits with 77 when no device is
    # present and ctest reports the whole suite as SKIPPED. The kernels live in
    # tests/cuda_test_kernels.cu; the suites themselves are host C++ because
    # GoogleTest does not survive nvcc's device pass (see the header comment).
    function(cuda_timer_add_device_test name source)
        cuda_timer_configure_test(${name} ${source} ${PROJECT_SOURCE_DIR}/tests/test_main_device.cpp)
        target_sources(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/cuda_test_kernels.cu)
    endfunction()

    # Unit tests
    cuda_timer_add_test(test_logger           tests/test_logger.cpp)
    cuda_timer_add_test(test_compare          tests/test_compare.cpp)
    cuda_timer_add_test(test_stats            tests/test_stats.cpp)
    cuda_timer_add_test(test_types            tests/test_types.cpp)
    cuda_timer_add_device_test(test_timer      tests/test_timer.cpp)
    cuda_timer_add_device_test(test_overhead   tests/test_overhead.cpp)
    cuda_timer_add_device_test(test_environment tests/test_environment.cpp)
    cuda_timer_add_device_test(test_benchmark  tests/test_benchmark.cpp)
else()
    # Still enable testing() so ctest works when tests are added later
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Package configuration for find_package(cuda_timer)
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_timer-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config.cmake
    INSTALL_DESTINATION lib/cmake/cuda_timer
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_timer-config-version.cmake
    DESTINATION lib/cmake/cuda_timer
)

# ---------------------------------------------------------------------------
# CPack for release packaging
# ---------------------------------------------------------------------------
include(CPack)
set(CPACK_PACKAGE_NAME "cuda-timer")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Header-only CUDA timing/profiling library")
set(CPACK_PACKAGE_VENDOR "Mardok-dotcom")
set(CPACK_PACKAGE_CONTACT "Mardok-dotcom")
set(CPACK_GENERATOR "ZIP;TGZ")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")