#pragma once

#include "types.cuh"

#include <cstdint>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef NVTX_ENABLED
// ABI types from the toolkit header. This include MUST stay at global scope:
// pulling a system or third-party header inside a namespace puts its
// declarations (and the ones it transitively pulls from glibc) into that
// namespace. glibc's include guards then report the headers as already
// included while the global declarations never appear, so any later
// <unistd.h>/<sys/stat.h> consumer - GoogleTest, for instance - fails with
// "identifier rmdir is undefined" / "incomplete class type stat".
#include <nvtx3/nvToolsExt.h>
#endif

namespace cuda_timer::nvtx {

#ifdef NVTX_ENABLED

// ---------------------------------------------------------------------------
// NVTX is loaded dynamically at runtime (no link-time dependency).
//
// Rationale: the CUDA toolkit ships the NVTX headers but NOT the import
// library (nvToolsExt64_1.lib), so linking against the NVTX C API would
// fail. Instead we LoadLibrary/GetProcAddress (Windows) or dlopen/dlsym
// (Linux) the NVTX injection library. If it is not present (no profiler
// attached, no driver support), all calls are silent no-ops.
// ---------------------------------------------------------------------------

namespace detail {

using push_ex_fn = int (*)(const nvtxEventAttributes_t*);
using pop_fn = int (*)();

struct NvtxLibrary {
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void* handle = nullptr;
#endif
    push_ex_fn push_ex = nullptr;
    pop_fn pop = nullptr;

    NvtxLibrary() {
#ifdef _WIN32
        // Ships with the NVIDIA driver / Nsight tools. Absence => no-op.
        handle = ::LoadLibraryA("nvToolsExt64_1.dll");
        if (handle) {
            push_ex = reinterpret_cast<push_ex_fn>(
                ::GetProcAddress(handle, "nvtxRangePushEx"));
            pop = reinterpret_cast<pop_fn>(::GetProcAddress(handle, "nvtxRangePop"));
        }
#else
        handle = ::dlopen("libnvToolsExt.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            handle = ::dlopen("libnvToolsExt.so", RTLD_NOW | RTLD_LOCAL);
        }
        if (handle) {
            push_ex = reinterpret_cast<push_ex_fn>(::dlsym(handle, "nvtxRangePushEx"));
            pop = reinterpret_cast<pop_fn>(::dlsym(handle, "nvtxRangePop"));
        }
#endif
    }

    // Intentionally leaked (Meyers singleton, destroyed never) so ranges
    // can still pop during static destruction.
    static const NvtxLibrary& instance() {
        static const NvtxLibrary lib;
        return lib;
    }

    bool available() const noexcept { return push_ex != nullptr && pop != nullptr; }
};

} // namespace detail

/**
 * @brief Pushes an NVTX range with the given name and color.
 *
 * No-op if the NVTX runtime library is not present.
 *
 * @param name Range name (ASCII). Must remain valid until pop_range().
 * @param color ARGB color (default: blue 0xFF0000FF)
 */
inline void push_range(const char* name, uint32_t color = 0xFF0000FF) noexcept {
    const auto& lib = detail::NvtxLibrary::instance();
    if (!lib.available() || name == nullptr) {
        return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    lib.push_ex(&attr);
}

/**
 * @brief Pops the most recent NVTX range. No-op if NVTX is unavailable.
 */
inline void pop_range() noexcept {
    const auto& lib = detail::NvtxLibrary::instance();
    if (lib.available()) {
        lib.pop();
    }
}

/**
 * @brief RAII wrapper for NVTX ranges.
 *
 * Pushes a range on construction, pops on destruction.
 *
 * Example:
 * @code
 * {
 *     cuda_timer::nvtx::ScopedRange range("my_kernel");
 *     my_kernel<<<...>>>(...);
 * } // range automatically popped here
 * @endcode
 */
class ScopedRange {
public:
    /**
     * @brief Constructs and pushes an NVTX range.
     *
     * @param name Range name (ASCII, must outlive this object)
     * @param color ARGB color (default: blue 0xFF0000FF)
     */
    explicit ScopedRange(const char* name, uint32_t color = 0xFF0000FF) noexcept {
        push_range(name, color);
    }

    ~ScopedRange() noexcept { pop_range(); }
    ScopedRange(ScopedRange&&) noexcept = default;
    ScopedRange& operator=(ScopedRange&&) noexcept = default;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

#else // NVTX_ENABLED

/**
 * @brief No-op when NVTX is disabled.
 */
inline void push_range(const char*, uint32_t = 0) noexcept {}

/**
 * @brief No-op when NVTX is disabled.
 */
inline void pop_range() noexcept {}

/**
 * @brief No-op RAII wrapper when NVTX is disabled.
 *
 * Compiles to empty operations, allowing NVTX code to compile without NVTX headers.
 */
class ScopedRange {
public:
    explicit ScopedRange(const char*, uint32_t = 0) noexcept {}
    ~ScopedRange() noexcept = default;
    ScopedRange(ScopedRange&&) noexcept = default;
    ScopedRange& operator=(ScopedRange&&) noexcept = default;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

#endif

} // namespace cuda_timer::nvtx
