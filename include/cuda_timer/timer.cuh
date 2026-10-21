#pragma once

#include "types.cuh"
#include <memory>

namespace cuda_timer {
namespace detail {

/**
 * @brief Custom deleter for cudaEvent_t using unique_ptr.
 *
 * Ensures cudaEventDestroy is called exactly once, even if exceptions occur.
 */
struct EventDeleter {
    void operator()(cudaEvent_t* ev) const noexcept {
        if (ev && *ev) {
            cudaEventDestroy(*ev);
        }
    }
};

using EventPtr = std::unique_ptr<cudaEvent_t, EventDeleter>;

/**
 * @brief Creates a CUDA event with the specified flags.
 *
 * @param flags Event creation flags (default: cudaEventDefault)
 * @return EventPtr owning the created event
 * @throws CudaError if event creation fails
 */
inline EventPtr make_event(unsigned int flags = cudaEventDefault) {
    cudaEvent_t ev;
    CUDA_TIMER_CHECK(cudaEventCreateWithFlags(&ev, flags));
    return EventPtr(new cudaEvent_t(ev), EventDeleter{});
}

} // namespace detail

/**
 * @brief RAII wrapper for CUDA event-based timing.
 *
 * Provides exception-safe timing using cudaEvent_t with unique_ptr management.
 * Captures the device at construction time and restores it before each operation
 * to support multi-device scenarios correctly.
 *
 * Example:
 * @code
 * cuda_timer::Timer timer(stream);
 * timer.start();
 * my_kernel<<<grid, block, 0, stream>>>(...);
 * float ms = timer.stop();
 * @endcode
 */
class Timer {
public:
    /**
     * @brief Constructs a Timer for the given stream.
     *
     * Captures the current CUDA device at construction time.
     *
     * @param stream CUDA stream to record events on (default: 0 = default stream)
     * @param flags Event creation flags (default: cudaEventDefault)
     * @throws CudaError if event creation or device query fails
     */
    explicit Timer(cudaStream_t stream = 0, unsigned int flags = cudaEventDefault)
        : start_(detail::make_event(flags))
        , stop_(detail::make_event(flags))
        , stream_(stream) {
        CUDA_TIMER_CHECK(cudaGetDevice(&device_id_));
    }

    ~Timer() noexcept = default;
    Timer(Timer&&) noexcept = default;
    Timer& operator=(Timer&&) noexcept = default;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    /**
     * @brief Records the start event on the associated stream.
     *
     * Must be called before launching the kernel to time.
     * @throws CudaError if event recording fails
     */
    void start() {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        CUDA_TIMER_CHECK(cudaEventRecord(*start_, stream_));
    }

    /**
     * @brief Records the stop event, synchronizes, and returns elapsed time.
     *
     * Blocks until the stop event completes (SyncMode::Blocking behavior).
     *
     * @return Elapsed time in milliseconds
     * @throws CudaError if event recording, synchronization, or elapsed query fails
     */
    float stop() {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        CUDA_TIMER_CHECK(cudaEventRecord(*stop_, stream_));
        CUDA_TIMER_CHECK(cudaEventSynchronize(*stop_));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, *start_, *stop_));
        return ms;
    }

    /**
     * @brief Returns elapsed time without synchronizing (non-blocking).
     *
     * Requires that stop event has already been recorded and completed.
     * Use with SyncMode::NonBlocking workflows.
     *
     * @return Elapsed time in milliseconds
     * @throws CudaError if elapsed time query fails
     */
    float elapsed() const {
        CUDA_TIMER_CHECK(cudaSetDevice(device_id_));
        float ms = 0;
        CUDA_TIMER_CHECK(cudaEventElapsedTime(&ms, *start_, *stop_));
        return ms;
    }

    /**
     * @brief Returns the raw start event handle.
     *
     * Null-safe: returns nullptr on moved-from timers.
     */
    cudaEvent_t event_start() const noexcept { return start_ ? *start_ : nullptr; }

    /**
     * @brief Returns the raw stop event handle.
     *
     * Null-safe: returns nullptr on moved-from timers.
     */
    cudaEvent_t event_stop() const noexcept { return stop_ ? *stop_ : nullptr; }

    /**
     * @brief Returns the device ID this timer is bound to.
     */
    int device_id() const noexcept { return device_id_; }

    /**
     * @brief Returns the stream this timer records events on.
     */
    cudaStream_t stream() const noexcept { return stream_; }

private:
    detail::EventPtr start_;
    detail::EventPtr stop_;
    cudaStream_t stream_ = 0;
    int device_id_ = 0;
};

} // namespace cuda_timer