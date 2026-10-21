// The only CUDA translation unit in the test suite: GoogleTest must not be
// compiled by nvcc (see cuda_test_kernels.cuh for the reason).

#include "cuda_test_kernels.cuh"

namespace {

__global__ void dummy_kernel() {}

}  // namespace

namespace cuda_test {

void launch_dummy(dim3 grid, dim3 block, cudaStream_t stream) {
    dummy_kernel<<<grid, block, 0, stream>>>();
}

}  // namespace cuda_test
