/*
    CUDA Wrapper for main program
    This file serves as a bridge between C++ main.cpp and CUDA libraries
    to resolve device code linking issues.
*/

#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"
#include "hausdorff/gpu_parallel_traverse.cuh"
#include "hausdorff/gpu_query_iface.hpp"

// This file intentionally includes CUDA headers to force device code linking
// The actual implementations are in the respective .cu files

// Dummy function to ensure this compilation unit is not optimized away
extern "C" void cuda_wrapper_init() {
    // This function does nothing but ensures CUDA device code is linked
}
