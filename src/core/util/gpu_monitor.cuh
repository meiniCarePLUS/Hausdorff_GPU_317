// GPU kernel launch configuration monitor
// Add this to gpu_batch_query.cu and gpu_initial_bounds.cu

#pragma once
#include <cuda_runtime.h>
#include <cstdio>

// Macro to print kernel launch configuration
#define PRINT_KERNEL_CONFIG(kernel_name, grid, block, shared_mem) \
    do { \
        printf("[GPU_KERNEL] %s: grid=(%d,%d,%d) block=(%d,%d,%d) threads=%d shared_mem=%zu bytes\n", \
               kernel_name, \
               grid.x, grid.y, grid.z, \
               block.x, block.y, block.z, \
               grid.x * grid.y * grid.z * block.x * block.y * block.z, \
               shared_mem); \
    } while(0)

// Function to print GPU device properties
inline void print_gpu_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\n");
    printf("========================================\n");
    printf("  GPU 设备信息\n");
    printf("========================================\n");
    printf("设备名称:           %s\n", prop.name);
    printf("计算能力:           %d.%d\n", prop.major, prop.minor);
    printf("多处理器数量:       %d\n", prop.multiProcessorCount);
    printf("每个SM的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个块的最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("全局内存:           %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("共享内存/块:        %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("理论最大并行线程数: %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("========================================\n\n");
}

// Function to print memory usage
inline void print_gpu_memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;

    printf("[GPU_MEMORY] 使用: %.2f MB / %.2f MB (%.1f%%)\n",
           used_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0),
           100.0 * used_mem / total_mem);
}
