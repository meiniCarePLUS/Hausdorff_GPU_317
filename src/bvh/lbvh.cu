#include "lbvh.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <cstring>
#include <stdexcept>

// ─── Morton code helpers ─────────────────────────────────────────────────────

// Expand a 10-bit integer into 30 bits by inserting 2 zeros between each bit.
__device__ __forceinline__ uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// 3D Morton code from normalised [0,1]^3 coordinates.
__device__ __forceinline__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.f, 0.f), 1023.f);
    y = fminf(fmaxf(y * 1024.f, 0.f), 1023.f);
    z = fminf(fmaxf(z * 1024.f, 0.f), 1023.f);
    return (expand_bits((uint32_t)x) << 2) |
           (expand_bits((uint32_t)y) << 1) |
            expand_bits((uint32_t)z);
}

// ─── Kernel: Morton codes ────────────────────────────────────────────────────

__global__ void kernel_compute_morton(
    const float3x3* __restrict__ tris,
    uint32_t* __restrict__ out_morton,
    int* __restrict__ out_idx,
    float3 scene_min, float3 scene_inv_extent,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // centroid = average of 3 vertices
    const float* v = tris[i].v;
    float cx = (v[0] + v[3] + v[6]) * (1.f / 3.f);
    float cy = (v[1] + v[4] + v[7]) * (1.f / 3.f);
    float cz = (v[2] + v[5] + v[8]) * (1.f / 3.f);

    // normalise to [0,1]
    float nx = (cx - scene_min.x) * scene_inv_extent.x;
    float ny = (cy - scene_min.y) * scene_inv_extent.y;
    float nz = (cz - scene_min.z) * scene_inv_extent.z;

    out_morton[i] = morton3D(nx, ny, nz);
    out_idx[i]    = i;
}

// ─── Kernel: binary radix tree (Karras 2012) ─────────────────────────────────

// delta(i,j): length of longest common prefix of morton[i] and morton[j].
// Returns -1 if j is out of range.
__device__ __forceinline__ int delta(
    const uint32_t* __restrict__ m, int i, int j, int n)
{
    if (j < 0 || j >= n) return -1;
    if (m[i] == m[j])
        // break ties with index
        return 32 + __clz(i ^ j);
    return __clz(m[i] ^ m[j]);
}

__global__ void kernel_build_tree(
    const uint32_t* __restrict__ sorted_morton,
    LBVHNode* __restrict__ nodes,
    int n)
{
    // Internal node index i covers [0, n-2].
    // Leaf node index i+n-1 covers [n-1, 2n-2].
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    const uint32_t* m = sorted_morton;

    // Determine direction of the range (+1 or -1).
    int d = (delta(m, i, i + 1, n) - delta(m, i, i - 1, n)) >= 0 ? 1 : -1;

    // Compute upper bound for the length of the range.
    int delta_min = delta(m, i, i - d, n);
    int l_max = 2;
    while (delta(m, i, i + l_max * d, n) > delta_min)
        l_max <<= 1;

    // Binary search for the exact end of the range.
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1)
        if (delta(m, i, i + (l + t) * d, n) > delta_min)
            l += t;
    int j = i + l * d;

    // Find the split position within [min(i,j), max(i,j)].
    int delta_node = delta(m, i, j, n);
    int s = 0;
    int step = l;
    do {
        step = (step + 1) >> 1;
        if (delta(m, i, i + (s + step) * d, n) > delta_node)
            s += step;
    } while (step > 1);
    int gamma = i + s * d + min(d, 0);

    // Assign children.
    int left_child  = (min(i, j) == gamma)     ? (gamma + n - 1)     : gamma;
    int right_child = (max(i, j) == gamma + 1) ? (gamma + 1 + n - 1) : (gamma + 1);

    nodes[i].left      = left_child;
    nodes[i].right     = right_child;
    nodes[i].prim_idx  = -1;

    nodes[left_child].parent  = i;
    nodes[right_child].parent = i;
}

// ─── Kernel: bottom-up AABB refit ────────────────────────────────────────────

__device__ __forceinline__ void aabb_of_tri(
    const float3x3& tri, float* bmin, float* bmax)
{
    for (int k = 0; k < 3; ++k) {
        float a = tri.v[k], b = tri.v[k + 3], c = tri.v[k + 6];
        bmin[k] = fminf(fminf(a, b), c);
        bmax[k] = fmaxf(fmaxf(a, b), c);
    }
}

__global__ void kernel_refit(
    const float3x3* __restrict__ tris,
    const int* __restrict__ sorted_idx,
    LBVHNode* __restrict__ nodes,
    int* __restrict__ flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Initialise leaf node (index = i + n - 1).
    int leaf = i + n - 1;
    nodes[leaf].prim_idx = sorted_idx[i];
    nodes[leaf].left     = -1;
    nodes[leaf].right    = -1;
    aabb_of_tri(tris[sorted_idx[i]],
                nodes[leaf].aabb_min,
                nodes[leaf].aabb_max);

    // Walk up the tree; the second thread to reach a node processes it.
    int node = nodes[leaf].parent;
    while (node != -1) {
        // atomicAdd returns old value; first thread gets 0 and exits.
        if (atomicAdd(&flags[node], 1) == 0) return;

        int lc = nodes[node].left;
        int rc = nodes[node].right;
        for (int k = 0; k < 3; ++k) {
            nodes[node].aabb_min[k] = fminf(nodes[lc].aabb_min[k],
                                            nodes[rc].aabb_min[k]);
            nodes[node].aabb_max[k] = fmaxf(nodes[lc].aabb_max[k],
                                            nodes[rc].aabb_max[k]);
        }
        node = nodes[node].parent;
    }
}

// ─── Host: LBVH::build ───────────────────────────────────────────────────────

void LBVH::build(const float3x3* h_tris, int n) {
    if (n <= 0) throw std::invalid_argument("LBVH::build: n must be > 0");
    n_prims = n;

    // Upload triangles.
    float3x3* d_tris;
    cudaMalloc(&d_tris, n * sizeof(float3x3));
    cudaMemcpy(d_tris, h_tris, n * sizeof(float3x3), cudaMemcpyHostToDevice);

    // Compute scene AABB on host (cheap for moderate n).
    float3 smin = {1e30f, 1e30f, 1e30f};
    float3 smax = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k) {
            float a = h_tris[i].v[k], b = h_tris[i].v[k+3], c = h_tris[i].v[k+6];
            float lo = fminf(fminf(a,b),c), hi = fmaxf(fmaxf(a,b),c);
            ((float*)&smin)[k] = fminf(((float*)&smin)[k], lo);
            ((float*)&smax)[k] = fmaxf(((float*)&smax)[k], hi);
        }
    }
    float3 inv_ext = {
        1.f / fmaxf(smax.x - smin.x, 1e-10f),
        1.f / fmaxf(smax.y - smin.y, 1e-10f),
        1.f / fmaxf(smax.z - smin.z, 1e-10f)
    };

    // Allocate device arrays.
    cudaMalloc(&d_morton,     n * sizeof(uint32_t));
    cudaMalloc(&d_sorted_idx, n * sizeof(int));
    cudaMalloc(&d_nodes,      (2*n-1) * sizeof(LBVHNode));

    // Initialise root parent sentinel.
    {
        LBVHNode root_sentinel{};
        root_sentinel.parent = -1;
        cudaMemcpy(d_nodes, &root_sentinel, sizeof(LBVHNode), cudaMemcpyHostToDevice);
    }

    // Step 1: Morton codes.
    int block = 256, grid = (n + block - 1) / block;
    kernel_compute_morton<<<grid, block>>>(
        d_tris, d_morton, d_sorted_idx, smin, inv_ext, n);

    // Step 2: Sort by Morton code (Thrust stable sort by key).
    thrust::device_ptr<uint32_t> t_morton(d_morton);
    thrust::device_ptr<int>      t_idx(d_sorted_idx);
    thrust::stable_sort_by_key(t_morton, t_morton + n, t_idx);

    // Step 3: Build binary radix tree.
    if (n > 1) {
        int igrid = (n - 1 + block - 1) / block;
        // Zero-init parent fields of all nodes first.
        cudaMemset(d_nodes, 0xFF, (2*n-1) * sizeof(LBVHNode)); // -1 = 0xFF...
        kernel_build_tree<<<igrid, block>>>(d_morton, d_nodes, n);
    }

    // Step 4: Refit AABBs bottom-up.
    int* d_flags;
    cudaMalloc(&d_flags, (n-1) * sizeof(int));
    cudaMemset(d_flags, 0, (n-1) * sizeof(int));
    kernel_refit<<<grid, block>>>(d_tris, d_sorted_idx, d_nodes, d_flags, n);

    cudaFree(d_flags);
    cudaFree(d_tris);
}

void LBVH::free() {
    cudaFree(d_nodes);      d_nodes      = nullptr;
    cudaFree(d_sorted_idx); d_sorted_idx = nullptr;
    cudaFree(d_morton);     d_morton     = nullptr;
}
