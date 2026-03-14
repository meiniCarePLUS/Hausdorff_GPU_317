#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ─── POD types (no zjucad::matrix on device) ────────────────────────────────

struct float3x3 {          // 3 triangle vertices, column-major
    float v[9];            // v[0..2]=p0, v[3..5]=p1, v[6..8]=p2
};

struct LBVHNode {
    float aabb_min[3];
    float aabb_max[3];
    int   left;            // child index (internal) or -1 (leaf)
    int   right;           // child index (internal) or -1 (leaf)
    int   prim_idx;        // primitive index if leaf, else -1
    int   parent;
};

// ─── Host-side LBVH container ───────────────────────────────────────────────

struct LBVH {
    // device pointers
    LBVHNode* d_nodes;     // [2*n-1] nodes: [0..n-2] internal, [n-1..2n-2] leaves
    int*      d_sorted_idx;// [n] primitive indices sorted by Morton code
    uint32_t* d_morton;    // [n] Morton codes
    int       n_prims;

    void build(const float3x3* h_tris, int n);
    void free();
};

// ─── Kernel declarations ────────────────────────────────────────────────────

// Compute 30-bit Morton code for each triangle centroid.
// tris[n], out_morton[n], out_idx[n] (identity permutation before sort)
__global__ void kernel_compute_morton(
    const float3x3* __restrict__ tris,
    uint32_t* __restrict__ out_morton,
    int* __restrict__ out_idx,
    float3 scene_min, float3 scene_inv_extent,
    int n);

// Build internal nodes of binary radix tree (Karras 2012).
// sorted_morton[n], nodes[2n-1]
__global__ void kernel_build_tree(
    const uint32_t* __restrict__ sorted_morton,
    LBVHNode* __restrict__ nodes,
    int n);

// Bottom-up AABB refit.
// tris[n] (original order), sorted_idx[n], nodes[2n-1], atomic flags[n-1]
__global__ void kernel_refit(
    const float3x3* __restrict__ tris,
    const int* __restrict__ sorted_idx,
    LBVHNode* __restrict__ nodes,
    int* __restrict__ flags,   // atomic counters, zero-initialised
    int n);
