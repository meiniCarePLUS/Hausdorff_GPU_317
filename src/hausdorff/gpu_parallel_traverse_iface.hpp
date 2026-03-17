#pragma once

// Forward declarations for GPU types (C++ compatible)
struct LBVH;
struct float3x3;

// Candidate triangle structure
struct CandidateTriangle {
    int tri_idx;
    double local_L;
    double local_U;
    float vertices[9];  // 三角形的3个顶点坐标 [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z]
};

// GPU parallel traverse interface (C++ compatible, no CUDA headers)
// This is a wrapper that can be included from .cpp files

void gpu_parallel_traverse(
    const LBVH& lbvh_A,
    const LBVH& lbvh_B,
    const float3x3* d_tris_A,
    const float3x3* d_tris_B,
    int nA, int nB,
    CandidateTriangle** d_candidates,
    int* h_candidate_count,
    double* h_global_L,
    double* h_global_U);

// Copy candidates from device to host
void gpu_parallel_traverse_copy_candidates(
    const CandidateTriangle* d_candidates,
    CandidateTriangle* h_candidates,
    int count);

void gpu_parallel_traverse_free(CandidateTriangle* d_candidates);
