#pragma once
#include <cuda_runtime.h>
#include "bvh/lbvh.cuh"

// ─── GPU端BVH遍历的数据结构 ──────────────────────────────────────────────────

// BVH节点对（用于遍历）
struct BVHNodePair {
    int node_A_idx;      // mesh A的BVH节点索引
    int node_B_idx;      // mesh B的BVH节点索引（始终为0，即根节点）
    double lower_bound;  // 该节点对的下界估计
};

// 候选三角形（通过裁剪后的结果）
struct CandidateTriangle {
    int tri_idx;         // 三角形索引（在mesh A中）
    double local_L;      // 局部下界
    double local_U;      // 局部上界
    float vertices[9];   // 三角形的3个顶点坐标 [v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z]
};

// GPU遍历的全局状态
struct TraversalState {
    double global_L;     // 全局下界（原子更新）
    double global_U;     // 全局上界（原子更新）
    int active_count;    // 当前活跃的节点对数量
    int candidate_count; // 候选三角形数量
};

// ─── GPU端BVH遍历接口 ────────────────────────────────────────────────────────

// 执行GPU端的BVH并行遍历
// 输入：
//   - lbvh_A: mesh A的LBVH树
//   - lbvh_B: mesh B的LBVH树
//   - d_tris_A: mesh A的三角形数据（device）
//   - d_tris_B: mesh B的三角形数据（device）
//   - nA, nB: 三角形数量
// 输出：
//   - d_candidates: 候选三角形列表（device）
//   - h_candidate_count: 候选三角形数量（host）
//   - h_global_L, h_global_U: 全局上下界（host）
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

// 释放GPU遍历分配的资源
void gpu_parallel_traverse_free(CandidateTriangle* d_candidates);
