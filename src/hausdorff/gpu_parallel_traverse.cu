#include "gpu_parallel_traverse.cuh"
#include "core/geometry/gpu_primitive_dis.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <float.h>
#include <cstdio>

// ─── 原子操作辅助函数 ─────────────────────────────────────────────────────────

__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) return;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) return;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// ─���─ 计算AABB到点的平方距离 ───────────────────────────────────────────────────

__device__ double aabb_point_sqr_dis(const LBVHNode& node, d3 p) {
    double d = 0;
    double px[3] = {p.x, p.y, p.z};
    for (int k = 0; k < 3; ++k) {
        if (px[k] < node.aabb_min[k]) {
            double diff = node.aabb_min[k] - px[k];
            d += diff * diff;
        } else if (px[k] > node.aabb_max[k]) {
            double diff = px[k] - node.aabb_max[k];
            d += diff * diff;
        }
    }
    return d;
}

// ─── 计算AABB的Hausdorff距离上界 ──────────────────────────────────────────────
// 对于AABB A和点p，计算A中任意点到p的最大可能距离的平方

__device__ double aabb_hausdorff_upper(const LBVHNode& node_A, d3 p) {
    // AABB的中心点
    d3 mid;
    mid.x = (node_A.aabb_min[0] + node_A.aabb_max[0]) * 0.5;
    mid.y = (node_A.aabb_min[1] + node_A.aabb_max[1]) * 0.5;
    mid.z = (node_A.aabb_min[2] + node_A.aabb_max[2]) * 0.5;

    // 从中心到p的向量
    d3 max_vec;
    max_vec.x = p.x - mid.x;
    max_vec.y = p.y - mid.y;
    max_vec.z = p.z - mid.z;

    // 加上AABB半径
    double half_size[3];
    for (int k = 0; k < 3; ++k) {
        half_size[k] = (node_A.aabb_max[k] - node_A.aabb_min[k]) * 0.5;
    }

    double max_dist_x = fabs(max_vec.x) + half_size[0];
    double max_dist_y = fabs(max_vec.y) + half_size[1];
    double max_dist_z = fabs(max_vec.z) + half_size[2];

    return max_dist_x * max_dist_x + max_dist_y * max_dist_y + max_dist_z * max_dist_z;
}

// ─── GPU Kernel: need_travel计算 ──────────────────────────────────────────────
// 对于mesh A的每个节点，计算其到mesh B的最近点，并估算下界

__device__ double gpu_need_travel(
    const LBVHNode& node_A,
    const LBVHNode* __restrict__ nodes_B,
    const float3x3* __restrict__ tris_B,
    double global_L)
{
    // 计算node_A的中心点
    d3 mid;
    mid.x = (node_A.aabb_min[0] + node_A.aabb_max[0]) * 0.5;
    mid.y = (node_A.aabb_min[1] + node_A.aabb_max[1]) * 0.5;
    mid.z = (node_A.aabb_min[2] + node_A.aabb_max[2]) * 0.5;

    // 在mesh B的BVH中查找最近三角形
    int stack[64];
    int top = 0;
    stack[top++] = 0;  // 从根节点开始
    double best = DBL_MAX;

    while (top > 0) {
        int ni = stack[--top];
        const LBVHNode& node = nodes_B[ni];

        // 叶子节点：计算到三角形的距离
        if (node.prim_idx >= 0) {
            const float* fv = tris_B[node.prim_idx].v;
            double dv[9];
            for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
            d3 cp;
            double d = pt_tri_sqr_dis(mid, dv, &cp);
            if (d < best) best = d;
            continue;
        }

        // 内部节点：按距离排序遍历子节点
        int lc = node.left, rc = node.right;
        double dl = aabb_point_sqr_dis(nodes_B[lc], mid);
        double dr = aabb_point_sqr_dis(nodes_B[rc], mid);

        if (dl <= dr) {
            if (dr < best) stack[top++] = rc;
            if (dl < best) stack[top++] = lc;
        } else {
            if (dl < best) stack[top++] = lc;
            if (dr < best) stack[top++] = rc;
        }
    }

    // 返回Hausdorff距离的下界估计
    return aabb_hausdorff_upper(node_A, mid);
}

// ─── Thrust谓词：用于stream compaction ───────────────────────────────────────

struct is_valid_pair {
    double global_L;

    __host__ __device__
    is_valid_pair(double L) : global_L(L) {}

    __host__ __device__
    bool operator()(const BVHNodePair& pair) const {
        return pair.lower_bound > global_L;
    }
};

// ─── GPU Kernel: 并行BVH遍历（层次化） ────────────────────────────────────────

__global__ void kernel_parallel_traverse_level(
    const LBVHNode* __restrict__ nodes_A,
    const LBVHNode* __restrict__ nodes_B,
    const float3x3* __restrict__ tris_A,
    const float3x3* __restrict__ tris_B,
    BVHNodePair* __restrict__ input_pairs,
    int input_count,
    BVHNodePair* __restrict__ output_pairs,
    int* __restrict__ output_count,
    CandidateTriangle* __restrict__ candidates,
    int* __restrict__ candidate_count,
    TraversalState* __restrict__ state,
    int max_candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_count) return;

    BVHNodePair pair = input_pairs[idx];
    const LBVHNode& node_A = nodes_A[pair.node_A_idx];

    double global_L = state->global_L;
    double global_U = state->global_U;

    // 如果是叶子节点，处理三角形
    if (node_A.prim_idx >= 0) {
        int tri_idx = node_A.prim_idx;
        const float3x3& tri_A = tris_A[tri_idx];

        // 对三角形的3个顶点 + 质心进行采样
        d3 samples[4];
        for (int k = 0; k < 3; ++k) {
            samples[k].x = (double)tri_A.v[k*3+0];
            samples[k].y = (double)tri_A.v[k*3+1];
            samples[k].z = (double)tri_A.v[k*3+2];
        }
        // 质心
        samples[3].x = (samples[0].x + samples[1].x + samples[2].x) / 3.0;
        samples[3].y = (samples[0].y + samples[1].y + samples[2].y) / 3.0;
        samples[3].z = (samples[0].z + samples[1].z + samples[2].z) / 3.0;

        // 查找每个采样点到mesh B的最近距离
        double local_U = 0;
        for (int s = 0; s < 4; ++s) {
            // 在mesh B的BVH中查找最近三角形
            int stack[64];
            int top = 0;
            stack[top++] = 0;
            double best = DBL_MAX;

            while (top > 0) {
                int ni = stack[--top];
                const LBVHNode& node = nodes_B[ni];

                if (node.prim_idx >= 0) {
                    const float* fv = tris_B[node.prim_idx].v;
                    double dv[9];
                    for (int k = 0; k < 9; ++k) dv[k] = (double)fv[k];
                    d3 cp;
                    double d = pt_tri_sqr_dis(samples[s], dv, &cp);
                    if (d < best) best = d;
                    continue;
                }

                int lc = node.left, rc = node.right;
                double dl = aabb_point_sqr_dis(nodes_B[lc], samples[s]);
                double dr = aabb_point_sqr_dis(nodes_B[rc], samples[s]);

                if (dl <= dr) {
                    if (dr < best) stack[top++] = rc;
                    if (dl < best) stack[top++] = lc;
                } else {
                    if (dl < best) stack[top++] = lc;
                    if (dr < best) stack[top++] = rc;
                }
            }

            if (best > local_U) local_U = best;
        }

        // 更新全局界
        atomicMaxDouble(&state->global_L, local_U);
        atomicMaxDouble(&state->global_U, local_U);

        // 如果local_U > global_L，加入候选队列
        if (local_U > global_L) {
            int cand_idx = atomicAdd(candidate_count, 1);
            if (cand_idx < max_candidates) {  // 防止溢出
                candidates[cand_idx].tri_idx = tri_idx;
                candidates[cand_idx].local_L = local_U;  // 简化：L=U
                candidates[cand_idx].local_U = local_U;
                for (int k = 0; k < 9; ++k) {
                    candidates[cand_idx].vertices[k] = tri_A.v[k];
                }
            }
        }

        return;
    }

    // 内部节点：计算子节点的need_travel并决定是否继续遍历
    int left_child = node_A.left;
    int right_child = node_A.right;

    double l_dis = gpu_need_travel(nodes_A[left_child], nodes_B, tris_B, global_L);
    double r_dis = gpu_need_travel(nodes_A[right_child], nodes_B, tris_B, global_L);

    // 按距离从大到小遍历（距离大的更可能提高下界）
    int children[2] = {left_child, right_child};
    double dists[2] = {l_dis, r_dis};

    if (l_dis < r_dis) {
        int tmp = children[0]; children[0] = children[1]; children[1] = tmp;
        double tmp_d = dists[0]; dists[0] = dists[1]; dists[1] = tmp_d;
    }

    // 将满足条件的子节点加入输出队列
    for (int i = 0; i < 2; ++i) {
        if (dists[i] > global_L) {
            int out_idx = atomicAdd(output_count, 1);
            if (out_idx < 10000000) {  // 防止溢出
                output_pairs[out_idx].node_A_idx = children[i];
                output_pairs[out_idx].node_B_idx = 0;  // 始终是根节点
                output_pairs[out_idx].lower_bound = dists[i];
            }
        }
    }
}

// ─── Host函数：GPU并行遍历 ────────────────────────────────────────────────────

void gpu_parallel_traverse(
    const LBVH& lbvh_A,
    const LBVH& lbvh_B,
    const float3x3* d_tris_A,
    const float3x3* d_tris_B,
    int nA, int nB,
    CandidateTriangle** d_candidates,
    int* h_candidate_count,
    double* h_global_L,
    double* h_global_U)
{
    printf("[GPU_TRAVERSE] Starting parallel BVH traversal...\n");
    printf("[GPU_TRAVERSE] Mesh A: %d triangles, Mesh B: %d triangles\n", nA, nB);

    // 分配设备内存
    TraversalState* d_state;
    cudaMalloc(&d_state, sizeof(TraversalState));

    TraversalState h_state;
    h_state.global_L = 0.0;
    h_state.global_U = 0.0;  // 修改：初始化为0，而不是DBL_MAX
    h_state.active_count = 1;
    h_state.candidate_count = 0;
    cudaMemcpy(d_state, &h_state, sizeof(TraversalState), cudaMemcpyHostToDevice);

    // 分配候选三角形缓冲区（预估最多nA个，但为安全起见分配2倍空间）
    const int MAX_CANDIDATES = nA * 2;
    cudaMalloc(d_candidates, MAX_CANDIDATES * sizeof(CandidateTriangle));
    int* d_candidate_count;
    cudaMalloc(&d_candidate_count, sizeof(int));
    cudaMemset(d_candidate_count, 0, sizeof(int));

    // 分配节点对缓冲区（双缓冲）
    const int MAX_PAIRS = 10000000;
    BVHNodePair* d_pairs[2];
    int* d_pair_count[2];
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&d_pairs[i], MAX_PAIRS * sizeof(BVHNodePair));
        cudaMalloc(&d_pair_count[i], sizeof(int));
    }

    // 初始化：根节点对
    BVHNodePair root_pair;
    root_pair.node_A_idx = 0;
    root_pair.node_B_idx = 0;
    root_pair.lower_bound = 0.0;
    cudaMemcpy(d_pairs[0], &root_pair, sizeof(BVHNodePair), cudaMemcpyHostToDevice);
    int init_count = 1;
    cudaMemcpy(d_pair_count[0], &init_count, sizeof(int), cudaMemcpyHostToDevice);

    // 层次化遍历
    int current = 0;
    int level = 0;

    while (true) {
        int h_input_count;
        cudaMemcpy(&h_input_count, d_pair_count[current], sizeof(int), cudaMemcpyDeviceToHost);

        if (h_input_count == 0) break;

        printf("[GPU_TRAVERSE] Level %d: processing %d node pairs\n", level, h_input_count);

        // 重置输出计数
        cudaMemset(d_pair_count[1 - current], 0, sizeof(int));

        // 启动kernel
        int block = 128;
        int grid = (h_input_count + block - 1) / block;

        kernel_parallel_traverse_level<<<grid, block>>>(
            lbvh_A.d_nodes,
            lbvh_B.d_nodes,
            d_tris_A,
            d_tris_B,
            d_pairs[current],
            h_input_count,
            d_pairs[1 - current],
            d_pair_count[1 - current],
            *d_candidates,
            d_candidate_count,
            d_state,
            MAX_CANDIDATES);

        cudaDeviceSynchronize();

        // 切换缓冲区
        current = 1 - current;
        level++;

        // 防止无限循环
        if (level > 100) {
            printf("[GPU_TRAVERSE] Warning: max level reached\n");
            break;
        }
    }

    // 读取结果
    cudaMemcpy(h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_state, d_state, sizeof(TraversalState), cudaMemcpyDeviceToHost);
    *h_global_L = h_state.global_L;
    *h_global_U = h_state.global_U;

    printf("[GPU_TRAVERSE] Completed: %d candidates, L=%.6f, U=%.6f\n",
           *h_candidate_count, sqrt(*h_global_L), sqrt(*h_global_U));

    // 释放临时内存
    cudaFree(d_state);
    cudaFree(d_candidate_count);
    for (int i = 0; i < 2; ++i) {
        cudaFree(d_pairs[i]);
        cudaFree(d_pair_count[i]);
    }
}

void gpu_parallel_traverse_free(CandidateTriangle* d_candidates) {
    cudaFree(d_candidates);
}

void gpu_parallel_traverse_copy_candidates(
    const CandidateTriangle* d_candidates,
    CandidateTriangle* h_candidates,
    int count) {
    cudaError_t err = cudaMemcpy(h_candidates, d_candidates,
               count * sizeof(CandidateTriangle),
               cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[GPU_TRAVERSE_ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }
}
