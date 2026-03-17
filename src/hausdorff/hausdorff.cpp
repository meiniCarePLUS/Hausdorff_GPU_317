/*
	State Key Lab of CAD&CG Zhejiang Unv.

	Author: 
          Yicun Zheng (3130104113@zju.edu.cn)
          Haoran Sun (hrsun@zju.edu.cn)
          Jin Huang (hj@cad.zju.edu.cn)

	Copyright (c) 2004-2021 <Jin Huang>
	All rights reserved.

	Licensed under the MIT License.
*/

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include "zjucad/matrix/io.h"

#include "bvh/nearest_query.h"
#include "core/geometry/primitive_dis.hpp"
#include "hausdorff.h"
#include "hausdorff/closest_cache.hpp"
#include "hausdorff/gpu_query_iface.hpp"
#include "hausdorff/gpu_parallel_traverse_iface.hpp"
#include "hausdorff_internal.h"
#include "mesh/adjacent_table.hpp"

// Enable GPU parallel traverse (set to 1 to use GPU, 0 to use CPU)
// NOTE: GPU traverse currently has correctness issues, use CPU for now
#define USE_GPU_TRAVERSE 0

using namespace std;
using namespace zjucad::matrix;
using namespace std::chrono;

void traverse(const bvh &A, const bvh &B, double &L, double &H, unique_ptr<hd_trait> &trait, point_t &max_point) {
    // leaf node
    if (A.is_leaf()) {
        trait->iterate_leaf(*A.get_primitive(), B, L, H, max_point);
        return;
    }

    // non leaf node
    if (!A.is_leaf()) { // TODO: why this if is necessary?  You have check this in previous lines

        // TODO: you used array (children_array, but not use array for dis.  It is better to make them consistent
        const double l_dis = trait->need_travel(*A.left_child(), B, L, H);
        const double r_dis = trait->need_travel(*A.right_child(), B, L, H);

        shared_ptr<bvh> children_array[2] = {A.left_child(), A.right_child()};
        const double *dis_array[2] = {&l_dis, &r_dis};

        // the node with larger distance has more possibility to improve the lower bound of hausdorff distance higher
        // so we travel the one with higher distance
        if (l_dis < r_dis) {
            std::swap(children_array[0], children_array[1]);
            std::swap(dis_array[0], dis_array[1]);
        }
        for (size_t i = 0; i < 2; ++i) {
            if (*dis_array[i] > L) {
                traverse(*children_array[i], B, L, H, trait, max_point);
            }
        }
    }
}

void subdivide(tri_mesh &A,
               closest_cache &cache,
               primitive_adjacent_table &adj_table,
               shared_ptr<bvh> B,
               const tri_mesh &model_B,
               const primitive_t &prim,
               bool use_voronoi,
               primitive_t result[4],
               size_t &voronoi_count,
               size_t &mid_count) {

    point_t new_vertex[3];
    size_t new_vertex_id[3];
    for (size_t i = 0; i < 3; ++i) {
        new_vertex_id[i] = A.is_subdivision_vertex_exist(prim.point_id[i], prim.point_id[(i + 1) % 3]);
        if (new_vertex_id[i] != -1) {
            new_vertex[i] = A.get_vertex(new_vertex_id[i]);
        } else {
            bool voronoi_valid = false;

            if (use_voronoi) {
                primitive_t *t[2];
                for (size_t point_iter = 0; point_iter < 2; ++point_iter) {
                    size_t point_id = (i + point_iter) % 3;
                    t[point_iter] = cache.get(prim.point_id[point_id], prim.points(colon(), point_id));
                }
                auto edge_itr = adj_table.table_.find(primitive_pair(t[0]->id, t[1]->id));
                if (edge_itr != adj_table.table_.cend()) {
                    voronoi_count++;
                    new_vertex[i] = voronoi_subdivide(
                        *t[0], *t[1],
                        (*model_B.v_)(colon(), edge_itr->second.first), (*model_B.v_)(colon(), edge_itr->second.second),
                        prim.points(colon(), i), prim.points(colon(), (i + 1) % 3));
                    voronoi_valid = true;
                }
            }
            if (!voronoi_valid) {
                mid_count++;
                new_vertex[i] = (prim.points(colon(), i) + prim.points(colon(), (i + 1) % 3)) / 2;
            }
            new_vertex_id[i] = A.add_subdivision_vertex(prim.point_id[i], prim.point_id[(i + 1) % 3], new_vertex[i]);
            // update closest cache
            cache.get(new_vertex_id[i], new_vertex[i]);
        }
    }

    for (size_t i = 0; i < 3; ++i) {
        result[i].points(colon(), 1) = new_vertex[i];
        result[i].points(colon(), 0) = prim.points(colon(), i);
        result[i].points(colon(), 2) = new_vertex[(i + 2) % 3];
        result[i].point_id[0] = prim.point_id[i];
        result[i].point_id[1] = new_vertex_id[i];
        result[i].point_id[2] = new_vertex_id[(i + 2) % 3];
        result[i].id = prim.id;
    }

    for (size_t d = 0; d < 3; ++d) {
        result[3].points(colon(), d) = new_vertex[d];
        result[3].point_id(d, 0) = new_vertex_id[d];
        result[3].id = prim.id;
    }
}

hausdorff_result hausdorff(tri_mesh &A, const tri_mesh &B,
                           shared_ptr<bvh> pbvh[2], unique_ptr<hd_trait> &trait,
                           bool use_voronoi, function<bool(double, double)> &stop_condition,
                           const LBVH* lbvh_A,
                           const LBVH* lbvh_B,
                           const float3x3* d_tris_A,
                           const float3x3* d_tris_B) {
    // build primitive adjacent table
    primitive_adjacent_table adjacent_table;
    build_primitive_adjacent_table_from_mesh(*B.v_, *B.t_, adjacent_table);

    // first iteration
    high_resolution_clock::time_point begin_clock = high_resolution_clock::now();
    double L = 0, U = std::numeric_limits<double>::max();

    point_t max_point = ones<double>(3, 1);

#if USE_GPU_TRAVERSE
    // Use GPU parallel traverse if LBVH data is available
    if (lbvh_A && lbvh_B && d_tris_A && d_tris_B) {
        logs(cout) << "[GPU_TRAVERSE] Using GPU parallel BVH traversal" << std::endl;

        CandidateTriangle* d_candidates = nullptr;
        int candidate_count = 0;

        gpu_parallel_traverse(
            *lbvh_A, *lbvh_B,
            d_tris_A, d_tris_B,
            A.t_->size(2), B.t_->size(2),
            &d_candidates,
            &candidate_count,
            &L, &U);

        logs(cout) << "[GPU_TRAVERSE] Candidates: " << candidate_count
                   << ", L=" << sqrt(L) << ", U=" << sqrt(U) << std::endl;

        // 将候选三角形加入优先队列
        if (candidate_count > 0 && d_candidates != nullptr) {
            logs(cout) << "[GPU_TRAVERSE] Copying " << candidate_count << " candidates from device to host..." << std::endl;
            logs(cout) << "[GPU_TRAVERSE] Candidate size: " << sizeof(CandidateTriangle) << " bytes" << std::endl;
            logs(cout) << "[GPU_TRAVERSE] Total memory: " << (candidate_count * sizeof(CandidateTriangle)) << " bytes" << std::endl;

            std::vector<CandidateTriangle> h_candidates(candidate_count);
            logs(cout) << "[GPU_TRAVERSE] Host vector allocated" << std::endl;

            gpu_parallel_traverse_copy_candidates(d_candidates, h_candidates.data(), candidate_count);
            logs(cout) << "[GPU_TRAVERSE] Copy completed" << std::endl;

            logs(cout) << "[GPU_TRAVERSE] tri_A_.size() = " << trait->tri_A_.size() << std::endl;

            int added_count = 0;
            for (int i = 0; i < candidate_count; ++i) {
                int tri_idx = h_candidates[i].tri_idx;
                if (tri_idx >= 0 && tri_idx < (int)trait->tri_A_.size()) {
                    trait->left_tris.push(primitive_with_hd(
                        trait->tri_A_[tri_idx],
                        h_candidates[i].local_L,
                        h_candidates[i].local_U));
                    added_count++;
                }
            }
            logs(cout) << "[GPU_TRAVERSE] Added " << added_count << " candidates to queue" << std::endl;
        }

        if (d_candidates != nullptr) {
            gpu_parallel_traverse_free(d_candidates);
        }
    } else {
        logs(cout) << "[CPU_TRAVERSE] Using CPU serial BVH traversal" << std::endl;
        traverse(*pbvh[0], *pbvh[1], L, U, trait, max_point);
    }
#else
    // Use CPU traverse
    logs(cout) << "[CPU_TRAVERSE] Using CPU serial BVH traversal" << std::endl;
    traverse(*pbvh[0], *pbvh[1], L, U, trait, max_point);
#endif

    // TODO: design a class for more convinient time measure: void
    // clock.start(), double clock.stop() returns the milisecond since
    // the last call of start.
    high_resolution_clock::time_point end_clock = high_resolution_clock::now();

    hausdorff_result result;
    // update result structure
    result.first_travel_bound = make_pair(sqrt(L), sqrt(U));
    result.first_travel_cost = duration_cast<duration<double>>(end_clock - begin_clock).count() * 1000;

    // subdivide to shrink error bound
    begin_clock = high_resolution_clock::now();

    size_t voronoi_count = 0, mid_count = 0;
    // vector<double> new_edge; vector<size_t> edge_index;

    // while( ((!use_relative_error && (sqrt(U)-sqrt(L) > error)) || (use_relative_error && ((sqrt(U)-sqrt(L)) >= (sqrt(U)+sqrt(L))/2 * 0.01)))
    logs(cout) << "[culling rate] " << 1 - (trait->left_tris.size() * 1.0 / A.t_->size(2)) << std::endl;

    // Batch size: process up to BATCH triangles per GPU round-trip.
    // Larger batches amortize kernel launch overhead; smaller batches preserve
    // priority-queue ordering. 8 is a good balance for typical meshes.
    static const int BATCH = 8;

    while (!stop_condition(sqrt(L), sqrt(U)) && (!trait->left_tris.empty())) {
        // Collect up to BATCH active triangles from the priority queue.
        std::vector<primitive_with_hd> batch;
        batch.reserve(BATCH);
        while ((int)batch.size() < BATCH && !trait->left_tris.empty()) {
            primitive_with_hd pwhd = trait->left_tris.top();
            trait->left_tris.pop();
            if (pwhd.U < L) { U = L; goto done; }
            batch.push_back(pwhd);
        }
        if (!trait->left_tris.empty()) U = trait->left_tris.top().U;

        // Subdivide all triangles in the batch.
        std::vector<std::array<primitive_t,4>> sub(batch.size());
        for (int b = 0; b < (int)batch.size(); ++b)
            subdivide(A, trait->closest_cache_, adjacent_table, pbvh[1], B,
                      batch[b].prim, use_voronoi, sub[b].data(), voronoi_count, mid_count);

        // Collect all cache-miss vertices across the entire batch for one GPU query.
        {
            // max 6 unique verts per triangle * BATCH triangles
            std::vector<double> pts_buf;
            std::vector<size_t> ids_buf;
            pts_buf.reserve(BATCH * 6 * 3);
            ids_buf.reserve(BATCH * 6);

            for (int b = 0; b < (int)batch.size(); ++b)
                for (int ti = 0; ti < 4; ++ti)
                    for (int vi = 0; vi < 3; ++vi) {
                        size_t id = sub[b][ti].point_id(vi, 0);
                        if (trait->closest_cache_.get(id) != nullptr) continue;
                        bool dup = false;
                        for (size_t j = 0; j < ids_buf.size(); ++j)
                            if (ids_buf[j] == id) { dup = true; break; }
                        if (dup) continue;
                        ids_buf.push_back(id);
                        pts_buf.push_back(sub[b][ti].points(0, vi));
                        pts_buf.push_back(sub[b][ti].points(1, vi));
                        pts_buf.push_back(sub[b][ti].points(2, vi));
                    }

            if (!ids_buf.empty()) {
                std::vector<int> nearest_buf(ids_buf.size());
                gpu_plain_query(pts_buf.data(), (int)ids_buf.size(), nearest_buf.data());
                for (int j = 0; j < (int)ids_buf.size(); ++j)
                    trait->closest_cache_.set(ids_buf[j],
                        const_cast<primitive_t*>(&trait->tri_B_[nearest_buf[j]]));
            }
        }

        // Compute local bounds for all sub-triangles.
        for (int b = 0; b < (int)batch.size(); ++b)
            for (size_t i = 0; i < 4; ++i) {
                if ((trait->closest_cache_.get(sub[b][i].point_id(0,0)) ==
                     trait->closest_cache_.get(sub[b][i].point_id(1,0))) &&
                    (trait->closest_cache_.get(sub[b][i].point_id(1,0)) ==
                     trait->closest_cache_.get(sub[b][i].point_id(2,0))))
                    continue;
                trait->shrink_bound(sub[b][i], batch[b].prim, *pbvh[1], L, U, batch[b].U, max_point);
            }

        if (!trait->left_tris.empty()) U = trait->left_tris.top().U;
    }
    done:;

    end_clock = high_resolution_clock::now();

    result.max_point = max_point;
    result.bound_reduce_cost = duration_cast<duration<double>>(end_clock - begin_clock).count() * 1000;
    result.hausdorff_bound = make_pair(sqrt(L), sqrt(U));
    result.voronoi_subdivision_count = voronoi_count;
    result.midpoint_subdivision_count = mid_count;
    // {
    //   ofstream debug_line("debug_line.vtk");
    //   line2vtk(debug_line, new_edge.data(), new_edge.size()/3, edge_index.data(), edge_index.size()/2);
    //   debug_line.close();
    // }

    return result;
}

// similar to closest_point
// calculate hausdorff distance from bvh node to point
// used in traversal check
double hausdorff(const bvh &node, const point_t &p) {
    {
        const aabb *pt = dynamic_cast<const aabb *>(&node);
        if (!!pt) {
            // aabb node
            point_t mid_point = pt->mid();
            point_t max_vector = p - mid_point;
            point_t mid_corner_vector = mid_point - pt->get_minmax()(colon(), 0);
            for (size_t d = 0; d < p.size(1); ++d) {
                max_vector(d, 0) = fabs(max_vector(d, 0)) + fabs(mid_corner_vector(d, 0));
            }
            return dot(max_vector, max_vector); //max_vector(0, 0) * max_vector(0, 0) + max_vector(1, 0) * max_vector(1, 0) + max_vector(2, 0) * max_vector(2, 0);
        }
    }

    logs(cout) << ERROR
               << "unknown bvh type" << std::endl;
    exit(-1);
}
