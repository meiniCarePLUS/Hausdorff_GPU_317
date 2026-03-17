#pragma once

// Forward declaration for LBVH (C++ compatible)
struct LBVH;
struct float3x3;

// C++ interface for LBVH functions (implemented in .cu files)
// Returns a pointer to LBVH that must be freed with lbvh_free
LBVH* lbvh_build_from_mesh(const double* vertices, int num_verts,
                           const int* triangles, int num_tris);

void lbvh_free(LBVH* lbvh);

void mesh_to_gpu_triangles(const double* vertices, const int* triangles,
                          int num_tris, float3x3** d_tris_out);

void free_gpu_triangles(float3x3* d_tris);
