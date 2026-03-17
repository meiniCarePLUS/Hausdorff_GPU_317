// Test GPU parallel BVH traversal
#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "bvh/lbvh.cuh"
#include "hausdorff/gpu_parallel_traverse.cuh"

static bool load_obj(const char* path, std::vector<float>& verts, std::vector<int>& tris) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='v' && line[1]==' ') {
            float x,y,z; sscanf(line+2,"%f %f %f",&x,&y,&z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0]=='f' && line[1]==' ') {
            int a,b,c;
            if (sscanf(line+2,"%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d//%*d %d//%*d %d//%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d/%*d %d/%*d %d/%*d",&a,&b,&c)==3 ||
                sscanf(line+2,"%d %d %d",&a,&b,&c)==3)
                tris.push_back(a-1), tris.push_back(b-1), tris.push_back(c-1);
        }
    }
    fclose(f);
    return !tris.empty();
}

static std::vector<float3x3> make_tris(const std::vector<float>& v, const std::vector<int>& t) {
    int n = (int)(t.size()/3);
    std::vector<float3x3> out(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            int vi = t[i*3+k];
            out[i].v[k*3+0] = v[vi*3+0];
            out[i].v[k*3+1] = v[vi*3+1];
            out[i].v[k*3+2] = v[vi*3+2];
        }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <meshA.obj> <meshB.obj>\n", argv[0]);
        return 1;
    }

    // Load meshes
    std::vector<float> vA, vB;
    std::vector<int> tA, tB;
    if (!load_obj(argv[1], vA, tA) || !load_obj(argv[2], vB, tB)) return 1;

    auto hA = make_tris(vA, tA), hB = make_tris(vB, tB);
    int nA = (int)hA.size(), nB = (int)hB.size();
    printf("Mesh A: %d tris   Mesh B: %d tris\n", nA, nB);

    // Build LBVH for both meshes
    LBVH lbvh_A{}, lbvh_B{};
    lbvh_A.build(hA.data(), nA);
    lbvh_B.build(hB.data(), nB);

    // Upload triangles to GPU
    float3x3 *d_tA, *d_tB;
    cudaMalloc(&d_tA, nA * sizeof(float3x3));
    cudaMalloc(&d_tB, nB * sizeof(float3x3));
    cudaMemcpy(d_tA, hA.data(), nA * sizeof(float3x3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tB, hB.data(), nB * sizeof(float3x3), cudaMemcpyHostToDevice);

    // Run GPU parallel traversal
    CandidateTriangle* d_candidates = nullptr;
    int candidate_count = 0;
    double global_L = 0, global_U = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    gpu_parallel_traverse(
        lbvh_A, lbvh_B,
        d_tA, d_tB,
        nA, nB,
        &d_candidates,
        &candidate_count,
        &global_L, &global_U);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n========================================\n");
    printf("GPU Parallel Traversal Results:\n");
    printf("========================================\n");
    printf("Time:       %.3f ms\n", ms);
    printf("Candidates: %d\n", candidate_count);
    printf("Global L:   %.6f\n", sqrt(global_L));
    printf("Global U:   %.6f\n", sqrt(global_U));
    printf("Culling rate: %.2f%%\n", 100.0 * (1.0 - candidate_count / (double)nA));

    // Cleanup
    gpu_parallel_traverse_free(d_candidates);
    cudaFree(d_tA);
    cudaFree(d_tB);
    lbvh_A.free();
    lbvh_B.free();

    return 0;
}
