#	State Key Lab of CAD&CG Zhejiang Unv.
#
#	Author: 
#          Yicun Zheng (3130104113@zju.edu.cn)
#          Haoran Sun (hrsun@zju.edu.cn)
#          Jin Huang (hj@cad.zju.edu.cn)
#
#	Copyright (c) 2004-2021 <Jin Huang>
#	All rights reserved.
#
#	Licensed under the MIT License.

file(GLOB zycgeo_hausdorff_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(zycgeo_hausdorff ${zycgeo_hausdorff_src})
target_link_libraries(zycgeo_hausdorff zycgeo_core
                                       zycgeo_mesh
                                       zycgeo_bvh
                                       zycgeo_gpu_bounds
                                       cudart)

# GPU initial bounds + batch query library
add_library(zycgeo_gpu_bounds STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_initial_bounds.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_batch_query.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu_parallel_traverse.cu
)
set_target_properties(zycgeo_gpu_bounds PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(zycgeo_gpu_bounds PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(zycgeo_gpu_bounds PRIVATE zycgeo_lbvh cudart)
