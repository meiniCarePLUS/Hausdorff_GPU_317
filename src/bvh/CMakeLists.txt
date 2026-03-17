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

file(GLOB zycgeo_bvh_src "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(zycgeo_bvh ${zycgeo_bvh_src})
target_link_libraries(zycgeo_bvh zycgeo_core)
target_include_directories(
    zycgeo_bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# GPU LBVH — separate CUDA library so the rest of the project stays C++.
add_library(zycgeo_lbvh STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/lbvh.cu
)
set_target_properties(zycgeo_lbvh PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(zycgeo_lbvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zycgeo_lbvh PRIVATE cudart)
