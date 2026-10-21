cmake_minimum_required(VERSION 3.24)
project(aljabr_mlx_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to MLX source in 3rdparty
set(MLX_DIR "../../../../../../../3rdparty/mlx")
add_subdirectory(${MLX_DIR} mlx_build)

# Create the bridge shared library
add_library(aljabr_mlx_bridge SHARED 
    mlx_bridge/aljabr_mlx_bridge.cpp
)

target_link_libraries(aljabr_mlx_bridge PRIVATE mlx)

# Ensure the output is a dylib on macOS
set_target_properties(aljabr_mlx_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
