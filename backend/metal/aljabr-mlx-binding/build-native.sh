#!/bin/bash
set -e

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build/native"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Run CMake
# We expect MLX to be in 3rdparty/mlx (6 levels up from src/main/cpp)
cmake "${SCRIPT_DIR}/src/main/cpp"

# Build
make -j$(sysctl -n hw.ncpu)

echo ""
echo "✅ Build complete: ${BUILD_DIR}/lib/libaljabr_mlx_bridge.dylib"
echo ""
echo "To use in Java, point your library path to this directory."
