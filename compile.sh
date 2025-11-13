#!/bin/bash
# Compilation script for memory copy benchmarks

set -e  # Exit on error

echo "=== Compiling Memory Copy Benchmarks ==="
echo ""

# Create build directory if it doesn't exist
mkdir -p build

# CUDA H2D Benchmark
echo "Compiling CUDA H2D benchmark..."
nvcc -O3 -std=c++11 \
    -o build/h2d_benchmark \
    src/h2d_benchmark.cu
echo "✓ CUDA H2D benchmark compiled successfully: build/h2d_benchmark"
echo ""

# C++ H2H Benchmark
echo "Compiling C++ H2H benchmark..."
g++ -O3 -std=c++11 \
    -o build/h2h_benchmark \
    src/h2h_benchmark.cpp
echo "✓ C++ H2H benchmark compiled successfully: build/h2h_benchmark"
echo ""

# System Info Utility
echo "Compiling system info utility..."
nvcc -O3 -std=c++11 \
    -o build/system_info \
    src/system_info.cu
echo "✓ System info utility compiled successfully: build/system_info"
echo ""

echo "=== Compilation Complete ==="
echo "Executables created:"
echo "  - build/h2d_benchmark (CUDA Host-to-Device)"
echo "  - build/h2h_benchmark (C++ Host-to-Host)"
echo "  - build/system_info (System Information Utility)"
