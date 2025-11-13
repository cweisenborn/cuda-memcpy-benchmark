#!/bin/bash

# Auto-detect GPU compute capability
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')

if [ -z "$COMPUTE_CAP" ]; then
    echo "Error: Could not detect GPU compute capability"
    exit 1
fi

echo "Detected GPU compute capability: $COMPUTE_CAP"

# Convert to SM architecture (e.g., 75 -> sm_75, 121 -> sm_121)
SM_ARCH="sm_${COMPUTE_CAP}"

echo "Compiling for ${SM_ARCH}..."

nvcc -G src/simple_memcpy_benchmark.cu -o simple_memcpy_benchmark \
    -arch=${SM_ARCH} \
    -O3 \
    -lpthread

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi
