#!/bin/bash
output_basename=$2
output_dir="$1/$output_basename"
mkdir -p "$output_dir"
num_elements=${3:-65536}
num_iterations=${4:-100000}
sudo nsys profile -o "$output_dir/$output_basename" \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  ./simple_memcpy_benchmark $num_elements $num_iterations | tee "$output_dir/$output_basename.md"
