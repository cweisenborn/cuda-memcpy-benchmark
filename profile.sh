#!/bin/bash
output_basename=$2
output_dir="$1/$output_basename"
mkdir -p "$output_dir"
num_elements=${3:-65536}
num_iterations=${4:-100000}
async_mode=${5:-True}
# sudo nsys profile -o "$output_dir/$output_basename" \
#   --trace=cuda,nvtx,osrt,cudnn,cublas \
#   --cuda-memory-usage=true \
#   --stats=true \
#   --force-overwrite=true \
#   ./.venv/bin/python src/memcpy_benchmark.py $num_elements $num_iterations "$async_mode" "$output_dir/$output_basename" | tee "$output_dir/$output_basename.md"


sudo nsys profile -o "$output_dir/$output_basename" \
  --stats=true \
  ./.venv/bin/python src/memcpy_benchmark.py $num_elements $num_iterations "$async_mode" "$output_dir/$output_basename" | tee "$output_dir/$output_basename.md"
