# Memory Copy Benchmark Suite

A comprehensive benchmarking suite for comparing CUDA Host-to-Device (H2D) and Host-to-Host (H2H) memory copy performance.

## Project Structure

```
cuda-memcpy-benchmark/
├── src/                          # Source files
│   ├── h2d_benchmark.cu         # CUDA host-to-device benchmark
│   └── h2h_benchmark.cpp        # C++ host-to-host benchmark
├── build/                        # Compiled executables (auto-generated)
│   ├── h2d_benchmark
│   └── h2h_benchmark
├── results/                      # Benchmark results (auto-generated)
│   └── benchmark_*_*/           # Timestamped result directories
├── compile.sh                    # Compilation script
├── run_benchmarks.sh            # Main benchmark runner
└── plot_benchmarks.py           # Plotting script
```

## Quick Start

### Basic Usage
```bash
# Run with default parameters (32K elements, 100K iterations, pageable memory)
./run_benchmarks.sh

# Run with custom parameters
./run_benchmarks.sh 65536 50000

# Run without nsys profiling (faster)
./run_benchmarks.sh --no-profile 32768 100000

# Run with pinned memory (faster transfers)
./run_benchmarks.sh --pinned 32768 100000

# Combine flags
./run_benchmarks.sh --no-profile --pinned 32768 100000
```

### Command Line Options

**run_benchmarks.sh** accepts the following arguments:
```
./run_benchmarks.sh [--no-profile] [--pinned] [num_elements] [num_iterations]

Arguments:
  --no-profile      Disable nsys profiling (runs faster)
  --pinned          Use CUDA pinned (page-locked) host memory instead of pageable memory
  num_elements      Number of float elements to transfer (default: 32768)
  num_iterations    Number of benchmark iterations (default: 100000)
```

**Memory Types:**
- **Pageable (default)**: Standard host memory allocated with `malloc()`. Requires OS paging for DMA transfers.
- **Pinned (--pinned)**: Page-locked memory allocated with `cudaMallocHost()`. Provides faster transfer speeds by avoiding paging overhead.

## Output

All results are saved to timestamped directories under `results/`:

### With Profiling (default)
```
results/benchmark_<elements>elem_<iterations>iter_<timestamp>/
├── h2d_results.json              # H2D benchmark data
├── h2h_results.json              # H2H benchmark data
├── h2d_results_plot.png          # H2D timing visualization
├── h2h_results_plot.png          # H2H timing visualization
├── comparison_plot.png           # Side-by-side comparison
├── h2d_profile.nsys-rep          # nsys profile data
├── h2d_profile.sqlite            # nsys database
└── h2d_profile_stats.txt         # nsys statistics output
```

### Without Profiling (--no-profile)
```
results/benchmark_noprofile_<elements>elem_<iterations>iter_<timestamp>/
├── h2d_results.json              # H2D benchmark data
├── h2h_results.json              # H2H benchmark data
├── h2d_results_plot.png          # H2D timing visualization
├── h2h_results_plot.png          # H2H timing visualization
└── comparison_plot.png           # Side-by-side comparison
```

## JSON Output Format

Both benchmarks output JSON files with the following structure:

**H2D Benchmark:**
```json
{
  "benchmark_type": "H2D",
  "memory_type": "pinned",           // or "pageable"
  "num_elements": 32768,
  "num_iterations": 100000,
  "element_size_bytes": 4,
  "total_bytes": 131072,
  "min_time_ms": 0.010832,
  "max_time_ms": 0.877602,
  "avg_time_ms": 0.018481,
  "timings_ms": [0.012, 0.015, ...]  // All timing measurements
}
```

**H2H Benchmark:**
```json
{
  "benchmark_type": "H2H",
  "num_elements": 32768,
  "num_iterations": 100000,
  "element_size_bytes": 4,
  "total_bytes": 131072,
  "min_time_ms": 0.001024,
  "max_time_ms": 0.108800,
  "avg_time_ms": 0.001283,
  "timings_ms": [0.001, 0.001, ...]  // All timing measurements
}
```

## Manual Usage

### Compile Only
```bash
./compile.sh
```
This creates executables in the `build/` directory.

### Run Benchmarks Individually
```bash
# H2D benchmark (pageable memory)
./build/h2d_benchmark <num_elements> <num_iterations> <output.json>

# H2D benchmark (pinned memory)
./build/h2d_benchmark <num_elements> <num_iterations> <output.json> --pinned

# H2H benchmark
./build/h2h_benchmark <num_elements> <num_iterations> <output.json>
```

### Generate Plots Only
```bash
# Single benchmark plot
python3 plot_benchmarks.py <output_dir> <benchmark.json>

# Multiple benchmarks (with comparison)
python3 plot_benchmarks.py <output_dir> <h2d.json> <h2h.json>
```

### Profile with nsys
```bash
sudo nsys profile \
    -o profile_output \
    --stats=true \
    ./build/h2d_benchmark 32768 100000 results.json
```

## Requirements

- **CUDA Toolkit** (with `nvcc` compiler)
- **g++** compiler
- **Python 3** with:
  - `matplotlib`
  - `numpy`
- **nsys** (NVIDIA Nsight Systems) for profiling

## Examples

### Quick 1 million iteration test
```bash
./run_benchmarks.sh 32768 1000000
```

### Fast test without profiling
```bash
./run_benchmarks.sh --no-profile 16384 10000
```

### Large transfer test
```bash
./run_benchmarks.sh 1048576 50000
```

### Compare pageable vs pinned memory
```bash
# First run with pageable memory
./run_benchmarks.sh --no-profile 32768 100000

# Then run with pinned memory
./run_benchmarks.sh --no-profile --pinned 32768 100000
```

### Profile pinned memory transfers
```bash
./run_benchmarks.sh --pinned 32768 100000
```

## Benchmark Details

### H2D Benchmark (CUDA Host-to-Device)
- Uses `cudaMemcpyAsync` with stream synchronization
- Supports both pageable and pinned host memory
- Measures transfer latency for each iteration
- Includes 100 warm-up iterations

### H2H Benchmark (Host-to-Host)
- Uses standard C++ `memcpy`
- Measures host memory copy performance
- Useful as a baseline comparison
- Includes 100 warm-up iterations

### Timing Methodology
- All times measured using `std::chrono::high_resolution_clock`
- Timing includes copy operation and synchronization
- Statistics: min, max, average calculated from all iterations
- Plots show individual iteration times, averages, and 50µs threshold

## Performance Notes

**Pinned Memory Benefits:**
- Typically 2-3x faster than pageable memory
- Direct DMA access without OS paging overhead
- More predictable latencies
- Limited by available physical RAM

**Profiling Overhead:**
- nsys profiling adds minimal overhead to actual transfers
- Use `--no-profile` for pure benchmark performance
- Profile data useful for analyzing CUDA API calls and memory operations

