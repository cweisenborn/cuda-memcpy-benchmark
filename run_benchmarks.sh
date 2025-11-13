#!/bin/bash
# Run memory copy benchmarks with profiling and visualization

set -e  # Exit on error

# Parse arguments
PROFILE_ENABLED=true
PINNED_MEMORY=false
NUM_ELEMENTS=32768
NUM_ITERATIONS=100000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-profile)
            PROFILE_ENABLED=false
            shift
            ;;
        --pinned)
            PINNED_MEMORY=true
            shift
            ;;
        *)
            if [ -z "${NUM_ELEMENTS_SET}" ]; then
                NUM_ELEMENTS=$1
                NUM_ELEMENTS_SET=true
                shift
            elif [ -z "${NUM_ITERATIONS_SET}" ]; then
                NUM_ITERATIONS=$1
                NUM_ITERATIONS_SET=true
                shift
            else
                echo "Unknown argument: $1"
                echo "Usage: $0 [--no-profile] [--pinned] [num_elements] [num_iterations]"
                exit 1
            fi
            ;;
    esac
done

# Default parameters
OUTPUT_DIR="results"

# Generate unique run identifier
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="benchmark_${NUM_ELEMENTS}elem_${NUM_ITERATIONS}iter"
if [ "$PINNED_MEMORY" = true ]; then
    RUN_NAME="${RUN_NAME}_pinned"
fi
if [ "$PROFILE_ENABLED" = false ]; then
    RUN_NAME="${RUN_NAME}_noprofile"
fi
RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"

echo "=== Memory Copy Benchmark Runner ==="
echo "Configuration:"
echo "  Number of elements: ${NUM_ELEMENTS}"
echo "  Number of iterations: ${NUM_ITERATIONS}"
echo "  Host memory type: $([ "$PINNED_MEMORY" = true ] && echo "Pinned (cudaMallocHost)" || echo "Pageable (malloc)")"
echo "  Profiling enabled: ${PROFILE_ENABLED}"
echo "  Output directory: ${RUN_DIR}"
echo ""

# Create output directory
mkdir -p "${RUN_DIR}"

# Output file paths
H2D_JSON="${RUN_DIR}/h2d_results.json"
H2H_JSON="${RUN_DIR}/h2h_results.json"
H2D_NSYS="${RUN_DIR}/h2d_profile"
H2D_STATS="${RUN_DIR}/h2d_profile_stats.txt"

echo "=== Step 1: Compiling Benchmarks ==="
./compile.sh
echo ""

echo "=== Step 2: Running H2D Benchmark ==="

# Build the benchmark command with optional pinned memory flag
H2D_CMD="./build/h2d_benchmark ${NUM_ELEMENTS} ${NUM_ITERATIONS} ${H2D_JSON}"
if [ "$PINNED_MEMORY" = true ]; then
    H2D_CMD="${H2D_CMD} --pinned"
fi

if [ "$PROFILE_ENABLED" = true ]; then
    echo "Running CUDA benchmark with nsys profiling..."
    sudo nsys profile \
        -o "${H2D_NSYS}" \
        --stats=true \
        --force-overwrite=true \
        ${H2D_CMD} 2>&1 | tee "${H2D_STATS}"
    
    echo ""
    echo "✓ H2D benchmark complete"
    echo "  JSON output: ${H2D_JSON}"
    echo "  nsys profile: ${H2D_NSYS}.nsys-rep"
    echo "  nsys stats: ${H2D_STATS}"
else
    echo "Running CUDA benchmark (no profiling)..."
    ${H2D_CMD}
    
    echo ""
    echo "✓ H2D benchmark complete"
    echo "  JSON output: ${H2D_JSON}"
fi
echo ""

echo "=== Step 3: Running H2H Benchmark ==="
./build/h2h_benchmark ${NUM_ELEMENTS} ${NUM_ITERATIONS} "${H2H_JSON}"
echo ""
echo "✓ H2H benchmark complete"
echo "  JSON output: ${H2H_JSON}"
echo ""

echo "=== Step 4: Generating Plots ==="
python3 plot_benchmarks.py "${RUN_DIR}" "${H2D_JSON}" "${H2H_JSON}"
echo ""

echo "=== Benchmark Run Complete ==="
echo ""
echo "Results saved to: ${RUN_DIR}"
echo ""
echo "Generated files:"
echo "  Benchmarks:"
echo "    - h2d_results.json (CUDA H2D results)"
echo "    - h2h_results.json (C++ H2H results)"
echo "  Plots:"
echo "    - h2d_results_plot.png (H2D timing plot)"
echo "    - h2h_results_plot.png (H2H timing plot)"
echo "    - comparison_plot.png (H2D vs H2H comparison)"
if [ "$PROFILE_ENABLED" = true ]; then
    echo "  Profiling:"
    echo "    - h2d_profile.nsys-rep (nsys profile data)"
    echo "    - h2d_profile.sqlite (nsys database)"
    echo "    - h2d_profile_stats.txt (nsys statistics output)"
    echo ""
    echo "To view nsys profile:"
    echo "  nsys-ui ${H2D_NSYS}.nsys-rep"
fi
echo ""
echo "=== Summary ==="
echo ""

# Extract and display key metrics
if command -v jq &> /dev/null; then
    echo "H2D (CUDA) Results:"
    jq -r '"  Min: \(.min_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2D_JSON}"
    jq -r '"  Avg: \(.avg_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2D_JSON}"
    jq -r '"  Max: \(.max_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2D_JSON}"
    echo ""
    echo "H2H (memcpy) Results:"
    jq -r '"  Min: \(.min_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2H_JSON}"
    jq -r '"  Avg: \(.avg_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2H_JSON}"
    jq -r '"  Max: \(.max_time_ms * 1000 | tostring | .[0:8]) µs"' "${H2H_JSON}"
else
    echo "Install 'jq' to see formatted summary"
    echo "Raw results available in:"
    echo "  ${H2D_JSON}"
    echo "  ${H2H_JSON}"
fi
