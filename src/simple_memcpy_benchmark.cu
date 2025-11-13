#include <iostream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

// ============================================================================
// Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(error) << std::endl;     \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while(0)

// ============================================================================
// Statistics Structure
// ============================================================================

struct MemcpyStats {
    float min_time;
    float max_time;
    float avg_time;
    float total_time;
    size_t num_elements;
    size_t bytes_transferred;
    float bandwidth_gbps;
};

MemcpyStats simple_h2d_benchmark(size_t num_elements, int num_iterations) {
    MemcpyStats stats = {};

    int16_t *h_data, *d_data;
    h_data = (int16_t *)malloc(num_elements * sizeof(int16_t));
    // CUDA_CHECK(cudaMallocHost(&h_data, num_elements * sizeof(int16_t)));
    CUDA_CHECK(cudaHostAlloc(&h_data, num_elements * sizeof(int16_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(int16_t)));

    for (size_t i = 0; i < num_elements; i++)
        h_data[i] = static_cast<int16_t>(i % 32767);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm up
    for (int i = 0; i < 25; i++)
        cudaMemcpyAsync(d_data, h_data, num_elements * sizeof(int16_t),
                        cudaMemcpyHostToDevice, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<cudaEvent_t> start(num_iterations), stop(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }

    // Record all operations first
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start[i], stream);
        cudaMemcpyAsync(d_data, h_data, num_elements * sizeof(int16_t),
                        cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stop[i], stream);
        cudaEventSynchronize(stop[i]); 
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Collect timings
    std::vector<float> timings;
    timings.reserve(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        float ms;
        cudaEventElapsedTime(&ms, start[i], stop[i]);
        timings.push_back(ms);
    }

    // Calculate statistics
    stats.min_time = std::numeric_limits<float>::max();
    stats.max_time = 0.0f;
    stats.total_time = 0.0f;
    
    for (float time : timings) {
        stats.min_time = std::min(stats.min_time, time);
        stats.max_time = std::max(stats.max_time, time);
        stats.total_time += time;
    }
    
    stats.avg_time = stats.total_time / num_iterations;
    
    // Calculate bandwidth
    stats.num_elements = num_elements;
    stats.bytes_transferred = num_elements * sizeof(int16_t);
    float bytes_per_ms = stats.bytes_transferred / stats.avg_time;
    stats.bandwidth_gbps = bytes_per_ms / 1e6;

    // Cleanup events
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaEventDestroy(start[i]));
        CUDA_CHECK(cudaEventDestroy(stop[i]));
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
    // free(h_data);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return stats;
}




// ============================================================================
// Print Functions
// ============================================================================

void print_stats(const MemcpyStats& stats, int num_iterations) {
    std::cout << "\nHost-to-Device Statistics:\n";
    std::cout << "  Number of elements:     " << stats.num_elements << "\n";
    std::cout << "  Bytes transferred:      " << stats.bytes_transferred 
              << " (" << (stats.bytes_transferred / 1024.0) << " KB, "
              << (stats.bytes_transferred / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "  Number of iterations:   " << num_iterations << "\n";
    std::cout << "\n";
    std::cout << "  Min time:               " << stats.min_time << " ms  (" 
              << (stats.min_time * 1000.0) << " µs)\n";
    std::cout << "  Avg time:               " << stats.avg_time << " ms  (" 
              << (stats.avg_time * 1000.0) << " µs)\n";
    std::cout << "  Max time:               " << stats.max_time << " ms  (" 
              << (stats.max_time * 1000.0) << " µs)\n";
    std::cout << "  Range:                  " << (stats.max_time - stats.min_time) << " ms\n";
    std::cout << "  Total time:             " << stats.total_time << " ms\n";
    std::cout << "\n";
    std::cout << "  Bandwidth (avg):        " << stats.bandwidth_gbps << " GB/s\n";
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [num_elements] [num_iterations]\n";
    std::cerr << "  num_elements:     Number of int16_t elements to transfer (default: 1048576 = 1M)\n";
    std::cerr << "  num_iterations:   Number of iterations per test (default: 1000)\n";
    std::cerr << "\nExamples:\n";
    std::cerr << "  " << program_name << " 1000000 1000\n";
    std::cerr << "  " << program_name << " 8192 10000\n";
}

int main(int argc, char** argv) {
    size_t num_elements = 1048576;  // 1M elements = 2MB
    int num_iterations = 1000;
    
    // Parse command line arguments
    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        num_elements = std::stoull(argv[1]);
        if (num_elements == 0) {
            std::cerr << "Error: num_elements must be greater than 0\n";
            return 1;
        }
    }
    
    if (argc > 2) {
        num_iterations = std::atoi(argv[2]);
        if (num_iterations <= 0) {
            std::cerr << "Error: num_iterations must be greater than 0\n";
            return 1;
        }
    }
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Simple CUDA H2D Memory Copy Benchmark                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Get GPU information
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Buffer size: " << (num_elements * sizeof(int16_t) / 1024.0) << " KB ("
              << (num_elements * sizeof(int16_t) / (1024.0 * 1024.0)) << " MB)\n";
    std::cout << "Iterations: " << num_iterations << "\n";
    std::cout << "────────────────────────────────────────────────────────────────\n";
    
    // Run benchmark
    MemcpyStats stats = simple_h2d_benchmark(num_elements, num_iterations);
    print_stats(stats, num_iterations);
    
    std::cout << "\n────────────────────────────────────────────────────────────────\n";
    std::cout << "Benchmark complete.\n\n";
    
    return 0;
}
