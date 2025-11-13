/**
 * CUDA Host-to-Device Memory Copy Benchmark
 * 
 * Measures performance of cudaMemcpy operations from host to device memory.
 * Collects timing data for multiple iterations and outputs statistics in JSON format.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void write_json_output(const std::string& output_path, 
                       int num_elements, 
                       int num_iterations,
                       bool use_pinned_memory,
                       const std::vector<double>& timings_ms,
                       double min_time,
                       double max_time,
                       double avg_time,
                       double total_time_seconds) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path.c_str());
        exit(EXIT_FAILURE);
    }
    
    out << "{\n";
    out << "  \"benchmark_type\": \"H2D\",\n";
    out << "  \"memory_type\": \"" << (use_pinned_memory ? "pinned" : "pageable") << "\",\n";
    out << "  \"num_elements\": " << num_elements << ",\n";
    out << "  \"num_iterations\": " << num_iterations << ",\n";
    out << "  \"element_size_bytes\": " << sizeof(float) << ",\n";
    out << "  \"total_bytes\": " << (num_elements * sizeof(float)) << ",\n";
    out << "  \"min_time_ms\": " << min_time << ",\n";
    out << "  \"max_time_ms\": " << max_time << ",\n";
    out << "  \"avg_time_ms\": " << avg_time << ",\n";
    out << "  \"total_time_seconds\": " << total_time_seconds << ",\n";
    out << "  \"timings_ms\": [";
    
    for (size_t i = 0; i < timings_ms.size(); ++i) {
        if (i > 0) out << ", ";
        out << timings_ms[i];
    }
    
    out << "]\n";
    out << "}\n";
    out.close();
    
    printf("Results written to: %s\n", output_path.c_str());
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <num_elements> <num_iterations> <output_json_path> [--pinned]\n", argv[0]);
        fprintf(stderr, "  --pinned: Use CUDA pinned (page-locked) host memory instead of pageable memory\n");
        return EXIT_FAILURE;
    }
    
    int num_elements = atoi(argv[1]);
    int num_iterations = atoi(argv[2]);
    std::string output_path = argv[3];
    bool use_pinned_memory = false;
    
    // Check for --pinned flag
    if (argc == 5 && std::string(argv[4]) == "--pinned") {
        use_pinned_memory = true;
    }
    
    if (num_elements <= 0 || num_iterations <= 0) {
        fprintf(stderr, "Error: num_elements and num_iterations must be positive\n");
        return EXIT_FAILURE;
    }
    
    printf("=== CUDA Host-to-Device Memory Copy Benchmark ===\n");
    printf("Number of elements: %d\n", num_elements);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Host memory type: %s\n", use_pinned_memory ? "Pinned (cudaMallocHost)" : "Pageable (malloc)");
    printf("Element size: %zu bytes\n", sizeof(float));
    printf("Total transfer size: %zu bytes (%.2f KB)\n", 
           num_elements * sizeof(float), 
           (num_elements * sizeof(float)) / 1024.0);
    printf("=================================================\n\n");
    
    // Allocate host memory
    size_t size = num_elements * sizeof(float);
    float* h_data;
    
    if (use_pinned_memory) {
        // Allocate pinned (page-locked) host memory
        // CUDA_CHECK(cudaMallocHost(&h_data, size));
        CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));
        printf("Allocated pinned host memory: %zu bytes\n", size);
    } else {
        // Allocate pageable host memory
        h_data = (float*)malloc(size);
        if (!h_data) {
            fprintf(stderr, "Failed to allocate host memory\n");
            return EXIT_FAILURE;
        }
        printf("Allocated pageable host memory: %zu bytes\n", size);
    }
    
    // Initialize host data
    for (int i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<float>(i);
    }
    
    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Warm-up
    printf("Performing warm-up iterations...\n");
    for (int i = 0; i < 100; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Benchmark iterations
    printf("Running %d benchmark iterations...\n", num_iterations);
    std::vector<double> timings_ms;
    timings_ms.reserve(num_iterations);
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timings_ms.push_back(elapsed.count());
        
        if ((i + 1) % 10000 == 0) {
            printf("  Completed %d/%d iterations\n", i + 1, num_iterations);
        }
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = benchmark_end - benchmark_start;
    double total_time_seconds = total_duration.count();
    
    // Calculate statistics
    double min_time = *std::min_element(timings_ms.begin(), timings_ms.end());
    double max_time = *std::max_element(timings_ms.begin(), timings_ms.end());
    double avg_time = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / timings_ms.size();
    
    printf("\n=== Results ===\n");
    printf("Min time: %.6f ms (%.2f µs)\n", min_time, min_time * 1000);
    printf("Max time: %.6f ms (%.2f µs)\n", max_time, max_time * 1000);
    printf("Avg time: %.6f ms (%.2f µs)\n", avg_time, avg_time * 1000);
    printf("Total benchmark time: %.3f seconds\n", total_time_seconds);
    printf("===============\n\n");
    
    // Write JSON output
    write_json_output(output_path, num_elements, num_iterations, use_pinned_memory,
                      timings_ms, min_time, max_time, avg_time, total_time_seconds);
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    
    if (use_pinned_memory) {
        CUDA_CHECK(cudaFreeHost(h_data));
    } else {
        free(h_data);
    }
    
    return EXIT_SUCCESS;
}
