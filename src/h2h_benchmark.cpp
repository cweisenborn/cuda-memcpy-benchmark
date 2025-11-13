/**
 * Host-to-Host Memory Copy Benchmark
 * 
 * Measures performance of memcpy operations between host memory buffers.
 * Collects timing data for multiple iterations and outputs statistics in JSON format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>

void write_json_output(const std::string& output_path, 
                       int num_elements, 
                       int num_iterations,
                       const std::vector<double>& timings_ms,
                       double min_time,
                       double max_time,
                       double avg_time) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path.c_str());
        exit(EXIT_FAILURE);
    }
    
    out << "{\n";
    out << "  \"benchmark_type\": \"H2H\",\n";
    out << "  \"num_elements\": " << num_elements << ",\n";
    out << "  \"num_iterations\": " << num_iterations << ",\n";
    out << "  \"element_size_bytes\": " << sizeof(float) << ",\n";
    out << "  \"total_bytes\": " << (num_elements * sizeof(float)) << ",\n";
    out << "  \"min_time_ms\": " << min_time << ",\n";
    out << "  \"max_time_ms\": " << max_time << ",\n";
    out << "  \"avg_time_ms\": " << avg_time << ",\n";
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
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <num_elements> <num_iterations> <output_json_path>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    int num_elements = atoi(argv[1]);
    int num_iterations = atoi(argv[2]);
    std::string output_path = argv[3];
    
    if (num_elements <= 0 || num_iterations <= 0) {
        fprintf(stderr, "Error: num_elements and num_iterations must be positive\n");
        return EXIT_FAILURE;
    }
    
    printf("=== Host-to-Host Memory Copy Benchmark ===\n");
    printf("Number of elements: %d\n", num_elements);
    printf("Number of iterations: %d\n", num_iterations);
    printf("Element size: %zu bytes\n", sizeof(float));
    printf("Total transfer size: %zu bytes (%.2f KB)\n", 
           num_elements * sizeof(float), 
           (num_elements * sizeof(float)) / 1024.0);
    printf("==========================================\n\n");
    
    // Allocate host memory buffers
    size_t size = num_elements * sizeof(float);
    float* src_data = (float*)malloc(size);
    float* dst_data = (float*)malloc(size);
    
    if (!src_data || !dst_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // Initialize source data
    for (int i = 0; i < num_elements; ++i) {
        src_data[i] = static_cast<float>(i);
    }
    
    // Warm-up
    printf("Performing warm-up iterations...\n");
    for (int i = 0; i < 100; ++i) {
        memcpy(dst_data, src_data, size);
    }
    
    // Benchmark iterations
    printf("Running %d benchmark iterations...\n", num_iterations);
    std::vector<double> timings_ms;
    timings_ms.reserve(num_iterations);
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        memcpy(dst_data, src_data, size);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timings_ms.push_back(elapsed.count());
        
        if ((i + 1) % 10000 == 0) {
            printf("  Completed %d/%d iterations\n", i + 1, num_iterations);
        }
    }
    
    // Calculate statistics
    double min_time = *std::min_element(timings_ms.begin(), timings_ms.end());
    double max_time = *std::max_element(timings_ms.begin(), timings_ms.end());
    double avg_time = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / timings_ms.size();
    
    printf("\n=== Results ===\n");
    printf("Min time: %.6f ms (%.2f µs)\n", min_time, min_time * 1000);
    printf("Max time: %.6f ms (%.2f µs)\n", max_time, max_time * 1000);
    printf("Avg time: %.6f ms (%.2f µs)\n", avg_time, avg_time * 1000);
    printf("===============\n\n");
    
    // Write JSON output
    write_json_output(output_path, num_elements, num_iterations, timings_ms, 
                      min_time, max_time, avg_time);
    
    // Cleanup
    free(src_data);
    free(dst_data);
    
    return EXIT_SUCCESS;
}
