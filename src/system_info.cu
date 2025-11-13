/**
 * System Information Utility
 * 
 * Captures CPU, GPU, and memory information and outputs to JSON file.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <fstream>
#include <sstream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::string get_cpu_model() {
    // Try using lscpu command to get all model names
    FILE* pipe = popen("lscpu | grep 'Model name:' | cut -d ':' -f 2 | sed 's/^[ \\t]*//'", "r");
    if (pipe) {
        char buffer[256];
        std::string result = "";
        bool first = true;
        
        // Read all model names (may be multiple in heterogeneous systems)
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string model = buffer;
            // Remove trailing newline
            if (!model.empty() && model[model.length()-1] == '\n') {
                model.erase(model.length()-1);
            }
            if (!model.empty()) {
                if (!first) {
                    result += " + ";
                }
                result += model;
                first = false;
            }
        }
        pclose(pipe);
        if (!result.empty()) {
            return result;
        }
    }
    
    // Fallback to /proc/cpuinfo - collect unique model names
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::vector<std::string> models;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string model = line.substr(pos + 2);
                // Check if this model is already in the list
                bool found = false;
                for (const auto& m : models) {
                    if (m == model) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    models.push_back(model);
                }
            }
        }
    }
    
    // Combine all unique models
    if (!models.empty()) {
        std::string result = models[0];
        for (size_t i = 1; i < models.size(); i++) {
            result += " + " + models[i];
        }
        return result;
    }
    
    return "Unknown";
}

int get_cpu_cores() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int cores = 0;
    while (std::getline(cpuinfo, line)) {
        if (line.find("processor") != std::string::npos) {
            cores++;
        }
    }
    return cores;
}

void write_system_info_json(const std::string& output_path) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path.c_str());
        exit(EXIT_FAILURE);
    }
    
    // Get system info
    struct utsname sys_info;
    uname(&sys_info);
    
    struct sysinfo mem_info;
    sysinfo(&mem_info);
    
    // Get CPU info
    std::string cpu_model = get_cpu_model();
    int cpu_cores = get_cpu_cores();
    
    // Get CUDA device count
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    // Start JSON output
    out << "{\n";
    out << "  \"timestamp\": \"" << __DATE__ << " " << __TIME__ << "\",\n";
    
    // System info
    out << "  \"system\": {\n";
    out << "    \"os\": \"" << sys_info.sysname << "\",\n";
    out << "    \"os_release\": \"" << sys_info.release << "\",\n";
    out << "    \"os_version\": \"" << sys_info.version << "\",\n";
    out << "    \"machine\": \"" << sys_info.machine << "\",\n";
    out << "    \"hostname\": \"" << sys_info.nodename << "\"\n";
    out << "  },\n";
    
    // CPU info
    out << "  \"cpu\": {\n";
    out << "    \"model\": \"" << cpu_model << "\",\n";
    out << "    \"cores\": " << cpu_cores << "\n";
    out << "  },\n";
    
    // Memory info (convert to GB)
    double total_ram_gb = mem_info.totalram / (1024.0 * 1024.0 * 1024.0);
    double free_ram_gb = mem_info.freeram / (1024.0 * 1024.0 * 1024.0);
    
    out << "  \"memory\": {\n";
    out << "    \"total_ram_gb\": " << total_ram_gb << ",\n";
    out << "    \"free_ram_gb\": " << free_ram_gb << ",\n";
    out << "    \"total_ram_bytes\": " << mem_info.totalram << ",\n";
    out << "    \"free_ram_bytes\": " << mem_info.freeram << "\n";
    out << "  },\n";
    
    // CUDA info
    out << "  \"cuda\": {\n";
    out << "    \"device_count\": " << device_count << ",\n";
    
    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    out << "    \"runtime_version\": " << runtime_version << ",\n";
    
    int driver_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    out << "    \"driver_version\": " << driver_version << ",\n";
    
    out << "    \"devices\": [\n";
    
    // Get info for each GPU
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        out << "      {\n";
        out << "        \"id\": " << i << ",\n";
        out << "        \"name\": \"" << prop.name << "\",\n";
        out << "        \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
        out << "        \"total_memory_gb\": " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << ",\n";
        out << "        \"total_memory_bytes\": " << prop.totalGlobalMem << ",\n";
        out << "        \"multiprocessors\": " << prop.multiProcessorCount << ",\n";
        out << "        \"memory_bus_width\": " << prop.memoryBusWidth << ",\n";
        out << "        \"l2_cache_size_bytes\": " << prop.l2CacheSize << ",\n";
        out << "        \"max_threads_per_block\": " << prop.maxThreadsPerBlock << ",\n";
        out << "        \"warp_size\": " << prop.warpSize << ",\n";
        out << "        \"can_map_host_memory\": " << (prop.canMapHostMemory ? "true" : "false") << ",\n";
        out << "        \"concurrent_kernels\": " << (prop.concurrentKernels ? "true" : "false") << ",\n";
        out << "        \"ecc_enabled\": " << (prop.ECCEnabled ? "true" : "false") << ",\n";
        out << "        \"pci_bus_id\": " << prop.pciBusID << ",\n";
        out << "        \"pci_device_id\": " << prop.pciDeviceID << "\n";
        out << "      }";
        
        if (i < device_count - 1) {
            out << ",\n";
        } else {
            out << "\n";
        }
    }
    
    out << "    ]\n";
    out << "  }\n";
    out << "}\n";
    
    out.close();
    printf("System information written to: %s\n", output_path.c_str());
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <output_json_path>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    std::string output_path = argv[1];
    
    printf("=== Collecting System Information ===\n");
    write_system_info_json(output_path);
    printf("=====================================\n");
    
    return EXIT_SUCCESS;
}
