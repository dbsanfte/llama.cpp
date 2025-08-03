#include <iostream>
#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>

// Include common.h to get our GPU-NUMA functions
#include "../common/common.h"

int main() {
    printf("=== CPU-GPU NUMA Affinity Test ===\n\n");
    
    // Test GPU-NUMA detection
    printf("1. Detecting GPU-NUMA topology...\n");
    std::vector<gpu_numa_info> gpu_infos = detect_gpu_numa_affinity();
    
    printf("Found %zu GPU devices:\n", gpu_infos.size());
    for (size_t i = 0; i < gpu_infos.size(); i++) {
        const auto& gpu = gpu_infos[i];
        printf("  GPU %d: %s\n", gpu.gpu_id, gpu.device_name.c_str());
        printf("    Backend: %s (%s)\n", gpu.backend_name.c_str(), gpu.backend_available ? "available" : "not available");
        printf("    NUMA Node: %d (%s)\n", gpu.numa_node, gpu.affinity_detected ? "detected" : "unknown");
        printf("    Virtual GPU: %s\n", gpu.is_virtual_gpu ? "yes" : "no");
        printf("    Local CPU cores: %zu\n", gpu.local_cpu_cores.size());
        if (!gpu.local_cpu_cores.empty()) {
            printf("    CPU cores on NUMA node %d: ", gpu.numa_node);
            for (size_t j = 0; j < gpu.local_cpu_cores.size(); j++) {
                printf("%d", gpu.local_cpu_cores[j]);
                if (j < gpu.local_cpu_cores.size() - 1) printf(", ");
            }
            printf("\n");
        }
        if (gpu.total_memory > 0) {
            printf("    Memory: %.1f GB total, %.1f GB free\n", 
                   gpu.total_memory / (1024.0 * 1024.0 * 1024.0),
                   gpu.free_memory / (1024.0 * 1024.0 * 1024.0));
        }
        printf("    CPU affinity verified: %s\n", gpu.cpu_affinity_verified ? "yes" : "no");
        printf("    Memory locality verified: %s\n", gpu.memory_locality_verified ? "yes" : "no");
        printf("\n");
    }
    
    // Test cross-socket traffic monitoring
    printf("2. Monitoring cross-socket GPU traffic...\n");
    bool locality_ok = monitor_cross_socket_gpu_traffic(gpu_infos);
    printf("GPU-CPU locality check: %s\n\n", locality_ok ? "GOOD" : "WARNING");
    
    // Test GPU-NUMA memory allocation
    printf("3. Testing NUMA-local memory allocation...\n");
    if (!gpu_infos.empty() && !gpu_infos[0].is_virtual_gpu) {
        void* buffer = allocate_gpu_numa_local_memory(1024 * 1024, gpu_infos[0].numa_node);  // 1MB
        printf("Allocated 1MB buffer on NUMA node %d: %p\n", gpu_infos[0].numa_node, buffer);
        if (buffer) {
            free_gpu_numa_local_memory(buffer, 1024 * 1024);
            printf("Buffer freed successfully\n");
        }
    } else {
        printf("No real GPUs detected - testing regular allocation\n");
        void* buffer = allocate_gpu_numa_local_memory(1024 * 1024, -1);  // 1MB
        printf("Allocated 1MB buffer: %p\n", buffer);
        if (buffer) {
            free_gpu_numa_local_memory(buffer, 1024 * 1024);
            printf("Buffer freed successfully\n");
        }
    }
    
    // Test thread binding for GPU 0 if available
    printf("\n4. Testing CPU thread binding to GPU NUMA node...\n");
    if (!gpu_infos.empty()) {
        printf("Attempting to bind thread to GPU 0's NUMA node %d...\n", gpu_infos[0].numa_node);
        bind_thread_to_gpu_numa_node(0, gpu_infos);
        printf("Thread binding attempted\n");
    } else {
        printf("No GPUs available for thread binding test\n");
    }
    
    // Test the enforce function for each GPU
    printf("\n5. Testing comprehensive GPU-CPU NUMA affinity enforcement...\n");
    for (size_t i = 0; i < gpu_infos.size(); i++) {
        printf("Enforcing affinity for GPU %d...\n", gpu_infos[i].gpu_id);
        bool result = enforce_gpu_cpu_numa_affinity(gpu_infos[i].gpu_id, gpu_infos[i]);
        printf("GPU %d enforcement: %s\n", gpu_infos[i].gpu_id, result ? "success" : "failed");
    }
    printf("Enforcement complete\n");
    
    // Display final status
    printf("\n6. Final GPU-NUMA status:\n");
    for (size_t i = 0; i < gpu_infos.size(); i++) {
        const auto& gpu = gpu_infos[i];
        printf("  GPU %d: NUMA node %d, CPU affinity %s, memory locality %s\n",
               gpu.gpu_id, gpu.numa_node,
               gpu.cpu_affinity_verified ? "verified" : "not verified",
               gpu.memory_locality_verified ? "verified" : "not verified");
    }
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
