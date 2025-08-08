/*
 * NUMA Coordinator CPU Mask Test
 * 
 * This test demonstrates the enhanced CPU mask logging and optimal assignment
 * features of the NUMA coordinator.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cpu/ggml-numa-coordinator.h"  // Internal header with params support
#include "common.h"

void test_default_coordinator_logging() {
    printf("🔍 Testing Default Coordinator CPU Assignment\n");
    printf("=============================================\n\n");
    
    printf("📊 Creating coordinator with all CPUs (default behavior)...\n");
    
    // Create context
    struct ggml_init_params params = {};
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return;
    }
    
    // Create a computation that will use the coordinator
    const int64_t size = 1000000;
    struct ggml_tensor *a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor *b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
    struct ggml_tensor *result = ggml_add(ctx, a, b);
    
    // Initialize data
    float *a_data = (float *)ggml_get_data(a);
    float *b_data = (float *)ggml_get_data(b);
    
    for (int64_t i = 0; i < size; i++) {
        a_data[i] = 1.0f;
        b_data[i] = 2.0f;
    }
    
    // Build computation graph
    struct ggml_cgraph *cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, result);
    
    // Create coordinator with default parameters
    printf("\n🚀 Creating coordinator (this should show enhanced logging):\n");
    printf("------------------------------------------------------------\n");
    
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(22);
    tpp.force_multi_socket = true;  // Force virtual NUMA nodes
    
    struct ggml_numa_coordinator_manager *mgr = 
        ggml_numa_coordinator_manager_new_with_params(&tpp);
    
    if (mgr) {
        printf("✅ Coordinator created successfully\n");
        
        // Submit work to see the coordinator in action
        printf("\n📈 Submitting work to coordinator...\n");
        int result_code = ggml_numa_coordinator_manager_compute_graph(mgr, cgraph);
        
        if (result_code == 0) {
            printf("✅ Work completed successfully\n");
            
            // Verify result
            float *result_data = (float *)ggml_get_data(result);
            printf("   Result verification: %.2f (expected 3.00) %s\n", 
                   result_data[0], 
                   (fabs(result_data[0] - 3.0f) < 0.01f) ? "✅" : "❌");
        } else {
            printf("❌ Work failed with code %d\n", result_code);
        }
        
        ggml_numa_coordinator_manager_free(mgr);
    } else {
        printf("❌ Failed to create coordinator\n");
    }
    
    ggml_free(ctx);
    printf("\n");
}

void test_custom_cpu_mask() {
    printf("🎯 Testing Custom CPU Mask Assignment\n");
    printf("=====================================\n\n");
    
    printf("📊 Creating coordinator with custom CPU mask to avoid hyperthreading...\n");
    
    // Create context
    struct ggml_init_params params = {};
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = NULL;
    params.no_alloc = false;
    
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        printf("❌ Failed to create context\n");
        return;
    }
    
    // Create threadpool parameters with custom CPU mask
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(12);
    tpp.force_multi_socket = true;
    
    // Manually set CPU mask to avoid hyperthreading conflicts
    // Use primary threads only: CPUs 0,2,4,6,8,10,12,14,16,18,20
    memset(tpp.cpumask, false, sizeof(tpp.cpumask));
    for (int i = 0; i <= 20; i += 2) {  // Even CPUs (primary threads)
        tpp.cpumask[i] = true;
    }
    
    printf("🔧 Custom CPU mask: Primary threads only (0,2,4,6,8,10,12,14,16,18,20)\n");
    printf("   This should eliminate hyperthreading conflicts\n\n");
    
    printf("🚀 Creating coordinator with custom mask:\n");
    printf("-----------------------------------------\n");
    
    struct ggml_numa_coordinator_manager *mgr = 
        ggml_numa_coordinator_manager_new_with_params(&tpp);
    
    if (mgr) {
        printf("✅ Custom coordinator created successfully\n");
        ggml_numa_coordinator_manager_free(mgr);
    } else {
        printf("❌ Failed to create custom coordinator\n");
    }
    
    ggml_free(ctx);
    printf("\n");
}

int main() {
    printf("🧪 NUMA Coordinator CPU Mask Enhancement Test\n");
    printf("==============================================\n");
    printf("Testing enhanced CPU assignment logging and optimization\n\n");
    
    // Test 1: Default behavior with enhanced logging
    test_default_coordinator_logging();
    
    printf("============================================================\n\n");
    
    // Test 2: Custom CPU mask to avoid hyperthreading
    test_custom_cpu_mask();
    
    printf("💡 Key Features Demonstrated:\n");
    printf("=============================\n");
    printf("• Enhanced CPU assignment logging with physical core mapping\n");
    printf("• Hyperthreading conflict detection and warnings\n");
    printf("• Custom CPU mask support for optimal thread placement\n");
    printf("• Automatic optimization when no custom mask is provided\n");
    
    printf("\n📋 Check the log output above for detailed CPU assignments!\n");
    return 0;
}
