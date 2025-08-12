#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-numa-coordinator.h"
#include <numa.h>

// Test configuration
struct test_config {
    bool verbose;
    bool test_real_numa;
    bool test_virtual_numa;
};

static struct test_config g_config = {
    false,  // verbose
    true,   // test_real_numa
    true    // test_virtual_numa
};

// Test result tracking
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void test_assert(bool condition, const char* test_name, const char* message) {
    tests_run++;
    if (condition) {
        tests_passed++;
        if (g_config.verbose) {
            printf("  ✅ PASS: %s - %s\n", test_name, message);
        }
    } else {
        tests_failed++;
        printf("  ❌ FAIL: %s - %s\n", test_name, message);
    }
}

// Capture GGML log output to check for specific messages
static char last_log_message[1024] = {0};
static void log_capture_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    strncpy(last_log_message, text, sizeof(last_log_message) - 1);
    last_log_message[sizeof(last_log_message) - 1] = '\0';
    
    if (g_config.verbose) {
        printf("    LOG: %s", text);
    }
}

// Test 1: Basic parameter initialization
static void test_max_numa_nodes_initialization() {
    printf("\n📋 Test 1: max_numa_nodes Parameter Initialization\n");
    printf("================================================\n");
    
    struct ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, 8);
    
    test_assert(tpp.max_numa_nodes == 0, "default_value", 
                "max_numa_nodes should default to 0 (auto-detect)");
    
    // Test setting different values
    tpp.max_numa_nodes = 1;
    test_assert(tpp.max_numa_nodes == 1, "manual_set_1", 
                "max_numa_nodes should be settable to 1");
                
    tpp.max_numa_nodes = 2;
    test_assert(tpp.max_numa_nodes == 2, "manual_set_2", 
                "max_numa_nodes should be settable to 2");
                
    tpp.max_numa_nodes = 4;
    test_assert(tpp.max_numa_nodes == 4, "manual_set_4", 
                "max_numa_nodes should be settable to 4");
}

// Test 2: Parameter comparison function
static void test_max_numa_nodes_comparison() {
    printf("\n🔍 Test 2: max_numa_nodes Parameter Comparison\n");
    printf("==============================================\n");
    
    struct ggml_threadpool_params tpp1, tpp2;
    ggml_threadpool_params_init(&tpp1, 8);
    ggml_threadpool_params_init(&tpp2, 8);
    
    // Same parameters should match
    test_assert(ggml_threadpool_params_match(&tpp1, &tpp2), "identical_params",
                "Identical parameters should match");
    
    // Different max_numa_nodes should not match
    tpp1.max_numa_nodes = 1;
    tpp2.max_numa_nodes = 2;
    test_assert(!ggml_threadpool_params_match(&tpp1, &tpp2), "different_max_numa",
                "Different max_numa_nodes values should not match");
                
    // Same max_numa_nodes should match again
    tpp2.max_numa_nodes = 1;
    test_assert(ggml_threadpool_params_match(&tpp1, &tpp2), "same_max_numa",
                "Same max_numa_nodes values should match");
}

// Test 3: NUMA node limiting with real NUMA detection
static void test_numa_node_limiting_real() {
    printf("\n🖥️  Test 3: NUMA Node Limiting (Real NUMA Hardware)\n");
    printf("===================================================\n");
    
#ifdef GGML_NUMA_MIRROR
    // Check if NUMA is available
    if (numa_available() == -1) {
        printf("  ⚠️  NUMA not available - skipping real NUMA tests\n");
        return;
    }
    
    int real_numa_nodes = numa_max_node() + 1;
    printf("  🔧 Detected %d real NUMA nodes\n", real_numa_nodes);
    
    if (real_numa_nodes < 2) {
        printf("  ⚠️  Less than 2 NUMA nodes detected - skipping limiting tests\n");
        return;
    }
    
    // Set up log capture to verify limiting messages
    ggml_log_set(log_capture_callback, NULL);
    
    // Test limiting to 1 node when 2+ are available
    printf("  🧪 Testing limitation to 1 node...\n");
    struct ggml_threadpool_params tpp1;
    ggml_threadpool_params_init(&tpp1, -1); // Auto-detect threads
    tpp1.max_numa_nodes = 1;
    tpp1.numa_aware = true;
    
    memset(last_log_message, 0, sizeof(last_log_message));
    struct ggml_numa_coordinator_manager* mgr1 = ggml_numa_coordinator_manager_new_with_params(&tpp1);
    
    test_assert(mgr1 != NULL, "limit_to_1_creation", 
                "Coordinator should be created with max_numa_nodes=1");
    
    bool found_limiting_msg = strstr(last_log_message, "Limiting NUMA nodes") != NULL ||
                             strstr(last_log_message, "max_numa_nodes constraint") != NULL;
    test_assert(found_limiting_msg, "limit_to_1_logging",
                "Should log NUMA node limiting message");
    
    if (mgr1) ggml_numa_coordinator_manager_free(mgr1);
    
    // Test no limiting when max_numa_nodes >= available nodes
    printf("  🧪 Testing no limitation when max >= available...\n");
    struct ggml_threadpool_params tpp2;
    ggml_threadpool_params_init(&tpp2, -1);
    tpp2.max_numa_nodes = real_numa_nodes + 1; // More than available
    tpp2.numa_aware = true;
    
    memset(last_log_message, 0, sizeof(last_log_message));
    struct ggml_numa_coordinator_manager* mgr2 = ggml_numa_coordinator_manager_new_with_params(&tpp2);
    
    test_assert(mgr2 != NULL, "no_limit_creation",
                "Coordinator should be created when max_numa_nodes >= available");
                
    bool no_limiting_msg = strstr(last_log_message, "Limiting NUMA nodes") == NULL;
    test_assert(no_limiting_msg, "no_limit_logging",
                "Should NOT log limiting message when no limiting occurs");
    
    if (mgr2) ggml_numa_coordinator_manager_free(mgr2);
    
    // Restore default logging
    ggml_log_set(NULL, NULL);
#else
    printf("  ⚠️  GGML_NUMA_MIRROR not enabled - skipping real NUMA tests\n");
#endif
}

// Test 4: NUMA node limiting with force_multi_socket
static void test_numa_node_limiting_virtual() {
    printf("\n🔄 Test 4: NUMA Node Limiting (Virtual/force_multi_socket)\n");
    printf("=========================================================\n");
    
    // Set up log capture
    ggml_log_set(log_capture_callback, NULL);
    
    // Test limiting in force_multi_socket mode (simulates 2 nodes by default)
    printf("  🧪 Testing limitation from 2 to 1 in force_multi_socket mode...\n");
    struct ggml_threadpool_params tpp1;
    ggml_threadpool_params_init(&tpp1, 8);
    tpp1.max_numa_nodes = 1;
    tpp1.force_multi_socket = true;
    tpp1.numa_aware = true;
    
    memset(last_log_message, 0, sizeof(last_log_message));
    struct ggml_numa_coordinator_manager* mgr1 = ggml_numa_coordinator_manager_new_with_params(&tpp1);
    
    test_assert(mgr1 != NULL, "virtual_limit_creation",
                "Coordinator should be created in force_multi_socket mode with limiting");
    
    bool found_limiting_msg = strstr(last_log_message, "Limiting NUMA nodes from 2 to 1") != NULL;
    test_assert(found_limiting_msg, "virtual_limit_logging",
                "Should log limiting from 2 to 1 nodes in force_multi_socket mode");
    
    if (mgr1) ggml_numa_coordinator_manager_free(mgr1);
    
    // Test no limiting when max_numa_nodes >= 2 in force_multi_socket
    printf("  🧪 Testing no limitation when max_numa_nodes >= 2...\n");
    struct ggml_threadpool_params tpp2;
    ggml_threadpool_params_init(&tpp2, 8);
    tpp2.max_numa_nodes = 4;
    tpp2.force_multi_socket = true;
    tpp2.numa_aware = true;
    
    memset(last_log_message, 0, sizeof(last_log_message));
    struct ggml_numa_coordinator_manager* mgr2 = ggml_numa_coordinator_manager_new_with_params(&tpp2);
    
    test_assert(mgr2 != NULL, "virtual_no_limit_creation",
                "Coordinator should be created without limiting when max >= force_multi_socket default");
    
    bool no_limiting_msg = strstr(last_log_message, "Limiting NUMA nodes") == NULL;
    test_assert(no_limiting_msg, "virtual_no_limit_logging",
                "Should NOT log limiting when max_numa_nodes >= available in force_multi_socket");
    
    if (mgr2) ggml_numa_coordinator_manager_free(mgr2);
    
    // Restore default logging
    ggml_log_set(NULL, NULL);
}

// Test 5: Edge cases and error conditions
static void test_edge_cases() {
    printf("\n⚠️  Test 5: Edge Cases and Error Conditions\n");
    printf("==========================================\n");
    
    // Test max_numa_nodes = 0 (auto-detect)
    struct ggml_threadpool_params tpp1;
    ggml_threadpool_params_init(&tpp1, 4);
    tpp1.max_numa_nodes = 0; // Auto-detect
    tpp1.force_multi_socket = true;
    
    struct ggml_numa_coordinator_manager* mgr1 = ggml_numa_coordinator_manager_new_with_params(&tpp1);
    test_assert(mgr1 != NULL, "auto_detect_creation",
                "Coordinator should be created with max_numa_nodes=0 (auto-detect)");
    if (mgr1) ggml_numa_coordinator_manager_free(mgr1);
    
    // Test very large max_numa_nodes value
    struct ggml_threadpool_params tpp2;
    ggml_threadpool_params_init(&tpp2, 4);
    tpp2.max_numa_nodes = 1000; // Very large value
    tpp2.force_multi_socket = true;
    
    struct ggml_numa_coordinator_manager* mgr2 = ggml_numa_coordinator_manager_new_with_params(&tpp2);
    test_assert(mgr2 != NULL, "large_max_numa_creation",
                "Coordinator should handle very large max_numa_nodes values gracefully");
    if (mgr2) ggml_numa_coordinator_manager_free(mgr2);
}

static void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --verbose, -v     Enable verbose output\n");
    printf("  --no-real-numa    Skip real NUMA hardware tests\n");
    printf("  --no-virtual      Skip virtual NUMA (force_multi_socket) tests\n");
    printf("  --help, -h        Show this help message\n");
    printf("\n");
    printf("This test validates the max_numa_nodes parameter implementation:\n");
    printf("1. Parameter initialization and default values\n");
    printf("2. Parameter comparison function\n");
    printf("3. NUMA node limiting with real NUMA hardware\n");
    printf("4. NUMA node limiting with virtual NUMA (force_multi_socket)\n");
    printf("5. Edge cases and error conditions\n");
}

int main(int argc, char** argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_config.verbose = true;
        } else if (strcmp(argv[i], "--no-real-numa") == 0) {
            g_config.test_real_numa = false;
        } else if (strcmp(argv[i], "--no-virtual") == 0) {
            g_config.test_virtual_numa = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    printf("🧪 max_numa_nodes Parameter Test Suite\n");
    printf("======================================\n");
    printf("Testing the max_numa_nodes parameter implementation in ggml_threadpool_params.\n");
    printf("This test validates NUMA node count limiting functionality.\n");
    
#ifdef GGML_NUMA_MIRROR
    printf("✅ NUMA support enabled (GGML_NUMA_MIRROR)\n");
#else
    printf("⚠️  NUMA support disabled - limited testing available\n");
#endif
    
    // Run test suite
    test_max_numa_nodes_initialization();
    test_max_numa_nodes_comparison();
    
    if (g_config.test_real_numa) {
        test_numa_node_limiting_real();
    } else {
        printf("\n🖥️  Test 3: NUMA Node Limiting (Real NUMA Hardware) - SKIPPED\n");
    }
    
    if (g_config.test_virtual_numa) {
        test_numa_node_limiting_virtual();
    } else {
        printf("\n🔄 Test 4: NUMA Node Limiting (Virtual/force_multi_socket) - SKIPPED\n");
    }
    
    test_edge_cases();
    
    // Print summary
    printf("\n📊 Test Results Summary\n");
    printf("======================\n");
    printf("Total tests: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    if (tests_failed == 0) {
        printf("🎉 All tests PASSED!\n");
        printf("\nmax_numa_nodes parameter implementation is working correctly.\n");
        return 0;
    } else {
        printf("❌ %d tests FAILED!\n", tests_failed);
        printf("\nPlease check the implementation and fix the failing tests.\n");
        return 1;
    }
}
