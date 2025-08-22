/**
 * NUMA Memory Locality Verification Tool
 * 
 * This tool tests various memory allocation strategies and verifies 
 * where memory actually ends up in terms of NUMA node placement.
 * Critical for debugging the 3x performance difference between nodes.
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>

// NUMA headers
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

class NumaMemoryLocalityTester {
private:
    int num_numa_nodes;
    size_t page_size;
    
    struct AllocationInfo {
        void* ptr;
        size_t size;
        std::string strategy;
        int requested_node;
        std::vector<int> actual_nodes;
        double locality_ratio;
    };
    
public:
    NumaMemoryLocalityTester() {
#ifdef __linux__
        if (numa_available() == -1) {
            std::cerr << "❌ NUMA not available on this system" << std::endl;
            exit(1);
        }
        
        num_numa_nodes = numa_num_configured_nodes();
        page_size = getpagesize();
        
        std::cout << "🔍 NUMA Memory Locality Verification Tool" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "📋 System Info:" << std::endl;
        std::cout << "   NUMA nodes: " << num_numa_nodes << std::endl;
        std::cout << "   Page size: " << page_size << " bytes" << std::endl;
        std::cout << "   Available nodes: ";
        
        struct bitmask* available = numa_get_run_node_mask();
        for (int i = 0; i < num_numa_nodes; i++) {
            if (numa_bitmask_isbitset(available, i)) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl << std::endl;
        numa_free_nodemask(available);
#else
        std::cerr << "❌ This tool requires Linux with NUMA support" << std::endl;
        exit(1);
#endif
    }
    
    /**
     * Check which NUMA node(s) a memory region actually resides on
     */
    std::vector<int> check_memory_location(void* ptr, size_t size) {
#ifdef __linux__
        std::vector<int> node_counts(num_numa_nodes, 0);
        size_t num_pages = (size + page_size - 1) / page_size;
        
        std::vector<void*> pages;
        std::vector<int> status;
        std::vector<int> nodes;
        
        // Get page addresses
        for (size_t i = 0; i < num_pages; i++) {
            pages.push_back((char*)ptr + i * page_size);
        }
        
        status.resize(num_pages);
        nodes.resize(num_pages);
        
        // Query page locations
        if (move_pages(0, num_pages, pages.data(), nullptr, status.data(), 0) == 0) {
            for (size_t i = 0; i < num_pages; i++) {
                if (status[i] >= 0 && status[i] < num_numa_nodes) {
                    node_counts[status[i]]++;
                }
            }
        }
        
        // Return nodes that have pages
        std::vector<int> result;
        for (int i = 0; i < num_numa_nodes; i++) {
            if (node_counts[i] > 0) {
                result.push_back(i);
            }
        }
        
        return result;
#else
        return {};
#endif
    }
    
    /**
     * Calculate locality ratio - percentage of memory on requested node
     */
    double calculate_locality_ratio(void* ptr, size_t size, int requested_node) {
#ifdef __linux__
        size_t num_pages = (size + page_size - 1) / page_size;
        size_t pages_on_requested_node = 0;
        
        std::vector<void*> pages;
        std::vector<int> status;
        
        for (size_t i = 0; i < num_pages; i++) {
            pages.push_back((char*)ptr + i * page_size);
        }
        
        status.resize(num_pages);
        
        if (move_pages(0, num_pages, pages.data(), nullptr, status.data(), 0) == 0) {
            for (size_t i = 0; i < num_pages; i++) {
                if (status[i] == requested_node) {
                    pages_on_requested_node++;
                }
            }
        }
        
        return (double)pages_on_requested_node / num_pages;
#else
        return 0.0;
#endif
    }
    
    /**
     * Test allocation strategy 1: Regular malloc
     */
    AllocationInfo test_malloc(size_t size, int requested_node) {
        AllocationInfo info;
        info.strategy = "malloc";
        info.requested_node = requested_node;
        info.size = size;
        
        // Set NUMA policy before allocation
#ifdef __linux__
        numa_set_preferred(requested_node);
#endif
        
        info.ptr = malloc(size);
        if (!info.ptr) {
            std::cerr << "❌ malloc failed" << std::endl;
            info.locality_ratio = 0.0;
            return info;
        }
        
        // Touch all pages to ensure allocation
        memset(info.ptr, 0x42, size);
        
        info.actual_nodes = check_memory_location(info.ptr, size);
        info.locality_ratio = calculate_locality_ratio(info.ptr, size, requested_node);
        
        return info;
    }
    
    /**
     * Test allocation strategy 2: numa_alloc_onnode
     */
    AllocationInfo test_numa_alloc_onnode(size_t size, int requested_node) {
        AllocationInfo info;
        info.strategy = "numa_alloc_onnode";
        info.requested_node = requested_node;
        info.size = size;
        
#ifdef __linux__
        info.ptr = numa_alloc_onnode(size, requested_node);
        if (!info.ptr) {
            std::cerr << "❌ numa_alloc_onnode failed" << std::endl;
            info.locality_ratio = 0.0;
            return info;
        }
        
        // Touch all pages
        memset(info.ptr, 0x42, size);
        
        info.actual_nodes = check_memory_location(info.ptr, size);
        info.locality_ratio = calculate_locality_ratio(info.ptr, size, requested_node);
#else
        info.ptr = nullptr;
        info.locality_ratio = 0.0;
#endif
        
        return info;
    }
    
    /**
     * Test allocation strategy 3: mmap with mbind
     */
    AllocationInfo test_mmap_mbind(size_t size, int requested_node) {
        AllocationInfo info;
        info.strategy = "mmap+mbind";
        info.requested_node = requested_node;
        info.size = size;
        
        // Allocate with mmap
        info.ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (info.ptr == MAP_FAILED) {
            std::cerr << "❌ mmap failed" << std::endl;
            info.ptr = nullptr;
            info.locality_ratio = 0.0;
            return info;
        }
        
#ifdef __linux__
        // Bind to specific node
        struct bitmask* nodemask = numa_allocate_nodemask();
        numa_bitmask_setbit(nodemask, requested_node);
        
        if (mbind(info.ptr, size, MPOL_BIND, nodemask->maskp, nodemask->size + 1, 0) != 0) {
            std::cerr << "❌ mbind failed" << std::endl;
        }
        
        numa_free_nodemask(nodemask);
#endif
        
        // Touch all pages
        memset(info.ptr, 0x42, size);
        
        info.actual_nodes = check_memory_location(info.ptr, size);
        info.locality_ratio = calculate_locality_ratio(info.ptr, size, requested_node);
        
        return info;
    }
    
    /**
     * Test allocation strategy 4: numa_alloc_local (current node)
     */
    AllocationInfo test_numa_alloc_local(size_t size) {
        AllocationInfo info;
        info.strategy = "numa_alloc_local";
        info.requested_node = numa_node_of_cpu(sched_getcpu());
        info.size = size;
        
#ifdef __linux__
        info.ptr = numa_alloc_local(size);
        if (!info.ptr) {
            std::cerr << "❌ numa_alloc_local failed" << std::endl;
            info.locality_ratio = 0.0;
            return info;
        }
        
        // Touch all pages
        memset(info.ptr, 0x42, size);
        
        info.actual_nodes = check_memory_location(info.ptr, size);
        info.locality_ratio = calculate_locality_ratio(info.ptr, size, info.requested_node);
#else
        info.ptr = nullptr;
        info.locality_ratio = 0.0;
#endif
        
        return info;
    }
    
    /**
     * Free allocated memory
     */
    void free_allocation(const AllocationInfo& info) {
        if (!info.ptr) return;
        
        if (info.strategy == "malloc") {
            free(info.ptr);
        } else if (info.strategy == "numa_alloc_onnode" || info.strategy == "numa_alloc_local") {
#ifdef __linux__
            numa_free(info.ptr, info.size);
#endif
        } else if (info.strategy == "mmap+mbind") {
            munmap(info.ptr, info.size);
        }
    }
    
    /**
     * Print allocation results
     */
    void print_allocation_info(const AllocationInfo& info) {
        std::cout << "📋 Strategy: " << std::setw(20) << info.strategy 
                  << " | Requested Node: " << info.requested_node
                  << " | Size: " << std::setw(8) << (info.size / 1024 / 1024) << " MB" << std::endl;
        
        if (!info.ptr) {
            std::cout << "   ❌ Allocation failed" << std::endl;
            return;
        }
        
        std::cout << "   📍 Actual nodes: [";
        for (size_t i = 0; i < info.actual_nodes.size(); i++) {
            std::cout << info.actual_nodes[i];
            if (i < info.actual_nodes.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "   📊 Locality ratio: " << std::fixed << std::setprecision(1) 
                  << (info.locality_ratio * 100) << "%" << std::endl;
        
        if (info.locality_ratio >= 0.95) {
            std::cout << "   ✅ EXCELLENT locality" << std::endl;
        } else if (info.locality_ratio >= 0.7) {
            std::cout << "   ⚠️  GOOD locality" << std::endl;
        } else if (info.locality_ratio >= 0.3) {
            std::cout << "   🔸 POOR locality" << std::endl;
        } else {
            std::cout << "   ❌ TERRIBLE locality" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    /**
     * Run comprehensive memory locality tests
     */
    void run_comprehensive_test() {
        std::vector<size_t> test_sizes = {
            1 * 1024 * 1024,    // 1 MB
            16 * 1024 * 1024,   // 16 MB  
            64 * 1024 * 1024,   // 64 MB
            256 * 1024 * 1024   // 256 MB
        };
        
        std::cout << "🧪 Testing allocation strategies across NUMA nodes..." << std::endl;
        std::cout << "=====================================================" << std::endl;
        
        for (size_t size : test_sizes) {
            std::cout << "\n🎯 Testing " << (size / 1024 / 1024) << " MB allocations:" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            for (int node = 0; node < num_numa_nodes; node++) {
                std::cout << "\n🔸 Target NUMA Node " << node << ":" << std::endl;
                
                // Test different allocation strategies
                std::vector<AllocationInfo> allocations = {
                    test_malloc(size, node),
                    test_numa_alloc_onnode(size, node),
                    test_mmap_mbind(size, node)
                };
                
                for (const auto& info : allocations) {
                    print_allocation_info(info);
                    free_allocation(info);
                }
            }
            
            // Test local allocation
            std::cout << "\n🔸 Local allocation (current CPU's node):" << std::endl;
            AllocationInfo local_info = test_numa_alloc_local(size);
            print_allocation_info(local_info);
            free_allocation(local_info);
        }
    }
    
    /**
     * Test the current NUMA setup for our ADD kernel issue
     */
    void test_add_kernel_scenario() {
        std::cout << "\n🎯 ADD Kernel NUMA Scenario Test" << std::endl;
        std::cout << "=================================" << std::endl;
        
        // Simulate the allocation sizes from our ADD benchmark
        size_t huge_tensor_size = 256 * 1024 * 1024; // 256 MB like the HUGE test
        
        std::cout << "📋 Simulating ADD kernel memory allocation..." << std::endl;
        std::cout << "   Tensor size: " << (huge_tensor_size / 1024 / 1024) << " MB" << std::endl;
        std::cout << "   Strategy: numa_alloc_onnode (current implementation)" << std::endl;
        
        for (int node = 0; node < num_numa_nodes; node++) {
            std::cout << "\n🔸 Allocating on NUMA Node " << node << ":" << std::endl;
            
            // Allocate 3 tensors like ADD operation (src0, src1, dst)
            AllocationInfo src0 = test_numa_alloc_onnode(huge_tensor_size, node);
            AllocationInfo src1 = test_numa_alloc_onnode(huge_tensor_size, node);
            AllocationInfo dst = test_numa_alloc_onnode(huge_tensor_size, node);
            
            std::cout << "   src0 tensor:" << std::endl;
            print_allocation_info(src0);
            
            std::cout << "   src1 tensor:" << std::endl;
            print_allocation_info(src1);
            
            std::cout << "   dst tensor:" << std::endl;
            print_allocation_info(dst);
            
            // Calculate overall locality
            double avg_locality = (src0.locality_ratio + src1.locality_ratio + dst.locality_ratio) / 3.0;
            std::cout << "   📊 Average locality: " << std::fixed << std::setprecision(1) 
                      << (avg_locality * 100) << "%" << std::endl;
            
            if (avg_locality >= 0.95) {
                std::cout << "   ✅ This node should have EXCELLENT performance" << std::endl;
            } else {
                std::cout << "   ❌ This node will have POOR performance due to cross-node access" << std::endl;
            }
            
            free_allocation(src0);
            free_allocation(src1);
            free_allocation(dst);
        }
    }
    
    /**
     * Check current process NUMA affinity
     */
    void check_process_affinity() {
        std::cout << "\n🎯 Process NUMA Affinity Check" << std::endl;
        std::cout << "===============================" << std::endl;
        
#ifdef __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            std::cout << "📋 CPU affinity: ";
            bool first = true;
            for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
                if (CPU_ISSET(cpu, &cpuset)) {
                    if (!first) std::cout << ", ";
                    std::cout << cpu;
                    first = false;
                }
            }
            std::cout << std::endl;
        }
        
        int current_cpu = sched_getcpu();
        int current_node = numa_node_of_cpu(current_cpu);
        std::cout << "📋 Current CPU: " << current_cpu << std::endl;
        std::cout << "📋 Current NUMA node: " << current_node << std::endl;
        
        struct bitmask* allowed_nodes = numa_get_run_node_mask();
        std::cout << "📋 Allowed NUMA nodes: ";
        for (int i = 0; i < num_numa_nodes; i++) {
            if (numa_bitmask_isbitset(allowed_nodes, i)) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
        numa_free_nodemask(allowed_nodes);
#endif
    }
};

int main() {
    try {
        NumaMemoryLocalityTester tester;
        
        // Check current process setup
        tester.check_process_affinity();
        
        // Run comprehensive allocation tests
        tester.run_comprehensive_test();
        
        // Test specific ADD kernel scenario
        tester.test_add_kernel_scenario();
        
        std::cout << "\n🏁 Memory locality verification complete!" << std::endl;
        std::cout << "Check the locality ratios to identify allocation strategy issues." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
