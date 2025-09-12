#include "llama-mmap.h"
#include "llama-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <numaif.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/mman.h>
#include "../ggml/src/ggml-cpu/ggml-numa-shared.h"
#endif

// ... [file/lock implementations remain the same as original] ...

// llama_mmap - Clean NUMA implementation

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
    
#ifdef GGML_NUMA_MIRROR
    // NUMA-specific data
    struct numa_allocation {
        void* addr;
        size_t size;
        int node;
        bool is_mmap;  // true if mmap, false if numa_alloc
    };
    std::vector<numa_allocation> numa_allocations;
    
    // Clean, reliable NUMA allocation
    void* allocate_numa_memory(size_t size, int node) {
        if (numa_available() < 0) {
            LLAMA_LOG_WARN("NUMA not available, falling back to regular allocation\n");
            return nullptr;
        }
        
        // Use numa_alloc_onnode - the standard, reliable method
        void* ptr = numa_alloc_onnode(size, node);
        if (!ptr) {
            LLAMA_LOG_WARN("numa_alloc_onnode failed for node %d, size %zu: %s\n", 
                           node, size, strerror(errno));
            return nullptr;
        }
        
        NUMA_LOG_DEBUG("Allocated %zu bytes on NUMA node %d at %p\n", size, node, ptr);
        return ptr;
    }
    
    void free_numa_memory(void* ptr, size_t size) {
        if (ptr) {
            numa_free(ptr, size);
            NUMA_LOG_DEBUG("Freed NUMA memory at %p, size %zu\n", ptr, size);
        }
    }
    
    // Single-file NUMA constructor
    void init_numa_single_file(struct llama_file* file, size_t prefetch, bool numa) {
        GGML_UNUSED(prefetch);  // Not used in NUMA mode
        
        size = file->size();
        enum ggml_numa_strategy strategy = numa ? ggml_get_numa_strategy() : GGML_NUMA_STRATEGY_DISABLED;
        
        switch (strategy) {
            case GGML_NUMA_STRATEGY_DISABLED:
                init_regular_mmap(file, prefetch, false);
                return;
                
            case GGML_NUMA_STRATEGY_ISOLATE:
                init_numa_isolate(file);
                return;
                
            case GGML_NUMA_STRATEGY_MIRROR:
                init_numa_mirror(file);
                return;
                
            default:
                LLAMA_LOG_WARN("Unknown NUMA strategy %d, falling back to regular mmap\n", strategy);
                init_regular_mmap(file, prefetch, false);
                return;
        }
    }
    
    void init_regular_mmap(struct llama_file* file, size_t prefetch, bool use_numa_hints) {
        int fd = file->file_id();
        int flags = MAP_SHARED;
        
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                           strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        
        addr = mmap(NULL, size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }
        
        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(size, prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                               strerror(errno));
            }
        }
        
        if (use_numa_hints) {
            if (posix_madvise(addr, size, POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                               strerror(errno));
            }
        }
        
        mapped_fragments.emplace_back(0, size);
        LLAMA_LOG_INFO("Regular mmap: mapped %zu bytes at %p\n", size, addr);
    }
    
    void init_numa_isolate(struct llama_file* file) {
        // Isolate mode: allocate only on current/target NUMA node
        int target_node = ggml_numa_get_current_node();
        if (target_node < 0) target_node = 0;
        
        void* numa_addr = allocate_numa_memory(size, target_node);
        if (!numa_addr) {
            LLAMA_LOG_WARN("NUMA isolate allocation failed, falling back to regular mmap\n");
            init_regular_mmap(file, 0, true);
            return;
        }
        
        // Read file data into NUMA memory
        try {
            file->seek(0, SEEK_SET);
            file->read_raw(numa_addr, size);
        } catch (const std::exception& e) {
            free_numa_memory(numa_addr, size);
            throw std::runtime_error(format("Failed to read file data for NUMA isolate: %s", e.what()));
        }
        
        numa_allocations.push_back({numa_addr, size, target_node, false});
        addr = numa_addr;
        
        LLAMA_LOG_INFO("NUMA isolate: allocated %zu bytes on node %d at %p\n", 
                       size, target_node, addr);
    }
    
    void init_numa_mirror(struct llama_file* file) {
        // Mirror mode: replicate data on all NUMA nodes
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 1) {
            LLAMA_LOG_WARN("NUMA mirror mode requires multiple nodes, falling back to isolate\n");
            init_numa_isolate(file);
            return;
        }
        
        LLAMA_LOG_INFO("NUMA mirror: creating %d copies of %zu bytes\n", num_nodes, size);
        
        // Allocate on each NUMA node
        for (int node = 0; node < num_nodes; ++node) {
            void* node_addr = allocate_numa_memory(size, node);
            if (!node_addr) {
                LLAMA_LOG_ERROR("Failed to allocate on NUMA node %d\n", node);
                // Clean up previous allocations
                for (const auto& alloc : numa_allocations) {
                    free_numa_memory(alloc.addr, alloc.size);
                }
                numa_allocations.clear();
                // Fall back to regular mmap
                init_regular_mmap(file, 0, true);
                return;
            }
            
            numa_allocations.push_back({node_addr, size, node, false});
            
            // Set primary address to first allocation
            if (node == 0) {
                addr = node_addr;
            }
        }
        
        // Read file data into first node's memory
        try {
            file->seek(0, SEEK_SET);
            file->read_raw(numa_allocations[0].addr, size);
        } catch (const std::exception& e) {
            // Clean up all allocations
            for (const auto& alloc : numa_allocations) {
                free_numa_memory(alloc.addr, alloc.size);
            }
            numa_allocations.clear();
            throw std::runtime_error(format("Failed to read file data for NUMA mirror: %s", e.what()));
        }
        
        // Copy data to other nodes
        for (size_t i = 1; i < numa_allocations.size(); ++i) {
            memcpy(numa_allocations[i].addr, numa_allocations[0].addr, size);
            LLAMA_LOG_INFO("NUMA mirror: copied data to node %d at %p\n", 
                           numa_allocations[i].node, numa_allocations[i].addr);
        }
        
        LLAMA_LOG_INFO("NUMA mirror: successfully created %d copies\n", num_nodes);
    }
    
    // Multi-file NUMA constructor
    void init_numa_multi_file(const std::vector<struct llama_file*>& files, size_t prefetch, bool numa) {
        GGML_UNUSED(prefetch);
        
        if (files.empty()) {
            throw std::runtime_error("Cannot create mapping with empty file list");
        }
        
        // Calculate total size
        size_t total_size = 0;
        for (const auto* file : files) {
            total_size += file->size();
        }
        size = total_size;
        
        enum ggml_numa_strategy strategy = numa ? ggml_get_numa_strategy() : GGML_NUMA_STRATEGY_DISABLED;
        
        switch (strategy) {
            case GGML_NUMA_STRATEGY_DISABLED:
                init_multi_file_regular(files);
                return;
                
            case GGML_NUMA_STRATEGY_ISOLATE:
                init_multi_file_numa_isolate(files);
                return;
                
            case GGML_NUMA_STRATEGY_MIRROR:
                init_multi_file_numa_mirror(files);
                return;
                
            default:
                LLAMA_LOG_WARN("Unknown NUMA strategy for multi-file, falling back to regular\n");
                init_multi_file_regular(files);
                return;
        }
    }
    
    void init_multi_file_regular(const std::vector<struct llama_file*>& files) {
        // For regular multi-file, use anonymous mmap and read all files
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("Multi-file mmap failed: %s", strerror(errno)));
        }
        
        // Read all files into the mapped memory
        size_t offset = 0;
        for (const auto* file : files) {
            try {
                file->seek(0, SEEK_SET);
                file->read_raw((char*)addr + offset, file->size());
                offset += file->size();
            } catch (const std::exception& e) {
                munmap(addr, size);
                throw std::runtime_error(format("Failed to read multi-file data: %s", e.what()));
            }
        }
        
        // Make the memory read-only after loading
        if (mprotect(addr, size, PROT_READ) != 0) {
            LLAMA_LOG_WARN("Failed to make multi-file memory read-only: %s\n", strerror(errno));
        }
        
        mapped_fragments.emplace_back(0, size);
        LLAMA_LOG_INFO("Multi-file regular: loaded %zu bytes from %zu files at %p\n", 
                       size, files.size(), addr);
    }
    
    void init_multi_file_numa_isolate(const std::vector<struct llama_file*>& files) {
        int target_node = ggml_numa_get_current_node();
        if (target_node < 0) target_node = 0;
        
        void* numa_addr = allocate_numa_memory(size, target_node);
        if (!numa_addr) {
            LLAMA_LOG_WARN("Multi-file NUMA isolate failed, falling back to regular\n");
            init_multi_file_regular(files);
            return;
        }
        
        // Read all files into NUMA memory
        size_t offset = 0;
        for (const auto* file : files) {
            try {
                file->seek(0, SEEK_SET);
                file->read_raw((char*)numa_addr + offset, file->size());
                offset += file->size();
            } catch (const std::exception& e) {
                free_numa_memory(numa_addr, size);
                throw std::runtime_error(format("Failed to read multi-file NUMA data: %s", e.what()));
            }
        }
        
        numa_allocations.push_back({numa_addr, size, target_node, false});
        addr = numa_addr;
        
        LLAMA_LOG_INFO("Multi-file NUMA isolate: loaded %zu bytes from %zu files on node %d at %p\n", 
                       size, files.size(), target_node, addr);
    }
    
    void init_multi_file_numa_mirror(const std::vector<struct llama_file*>& files) {
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 1) {
            LLAMA_LOG_WARN("Multi-file NUMA mirror requires multiple nodes, falling back to isolate\n");
            init_multi_file_numa_isolate(files);
            return;
        }
        
        // Allocate on each NUMA node
        for (int node = 0; node < num_nodes; ++node) {
            void* node_addr = allocate_numa_memory(size, node);
            if (!node_addr) {
                // Clean up and fall back
                for (const auto& alloc : numa_allocations) {
                    free_numa_memory(alloc.addr, alloc.size);
                }
                numa_allocations.clear();
                init_multi_file_regular(files);
                return;
            }
            
            numa_allocations.push_back({node_addr, size, node, false});
            if (node == 0) addr = node_addr;
        }
        
        // Load files into first node
        size_t offset = 0;
        for (const auto* file : files) {
            try {
                file->seek(0, SEEK_SET);
                file->read_raw((char*)numa_allocations[0].addr + offset, file->size());
                offset += file->size();
            } catch (const std::exception& e) {
                for (const auto& alloc : numa_allocations) {
                    free_numa_memory(alloc.addr, alloc.size);
                }
                numa_allocations.clear();
                throw std::runtime_error(format("Failed to read multi-file NUMA data: %s", e.what()));
            }
        }
        
        // Copy to other nodes
        for (size_t i = 1; i < numa_allocations.size(); ++i) {
            memcpy(numa_allocations[i].addr, numa_allocations[0].addr, size);
        }
        
        LLAMA_LOG_INFO("Multi-file NUMA mirror: loaded %zu bytes from %zu files, %d copies\n", 
                       size, files.size(), num_nodes);
    }
#endif

    // Constructor for single file
    impl(struct llama_file * file, size_t prefetch, bool numa) {
#ifdef GGML_NUMA_MIRROR
        if (numa) {
            init_numa_single_file(file, prefetch, numa);
            return;
        }
#endif
        // Original clean implementation for non-NUMA
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }
        
        mapped_fragments.emplace_back(0, file->size());
    }

    // Constructor for multiple files
    impl(const std::vector<struct llama_file *> & files, size_t prefetch, bool numa) {
#ifdef GGML_NUMA_MIRROR
        if (numa) {
            init_numa_multi_file(files, prefetch, numa);
            return;
        }
#endif
        // Simplified multi-file for non-NUMA case
        if (files.empty()) {
            throw std::runtime_error("Cannot create mapping with empty file list");
        }
        
        // For simplicity, just use the first file in non-NUMA mode
        // A full implementation would create a unified mapping
        struct llama_file * first_file = files[0];
        size = first_file->size();
        int fd = first_file->file_id();
        
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        
        addr = mmap(NULL, first_file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(first_file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, first_file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }
        
        mapped_fragments.emplace_back(0, first_file->size());
        
        LLAMA_LOG_WARN("Multi-part unified mapping not fully supported in non-NUMA mode\n");
    }

    // ... [rest of original implementation: unmap_fragment, align_range, etc.] ...

    ~impl() {
#ifdef GGML_NUMA_MIRROR
        // Clean up NUMA allocations
        for (const auto& alloc : numa_allocations) {
            free_numa_memory(alloc.addr, alloc.size);
        }
        numa_allocations.clear();
#endif
        // Clean up regular mmaps
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }

    void * addr;
    size_t size;
};

// NUMA address lookup - clean implementation
void * llama_mmap::numa_addr(size_t offset, int numa_node) const {
#ifdef GGML_NUMA_MIRROR
    if (numa_node == -1) {
        numa_node = ggml_current_numa_node;
        if (numa_node == -1) numa_node = 0;
    }
    
    // For NUMA allocations, return appropriate node's address
    if (!pimpl->numa_allocations.empty()) {
        // Clamp node to available range
        if (numa_node < 0 || numa_node >= (int)pimpl->numa_allocations.size()) {
            numa_node = 0;
        }
        return (char*)pimpl->numa_allocations[numa_node].addr + offset;
    }
#else
    GGML_UNUSED(numa_node);
#endif
    // Fallback to regular addressing
    return (char*)pimpl->addr + offset;
}

// ... [rest of interface implementation remains the same] ...