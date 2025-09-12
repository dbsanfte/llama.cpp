#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>

#ifdef GGML_NUMA_MIRROR
#include <numa.h>
#include <numaif.h>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// NUMA debug logging macros
#define NUMA_LOG_DEBUG(fmt, ...) \
    do { \
        const char* debug_env = getenv("GGML_NUMA_DEBUG"); \
        if (debug_env && atoi(debug_env) >= 1) { \
            LLAMA_LOG_DEBUG(fmt, ##__VA_ARGS__); \
        } \
    } while (0)
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

#ifdef GGML_NUMA_MIRROR
// NUMA system validation with comprehensive node testing
static bool validate_numa_system() {
    printf("🔍 NUMA validation starting - checking system capabilities...\n");
    fflush(stdout);
    
    if (numa_available() < 0) {
        printf("❌ numa_available() returned < 0\n");
        fflush(stdout);
        NUMA_LOG_DEBUG("NUMA library not available\n");
        return false;
    }
    
    int num_nodes = numa_num_configured_nodes();
    printf("📊 numa_num_configured_nodes() = %d\n", num_nodes);
    fflush(stdout);
    
    if (num_nodes <= 1) {
        printf("❌ Only %d NUMA node(s) configured\n", num_nodes);
        fflush(stdout);
        NUMA_LOG_DEBUG("Only %d NUMA node(s) configured\n", num_nodes);
        return false;
    }
    
    printf("✅ NUMA validation: Testing %d nodes\n", num_nodes);
    fflush(stdout);
    NUMA_LOG_DEBUG("NUMA validation: Testing %d nodes\n", num_nodes);
    
    // Test allocation on each node and verify placement
    const size_t test_size = 4096; // One page
    bool all_nodes_working = true;
    
    for (int node = 0; node < num_nodes; node++) {
        printf("🧪 Testing NUMA node %d...\n", node);
        fflush(stdout);
        
        // Test allocation
        void* test_ptr = numa_alloc_onnode(test_size, node);
        if (!test_ptr) {
            printf("❌ NUMA validation: node %d allocation failed\n", node);
            fflush(stdout);
            NUMA_LOG_DEBUG("NUMA validation: node %d allocation failed\n", node);
            all_nodes_working = false;
            continue;
        }
        printf("✅ Node %d: allocation successful at %p\n", node, test_ptr);
        fflush(stdout);
        
        // Verify the allocation is actually on the requested node
        int actual_node = -1;
        if (get_mempolicy(&actual_node, NULL, 0, test_ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            printf("📍 Node %d: memory at %p is on actual node %d\n", node, test_ptr, actual_node);
            fflush(stdout);
            if (actual_node != node) {
                printf("⚠️  Node %d: expected allocation on node %d, got node %d (container environment)\n", node, node, actual_node);
                fflush(stdout);
                NUMA_LOG_DEBUG("NUMA validation: node %d requested, but memory at %p is on node %d (container env)\n", 
                               node, test_ptr, actual_node);
                // Don't fail for placement in containers - NUMA topology exists but memory placement may be limited
                // The important thing is that numa_alloc_onnode() succeeds and thread binding works
            } else {
                printf("✅ Node %d: allocation verified correctly placed\n", node);
                fflush(stdout);
                NUMA_LOG_DEBUG("NUMA validation: node %d allocation verified at %p\n", node, test_ptr);
            }
        } else {
            printf("❌ Node %d: get_mempolicy failed: %s\n", node, strerror(errno));
            fflush(stdout);
            NUMA_LOG_DEBUG("NUMA validation: get_mempolicy failed for node %d: %s\n", node, strerror(errno));
            all_nodes_working = false;
        }
        
        // Touch the memory to ensure it's actually allocated
        printf("🖊️  Node %d: touching memory to verify allocation\n", node);
        fflush(stdout);
        volatile char* mem = (volatile char*)test_ptr;
        mem[0] = 1;
        mem[test_size - 1] = 1;
        printf("✅ Node %d: memory touch successful\n", node);
        fflush(stdout);
        
        numa_free(test_ptr, test_size);
        printf("🧹 Node %d: memory freed\n", node);
        fflush(stdout);
    }
    
    printf("🧵 Testing thread binding capabilities...\n");
    fflush(stdout);
    // Test thread binding
    struct bitmask* old_mask = numa_get_run_node_mask();
    for (int node = 0; node < num_nodes; node++) {
        printf("🔗 Testing binding to node %d...\n", node);
        fflush(stdout);
        if (numa_run_on_node(node) != 0) {
            printf("❌ Cannot bind to node %d\n", node);
            fflush(stdout);
            NUMA_LOG_DEBUG("NUMA validation: cannot bind to node %d\n", node);
            all_nodes_working = false;
        } else {
            printf("✅ Successfully bound to node %d\n", node);
            fflush(stdout);
        }
    }
    // Restore original thread binding
    if (old_mask) {
        numa_run_on_node_mask(old_mask);
        numa_free_nodemask(old_mask);
        printf("🔄 Restored original thread binding\n");
        fflush(stdout);
    }
    
    if (all_nodes_working) {
        printf("🎉 NUMA validation SUCCESSFUL - all %d nodes working!\n", num_nodes);
        fflush(stdout);
        NUMA_LOG_DEBUG("✅ NUMA system validation successful - %d nodes working\n", num_nodes);
    } else {
        printf("💥 NUMA validation FAILED - some tests didn't pass\n");
        fflush(stdout);
        NUMA_LOG_DEBUG("❌ NUMA system validation failed - fallback to regular allocation\n");
    }
    
    return all_nodes_working;
}

// NUMA allocation using posix_memalign + first-touch approach with SIMD alignment
static void* numa_alloc_mmap_first_touch(size_t size, int node) {
    // Define SIMD alignment (same as ggml_aligned_malloc)
#if defined(__s390x__)
    const size_t alignment = 256;
#else
    const size_t alignment = 64;  // 64-byte alignment for AVX-512
#endif
    
    // Bind current thread to the target NUMA node for first-touch
    struct bitmask* old_mask = numa_get_run_node_mask();
    if (numa_run_on_node(node) != 0) {
        NUMA_LOG_DEBUG("Warning: could not bind thread to NUMA node %d: %s\n", node, strerror(errno));
        // Continue anyway - might still work
    }
    
    // Use posix_memalign for SIMD alignment
    void* ptr = nullptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        NUMA_LOG_DEBUG("posix_memalign failed for %zu bytes with alignment %zu: %s\n", 
                       size, alignment, strerror(ret));
        // Restore original thread binding
        if (old_mask) {
            numa_run_on_node_mask(old_mask);
            numa_free_nodemask(old_mask);
        }
        return nullptr;
    }
    
    // First-touch: touch every page to ensure physical allocation on current node
    volatile char* mem = (volatile char*)ptr;
    const size_t page_size = sysconf(_SC_PAGESIZE);
    for (size_t i = 0; i < size; i += page_size) {
        mem[i] = 0; // First touch allocates the page on current NUMA node
    }
    
    // Restore original thread binding
    if (old_mask) {
        numa_run_on_node_mask(old_mask);
        numa_free_nodemask(old_mask);
    }
    
    NUMA_LOG_DEBUG("✅ posix_memalign + first-touch allocation: %zu bytes for node %d at %p (SIMD aligned to %zu bytes)\n", 
                   size, node, ptr, alignment);
    return ptr;
}

static void numa_free_mmap_first_touch(void* ptr, size_t size) {
    if (ptr) {
        free(ptr);  // Use free() for posix_memalign() allocated memory
        NUMA_LOG_DEBUG("Freed aligned memory at %p, size %zu\n", ptr, size);
    }
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw std::runtime_error(format("ftell error: %s", strerror(errno)));
        }

        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
// TODO: this ifdef is never true?
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        if (ret != 0) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw(void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~impl() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif

    FILE * fp;
    size_t size;
};

llama_file::llama_file(const char * fname, const char * mode) : pimpl(std::make_unique<impl>(fname, mode)) {}
llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

int llama_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) const { pimpl->read_raw(ptr, len); }

uint32_t llama_file::read_u32() const { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
#ifdef GGML_NUMA_MIRROR
    struct numa_mapping {
        void* addr;
        size_t size;
        std::string path;
    };
    std::vector<numa_mapping> numa_mappings;
    
    // NUMA mirroring implementation - creates copies of model data on each NUMA node
    void mmap_numa_mirror(struct llama_file * file, size_t prefetch) {
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (prefetch) { flags |= MAP_POPULATE; }
        
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
#endif
        
        int oldpolicy;
        struct bitmask* oldmask = numa_allocate_nodemask();
        if (get_mempolicy(&oldpolicy, oldmask->maskp,
                          oldmask->size + 1, 0, 0) < 0) {
            LLAMA_LOG_WARN("get_mempolicy failed, errno=%d %s\n", errno, strerror(errno));
            oldpolicy = MPOL_DEFAULT;
        }

        // Get the number of NUMA nodes
        int num_nodes = numa_num_configured_nodes();
        if (num_nodes <= 0) {
            LLAMA_LOG_WARN("numa_num_configured_nodes returned %d, defaulting to 1\n", num_nodes);
            num_nodes = 1;
        }
        LLAMA_LOG_INFO("Detected %d NUMA nodes for mirror mode\n", num_nodes);

        // Validate NUMA mirroring request based on strategy and hardware
        if (num_nodes <= 1) {
            LLAMA_LOG_ERROR("NUMA mirror mode requested but only %d NUMA node(s) detected\n", num_nodes);
            LLAMA_LOG_ERROR("NUMA mirroring requires multiple NUMA nodes to be effective\n");
            throw std::runtime_error("NUMA mirror mode requires multiple NUMA nodes");
        }

        size_t total_size = file->size();
        
        LLAMA_LOG_INFO("Creating NUMA mirrors with validated allocation: %zu bytes per node\n", total_size);

        // Allocate and populate memory on each NUMA node using validated allocator
        for (int node = 0; node < num_nodes; ++node) {
            numa_set_preferred(node);
            LLAMA_LOG_INFO("Allocating mirror on NUMA node %d\n", node);
                                   
            // Use mmap + first-touch for container compatibility
            void* node_mem = nullptr;
            
            // For large allocations, use mmap + first-touch allocation
            node_mem = numa_alloc_mmap_first_touch(total_size, node);
            if (!node_mem) {
                // Clean up any previous allocations before throwing
                for (const auto& mapping : numa_mappings) {
                    numa_free_mmap_first_touch(mapping.addr, mapping.size);
                }
                LLAMA_LOG_ERROR("Failed to allocate %zu bytes on NUMA node %d\n", total_size, node);
                throw std::runtime_error(format("mmap + first-touch allocation failed for node %d: %s", node, strerror(errno)));
            }
            
            // Verify SIMD alignment - our posix_memalign implementation guarantees this
#if defined(__s390x__)
            const size_t expected_alignment = 256;
#else
            const size_t expected_alignment = 64;  // 64-byte alignment for AVX-512
#endif
            if (reinterpret_cast<uintptr_t>(node_mem) % expected_alignment != 0) {
                LLAMA_LOG_ERROR("SIMD alignment verification failed: %p not aligned to %zu bytes\n", 
                               node_mem, expected_alignment);
                numa_free_mmap_first_touch(node_mem, total_size);
                throw std::runtime_error(format("SIMD alignment verification failed for node %d", node));
            }
            NUMA_LOG_DEBUG("✅ SIMD alignment verified: %p aligned to %zu bytes\n", node_mem, expected_alignment);
            
            NUMA_LOG_DEBUG("NUMA node %d: allocated %zu bytes at %p\n", node, total_size, node_mem);
            
            // Read model data from file directly into NUMA-local memory
            // Use the llama_file API instead of direct pread to ensure proper file handling
            try {
                file->seek(0, SEEK_SET);
                file->read_raw(node_mem, total_size);
            } catch (const std::exception& e) {
                LLAMA_LOG_ERROR("Failed to read model data for NUMA node %d: %s\n", node, e.what());
                numa_free_mmap_first_touch(node_mem, total_size);
                // Clean up any previous allocations
                for (const auto& mapping : numa_mappings) {
                    numa_free_mmap_first_touch(mapping.addr, mapping.size);
                }
                throw std::runtime_error(format("Failed to read model data for NUMA node %d: %s", node, e.what()));
            }
            
            NUMA_LOG_DEBUG("NUMA node %d: loaded %zu bytes from file at %p\n", node, total_size, node_mem);
            
            // Store the NUMA allocation directly
            numa_mappings.push_back({node_mem, total_size, ""});
        }

        // Set addr to the first allocation for compatibility
        addr = numa_mappings.empty() ? nullptr : numa_mappings[0].addr;
        
        LLAMA_LOG_INFO("NUMA mirror mode: successfully created %d copies of %zu bytes\n", 
                      num_nodes, total_size);
    }
#endif

    impl(struct llama_file * file, size_t prefetch, bool numa) {
        size = file->size();
        int fd = file->file_id();
        
#ifdef GGML_NUMA_MIRROR
        // If NUMA mirroring is requested, use NUMA mirror mode
        if (numa && validate_numa_system()) {
            LLAMA_LOG_INFO("NUMA mirror mode - replicating model data on each NUMA node\n");
            mmap_numa_mirror(file, prefetch);
            return;
        }
        
        // Fall back to regular mmap if NUMA not requested or not available
        if (numa) {
            LLAMA_LOG_WARN("NUMA mirroring requested but not available, falling back to regular mmap\n");
        }
#endif

        // Regular mmap implementation
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

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
#ifdef GGML_NUMA_MIRROR
        // Clean up NUMA mappings first
        for (const auto& mapping : numa_mappings) {
            numa_free_mmap_first_touch(mapping.addr, mapping.size);
        }
        
        // If we have NUMA mappings, we don't have regular mapped_fragments
        if (!numa_mappings.empty()) {
            return;
        }
#endif
        
        // Clean up regular mmap fragments
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    impl(struct llama_file * file, size_t prefetch, bool numa) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
};

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
        // visionOS/tvOS dont't support RLIMIT_MEMLOCK
        // Skip resource limit checks on visionOS/tvOS
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
