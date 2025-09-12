# NUMA Infrastructure Improvements - August 23, 2025

## 🎯 Lessons Learned from Physical Core Testing

### Key Discoveries:
1. **Memory Locality**: `numa_alloc_onnode()` works correctly, but pages must be initialized (memset) before `move_pages()` verification works
2. **Physical Core Binding**: `pthread_setaffinity_np()` with physical cores only provides consistent performance
3. **Hardware Asymmetry**: 10% performance difference between NUMA nodes is acceptable for now
4. **Memory Verification**: Need proper error handling for `move_pages()` status codes (-2 = uninitialized pages)

## 📋 Todo List - NUMA Infrastructure Fixes

### 🔥 Critical (Immediate)
- [ ] **NUMA Allocator**: Fix `ggml-cpu-numa-buffer.cpp` to properly initialize allocated pages
- [ ] **Coordinator Thread Binding**: Implement physical core binding in `ggml-numa-coordinator.c`
- [ ] **Memory Locality Verification**: Add proper `move_pages()` validation with initialization
- [ ] **CPU Affinity Masks**: Ensure llama-cpp passes correct CPU masks to coordinator

### 🚨 High Priority  
- [ ] **llama-mmap.cpp**: Add NUMA-aware memory mapping with proper page initialization
- [ ] **Threadpool Binding**: Bind coordinator threadpools to physical cores only
- [ ] **NUMA Mirror Mode**: Ensure mirror mode uses physical core binding
- [ ] **Error Handling**: Improve NUMA allocation error reporting and fallbacks

### 📊 Medium Priority
- [ ] **Integration Testing**: Update NUMA tests to use physical cores consistently
- [ ] **Documentation**: Update architecture docs with physical core requirements
- [ ] **Performance Monitoring**: Add runtime NUMA locality verification
- [ ] **Configuration**: Allow physical-core-only mode configuration

### 🔮 Future (Post-Fix)
- [ ] **Hardware Asymmetry**: Implement node-specific performance tuning
- [ ] **Dynamic Binding**: Runtime CPU topology discovery and binding
- [ ] **NUMA Policies**: More sophisticated memory allocation policies
- [ ] **Performance Profiling**: Per-node performance characterization

## 🎯 Implementation Priority Order

1. Fix NUMA allocator page initialization
2. Implement coordinator physical core binding  
3. Update llama-mmap.cpp NUMA awareness
4. Verify integration with llama-cpp CPU mask passing
5. Comprehensive testing and validation
