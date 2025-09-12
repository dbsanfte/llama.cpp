# 3-Tier NUMA Coordinator Architecture Implementation

## Date: 2025-08-06

## Summary
Implemented the user's proposed 3-tier NUMA coordinator architecture to replace the problematic cgraph sharing approach with a cleaner hierarchical design.

## Changes Made

### 1. New Architecture Design
- **Main Thread** → **Coordinator Threads** → **NUMA Node Threadpools**
- Each NUMA node gets its own cgraph copy (eliminates race conditions)
- Work flows through hierarchical queues
- Proper cleanup sequence: NUMA → coordinator → main

### 2. New Files Created
- `/ggml/include/ggml-numa-coordinator.h` - Public API header
- `/ggml/src/ggml-cpu/ggml-numa-coordinator.c` - Implementation

### 3. Core Components

#### Work Queue System
- `struct ggml_work_item` - Individual work units
- `struct ggml_work_queue` - Thread-safe work queues with mutexes/conditions
- Work flows: main → global queue → coordinator → NUMA pool

#### Coordinator Threads
- `struct ggml_coordinator_thread` - One per NUMA node
- Each coordinator has its own NUMA-specific threadpool
- Each coordinator has its own full copy of cgraph (no sharing!)

#### Manager
- `struct ggml_numa_coordinator_manager` - Overall coordinator
- Manages all coordinator threads
- Handles main thread synchronization

### 4. Integration Points
- Modified `ggml_threadpool_new_impl()` to use coordinator when appropriate
- Added `use_coordinator` flag to threadpool structure
- Updated `ggml_threadpool_free()` for hierarchical cleanup
- Added to CMake build system

### 5. Key Improvements Over Previous Approach
- **No shared cgraph references** - each NUMA node owns its copy
- **Hierarchical cleanup** - proper shutdown sequence
- **Work queue based** - scalable work distribution
- **Thread safety** - atomic operations and proper synchronization
- **CPU affinity** - coordinator threads bound to their NUMA nodes

## Current Status
- ✅ Compilation successful
- ✅ Coordinator creation working
- ✅ Cgraph copying working  
- ❌ Thread creation has segfault (null pointer issue)

## Next Steps
1. Fix thread parameter passing in coordinator thread creation
2. Implement work submission and processing
3. Test performance compared to legacy approach
4. Remove legacy NUMA manager once stable

## Architecture Benefits
- **Eliminates race conditions** - no shared state between NUMA nodes
- **Better resource isolation** - each NUMA node is independent
- **Cleaner shutdown** - hierarchical cleanup prevents orphaned threads
- **Scalable design** - work queue system handles variable loads
- **Maintainable code** - clear separation of concerns

## User Flow Verification
✅ 1. Main thread creates coordinator threads (one per NUMA node)
✅ 2. Each coordinator gets its own NUMA threadpool  
✅ 3. Each coordinator gets full cgraph copy
❌ 4. Coordinator threads start successfully (FIXING)
⏳ 5-12. Work distribution and cleanup (TO IMPLEMENT)

The core architecture is working as designed - just need to fix the thread startup issue.
