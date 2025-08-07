# Global Singleton Coordinator Implementation

## Progress Made

✅ **Global Singleton Pattern**: Implemented global coordinator manager that persists for program lifetime
✅ **Proper Cleanup**: Coordinator only freed at program exit via `atexit()`
✅ **Reference Management**: Threadpools now release references instead of freeing coordinator
✅ **Thread Safety**: Added mutex protection for singleton initialization

## Current Issue

The coordinator's **NUMA threadpools** still create worker threads that persist after coordinator reference is released. These old worker threads conflict with new threadpools created in subsequent tests.

## Root Cause

1. Coordinator (singleton) creates NUMA threadpools with `ggml_threadpool_new()` 
2. These NUMA threadpools create worker threads
3. When threadpool releases coordinator reference, NUMA threadpools still exist 
4. New tests create new threadpools, but old worker threads are still running
5. Old worker threads try to access freed memory from new threadpools

## Solution

The **NUMA threadpools** created by coordinators should also be persistent/singleton, not recreated for each operation. Or we need to ensure that NUMA threadpool cleanup is fully synchronous and blocks until all worker threads terminate.

## Architecture Validation

The 3-tier singleton approach is working correctly:
- ✅ Main Thread → Global Coordinator Manager (singleton)  
- ✅ Coordinator Threads (persistent)
- ❌ NUMA Threadpools (still creating/destroying worker threads)

Next: Make NUMA threadpools persistent or fix worker thread cleanup synchronization.
