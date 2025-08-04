# DUAL CPU ASSIGNMENT PROBLEM ANALYSIS

## 🚨 Problem Overview

The llama.cpp codebase currently has a **dual-layer CPU assignment system** that creates confusing behavior and potential conflicts. This analysis expands on the architectural issues identified in the CPU strict placement investigation.

## 📋 The Two Assignment Layers

### Layer 1: Threadpool Parameter Assignment (User-Controlled)
**Location**: `ggml/src/ggml-cpu/ggml-cpu.c:3236-3250`

```c
// Spin the threads for all workers, and update CPU placements.
// Place the main thread last (towards the higher numbered CPU cores).

int32_t cpumask_iter = 0;

for (int j = 1; j < tpp->n_threads; j++) {
    ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    // Creates secondary threads...
}

ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);
```

**Behavior**:
- Uses `ggml_thread_cpumask_next()` to assign CPUs based on threadpool parameters
- Respects user's `--cpu-strict` and `--cpu-strict-batch` settings
- Assigns individual CPUs when `strict_cpu=true`, shared CPU mask when `strict_cpu=false`
- **User Intent**: Controlled via `--cpu-strict` and `--cpu-strict-batch` options

### Layer 2: NUMA Runtime Override (System-Controlled)
**Location**: `ggml/src/ggml-cpu/ggml-cpu.c:2947-3010`

```c
#ifdef GGML_NUMA_MIRROR
if (GGML_UNLIKELY(ggml_current_numa_node == -1)) {
    int thread_id = state->ith;
    int target_numa_node = thread_id % num_numa_nodes;
    
    // ... NUMA node assignment logic ...
    
    // Fallback: if we couldn't find a CPU on the target node, use the original algorithm
    if (cpuid == -1) {
        bool local_mask[GGML_MAX_N_THREADS];
        int iter = 0;
        for (int j = 0; j < thread_id; ++j) {
            ggml_thread_cpumask_next(cpumask, local_mask, true, &iter);  // FORCES strict=true!
        }
        // ... assigns single CPU per thread ...
    }
    
    if (cpuid != -1) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuid, &cpuset);
        sched_setaffinity(gettid(), sizeof(cpuset), &cpuset);  // RUNTIME OVERRIDE
    }
}
#endif
```

**Behavior**:
- **Silently overrides** threadpool CPU assignments at runtime
- **Always forces strict placement** regardless of user's `--cpu-strict` setting
- Distributes threads across NUMA nodes for memory locality
- **System Intent**: NUMA performance optimization

## 🔥 Critical Issues

### 1. **Silent User Setting Override**
```bash
# User runs this expecting shared CPU masks:
./llama-server --cpu-strict false --threads 8

# But NUMA code silently forces strict=true and assigns individual CPUs!
# User's explicit preference is completely ignored.
```

### 2. **Inconsistent Behavior Across Environments**
- **Single NUMA node**: User settings respected
- **Multi-NUMA system**: User settings silently ignored
- **No documentation** warns users about this behavior change

### 3. **Conflicting Assignment Logic**
The two layers use different algorithms:

**Layer 1 (Threadpool)**:
```c
// Non-strict: All threads share the same CPU mask
if (!strict) {
    memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
    return;
}

// Strict: Round-robin assignment of individual CPUs
else {
    // ... assigns one CPU per thread via round-robin ...
}
```

**Layer 2 (NUMA)**:
```c
// ALWAYS strict, but uses NUMA-aware assignment
int target_numa_node = thread_id % num_numa_nodes;
// ... finds CPU on specific NUMA node ...
// Fallback uses strict assignment even if user wanted non-strict
```

### 4. **Debugging Nightmare**
When users report unexpected CPU affinity behavior:
- CPU assignment appears to work correctly in single-NUMA environments
- Behavior changes mysteriously on multi-NUMA systems
- No logging indicates the override is happening
- Performance implications aren't visible to users

## 📊 Code Flow Analysis

### Normal Flow (Single NUMA Node)
```
User Options → cpu_params → ggml_threadpool_params → Thread Creation → CPU Assignment
     ↓              ↓              ↓                        ↓              ↓
--cpu-strict → strict_cpu → tpp.strict_cpu → workers[].cpumask → Applied as intended
```

### NUMA Override Flow (Multi-NUMA System)
```
User Options → cpu_params → ggml_threadpool_params → Thread Creation → CPU Assignment → NUMA OVERRIDE
     ↓              ↓              ↓                        ↓              ↓               ↓
--cpu-strict → strict_cpu → tpp.strict_cpu → workers[].cpumask → Applied → COMPLETELY REPLACED
                                                                            by sched_setaffinity()
```

## 🎯 Specific Problems Found

### 1. **Fallback Logic Forces Strict Mode**
```c
// In NUMA fallback code - LINE 2989-2997
for (int j = 0; j < thread_id; ++j) {
    ggml_thread_cpumask_next(cpumask, local_mask, true, &iter);  // <-- HARDCODED true!
}
```
Even when user explicitly set `--cpu-strict false`, the NUMA fallback **hardcodes `strict=true`**.

### 2. **Double CPU Assignment Work**
- Threadpool creation does expensive CPU mask calculations
- NUMA code completely replaces this with its own assignment
- Wasted computation and confusing code paths

### 3. **Inconsistent Strict Placement Semantics**
- **User's strict=false**: "I want threads to share CPU cores for flexibility"
- **NUMA's forced strict=true**: "Each thread gets exactly one CPU for NUMA locality"
- These are **conflicting goals** with no user control

## 🛠️ Technical Debt Impact

### Code Maintenance
- Two separate CPU assignment algorithms to maintain
- Complex interaction between threadpool and NUMA systems
- Difficult to test all combinations of settings

### User Experience
- Unexpected performance changes between environments
- Silent behavior changes without user consent
- Documentation doesn't match actual behavior

### Performance Implications
- Users cannot opt out of NUMA strict placement for workloads where flexibility matters
- No way to balance NUMA locality vs thread flexibility
- Advanced users cannot fine-tune CPU assignment behavior

## 💡 Architectural Solutions

### Option 1: Unified CPU Assignment (Recommended)
```c
struct cpu_assignment_params {
    bool strict_cpu;           // User preference
    bool numa_aware;          // NUMA locality preference  
    bool allow_numa_override; // Can NUMA override user strict setting?
};

// Single function handles all CPU assignment logic
void assign_thread_cpu(thread_id, assignment_params, available_cpus) {
    if (numa_aware && numa_nodes > 1) {
        int target_node = select_numa_node(thread_id);
        if (strict_cpu || !assignment_params.allow_numa_override) {
            assign_single_cpu_on_node(thread_id, target_node);
        } else {
            assign_node_cpu_mask(thread_id, target_node);
        }
    } else {
        if (strict_cpu) {
            assign_single_cpu_round_robin(thread_id);
        } else {
            assign_shared_cpu_mask(thread_id);
        }
    }
}
```

### Option 2: Explicit NUMA Override Control
Add new command-line options:
```bash
--numa-override-strict    # Allow NUMA to force strict placement (default: true)
--numa-locality-priority  # NUMA locality vs user preferences (default: numa)
```

### Option 3: Transparent NUMA Enhancement
Modify NUMA code to respect user strict preferences:
```c
// Instead of hardcoding strict=true, use user preference
bool effective_strict = user_strict_preference || numa_locality_required;
ggml_thread_cpumask_next(cpumask, local_mask, effective_strict, &iter);
```

## 🚦 Immediate Recommendations

### 1. **Add Visibility**
```c
LOG_WRN("NUMA: Overriding user CPU strict setting (%s -> true) for NUMA locality\n", 
        user_strict ? "true" : "false");
```

### 2. **Document the Behavior**
Update help text and documentation to clearly explain:
- NUMA systems may override `--cpu-strict` settings
- When and why this override occurs
- Performance implications of different settings

### 3. **Add Override Control**
Implement `--numa-respect-strict` option to give users control over the override behavior.

## 🔍 Code Locations for Fixes

### Primary Files to Modify
1. **`ggml/src/ggml-cpu/ggml-cpu.c:2989-2997`** - Remove hardcoded `true` in fallback
2. **`ggml/src/ggml-cpu/ggml-cpu.c:2947-3010`** - Add user preference checks
3. **`common/arg.cpp`** - Add new NUMA override control options
4. **`common/common.h`** - Extend cpu_params with NUMA override settings

### Testing Requirements
- Test strict vs non-strict behavior on single-NUMA systems
- Verify NUMA override behavior is consistent and documented
- Test performance impact of different assignment strategies
- Validate user settings are respected when technically feasible

## 📈 Benefits of Fixing This

### User Experience
- Predictable behavior across all hardware configurations
- User control over CPU assignment vs NUMA locality trade-offs
- Clear documentation of when overrides occur

### Code Quality
- Single source of truth for CPU assignment logic
- Reduced complexity in threadpool and NUMA interactions
- Better testability and maintainability

### Performance
- Users can choose optimal strategy for their specific workloads
- Advanced users can fine-tune CPU assignment behavior
- Better understanding of performance characteristics

---

## 🎯 Conclusion

The dual CPU assignment system represents a significant architectural debt that affects user experience, code maintainability, and performance predictability. The silent override of user preferences is particularly problematic as it violates the principle of least surprise.

**Priority Level**: **HIGH** - This affects user trust and system predictability.

**Recommended Action**: Implement Option 1 (Unified CPU Assignment) with explicit user controls for NUMA override behavior.
