# CPU ASSIGNMENT CONTROL FLOW DIAGRAM

## Normal Single-NUMA Flow (User Settings Respected)

```
User Command Line
       │
       ▼
  --cpu-strict false
       │
       ▼
    cpu_params
   {strict_cpu: false}
       │
       ▼
 ggml_threadpool_params
   {strict_cpu: false}
       │
       ▼
 ggml_thread_cpumask_next()
  with strict=false
       │
       ▼
  memcpy(local_mask, global_mask)
       │
       ▼
   ALL THREADS SHARE
   SAME CPU MASK ✓
```

## Multi-NUMA Flow (User Settings Silently Overridden)

```
User Command Line
       │
       ▼
  --cpu-strict false
       │
       ▼
    cpu_params
   {strict_cpu: false}
       │
       ▼
 ggml_threadpool_params
   {strict_cpu: false}
       │
       ▼
 ggml_thread_cpumask_next()     ←── Layer 1: User intent honored
  with strict=false                  (temporarily)
       │
       ▼
  memcpy(local_mask, global_mask)
       │
       ▼
   Threads assigned shared mask   ←── Works as expected so far
       │
       ▼
  ┌─────────────────────────────┐
  │   NUMA RUNTIME OVERRIDE     │  ←── Layer 2: System overrides user
  │  (ggml-cpu.c:2947-3010)    │
  └─────────────────────────────┘
       │
       ▼
 #ifdef GGML_NUMA_MIRROR
 if (numa_nodes > 1)  ←── Multi-NUMA detected
       │
       ▼
  ggml_thread_cpumask_next()
  with strict=TRUE     ←── ❌ HARDCODED! Ignores user setting
       │
       ▼
  sched_setaffinity(single_cpu)  ←── Forces individual CPU per thread
       │
       ▼
   EACH THREAD BOUND TO
   SINGLE CPU ❌
   
   User's --cpu-strict false
   completely ignored!
```

## The Problem Locations

### Location 1: Threadpool Creation (Layer 1)
**File**: `ggml/src/ggml-cpu/ggml-cpu.c:3240-3248`
```c
for (int j = 1; j < tpp->n_threads; j++) {
    ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, 
                           tpp->strict_cpu, &cpumask_iter);  // ✓ Respects user setting
    // ...
}
```

### Location 2: NUMA Override (Layer 2)  
**File**: `ggml/src/ggml-cpu/ggml-cpu.c:2989-2997`
```c
if (cpuid == -1) {
    bool local_mask[GGML_MAX_N_THREADS];
    int iter = 0;
    for (int j = 0; j < thread_id; ++j) {
        ggml_thread_cpumask_next(cpumask, local_mask, 
                               true, &iter);          // ❌ HARDCODED true!
    }
    // ... assigns single CPU regardless of user preference
}
```

## Behavioral Matrix

| Environment | User Setting | Actual Behavior | Expected Behavior | Issue |
|-------------|--------------|-----------------|-------------------|-------|
| Single NUMA | `--cpu-strict false` | Shared CPU mask | Shared CPU mask | ✓ OK |
| Single NUMA | `--cpu-strict true` | Individual CPUs | Individual CPUs | ✓ OK |
| Multi NUMA | `--cpu-strict false` | Individual CPUs | Shared CPU mask | ❌ BROKEN |
| Multi NUMA | `--cpu-strict true` | Individual CPUs | Individual CPUs | ✓ OK (by accident) |

## User Experience Problems

### Problem 1: Silent Override
```bash
# User runs this expecting shared CPU access:
./llama-server --cpu-strict false --threads 8

# On single-NUMA: Works as expected (threads share CPUs)
# On multi-NUMA: Silently forces strict placement (each thread gets 1 CPU)
# No warning, no documentation, no user control
```

### Problem 2: Environment-Dependent Behavior
```bash
# Same command, different behavior based on hardware:

# Desktop (1 NUMA node): 
#   8 threads all share CPUs 0-7 ✓

# Server (2 NUMA nodes):
#   Thread 0 → CPU 4 only
#   Thread 1 → CPU 5 only  
#   Thread 2 → CPU 6 only
#   ... ❌ User's flexibility request ignored
```

### Problem 3: Performance Mystery
Users report:
- "My inference is slower on the server than my desktop"
- "CPU utilization looks weird - threads aren't load balancing"
- "Why does --cpu-strict false not work on some machines?"

**Root cause**: NUMA code silently forces strict placement, preventing load balancing.

## Solution Architecture

### Current (Broken) Flow:
```
User Intent → Threadpool Assignment → NUMA Override → Final Assignment
     ↓               ↓                      ↓              ↓
 flexible        respects user         ignores user    strict only
```

### Proposed (Unified) Flow:
```
User Intent → Combined Assignment Logic → Final Assignment
     ↓                    ↓                      ↓
 flexible         considers both user       user choice
                  preference AND NUMA       respected
                  requirements
```

### Implementation Strategy:
1. **Merge** the two assignment layers into one unified function
2. **Add** user control over NUMA override behavior  
3. **Provide** clear logging when overrides occur
4. **Document** the trade-offs between flexibility and NUMA locality
