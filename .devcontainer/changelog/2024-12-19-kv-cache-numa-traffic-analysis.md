# KV Cache NUMA Architecture Analysis: Traffic, Distribution, and Synchronization

## 🎯 Executive Summary

Our NUMA-aware KV cache implementation will have **SIGNIFICANT** implications for UPI traffic and data distribution patterns. The current implementation creates a **single shared KV cache per context**, which means:

1. **Each NUMA node gets its own KV cache allocation** (via our NUMA buffer type)
2. **BUT** - all nodes access the **same logical cache** for a given sequence  
3. **Cross-NUMA traffic will be HIGH** unless we implement cache sharding

## 🏗️ Current KV Cache Architecture

### Single Cache Model
```cpp
// Current llama.cpp design: ONE KV cache per context
llama_context ctx;
├── llama_kv_cache_unified memory;    // Single cache object
    ├── Multiple streams (for parallel sequences)
    ├── Each stream has cells for positions
    └── All threads access same cache data
```

### With Our NUMA Buffer Type
```cpp
// After our changes:
if (ggml_is_numa()) {
    buft = ggml_backend_cpu_numa_buffer_type();  // NUMA-aware allocation
} else {
    buft = ggml_backend_cpu_buffer_type();       // Standard allocation  
}
```

**Result**: KV cache memory gets allocated on specific NUMA nodes, but **logical structure remains shared**.

## 🌐 NUMA Coordinator Integration Analysis

### Data Parallelism Scenarios

#### Scenario 1: **Batch Processing (Different Sequences)**
```
NUMA Node 0: Processing sequence A 
├── Accesses KV cache for sequence A  ✅ LOCAL
├── Never touches KV cache for sequence B  ✅ NO CROSS-TRAFFIC

NUMA Node 1: Processing sequence B
├── Accesses KV cache for sequence B  ✅ LOCAL  
├── Never touches KV cache for sequence A  ✅ NO CROSS-TRAFFIC
```
**UPI Traffic**: ✅ **MINIMAL** - Each node only accesses its own sequence's cache

#### Scenario 2: **Single Sequence, Multi-NUMA Processing**
```
NUMA Node 0: Processing tokens 0-512 of sequence A
├── Needs KV cache positions 0-512    ✅ LOCAL (if allocated here)
├── Needs KV cache positions 513-1024 ❌ REMOTE ACCESS → HIGH UPI TRAFFIC

NUMA Node 1: Processing tokens 513-1024 of sequence A  
├── Needs KV cache positions 513-1024  ✅ LOCAL (if allocated here)
├── Needs KV cache positions 0-512     ❌ REMOTE ACCESS → HIGH UPI TRAFFIC
```
**UPI Traffic**: ⚠️ **VERY HIGH** - Constant cross-node memory access

### Current KV Cache Access Patterns

From the code analysis, here's how KV cache works:

```cpp
// Each sequence gets its own stream
seq_id → stream_id mapping
│
├── Stream 0: Sequence A's KV data (keys + values for all positions)
├── Stream 1: Sequence B's KV data  
└── Stream N: Sequence N's KV data

// During attention computation:
for (each_token_in_batch) {
    seq_id = token.sequence_id;
    stream = seq_to_stream[seq_id];
    
    // Access ALL previous positions for this sequence
    for (pos = 0; pos < current_position; pos++) {
        key = kv_cache.get_k(stream, layer, pos);    // May be remote!
        val = kv_cache.get_v(stream, layer, pos);    // May be remote!
        attention_score += query * key;
    }
}
```

## 🚨 Critical Issues with Current Implementation

### Issue 1: Cache Allocation Placement
```cpp
// Our NUMA buffer allocates KV cache on ONE node:
ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(numa_buft, cache_size);
// ↑ This entire cache lives on whichever node the coordinator selects
```

### Issue 2: Attention Computation Cross-Node Access
```cpp
// When computing attention:
// Thread on NUMA Node 1 processing sequence allocated on NUMA Node 0:
float attention = 0.0f;
for (int pos = 0; pos < seq_len; pos++) {
    float key = kv_cache[pos];     // ❌ Remote memory access via UPI!
    float val = kv_cache[pos];     // ❌ Another remote access!
    attention += query * key;      // Massive bandwidth waste
}
```

### Issue 3: Memory Bandwidth Bottleneck
- **Single-node allocation** → Only one node's memory controllers utilized
- **All other nodes** → Must access via UPI links (50-70% slower)
- **Effective bandwidth** → Limited to single node's capacity, not aggregate

## 💡 Solutions and Trade-offs

### Option 1: **Cache Replication** (Current Best Option)
```cpp
// Each NUMA node gets FULL copy of each sequence's KV cache
NUMA Node 0: Complete KV cache copy for active sequences
NUMA Node 1: Complete KV cache copy for active sequences  
NUMA Node 2: Complete KV cache copy for active sequences
```

**Pros**:
- ✅ Zero cross-node traffic during inference
- ✅ Maximum memory bandwidth utilization
- ✅ Linear scaling with NUMA nodes

**Cons**:  
- ❌ N×memory usage (where N = NUMA nodes)
- ❌ Cache coherency complexity
- ❌ Write amplification during cache updates

### Option 2: **Cache Sharding by Position**
```cpp
// Split KV cache positions across nodes
NUMA Node 0: KV positions 0-512
NUMA Node 1: KV positions 513-1024
NUMA Node 2: KV positions 1025-1536
```

**Pros**:
- ✅ Memory usage scales efficiently
- ✅ No replication overhead

**Cons**:
- ❌ HIGH UPI traffic during attention (every token needs all positions)
- ❌ Poor performance for autoregressive generation
- ❌ Complex attention computation

### Option 3: **Sequence Affinity** (Recommended for Data Parallelism)
```cpp
// Each sequence "sticks" to one NUMA node  
Sequence A → Always processed on NUMA Node 0 → Cache allocated on Node 0
Sequence B → Always processed on NUMA Node 1 → Cache allocated on Node 1
```

**Pros**:
- ✅ Zero cross-node traffic for batch processing
- ✅ Simple implementation (already mostly works!)
- ✅ Optimal memory locality

**Cons**:
- ❌ Load balancing challenges
- ❌ Single sequence can't utilize multiple nodes

## 🎯 Recommended Implementation Strategy

### Phase 1: **Sequence Affinity** (Immediate)
Our current implementation already provides this! Here's why it works:

```cpp
// Current behavior with our NUMA buffer:
Context 1 (Sequence A) → NUMA Node 0 allocation → Node 0 threads process it
Context 2 (Sequence B) → NUMA Node 1 allocation → Node 1 threads process it
```

### Phase 2: **Intelligent Cache Placement** (Future)
```cpp
// Enhanced coordinator integration:
int optimal_node = coordinator->select_node_for_sequence(seq_id, cache_size);
ggml_backend_buffer_t buffer = ggml_backend_numa_buft_alloc_on_node(numa_buft, cache_size, optimal_node);
```

### Phase 3: **Cache Replication** (Advanced)
For scenarios requiring single sequence across multiple NUMA nodes.

## 📊 Performance Implications

### Data Parallelism (Multiple Sequences)
- **Memory Traffic**: ✅ Excellent - Each node accesses local cache only
- **UPI Utilization**: ✅ Minimal - Only for model weights and coordination  
- **Scaling**: ✅ Linear - Each node handles its sequences independently

### Single Sequence Processing
- **Current State**: ⚠️ All cache access on one node → bandwidth limited
- **With Replication**: ✅ All nodes can process different parts → maximum bandwidth
- **Trade-off**: ❌ N×memory usage but massive throughput gains

## 🔧 Implementation Status

### ✅ What Works Now
1. **NUMA-aware allocation**: KV cache allocated on specific nodes
2. **Sequence isolation**: Different sequences naturally use different cache instances  
3. **Fallback compatibility**: Works perfectly on single-NUMA systems

### 🚧 What Needs Enhancement  
1. **Node selection strategy**: Currently node selection is default, not optimized
2. **Cache replication**: For single-sequence multi-NUMA processing
3. **Coordinator integration**: Smarter cache placement decisions

### 🎉 Key Insight
**Our current implementation already provides the optimal solution for the most common use case**: batch processing with multiple sequences. Each sequence gets its own NUMA-local cache, eliminating cross-node traffic!

---

**Bottom Line**: Your NUMA-aware KV cache implementation will **dramatically reduce UPI traffic** for multi-sequence workloads while providing a solid foundation for more advanced cache strategies. The key insight is that llama.cpp's sequence-based architecture naturally aligns with NUMA node boundaries!
