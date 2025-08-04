# Layer Assignment Visualization Enhancement

**Date:** August 4, 2025  
**Task:** Enhanced topology display to show detailed model layer assignments to GPUs/CPUs with realistic planning

## Summary

Enhanced the `--cpu-topology` command to display comprehensive layer assignment visualization that accurately reflects actual execution behavior based on GPU backend availability.

## Key Requirements Met

1. **Layer Assignment Visualization**: "When we print our numa topology, I'd like to also show which model layers are being proposed to be assigned to which GPUs / CPUs"

2. **Realistic Planning**: "The plan should always represent what actually ends up happening. If there are no GPUs, it makes no sense to try to show GPU offloading in the plan"

## Technical Implementation

### Enhanced Functions in `common/common.cpp`:

1. **`print_gpu_numa_topology()`**:
   - Added comprehensive layer distribution display
   - Shows realistic assignments based on GPU backend availability
   - Displays GPU detection with backend status
   - Provides helpful tips for GPU offloading

2. **`calculate_gpu_layer_distribution()`**:
   - Implements realistic layer assignment calculation
   - Checks GPU backend availability before proposing assignments
   - Only shows GPU assignments for GPUs with working backends
   - Falls back to CPU-only when no working GPUs available

3. **`get_model_layer_count()`**:
   - Parses GGUF files to determine actual layer count
   - Enables accurate layer distribution calculations
   - Handles cases where model file is not available

### Function Signatures Updated in `common/common.h`:

- Updated `calculate_gpu_layer_distribution()` to pass full `gpu_info` vector for backend checking
- Added `get_model_layer_count()` declaration

## Behavior Verification

### Test Cases Confirmed:

1. **CPU-only system**: ✅ Shows "GPU offloading: Disabled (CPU-only inference)"
2. **GPU without backend**: ✅ Shows "Backend: Not available" and realistic CPU-only planning
3. **User requests GPU layers**: ✅ Ignores unrealistic requests when no working GPUs available
4. **Topology display**: ✅ Shows comprehensive system overview with threading plan

### Key Features:

- **Realistic Planning**: Never shows GPU assignments when backends aren't available
- **User Guidance**: Provides helpful tips like "[TIP] Consider using -ngl <layers> to offload to GPU"
- **Backend Validation**: Checks actual GPU backend availability vs just device detection
- **Graceful Fallback**: Falls back to CPU-only inference when appropriate

## Testing Results

```bash
# Basic topology display
./build/bin/llama-server --cpu-topology
# Shows: "GPU offloading: Disabled (CPU-only inference)"

# With GPU layer request but no working backend
./build/bin/llama-server --cpu-topology --n-gpu-layers 10
# Shows: "Backend: Not available" and realistic CPU-only planning
```

## Code Quality

- ✅ Build successful with no compilation errors
- ✅ Maintains backward compatibility
- ✅ Follows existing code patterns and style
- ✅ Comprehensive error handling for missing model files
- ✅ Realistic behavior that matches actual execution

## Impact

This enhancement provides users with accurate, actionable information about their system's capabilities and how layers will actually be assigned during inference, eliminating confusion between user requests and actual execution behavior.
