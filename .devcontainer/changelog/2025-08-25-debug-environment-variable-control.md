# 2025-08-25: NUMA Debug Environment Variable Control System

## 🎯 Summary
Implemented comprehensive debug message control system to eliminate debug flooding during NUMA mirror mode inference while preserving diagnostic capabilities for development.

## ✅ Implementation Details

### Environment Variable Control System
- **`GGML_NUMA_DEBUG=0` or unset**: Clean, silent operation (default) - no debug messages
- **`GGML_NUMA_DEBUG=1`**: Debug messages enabled - shows NUMA strategy decisions and execution details  
- **`GGML_NUMA_DEBUG=2`**: Verbose debug messages - shows additional internal details

### Centralized Debug Macros
- **`NUMA_LOG_DEBUG()`**: Controlled debug messages respecting environment variable
- **`NUMA_LOG_VERBOSE()`**: Verbose debug messages for detailed troubleshooting
- **`ggml_numa_debug_enabled()`**: Central function checking environment variable state

### Files Modified
- **`ggml/src/ggml-cpu/ggml-numa-shared.h`**: Added debug control system and macros
- **`ggml/src/ggml-cpu/ggml-numa-executor.c`**: Converted printf to NUMA_LOG_DEBUG macros
- **`ggml/src/ggml-cpu/numa-kernels/numa-add-kernel.c`**: Converted ADD kernel debug messages
- **`ggml/src/ggml-cpu/ggml-numa-simple-coordinator.c`**: Converted all coordinator debug messages including:
  - Threadpool verification messages
  - NUMA binding verification messages  
  - Fallback threadpool creation messages
  - Dispatch thread lifecycle messages
- **`src/llama-mmap.cpp`**: Converted NUMA allocation success messages
- **`.github/copilot-instructions.md`**: Added comprehensive debug system documentation

## 🚀 Performance Impact
- **Debug flooding eliminated**: No printf overhead during normal inference
- **Clean inference loops**: Debug messages only appear when explicitly enabled
- **Production ready**: Default behavior is silent operation for maximum performance

## 🧪 Testing Verification
- ✅ **Clean mode**: No debug messages, fast inference performance restored
- ✅ **Debug mode**: Rich debugging information available when `GGML_NUMA_DEBUG=1`
- ✅ **llama-bench**: Completes successfully without debug flooding
- ✅ **llama-server**: Responds correctly to API requests without debug noise
- ✅ **Segfault resolved**: Tensor cleanup bypass prevents race condition crashes

## 📋 Usage Examples

### Production Inference (Silent)
```bash
./build/bin/llama-server -m model.gguf --numa mirror
```

### Development Debugging
```bash
GGML_NUMA_DEBUG=1 ./build/bin/llama-bench -m model.gguf --numa mirror
```

### Detailed Troubleshooting  
```bash
GGML_NUMA_DEBUG=2 ./build/bin/llama-server -m model.gguf --numa mirror
```

## 🎉 Result
User reported issue of "LOT of debug logging slowing down the inferencing loop" has been completely resolved. NUMA mirror mode inference now runs cleanly without debug flooding while maintaining full diagnostic capabilities when needed.
