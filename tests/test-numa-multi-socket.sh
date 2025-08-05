#!/bin/bash

# NUMA Multi-Socket Test Script
# This script tests the multi-socket NUMA functionality

echo "NUMA Multi-Socket Functionality Test"
echo "===================================="

# Check if the test executable exists
if [ ! -f "../build/bin/test-numa-multi-socket" ]; then
    echo "❌ Test executable not found. Build it first with:"
    echo "   cmake --build build --target test-numa-multi-socket"
    exit 1
fi

echo "Running basic functionality test..."
if ./build/bin/test-numa-multi-socket; then
    echo "✅ Basic test passed!"
else
    echo "❌ Basic test failed!"
    exit 1
fi

echo ""
echo "Testing with different environment configurations..."

# Test with different thread counts
for threads in 1 2 4 8; do
    echo "Testing with OMP_NUM_THREADS=$threads..."
    if OMP_NUM_THREADS=$threads ../build/bin/test-numa-multi-socket >/dev/null 2>&1; then
        echo "✅ Thread count $threads: OK"
    else
        echo "❌ Thread count $threads: FAILED"
    fi
done

echo ""
echo "=== Multi-Socket Code Path Status ==="
echo "The multi-socket matrix multiplication code is now available in:"
echo "  - ggml/src/ggml-cpu/ggml-cpu.c"
echo ""
echo "Key functions implemented:"
echo "  - ggml_compute_forward_mul_mat_multi_socket()"
echo "  - ggml_numa_socket_compute_mul_mat_chunk()"
echo "  - ggml_numa_socket_get_thread_count()"
echo ""
echo "The code will be used when:"
echo "  1. NUMA threadpools are created with numa_aware=true"
echo "  2. Multi-socket mode is enabled (enable_multi_socket=true)"
echo "  3. Matrix multiplication operations are performed"
echo ""