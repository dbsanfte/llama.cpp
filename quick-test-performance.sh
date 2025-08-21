#!/bin/bash
# Quick test of redesigned performance benchmark - just TINY complexity
cd /workspaces/llama-cpp-dbsanfte-dev
timeout 60s ./build/bin/test-numa-performance-benchmark-add 2>&1 | \
  GGML_NUMA_COORDINATOR_DEBUG=0 \
  grep -A 100 "Performance Matrix.*Configuration" | \
  head -50
