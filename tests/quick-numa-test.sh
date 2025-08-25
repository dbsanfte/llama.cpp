#!/bin/bash

# Quick NUMA Performance Test - Just a few key configurations
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
TEST_EXECUTABLE="$BUILD_DIR/bin/test-numa-execution-modes"

echo "🚀 Quick NUMA Performance Test"
echo "=============================="

# Check if test executable exists
if [ ! -f "$TEST_EXECUTABLE" ]; then
    echo "❌ Test executable not found: $TEST_EXECUTABLE"
    exit 1
fi

# Results will be stored here
RESULTS_FILE="/tmp/numa_quick_test_$(date +%Y%m%d_%H%M%S).csv"
echo "Results will be written to: $RESULTS_FILE"
echo "Operation,Strategy,Size,Time_ms" > "$RESULTS_FILE"

# Quick test configurations
CONFIGURATIONS=(
    "ADD ISOLATE_NODE_0 SMALL"
    "ADD ISOLATE_NODE_1 SMALL"  
    "ADD MIRROR SMALL"
    "ADD ISOLATE_NODE_0 LARGE"
    "ADD MIRROR LARGE"
)

echo ""
echo "Running quick tests:"
echo "-------------------"

for config in "${CONFIGURATIONS[@]}"; do
    echo ""
    echo "🔹 Testing: $config"
    
    # Run the test with a timeout and capture output
    if timeout 30 "$TEST_EXECUTABLE" $config > /tmp/test_output.log 2>&1; then
        # Extract the result line
        result_line=$(grep "📊 RESULT:" /tmp/test_output.log | tail -1)
        if [ -n "$result_line" ]; then
            # Extract CSV data (remove "📊 RESULT: " prefix)  
            csv_data=$(echo "$result_line" | sed 's/📊 RESULT: //')
            echo "$csv_data" >> "$RESULTS_FILE"
            echo "   ✅ Success: $csv_data"
        else
            echo "   ❌ Failed: No result line found"
            echo "ADD,ERROR,ERROR,ERROR" >> "$RESULTS_FILE"
        fi
    else
        echo "   ❌ Failed: Test timeout or error"
        echo "ADD,ERROR,ERROR,ERROR" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "📊 Quick Results Summary"
echo "======================="

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    printf "%-12s %-15s %-8s %-12s\n" "Operation" "Strategy" "Size" "Time (ms)"
    echo "---------------------------------------------------"
    
    # Skip header line and print results
    tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r operation strategy size time; do
        if [ "$time" = "ERROR" ]; then
            printf "%-12s %-15s %-8s %-12s\n" "$operation" "$strategy" "$size" "❌ ERROR"
        else
            printf "%-12s %-15s %-8s %-12.3f\n" "$operation" "$strategy" "$size" "$time"
        fi
    done
    
    echo ""
    echo "✅ Quick test completed!"
    echo "📄 Full results: $RESULTS_FILE"
    
    # Show performance comparison
    echo ""
    echo "💡 Performance Notes:"
    node0_small=$(grep "ISOLATE_NODE_0,SMALL" "$RESULTS_FILE" | cut -d',' -f4 2>/dev/null || echo "N/A")
    node1_small=$(grep "ISOLATE_NODE_1,SMALL" "$RESULTS_FILE" | cut -d',' -f4 2>/dev/null || echo "N/A")
    mirror_small=$(grep "MIRROR,SMALL" "$RESULTS_FILE" | cut -d',' -f4 2>/dev/null || echo "N/A")
    
    if [[ "$node0_small" != "N/A" && "$node1_small" != "N/A" ]]; then
        echo "   Node 0 (SMALL): ${node0_small}ms"
        echo "   Node 1 (SMALL): ${node1_small}ms"
        if [[ "$mirror_small" != "N/A" ]]; then
            echo "   Mirror (SMALL): ${mirror_small}ms"
        fi
    fi
else
    echo "❌ Results file not found"
    exit 1
fi

# Clean up
rm -f /tmp/test_output.log
