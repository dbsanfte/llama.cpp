#!/bin/bash

# NUMA Performance Test Suite Runner
# Runs all NUMA performance benchmark tests and generates comprehensive reports
# Compares NUMA kernel performance against CPU fallback implementations

# Parse command line arguments
VERBOSE_MODE=false
QUICK_MODE=false
OPERATION_FILTER=""
OUTPUT_FORMAT="summary"  # summary, detailed, csv, json

for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --operation=*)
            OPERATION_FILTER="${arg#*=}"
            shift
            ;;
        --output=*)
            OUTPUT_FORMAT="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "NUMA Performance Benchmark Suite Runner"
            echo "Compares NUMA kernel performance against CPU fallback implementations"
            echo ""
            echo "Options:"
            echo "  --verbose         Show detailed benchmark output from each test"
            echo "  --quick           Run reduced test suite for faster results"
            echo "  --operation=OP    Run benchmarks only for specific operation (e.g., ADD, MUL_MAT)"
            echo "  --output=FORMAT   Output format: summary, detailed, csv, json (default: summary)"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Output formats:"
            echo "  summary    - Human-readable summary with key metrics"
            echo "  detailed   - Comprehensive results with statistical analysis"
            echo "  csv        - CSV format for spreadsheet analysis"
            echo "  json       - JSON format for programmatic processing"
            echo ""
            echo "Examples:"
            echo "  $0                              # Run all performance tests"
            echo "  $0 --verbose --operation=ADD   # Detailed ADD operation benchmarks"
            echo "  $0 --quick --output=csv        # Quick test suite with CSV output"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$BUILD_DIR/bin"

# Available performance benchmark tests (discovered automatically)
PERFORMANCE_TESTS=()

# Function to discover performance benchmark tests
discover_performance_tests() {
    echo -e "${CYAN}🔍 Discovering NUMA performance benchmark tests...${NC}"
    
    if [ ! -d "$BIN_DIR" ]; then
        echo -e "${RED}❌ Error: Binary directory not found: $BIN_DIR${NC}"
        echo "Please run 'cmake --build build' first to build the tests."
        exit 1
    fi
    
    # Find all performance benchmark binaries
    for binary in "$BIN_DIR"/test-numa-performance-benchmark-*; do
        if [ -f "$binary" ] && [ -x "$binary" ]; then
            test_name=$(basename "$binary")
            operation_name=$(echo "$test_name" | sed 's/test-numa-performance-benchmark-//')
            
            # Apply operation filter if specified (case-insensitive comparison)
            if [ -n "$OPERATION_FILTER" ]; then
                operation_upper=$(echo "$operation_name" | tr '[:lower:]' '[:upper:]')
                filter_upper=$(echo "$OPERATION_FILTER" | tr '[:lower:]' '[:upper:]')
                if [ "$operation_upper" != "$filter_upper" ]; then
                    continue
                fi
            fi
            
            PERFORMANCE_TESTS+=("$test_name")
            echo "  Found: $operation_name"
        fi
    done
    
    if [ ${#PERFORMANCE_TESTS[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No performance benchmark tests found matching criteria${NC}"
        if [ -n "$OPERATION_FILTER" ]; then
            echo "Filter: $OPERATION_FILTER"
        fi
        echo "Available tests should match pattern: test-numa-performance-benchmark-*"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Discovered ${#PERFORMANCE_TESTS[@]} performance benchmark test(s)${NC}"
    echo ""
}

# Statistics tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
PERFORMANCE_RESULTS=()

# Performance results structure: "test_name:avg_speedup:max_speedup:throughput_improvement"
declare -A SPEEDUP_RESULTS

echo -e "${BLUE}🚀 NUMA Performance Benchmark Suite Runner${NC}"
echo "=============================================="
echo "Project: llama.cpp NUMA improvements"
echo "Build directory: $BUILD_DIR"
if [ "$VERBOSE_MODE" = true ]; then
    echo "Output mode: Verbose (detailed benchmark output)"
else
    echo "Output mode: Summary (use --verbose for detailed output)"
fi
if [ "$QUICK_MODE" = true ]; then
    echo "Test mode: Quick (reduced test suite)"
else
    echo "Test mode: Comprehensive (full benchmark suite)"
fi
if [ -n "$OPERATION_FILTER" ]; then
    echo "Operation filter: $OPERATION_FILTER"
fi
echo "Output format: $OUTPUT_FORMAT"
echo ""

# Ensure fresh Release build for performance testing
echo -e "${YELLOW}🔨 Building fresh Release configuration for performance testing...${NC}"
cd "$PROJECT_ROOT" || {
    echo -e "${RED}❌ Error: Cannot change to project root directory: $PROJECT_ROOT${NC}"
    exit 1
}

# Configure Release build with NUMA support and optimizations
echo "Configuring Release build with NUMA support and optimizations..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF || {
    echo -e "${RED}❌ Error: CMake configuration failed${NC}"
    exit 1
}

# Build with maximum parallelism
echo "Building NUMA performance suite in Release mode..."
cmake --build build --parallel || {
    echo -e "${RED}❌ Error: CMake build failed${NC}"
    exit 1
}

echo -e "${GREEN}✅ Release build completed successfully${NC}"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Error: Build directory not found: $BUILD_DIR${NC}"
    echo "Please run 'cmake --build build' first to build the tests."
    exit 1
fi

# Discover available tests
discover_performance_tests

TOTAL_TESTS=${#PERFORMANCE_TESTS[@]}

# System information gathering
echo -e "${CYAN}📋 System Information${NC}"
echo "===================="
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
if command -v numactl &> /dev/null; then
    echo "NUMA nodes: $(numactl --hardware | grep 'available:' | awk '{print $2}')"
    echo "NUMA topology: $(numactl --hardware | grep 'node.*cpus' | wc -l) nodes"
fi
echo "Timestamp: $(date)"
echo ""

# Function to format duration with appropriate units
format_duration() {
    local duration_seconds="$1"
    if (( $(echo "$duration_seconds >= 60" | bc -l) )); then
        local minutes=$(echo "scale=1; $duration_seconds / 60" | bc -l)
        echo "${minutes}m"
    else
        printf "%.1fs" "$duration_seconds"
    fi
}

# Function to run a single performance benchmark test
run_performance_test() {
    local test_name="$1"
    local test_binary="$BIN_DIR/$test_name"
    local test_number=$((PASSED_TESTS + FAILED_TESTS + 1))
    local operation_name=$(echo "$test_name" | sed 's/test-numa-performance-benchmark-//')
    
    echo -e "${BLUE}🎯 Running performance benchmark $test_number/$TOTAL_TESTS: $operation_name${NC}"
    echo "=================================================="
    
    # Check if binary exists and is executable
    if [ ! -f "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not found: $test_binary${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$test_name:BINARY_NOT_FOUND")
        return 1
    fi
    
    if [ ! -x "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not executable: $test_binary${NC}"
        chmod +x "$test_binary" 2>/dev/null || {
            echo "Failed to make binary executable."
            FAILED_TESTS=$((FAILED_TESTS + 1))
            PERFORMANCE_RESULTS+=("$test_name:NOT_EXECUTABLE")
            return 1
        }
        echo "Made binary executable and retrying..."
    fi
    
    # Record start time
    local start_time=$(date +%s.%N)
    
    # Determine test arguments based on mode
    local test_args=""
    if [ "$VERBOSE_MODE" = true ]; then
        test_args="--verbose"
    fi
    if [ "$QUICK_MODE" = true ]; then
        test_args="$test_args --quick"
    fi
    
    # Run the performance benchmark with timeout (10 minutes max per test)
    local exit_code=0
    local output_file=$(mktemp)
    
    timeout 600 "$test_binary" $test_args > "$output_file" 2>&1
    exit_code=$?
    
    # Record end time
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    local formatted_duration=$(format_duration "$duration")
    
    # Parse results from output
    local speedup_info=""
    if [ $exit_code -eq 0 ]; then
        # Extract performance metrics from new 2D matrix format
        # Look for our new performance analysis section
        local avg_speedup=$(grep "Average speedup:" "$output_file" | awk '{print $3}' | sed 's/x//' | head -1)
        local best_speedup=$(grep "Best speedup:" "$output_file" | awk '{print $3}' | sed 's/x//' | head -1)
        local successful_tests=$(grep "Successful tests:" "$output_file" | awk '{print $3}' | cut -d'/' -f1 | head -1)
        local total_benchmark_tests=$(grep "Successful tests:" "$output_file" | awk '{print $3}' | cut -d'/' -f2 | head -1)
        
        # Alternative: look for inline speedup results from our new format
        if [ -z "$avg_speedup" ] || [ -z "$best_speedup" ]; then
            # Parse individual speedup results from "Speedup=X.XXx" patterns
            local speedups=$(grep -o "Speedup=[0-9.]*x" "$output_file" | sed 's/Speedup=//g' | sed 's/x//g')
            if [ -n "$speedups" ]; then
                # Calculate average speedup from individual results
                local sum=0
                local count=0
                local max_speedup=0
                while read -r speedup; do
                    if [ -n "$speedup" ] && [ "$speedup" != "0" ]; then
                        sum=$(echo "$sum + $speedup" | bc -l 2>/dev/null || echo "$sum")
                        count=$((count + 1))
                        max_speedup=$(echo "if ($speedup > $max_speedup) $speedup else $max_speedup" | bc -l 2>/dev/null || echo "$max_speedup")
                    fi
                done <<< "$speedups"
                
                if [ $count -gt 0 ]; then
                    avg_speedup=$(echo "scale=2; $sum / $count" | bc -l 2>/dev/null || echo "0")
                    best_speedup="$max_speedup"
                    successful_tests="$count"
                    total_benchmark_tests="$count"
                fi
            fi
        fi
        
        # Check for completion indicators from new format
        local completed_successfully=$(grep -c "completed successfully" "$output_file")
        if [ "$completed_successfully" -gt 0 ] && [ -n "$avg_speedup" ] && [ -n "$best_speedup" ]; then
            speedup_info="avg=${avg_speedup}x, best=${best_speedup}x"
            SPEEDUP_RESULTS["$operation_name"]="$avg_speedup:$best_speedup:$successful_tests:$total_benchmark_tests"
        elif [ "$completed_successfully" -gt 0 ]; then
            # Test completed but no performance metrics extracted - still consider success
            speedup_info="completed_no_metrics"
            SPEEDUP_RESULTS["$operation_name"]="1.0:1.0:1:1"
        else
            speedup_info="results_parse_failed"
        fi
    fi
    
    # Display test output based on verbosity
    if [ "$VERBOSE_MODE" = true ] || [ $exit_code -ne 0 ]; then
        cat "$output_file"
    else
        # Show summary for non-verbose mode - updated for new 2D matrix format
        if [ $exit_code -eq 0 ]; then
            # Show key sections from new output format
            grep -E "(🎯 Test Design|📊 Performance Matrix|⚙️.*Configuration|Speedup=.*x|🎯 ADD OPERATION PERFORMANCE ANALYSIS|Average speedup|Best speedup|completed successfully)" "$output_file" | head -20 || true
        else
            echo "Test output (last 10 lines):"
            tail -10 "$output_file"
        fi
    fi
    
    # Cleanup output file
    rm -f "$output_file"
    
    # Process results
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $operation_name performance benchmark completed successfully${NC}"
        echo "   Duration: $formatted_duration"
        if [ -n "$speedup_info" ]; then
            echo "   Performance: $speedup_info"
        fi
        PASSED_TESTS=$((PASSED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$test_name:SUCCESS:$speedup_info")
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ $operation_name performance benchmark timed out after 10 minutes${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$test_name:TIMEOUT")
    else
        echo -e "${RED}❌ $operation_name performance benchmark failed (exit code: $exit_code)${NC}"
        echo "   Duration: $formatted_duration"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$test_name:FAILED:$exit_code")
    fi
    
    echo ""
}

# Function to generate performance summary
generate_performance_summary() {
    echo -e "${PURPLE}📊 PERFORMANCE BENCHMARK SUMMARY${NC}"
    echo "=================================="
    echo "Total tests run: $TOTAL_TESTS"
    echo "Successful: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo ""
    
    if [ ${#SPEEDUP_RESULTS[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No performance results available${NC}"
        return
    fi
    
    echo -e "${CYAN}🚀 Performance Results by Operation:${NC}"
    printf "%-15s %-12s %-12s %-15s %s\n" "Operation" "Avg Speedup" "Best Speedup" "Success Rate" "Status"
    printf "%-15s %-12s %-12s %-15s %s\n" "---------" "-----------" "------------" "------------" "------"
    
    local total_avg_speedup=0
    local total_operations=0
    local excellent_count=0
    local good_count=0
    local poor_count=0
    
    for operation in "${!SPEEDUP_RESULTS[@]}"; do
        local result_data="${SPEEDUP_RESULTS[$operation]}"
        local avg_speedup=$(echo "$result_data" | cut -d':' -f1)
        local best_speedup=$(echo "$result_data" | cut -d':' -f2)
        local successful=$(echo "$result_data" | cut -d':' -f3)
        local total_benchmarks=$(echo "$result_data" | cut -d':' -f4)
        
        local success_rate="$successful/$total_benchmarks"
        local status=""
        
        # Categorize performance
        if (( $(echo "$avg_speedup >= 1.5" | bc -l) )); then
            status="🚀 Excellent"
            excellent_count=$((excellent_count + 1))
        elif (( $(echo "$avg_speedup >= 1.1" | bc -l) )); then
            status="✅ Good"
            good_count=$((good_count + 1))
        elif (( $(echo "$avg_speedup >= 0.9" | bc -l) )); then
            status="⚠️  Marginal"
            poor_count=$((poor_count + 1))
        else
            status="❌ Poor"
            poor_count=$((poor_count + 1))
        fi
        
        printf "%-15s %-12s %-12s %-15s %s\n" \
            "$operation" "${avg_speedup}x" "${best_speedup}x" "$success_rate" "$status"
        
        total_avg_speedup=$(echo "$total_avg_speedup + $avg_speedup" | bc -l)
        total_operations=$((total_operations + 1))
    done
    
    echo ""
    
    if [ $total_operations -gt 0 ]; then
        local overall_avg=$(echo "scale=2; $total_avg_speedup / $total_operations" | bc -l)
        echo -e "${CYAN}📈 Overall Performance Analysis:${NC}"
        echo "  Overall average speedup: ${overall_avg}x"
        echo "  Excellent performance (≥1.5x): $excellent_count operations"
        echo "  Good performance (≥1.1x): $good_count operations"
        echo "  Poor performance (<1.1x): $poor_count operations"
        echo ""
        
        if (( $(echo "$overall_avg >= 1.3" | bc -l) )); then
            echo -e "  ${GREEN}🎉 EXCELLENT: NUMA kernels show significant performance improvements!${NC}"
        elif (( $(echo "$overall_avg >= 1.1" | bc -l) )); then
            echo -e "  ${GREEN}✅ GOOD: NUMA kernels provide meaningful performance benefits${NC}"
        elif (( $(echo "$overall_avg >= 0.95" | bc -l) )); then
            echo -e "  ${YELLOW}⚠️  MARGINAL: NUMA kernels perform similarly to fallback${NC}"
        else
            echo -e "  ${RED}❌ POOR: NUMA kernels underperform fallback - optimization needed${NC}"
        fi
    fi
}

# Function to output results in different formats
output_results() {
    case "$OUTPUT_FORMAT" in
        "csv")
            echo "Operation,AvgSpeedup,BestSpeedup,SuccessfulTests,TotalTests,Status"
            for operation in "${!SPEEDUP_RESULTS[@]}"; do
                local result_data="${SPEEDUP_RESULTS[$operation]}"
                local avg_speedup=$(echo "$result_data" | cut -d':' -f1)
                local best_speedup=$(echo "$result_data" | cut -d':' -f2)
                local successful=$(echo "$result_data" | cut -d':' -f3)
                local total_benchmarks=$(echo "$result_data" | cut -d':' -f4)
                local status="Good"
                if (( $(echo "$avg_speedup >= 1.5" | bc -l) )); then
                    status="Excellent"
                elif (( $(echo "$avg_speedup < 1.1" | bc -l) )); then
                    status="Poor"
                fi
                echo "$operation,$avg_speedup,$best_speedup,$successful,$total_benchmarks,$status"
            done
            ;;
        "json")
            echo "{"
            echo "  \"timestamp\": \"$(date -Iseconds)\","
            echo "  \"total_tests\": $TOTAL_TESTS,"
            echo "  \"passed_tests\": $PASSED_TESTS,"
            echo "  \"failed_tests\": $FAILED_TESTS,"
            echo "  \"operations\": {"
            local first=true
            for operation in "${!SPEEDUP_RESULTS[@]}"; do
                if [ "$first" = false ]; then
                    echo ","
                fi
                first=false
                local result_data="${SPEEDUP_RESULTS[$operation]}"
                local avg_speedup=$(echo "$result_data" | cut -d':' -f1)
                local best_speedup=$(echo "$result_data" | cut -d':' -f2)
                local successful=$(echo "$result_data" | cut -d':' -f3)
                local total_benchmarks=$(echo "$result_data" | cut -d':' -f4)
                echo -n "    \"$operation\": {"
                echo -n "\"avg_speedup\": $avg_speedup, \"best_speedup\": $best_speedup, \"successful_tests\": $successful, \"total_tests\": $total_benchmarks"
                echo -n "}"
            done
            echo ""
            echo "  }"
            echo "}"
            ;;
        "detailed"|"summary"|*)
            generate_performance_summary
            ;;
    esac
}

# Main execution
echo -e "${CYAN}🏃 Running Performance Benchmark Tests${NC}"
echo "======================================="

# Run all discovered performance tests
for test_name in "${PERFORMANCE_TESTS[@]}"; do
    run_performance_test "$test_name"
done

echo -e "${BLUE}🏁 Performance Benchmark Suite Completed${NC}"
echo "==========================================="

# Generate final results
output_results

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All performance benchmarks completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ $FAILED_TESTS out of $TOTAL_TESTS performance benchmarks failed${NC}"
    exit 1
fi
