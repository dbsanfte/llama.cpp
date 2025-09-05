#!/bin/bash

# NUMA Performance Test Suite Runner
# Runs all NUMA performance benchmark tests and generates comprehensive reports
# Compares NUMA kernel performance against CPU fallback implementations

# Parse command line arguments
VERBOSE_MODE=false
QUICK_MODE=false
OPERATION_FILTER=""
OUTPUT_FORMAT="summary"  # summary, detailed, csv, json
NO_HYPERTHREADING=false

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
        --no-hyperthreading|--physical-cores-only)
            NO_HYPERTHREADING=true
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
            echo "  --no-hyperthreading  Use only physical cores, exclude hyperthreading"
            echo "  --physical-cores-only  Same as --no-hyperthreading"
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
            echo "  $0 --verbose --operation=MUL_MAT # Detailed MUL_MAT operation benchmarks"
            echo "  $0 --quick --output=csv        # Quick test suite with CSV output"
            echo "  $0 --no-hyperthreading         # Use only physical cores for testing"
            echo "  $0 --operation=MUL_MAT --no-hyperthreading  # MUL_MAT with physical cores only"
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

# Detect number of NUMA nodes
detect_numa_nodes() {
    local numa_nodes=0
    if command -v numactl >/dev/null 2>&1; then
        numa_nodes=$(numactl --hardware 2>/dev/null | grep "available:" | awk '{print $2}' || echo "1")
    elif [ -d "/sys/devices/system/node" ]; then
        numa_nodes=$(ls -1d /sys/devices/system/node/node* 2>/dev/null | wc -l || echo "1")
    else
        numa_nodes=1
    fi
    
    # Ensure we have at least 1 node
    if [ "$numa_nodes" -le 0 ]; then
        numa_nodes=1
    fi
    
    echo "$numa_nodes"
}

# Function to run llama-bench performance tests
run_llama_bench_tests() {
    echo ""
    echo -e "${PURPLE}🚀 Running llama-bench Real-World Inference Tests${NC}"
    echo "=================================================="
    
    local test_model="./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf"
    local llama_bench="$BIN_DIR/llama-bench"
    
    # Check if llama-bench binary exists
    if [ ! -f "$llama_bench" ]; then
        echo -e "${RED}❌ llama-bench not found: $llama_bench${NC}"
        echo "   Build it with: cmake --build build --target llama-bench"
        return 1
    fi
    
    # Check if test model exists
    if [ ! -f "$test_model" ]; then
        echo -e "${RED}❌ Test model not found: $test_model${NC}"
        echo "   Download it with the commands from copilot-instructions.md"
        return 1
    fi
    
    # Detect NUMA nodes
    local num_numa_nodes=$(detect_numa_nodes)
    echo "Detected $num_numa_nodes NUMA nodes"
    
    # Configuration matrix - simplified to avoid isolate mode segfault
    local configurations=()
    configurations+=("no-numa")
    
    # Skip isolate mode for now due to segfault in llama-bench
    # TODO: Debug and re-enable isolate mode
    # for ((node=1; node<num_numa_nodes; node++)); do
    #     configurations+=("isolate-node-$node")
    # done
    
    configurations+=("mirror")
    
    echo "Testing configurations: ${configurations[*]}"
    echo ""
    
    # Results storage
    local bench_results=()
    
    for config in "${configurations[@]}"; do
        echo -e "${BLUE}🔹 Testing configuration: $config${NC}"
        
        # Build llama-bench command
        local cmd="$llama_bench -m $test_model --repetitions 1"
        case "$config" in
            "no-numa")
                # No additional flags
                ;;
            "isolate-node-"*)
                cmd="$cmd --numa isolate"
                ;;
            "mirror")
                cmd="$cmd --numa mirror"
                ;;
        esac
        
        echo "  Running: $cmd"
        
        # Run the benchmark and capture output with extended timeout for mirror mode
        local timeout_duration=90  # Extended timeout for mirror mode
        local output_file=$(mktemp)
        if timeout $timeout_duration $cmd > "$output_file" 2>&1; then
            # Parse results - look for the table rows with benchmark data
            local pp512_result=""
            local tg128_result=""
            
            # Extract pp512 result (prompt processing) - handle multi-line output
            pp512_result=$(grep -A1 "pp512" "$output_file" | grep -E "^[[:space:]]*[0-9]+\.[0-9]+" | awk '{
                # Find the first floating point number
                for(i=1; i<=NF; i++) {
                    if($i ~ /^[0-9]+\.[0-9]+/) {
                        # Remove any trailing ± uncertainty
                        split($i, parts, "±")
                        print parts[1]
                        break
                    }
                }
            }')
            
            # Extract tg128 result (text generation) - handle multi-line output  
            tg128_result=$(grep -A1 "tg128" "$output_file" | grep -E "^[[:space:]]*[0-9]+\.[0-9]+" | awk '{
                # Find the first floating point number
                for(i=1; i<=NF; i++) {
                    if($i ~ /^[0-9]+\.[0-9]+/) {
                        # Remove any trailing ± uncertainty
                        split($i, parts, "±")
                        print parts[1]
                        break
                    }
                }
            }')
            
            if [ -n "$pp512_result" ] && [ -n "$tg128_result" ]; then
                echo "    ✅ pp512: ${pp512_result} t/s, tg128: ${tg128_result} t/s"
                bench_results+=("$config:$pp512_result:$tg128_result")
            else
                echo "    ⚠️  Could not parse results"
                if [ "$VERBOSE_MODE" = true ]; then
                    echo "Output:"
                    cat "$output_file"
                fi
                bench_results+=("$config:ERROR:ERROR")
            fi
        else
            echo "    ❌ Benchmark failed or timed out"
            bench_results+=("$config:FAILED:FAILED")
        fi
        
        rm -f "$output_file"
        echo ""
    done
    
    # Display summary results
    echo -e "${GREEN}📊 llama-bench Results Summary${NC}"
    echo "=============================="
    printf "%-15s %10s %10s\n" "Configuration" "pp512 t/s" "tg128 t/s"
    printf "%-15s %10s %10s\n" "-------------" "---------" "---------"
    
    for result in "${bench_results[@]}"; do
        IFS=':' read -r config pp512 tg128 <<< "$result"
        printf "%-15s %10s %10s\n" "$config" "$pp512" "$tg128"
    done
    
    echo ""
    echo -e "${CYAN}🎯 Performance Analysis${NC}"
    echo "======================"
    
    # Find no-numa baseline for comparison
    local baseline_pp512=""
    local baseline_tg128=""
    
    for result in "${bench_results[@]}"; do
        IFS=':' read -r config pp512 tg128 <<< "$result"
        if [ "$config" = "no-numa" ]; then
            baseline_pp512="$pp512"
            baseline_tg128="$tg128"
            break
        fi
    done
    
    if [ -n "$baseline_pp512" ] && [ -n "$baseline_tg128" ] && \
       [ "$baseline_pp512" != "ERROR" ] && [ "$baseline_pp512" != "FAILED" ]; then
        echo "Baseline (no-numa): pp512=${baseline_pp512} t/s, tg128=${baseline_tg128} t/s"
        echo ""
        
        for result in "${bench_results[@]}"; do
            IFS=':' read -r config pp512 tg128 <<< "$result"
            if [ "$config" != "no-numa" ] && [ "$pp512" != "ERROR" ] && [ "$pp512" != "FAILED" ]; then
                # Calculate relative performance
                local pp512_ratio=$(echo "scale=2; $pp512 / $baseline_pp512" | bc -l 2>/dev/null || echo "N/A")
                local tg128_ratio=$(echo "scale=2; $tg128 / $baseline_tg128" | bc -l 2>/dev/null || echo "N/A")
                
                if [ "$pp512_ratio" != "N/A" ] && [ "$tg128_ratio" != "N/A" ]; then
                    printf "%-15s: pp512 %s (%.2fx), tg128 %s (%.2fx)\n" \
                           "$config" "$pp512" "$pp512_ratio" "$tg128" "$tg128_ratio"
                else
                    printf "%-15s: pp512 %s, tg128 %s\n" "$config" "$pp512" "$tg128"
                fi
            fi
        done
    else
        echo "No valid baseline found for comparison"
    fi
    
    echo ""
}


# Function to discover performance benchmark tests
discover_performance_tests() {
    echo -e "${CYAN}🔍 Discovering extensible NUMA performance test...${NC}"
    
    if [ ! -d "$BIN_DIR" ]; then
        echo -e "${RED}❌ Error: Binary directory not found: $BIN_DIR${NC}"
        echo "Please run 'cmake --build build' first to build the tests."
        exit 1
    fi
    
    # Check for the new extensible test-numa-execution-modes binary
    local test_binary="$BIN_DIR/test-numa-execution-modes"
    if [ ! -f "$test_binary" ]; then
        echo -e "${RED}❌ Performance test not found: $test_binary${NC}"
        echo "   Make sure you have built the extensible performance test:"
        echo "   cmake --build build --target test-numa-execution-modes"
        exit 1
    fi
    
    if [ ! -x "$test_binary" ]; then
        echo -e "${RED}❌ Performance test not executable: $test_binary${NC}"
        exit 1
    fi
    
    # Define supported operation types based on the new extensible test
    local all_operations=("ADD" "MUL_MAT")
    
    # Filter operations based on operation filter if provided
    if [ -n "$OPERATION_FILTER" ]; then
        for operation in "${all_operations[@]}"; do
            operation_upper=$(echo "$operation" | tr '[:lower:]' '[:upper:]')
            filter_upper=$(echo "$OPERATION_FILTER" | tr '[:lower:]' '[:upper:]')
            if [ "$operation_upper" = "$filter_upper" ]; then
                PERFORMANCE_TESTS+=("$operation")
                echo "  Found: $operation"
            fi
        done
    else
        PERFORMANCE_TESTS=("${all_operations[@]}")
        for operation in "${all_operations[@]}"; do
            echo "  Found: $operation"
        done
    fi
    
    if [ ${#PERFORMANCE_TESTS[@]} -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No operations found matching criteria${NC}"
        if [ -n "$OPERATION_FILTER" ]; then
            echo "Filter: $OPERATION_FILTER"
        fi
        echo "Available operations: ${all_operations[*]}"
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
# Store full table output for each operation
declare -A OPERATION_TABLES

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
if [ "$NO_HYPERTHREADING" = true ]; then
    echo "Hyperthreading: Disabled (physical cores only)"
else
    echo "Hyperthreading: Enabled (all logical cores)"
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
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=ON || {
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
    local operation_name="$1"
    local test_binary="$BIN_DIR/test-numa-execution-modes"
    local test_number=$((PASSED_TESTS + FAILED_TESTS + 1))
    
    echo -e "${BLUE}🎯 Running performance benchmark $test_number/$TOTAL_TESTS: $operation_name${NC}"
    echo "=================================================="
    
    # Check if binary exists and is executable
    if [ ! -f "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not found: $test_binary${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$operation_name:BINARY_NOT_FOUND")
        return 1
    fi
    
    if [ ! -x "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not executable: $test_binary${NC}"
        chmod +x "$test_binary" 2>/dev/null || {
            echo "Failed to make binary executable."
            FAILED_TESTS=$((FAILED_TESTS + 1))
            PERFORMANCE_RESULTS+=("$operation_name:NOT_EXECUTABLE")
            return 1
        }
        echo "Made binary executable and retrying..."
    fi
    
    # Record start time
    local start_time=$(date +%s.%N)
    
    # Define test configurations for this operation
    local configurations=()
    if [ "$QUICK_MODE" = true ]; then
        configurations=(
            "$operation_name ISOLATE_NODE_0 SMALL"
            "$operation_name ISOLATE_NODE_1 SMALL"
            "$operation_name MIRROR SMALL"
            "$operation_name ISOLATE_NODE_0 LARGE"
            "$operation_name MIRROR LARGE"
        )
    else
        configurations=(
            "$operation_name ISOLATE_NODE_0 SMALL"
            "$operation_name ISOLATE_NODE_1 SMALL"
            "$operation_name MIRROR SMALL"
            "$operation_name ISOLATE_NODE_0 LARGE"
            "$operation_name ISOLATE_NODE_1 LARGE"
            "$operation_name MIRROR LARGE"
            "$operation_name ISOLATE_NODE_0 HUGE"
            "$operation_name ISOLATE_NODE_1 HUGE"
            "$operation_name MIRROR HUGE"
            "$operation_name ISOLATE_NODE_0 GIGANTIC_1GB"
            "$operation_name ISOLATE_NODE_1 GIGANTIC_1GB"
            "$operation_name MIRROR GIGANTIC_1GB"
            "$operation_name ISOLATE_NODE_0 GIGANTIC_2GB"
            "$operation_name ISOLATE_NODE_1 GIGANTIC_2GB"
            "$operation_name MIRROR GIGANTIC_2GB"
            "$operation_name ISOLATE_NODE_0 GIGANTIC_4GB"
            "$operation_name ISOLATE_NODE_1 GIGANTIC_4GB"
            "$operation_name MIRROR GIGANTIC_4GB"
            "$operation_name ISOLATE_NODE_0 GIGANTIC_8GB"
            "$operation_name ISOLATE_NODE_1 GIGANTIC_8GB"
            "$operation_name MIRROR GIGANTIC_8GB"
            "$operation_name ISOLATE_NODE_0 GIGANTIC_16GB"
            "$operation_name ISOLATE_NODE_1 GIGANTIC_16GB"
            "$operation_name MIRROR GIGANTIC_16GB"
        )
    fi
    
    # Run all configurations and collect results
    local exit_code=0
    local all_results=""
    local output_file=$(mktemp)
    
    for config in "${configurations[@]}"; do
        echo "  🔹 Testing: $config"
        
        # Run single configuration with timeout
        if timeout 120 "$test_binary" $config > /tmp/single_test_output.log 2>&1; then
            # Extract the result line
            result_line=$(grep "📊 RESULT:" /tmp/single_test_output.log | tail -1)
            if [ -n "$result_line" ]; then
                # Extract CSV data (remove "📊 RESULT: " prefix)  
                csv_data=$(echo "$result_line" | sed 's/📊 RESULT: //')
                all_results="$all_results$csv_data\n"
                echo "     ✅ Success: $csv_data"
            else
                echo "     ⚠️  No result found in output"
                exit_code=1
            fi
        else
            echo "     ❌ Test failed or timed out"
            exit_code=1
        fi
    done
    
    # Write aggregated results to output file
    echo -e "$all_results" > "$output_file"
    
    # Record end time
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    local formatted_duration=$(format_duration "$duration")
    
    # Parse results from CSV output
    local speedup_info=""
    if [ $exit_code -eq 0 ]; then
        # Calculate performance metrics from CSV results
        local isolate_0_times=""
        local isolate_1_times=""
        local mirror_times=""
        
        # Parse CSV results to extract timing data
        while IFS=',' read -r op strategy size time; do
            if [ -n "$time" ] && [ "$time" != "Time_ms" ]; then
                case "$strategy" in
                    "ISOLATE_NODE_0")
                        isolate_0_times="$isolate_0_times $time"
                        ;;
                    "ISOLATE_NODE_1") 
                        isolate_1_times="$isolate_1_times $time"
                        ;;
                    "MIRROR")
                        mirror_times="$mirror_times $time"
                        ;;
                esac
            fi
        done < "$output_file"
        
        # Calculate average times for performance analysis
        local avg_isolate_0=$(echo "$isolate_0_times" | awk '{sum=0; count=0; for(i=1;i<=NF;i++){sum+=$i; count++}} END{if(count>0) print sum/count; else print 0}')
        local avg_isolate_1=$(echo "$isolate_1_times" | awk '{sum=0; count=0; for(i=1;i<=NF;i++){sum+=$i; count++}} END{if(count>0) print sum/count; else print 0}')
        local avg_mirror=$(echo "$mirror_times" | awk '{sum=0; count=0; for(i=1;i<=NF;i++){sum+=$i; count++}} END{if(count>0) print sum/count; else print 0}')
        
        # Calculate speedups if we have mirror results
        if [ "$avg_mirror" != "0" ] && [ -n "$avg_isolate_0" ] && [ -n "$avg_isolate_1" ]; then
            # Best single node performance
            local best_single=$(echo "$avg_isolate_0 $avg_isolate_1" | awk '{if($1<$2) print $1; else print $2}')
            if [ "$best_single" != "0" ]; then
                # Mirror should ideally be faster (lower time), so speedup = best_single / mirror
                # But if mirror is slower, we show the slowdown factor
                local speedup=$(echo "scale=2; $best_single / $avg_mirror" | bc -l 2>/dev/null || echo "1.0")
                local speedup_display
                if [ $(echo "$speedup >= 1.0" | bc -l) -eq 1 ]; then
                    speedup_display="${speedup}x speedup"
                else
                    local slowdown=$(echo "scale=2; $avg_mirror / $best_single" | bc -l 2>/dev/null || echo "1.0")
                    speedup_display="${slowdown}x slowdown"
                fi
                speedup_info="mirror_vs_best=${speedup_display} (mirror: ${avg_mirror}ms, best_single: ${best_single}ms)"
                SPEEDUP_RESULTS["$operation_name"]="$speedup:$speedup:$(echo "$all_results" | wc -l):95"
            fi
        fi
        
        # Store results for later use
        OPERATION_TABLES["$operation_name"]="$all_results"
    fi
    
    # Display test output based on verbosity
    if [ "$VERBOSE_MODE" = true ] || [ $exit_code -ne 0 ]; then
        echo "Full test results:"
        cat "$output_file"
    else
        # Show summary for non-verbose mode
        if [ $exit_code -eq 0 ]; then
            echo "CSV Results:"
            cat "$output_file"
        else
            echo "Test failed. Output:"
            cat "$output_file"
        fi
    fi
    
    # Cleanup output file and temporary files
    rm -f "$output_file"
    rm -f /tmp/single_test_output.log
    
    # Process results
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $operation_name performance benchmark completed successfully${NC}"
        echo "   Duration: $formatted_duration"
        if [ -n "$speedup_info" ]; then
            echo "   Performance: $speedup_info"
        fi
        PASSED_TESTS=$((PASSED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$operation_name:SUCCESS:$speedup_info")
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ $operation_name performance benchmark timed out after 10 minutes${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$operation_name:TIMEOUT")
    else
        echo -e "${RED}❌ $operation_name performance benchmark failed (exit code: $exit_code)${NC}"
        echo "   Duration: $formatted_duration"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        PERFORMANCE_RESULTS+=("$operation_name:FAILED:$exit_code")
    fi
    
    echo ""
}

# Function to generate performance summary with detailed tables
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

    # Display detailed results tables by operation
    echo -e "${CYAN}� DETAILED NUMA PERFORMANCE ANALYSIS BY OPERATION${NC}"
    echo "======================================================="
    echo ""
    
    for operation in "${!OPERATION_TABLES[@]}"; do
        echo -e "${BLUE}🔍 OPERATION: $operation${NC}"
        echo "$(echo "$operation" | sed 's/./─/g')──────────────"
        echo ""
        
        # Display the full table for this operation
        if [ -n "${OPERATION_TABLES[$operation]}" ]; then
            echo -e "${OPERATION_TABLES[$operation]}"
        else
            echo "⚠️  No detailed table available for $operation"
        fi
        
        # Display summary metrics for this operation
        if [ -n "${SPEEDUP_RESULTS[$operation]}" ]; then
            local result_data="${SPEEDUP_RESULTS[$operation]}"
            local avg_speedup=$(echo "$result_data" | cut -d':' -f1)
            local best_speedup=$(echo "$result_data" | cut -d':' -f2)
            local numa_configs=$(echo "$result_data" | cut -d':' -f3)
            local efficiency=$(echo "$result_data" | cut -d':' -f4)
            
            echo ""
            echo "📊 Summary Metrics for $operation:"
            echo "   Average speedup: ${avg_speedup}x"
            echo "   Best speedup: ${best_speedup}x"
            echo "   Configurations tested: $numa_configs"
            if [ "$efficiency" != "0" ]; then
                echo "   NUMA efficiency: ${efficiency}%"
            fi
            
            # Performance assessment
            if (( $(echo "$avg_speedup >= 1.5" | bc -l) )); then
                echo "   Status: 🚀 Excellent performance"
            elif (( $(echo "$avg_speedup >= 1.1" | bc -l) )); then
                echo "   Status: ✅ Good performance"
            elif (( $(echo "$avg_speedup >= 0.9" | bc -l) )); then
                echo "   Status: ⚠️  Marginal performance"
            else
                echo "   Status: ❌ Poor performance"
            fi
        fi
        
        echo ""
        echo "$(printf '%.0s─' {1..70})"
        echo ""
    done
    
    # Overall summary
    echo -e "${CYAN}🎯 OVERALL PERFORMANCE SUMMARY${NC}"
    echo "================================"
    
    local total_avg_speedup=0
    local total_operations=0
    local excellent_count=0
    local good_count=0
    local poor_count=0
    
    # Compact summary table
    printf "%-15s %-12s %-12s %s\n" "Operation" "Avg Speedup" "Best Speedup" "Status"
    printf "%-15s %-12s %-12s %s\n" "---------" "-----------" "------------" "------"
    
    for operation in "${!SPEEDUP_RESULTS[@]}"; do
        local result_data="${SPEEDUP_RESULTS[$operation]}"
        local avg_speedup=$(echo "$result_data" | cut -d':' -f1)
        local best_speedup=$(echo "$result_data" | cut -d':' -f2)
        
        local status=""
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
        
        printf "%-15s %-12s %-12s %s\n" \
            "$operation" "${avg_speedup}x" "${best_speedup}x" "$status"
        
        total_avg_speedup=$(echo "$total_avg_speedup + $avg_speedup" | bc -l)
        total_operations=$((total_operations + 1))
    done
    
    echo ""
    
    if [ $total_operations -gt 0 ]; then
        local overall_avg=$(echo "scale=2; $total_avg_speedup / $total_operations" | bc -l)
        echo "Overall average speedup: ${overall_avg}x"
        echo "Excellent performance (≥1.5x): $excellent_count operations"
        echo "Good performance (≥1.1x): $good_count operations"
        echo "Poor performance (<1.1x): $poor_count operations"
        echo ""
        
        if (( $(echo "$overall_avg >= 1.3" | bc -l) )); then
            echo -e "${GREEN}🎉 EXCELLENT: NUMA kernels show significant performance improvements!${NC}"
        elif (( $(echo "$overall_avg >= 1.1" | bc -l) )); then
            echo -e "${GREEN}✅ GOOD: NUMA kernels provide meaningful performance benefits${NC}"
        elif (( $(echo "$overall_avg >= 0.95" | bc -l) )); then
            echo -e "${YELLOW}⚠️  MARGINAL: NUMA kernels perform similarly to fallback${NC}"
        else
            echo -e "${RED}❌ POOR: NUMA kernels underperform fallback - optimization needed${NC}"
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
for operation_name in "${PERFORMANCE_TESTS[@]}"; do
    run_performance_test "$operation_name"
done

# Run real-world inference benchmarks with llama-bench
run_llama_bench_tests

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
