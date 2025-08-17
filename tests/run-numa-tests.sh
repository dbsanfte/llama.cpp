#!/bin/bash

# NUMA Test Suite Runner
# Runs all NUMA-related tests sequentially and reports results
# Exits with error code if any test fails

# Don't use set -e as we need to capture test exit codes manually

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$BUILD_DIR/bin"

# Test binaries to run (in order of complexity)
NUMA_TESTS=(
    "test-numa-coordinator"
    "test-numa-coordinator-wait"
    "test-numa-dispatcher"
    "test-numa-mathematical-correctness"
    "test-numa-mathematical-correctness-soft-max"
    "test-numa-mathematical-correctness-rope"
    "test-numa-mathematical-correctness-add"
)

# Statistics
TOTAL_TESTS=${#NUMA_TESTS[@]}
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

echo -e "${BLUE}🧪 NUMA Test Suite Runner${NC}"
echo "========================================"
echo "Project: llama.cpp NUMA improvements"
echo "Build directory: $BUILD_DIR"
echo "Total tests: $TOTAL_TESTS"
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}❌ Error: Build directory not found: $BUILD_DIR${NC}"
    echo "Please run 'cmake --build build' first to build the tests."
    exit 1
fi

# Check if bin directory exists
if [ ! -d "$BIN_DIR" ]; then
    echo -e "${RED}❌ Error: Binary directory not found: $BIN_DIR${NC}"
    echo "Please run 'cmake --build build' first to build the tests."
    exit 1
fi

# Function to format duration with 2 decimal places and leading zero
format_duration() {
    local duration="$1"
    if [ "$duration" = "N/A" ]; then
        echo "N/A"
    else
        # Format to 2 decimal places and ensure leading zero
        printf "%.2f" "$duration" 2>/dev/null || echo "N/A"
    fi
}

# Function to run a single test
run_test() {
    local test_name="$1"
    local test_binary="$BIN_DIR/$test_name"
    local test_number=$((PASSED_TESTS + FAILED_TESTS + 1))
    
    echo -e "${BLUE}🎯 Running test $test_number/$TOTAL_TESTS: $test_name${NC}"
    echo "========================================="
    
    # Check if binary exists
    if [ ! -f "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not found: $test_binary${NC}"
        echo "Please ensure the test is built with CMake."
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("$test_name: BINARY_NOT_FOUND")
        return 1
    fi
    
    # Check if binary is executable
    if [ ! -x "$test_binary" ]; then
        echo -e "${RED}❌ Error: Test binary not executable: $test_binary${NC}"
        chmod +x "$test_binary" 2>/dev/null || {
            echo "Failed to make binary executable."
            FAILED_TESTS=$((FAILED_TESTS + 1))
            TEST_RESULTS+=("$test_name: NOT_EXECUTABLE")
            return 1
        }
        echo "Made binary executable and retrying..."
    fi
    
    # Record start time
    local start_time=$(date +%s.%N)
    
    # Run the test with timeout (5 minutes max per test)
    local exit_code=0
    timeout 300 "$test_binary"
    exit_code=$?
    
    # Record end time
    local end_time=$(date +%s.%N)
    local raw_duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    local duration=$(format_duration "$raw_duration")
    
    # Check results
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ PASSED${NC} ($test_name) - Duration: ${duration}s"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        TEST_RESULTS+=("$test_name: PASSED (${duration}s)")
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}❌ TIMEOUT${NC} ($test_name) - Exceeded 5 minutes"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("$test_name: TIMEOUT")
    else
        echo -e "${RED}❌ FAILED${NC} ($test_name) - Exit code: $exit_code, Duration: ${duration}s"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        TEST_RESULTS+=("$test_name: FAILED (exit_code=$exit_code, ${duration}s)")
    fi
    
    echo ""
    return $exit_code
}

# Function to print summary
print_summary() {
    echo "========================================"
    echo -e "${BLUE}📊 NUMA Test Suite Results${NC}"
    echo "========================================"
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    echo ""
    
    echo "Detailed Results:"
    echo "----------------"
    for result in "${TEST_RESULTS[@]}"; do
        if [[ "$result" == *"PASSED"* ]]; then
            echo -e "${GREEN}✅${NC} $result"
        else
            echo -e "${RED}❌${NC} $result"
        fi
    done
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 All NUMA tests passed successfully!${NC}"
        echo "The NUMA coordinator system is working correctly."
    else
        echo -e "${RED}💥 $FAILED_TESTS test(s) failed.${NC}"
        echo "Please review the failing tests and fix any issues."
    fi
}

# Function to check system requirements
check_requirements() {
    echo -e "${YELLOW}🔍 Checking system requirements...${NC}"
    
    # Check for required commands
    local missing_commands=()
    
    if ! command -v timeout >/dev/null 2>&1; then
        missing_commands+=("timeout")
    fi
    
    if ! command -v bc >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Warning: 'bc' not found. Test durations will show as 'N/A'.${NC}"
    fi
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required commands: ${missing_commands[*]}${NC}"
        echo "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check NUMA system info
    if command -v numactl >/dev/null 2>&1; then
        echo -e "${BLUE}🏗️  NUMA system information:${NC}"
        numactl --hardware | head -3 || echo "NUMA hardware info not available"
    else
        echo -e "${YELLOW}⚠️  numactl not found. NUMA tests will run in simulated mode.${NC}"
    fi
    
    echo ""
}

# Main execution
main() {
    check_requirements
    
    echo -e "${YELLOW}🚀 Starting NUMA test suite...${NC}"
    echo ""
    
    # Run all tests
    for test_name in "${NUMA_TESTS[@]}"; do
        # Continue running even if individual tests fail
        run_test "$test_name" || true
    done
    
    # Print final summary
    print_summary
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}⚠️  Test run interrupted by user.${NC}"
    print_summary
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if script is being run from the right location
cd "$PROJECT_ROOT" || {
    echo -e "${RED}❌ Error: Could not change to project root: $PROJECT_ROOT${NC}"
    exit 1
}

# Run main function
main "$@"
