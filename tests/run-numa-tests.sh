#!/bin/bash

# NUMA Test Suite Runner
# Runs all NUMA-related tests sequentially and reports results
# Exits with error code if any test fails

# Parse command line arguments
VERBOSE_MODE=false
RUN_PERFORMANCE_TESTS=false
for arg in "$@"; do
    case $arg in
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --performance)
            RUN_PERFORMANCE_TESTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--performance] [--help]"
            echo ""
            echo "Options:"
            echo "  --verbose      Show full test output (default: summary only)"
            echo "  --performance  Also run performance benchmark tests"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "By default, tests run in summary-only mode for cleaner output."
            echo "Use --verbose to see full test execution details."
            echo "Use --performance to include comprehensive performance benchmarks."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

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
    "test-numa-mathematical-correctness-add"
)

# Performance benchmark tests (separate category)
NUMA_PERFORMANCE_TESTS=(
    "test-numa-performance-benchmark-add"
)

# Statistics
TOTAL_TESTS=${#NUMA_TESTS[@]}
if [ "$RUN_PERFORMANCE_TESTS" = true ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + ${#NUMA_PERFORMANCE_TESTS[@]}))
fi
PASSED_TESTS=0
FAILED_TESTS=0
TEST_RESULTS=()

echo -e "${BLUE}🧪 NUMA Test Suite Runner${NC}"
echo "========================================"
echo "Project: llama.cpp NUMA improvements"
echo "Build directory: $BUILD_DIR"
echo "Total tests: $TOTAL_TESTS"
if [ "$RUN_PERFORMANCE_TESTS" = true ]; then
    echo "Performance tests: Enabled (${#NUMA_PERFORMANCE_TESTS[@]} tests)"
else
    echo "Performance tests: Disabled (use --performance to enable)"
fi
if [ "$VERBOSE_MODE" = true ]; then
    echo "Output mode: Full verbose output"
else
    echo "Output mode: Summary only (use --verbose for full output)"
fi
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
    
    # Determine test arguments based on verbosity mode
    local test_args=""
    if [ "$VERBOSE_MODE" = false ]; then
        test_args="--summary-only"
    fi
    
    # Run the test with timeout (5 minutes max per test)
    local exit_code=0
    if [ -n "$test_args" ]; then
        # In summary-only mode, suppress both stdout and stderr from NUMA debug logs
        timeout 300 "$test_binary" $test_args 2>/dev/null
    else
        timeout 300 "$test_binary"
    fi
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

# Function to run integration test with llama-server
run_integration_test() {
    echo "========================================"
    echo -e "${BLUE}🧪 NUMA Integration Test with llama-server${NC}"
    echo "========================================"
    
    local model_path="./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf"
    local server_port=8080
    local test_prompt="Hello!"
    local expected_response_pattern="Hello"
    local server_pid=""
    
    # Check if model exists, download if needed
    if [ ! -f "$model_path" ]; then
        echo "📥 Downloading test model..."
        wget -c -O "$model_path" https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to download test model${NC}"
            return 1
        fi
    fi
    
    echo "    🚀 Starting llama-server with NUMA mirror mode..."
    
    # Start llama-server in background with NUMA mirror mode
    "$BIN_DIR/llama-server" -m "$model_path" --host 0.0.0.0 --numa mirror-force --port $server_port > /tmp/llama-server.log 2>&1 &
    server_pid=$!
    
    # Function to cleanup server
    cleanup_server() {
        if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
            echo "🛑 Stopping llama-server (PID: $server_pid)..."
            kill "$server_pid" 2>/dev/null
            sleep 2
            # Force kill if still running
            if kill -0 "$server_pid" 2>/dev/null; then
                kill -9 "$server_pid" 2>/dev/null
            fi
        fi
        # Also kill any other llama-server processes
        pkill -f "llama-server.*numa.*mirror" 2>/dev/null || true
    }
    
    # Set up cleanup trap
    trap cleanup_server EXIT
    
    echo "⏳ Waiting for server to start..."
    local max_attempts=60  # Increased timeout for model loading
    local attempt=0
    
    # Wait for server to become available
    while [ $attempt -lt $max_attempts ]; do
        # Check if server process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo -e "\n${RED}❌ Server process died during startup (PID: $server_pid)${NC}"
            echo "Server log:"
            cat /tmp/llama-server.log 2>/dev/null || echo "No log file found"
            return 1
        fi
        
        if curl --silent --fail-with-body --show-error http://localhost:$server_port/ >/dev/null 2>&1; then
            echo "✅ Server is ready!"
            break
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    echo ""
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}❌ Server failed to start within 60 seconds${NC}"
        echo "Server log:"
        cat /tmp/llama-server.log 2>/dev/null || echo "No log file found"
        cleanup_server
        return 1
    fi
    
    echo "⏳ Waiting for model to finish loading..."
    local model_loaded=false
    local load_attempts=30
    local load_attempt=0
    
    # Wait for model to be fully loaded by testing API endpoint
    while [ $load_attempt -lt $load_attempts ]; do
        # Check if server process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo -e "\n${RED}❌ Server process died during model loading (PID: $server_pid)${NC}"
            echo "Server log:"
            cat /tmp/llama-server.log 2>/dev/null || echo "No log file found"
            return 1
        fi
        
        local health_response=$(curl -s -X POST http://localhost:$server_port/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "test"}], "max_tokens": 1}' 2>/dev/null)
        
        # Check if we get a proper response (not 503 loading error)
        if echo "$health_response" | grep -q "choices\|content" && ! echo "$health_response" | grep -q "Loading model"; then
            echo "✅ Model is fully loaded!"
            model_loaded=true
            break
        fi
        
        sleep 2
        load_attempt=$((load_attempt + 1))
        echo -n "."
    done
    echo ""
    
    if [ "$model_loaded" = false ]; then
        echo -e "${RED}❌ Model failed to load within 60 seconds${NC}"
        echo "Last response: $health_response"
        echo "Server log:"
        tail -20 /tmp/llama-server.log 2>/dev/null || echo "No log file found"
        cleanup_server
        return 1
    fi
    
    echo "🔍 Testing deterministic response generation..."
    echo "   Prompt: \"$test_prompt\""
    echo "   Expected: Response containing \"$expected_response_pattern\""
    
    # Make API request with temperature=0.0 for deterministic output
    local response=$(curl -s -X POST http://localhost:$server_port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen2.5-0.5b-instruct", 
            "messages": [{"role": "user", "content": "'"$test_prompt"'"}], 
            "max_tokens": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42
        }' 2>/dev/null)
    
    if [ $? -ne 0 ] || [ -z "$response" ]; then
        echo -e "${RED}❌ Failed to get response from server${NC}"
        echo "Server log:"
        tail -20 /tmp/llama-server.log 2>/dev/null || echo "No log file found"
        cleanup_server
        return 1
    fi
    
    echo "📄 Raw response:"
    echo "$response"
    echo ""
    
    # Extract the content from the JSON response
    local content=""
    
    # Try jq first, fallback to grep/sed if jq is not available
    if command -v jq >/dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.choices[0].message.content' 2>/dev/null)
    else
        # Fallback JSON parsing using grep and sed
        content=$(echo "$response" | grep -o '"content":"[^"]*"' | sed 's/"content":"//' | sed 's/"$//' | head -1)
    fi
    
    if [ -z "$content" ] || [ "$content" = "null" ]; then
        echo -e "${RED}❌ Invalid JSON response or missing content${NC}"
        cleanup_server
        return 1
    fi
    
    echo "💬 Generated content: \"$content\""
    
    # Check if response contains expected pattern (case-insensitive)
    if echo "$content" | grep -i "$expected_response_pattern" >/dev/null; then
        echo -e "${GREEN}✅ Integration test PASSED: Response contains expected pattern${NC}"
        echo "🎯 NUMA-enabled llama-server is working correctly!"
        cleanup_server
        return 0
    else
        echo -e "${RED}❌ Integration test FAILED: Response does not contain expected pattern${NC}"
        echo "   Expected pattern: \"$expected_response_pattern\""
        echo "   Actual content: \"$content\""
        cleanup_server
        return 1
    fi
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
    
    # Run performance benchmark tests if requested
    if [ "$RUN_PERFORMANCE_TESTS" = true ]; then
        echo ""
        echo -e "${BLUE}🚀 Running Performance Benchmark Tests${NC}"
        echo "======================================"
        for test_name in "${NUMA_PERFORMANCE_TESTS[@]}"; do
            # Continue running even if individual tests fail
            run_test "$test_name" || true
        done
    fi
    
    # Print final summary
    print_summary
    
    # Run integration test only if all individual tests passed
    if [ $FAILED_TESTS -eq 0 ]; then
        echo ""
        echo -e "${BLUE}🔗 All NUMA tests passed! Running integration test...${NC}"
        run_integration_test
        integration_exit_code=$?
        
        if [ $integration_exit_code -eq 0 ]; then
            echo -e "${GREEN}🎉 Integration test passed! NUMA system fully validated.${NC}"
            exit 0
        else
            echo -e "${RED}❌ Integration test failed!${NC}"
            exit 1
        fi
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
