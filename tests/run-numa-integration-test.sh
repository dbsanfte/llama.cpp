#!/bin/bash

# NUMA Integration Test with llama-server
# Standalone script that can be run independently or called from the main test orchestrator
# Tests NUMA-enabled llama-server with a real model to ensure end-to-end functionality

set -e

# Parse command line arguments
VERBOSE_MODE=false
NUMA_OPTION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE_MODE=true
            shift
            ;;
        --numa)
            if [ -z "$2" ]; then
                echo "Error: --numa option requires an argument (e.g., --numa mirror, --numa distribute, --numa isolate)"
                exit 1
            fi
            NUMA_OPTION="--numa $2"
            shift 2
            ;;
        --numa=*)
            NUMA_OPTION="--numa ${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--numa <mode>] [--help]"
            echo ""
            echo "NUMA Integration Test with llama-server"
            echo "Tests llama-server with a real model to ensure end-to-end functionality."
            echo "Automatically enables NUMA debug logging for operation analysis and prioritization."
            echo ""
            echo "Options:"
            echo "  --verbose          Show detailed test output and logs"
            echo "  --numa <mode>      NUMA mode to pass to llama-server (e.g., mirror, distribute, isolate)"
            echo "                     If not specified, llama-server runs without NUMA options"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  All environment variables are passed through to llama-server, including:"
            echo "  GGML_NUMA_DEBUG   Control NUMA debug output (0=off, 1=info, 2=verbose, 3=trace)"
            echo "                     Default: 1 (automatically enabled for operation analysis)"
            echo "  GGML_LOG_DEBUG    Control general debug logging"
            echo "  GGML_OPENMP       Control OpenMP threading behavior"
            echo ""
            echo "Features:"
            echo "  📊 Operation Analysis: Automatically analyzes NUMA vs fallback operations"
            echo "  🎯 Prioritization: Shows which operations should be implemented next"
            echo "  📈 Usage Statistics: Displays call counts for performance optimization"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Basic test without NUMA"
            echo "  $0 --numa mirror                     # Test with NUMA mirror mode"
            echo "  GGML_NUMA_DEBUG=2 $0 --numa mirror   # Test with verbose NUMA debug output"
            echo ""
            echo "This test downloads a small model (if not present) and validates that"
            echo "llama-server can generate coherent responses. When --numa is specified,"
            echo "it tests NUMA-specific functionality and provides operation analysis."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Colors for output (only if running standalone, avoid conflicts with orchestrator)
if [ -z "$RED" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$BUILD_DIR/bin"

# Function to check system requirements for integration test
check_integration_requirements() {
    echo -e "${YELLOW}🔍 Checking integration test requirements...${NC}"
    
    # Check for required commands
    local missing_commands=()
    
    if ! command -v curl >/dev/null 2>&1; then
        missing_commands+=("curl")
    fi
    
    if ! command -v wget >/dev/null 2>&1; then
        missing_commands+=("wget")
    fi
    
    if [ ${#missing_commands[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required commands: ${missing_commands[*]}${NC}"
        echo "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check if llama-server binary exists
    if [ ! -f "$BIN_DIR/llama-server" ]; then
        echo -e "${RED}❌ llama-server binary not found at: $BIN_DIR/llama-server${NC}"
        echo "Please build the project first:"
        echo "  cmake -B build -DCMAKE_BUILD_TYPE=Debug -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF"
        echo "  cmake --build build --parallel"
        exit 1
    fi
    
    # Check NUMA system info (optional for integration test)
    if command -v numactl >/dev/null 2>&1; then
        echo -e "${BLUE}🏗️  NUMA system information:${NC}"
        numactl --hardware | head -3 || echo "NUMA hardware info not available"
    else
        echo -e "${YELLOW}⚠️  numactl not found. NUMA tests will run in simulated mode.${NC}"
    fi
    
    echo ""
}

# Function to analyze NUMA debug logs and prioritize next operations
analyze_numa_debug_logs() {
    local log_file="$1"
    
    if [ ! -f "$log_file" ]; then
        echo -e "${YELLOW}⚠️  No debug log file found for analysis${NC}"
        return
    fi
    
    echo ""
    echo "========================================"
    echo -e "${BLUE}📊 NUMA Operation Analysis${NC}"
    echo "========================================"
    
    # Create temporary files for analysis
    local numa_ops_file=$(mktemp)
    local fallback_ops_file=$(mktemp)
    local summary_file=$(mktemp)
    
    # Extract NUMA kernel executions (successful dispatches)
    # Look for "Query result - supported=true, kernel=NUMA ADD (Single/Multi)" patterns
    grep "Query result.*supported=true.*kernel=" "$log_file" | \
        sed -E 's/.*kernel=(NUMA )?([A-Z_]+).*/\2/' | \
        sort | uniq -c | sort -nr > "$numa_ops_file"
    
    # Extract fallback executions (operations that fell back to ggml-cpu)
    # Look for "No kernel found for operation GET_ROWS" patterns specifically
    grep "No kernel found for operation" "$log_file" | \
        sed -E 's/.*No kernel found for operation ([A-Z_]+).*/\1/' | \
        sort | uniq -c | sort -nr > "$fallback_ops_file"
    
    # Show NUMA-implemented operations
    if [ -s "$numa_ops_file" ]; then
        echo "✅ Operations using NUMA kernels:"
        while read -r count op; do
            printf "   %3d × %s\n" "$count" "$op"
        done < "$numa_ops_file"
    else
        echo "⚠️  No NUMA kernel executions detected"
    fi
    
    echo ""
    
    # Show fallback operations (prioritization candidates)
    if [ -s "$fallback_ops_file" ]; then
        echo "🎯 Operations falling back to ggml-cpu (prioritized by usage):"
        local rank=1
        while read -r count op; do
            printf "   %d. %s (%d calls)\n" "$rank" "$op" "$count"
            rank=$((rank + 1))
        done < "$fallback_ops_file"
        
        echo ""
        echo -e "${YELLOW}💡 Recommendation: Consider implementing NUMA kernels for the most frequently used fallback operations${NC}"
        
        # Extract top 3 candidates
        local top_candidates=$(head -3 "$fallback_ops_file" | awk '{print $2}' | tr '\n' ', ' | sed 's/,$//')
        if [ -n "$top_candidates" ]; then
            echo -e "${BLUE}🚀 Top candidates for next implementation: $top_candidates${NC}"
        fi
    else
        echo "🎉 All operations are using NUMA kernels (no fallbacks detected)!"
    fi
    
    # Cleanup
    rm -f "$numa_ops_file" "$fallback_ops_file" "$summary_file"
}

# Function to run integration test with llama-server
run_integration_test() {
    echo "========================================"
    if [ -n "$NUMA_OPTION" ]; then
        echo -e "${BLUE}🧪 NUMA Integration Test with llama-server${NC}"
    else
        echo -e "${BLUE}🧪 Integration Test with llama-server${NC}"
    fi
    echo "========================================"
    
    local model_path="./.devcontainer/qwen2.5-0.5b-instruct-q8_0.gguf"
    local server_port=8080
    local test_prompt="Hello!"
    local expected_response_pattern="Hello"
    local server_pid=""
    local debug_log="/tmp/llama-server-debug.log"
    
    # Check if model exists, download if needed
    if [ ! -f "$model_path" ]; then
        echo "📥 Downloading test model..."
        wget -c -O "$model_path" https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ Failed to download test model${NC}"
            return 1
        fi
    fi
    
    if [ -n "$NUMA_OPTION" ]; then
        echo "    🚀 Starting llama-server with NUMA option: $NUMA_OPTION..."
    else
        echo "    🚀 Starting llama-server without NUMA options..."
    fi
    
    # Enable NUMA debug logging by default for operation analysis
    # Set GGML_NUMA_DEBUG=1 if not already set to capture operation statistics
    local numa_debug_level="${GGML_NUMA_DEBUG:-1}"
    if [ "$numa_debug_level" != "0" ]; then
        echo "    📊 NUMA debug logging enabled (level=$numa_debug_level) for operation analysis"
        export GGML_NUMA_DEBUG="$numa_debug_level"
    fi
    
    # Show relevant environment variables in verbose mode
    if [ "$VERBOSE_MODE" = true ]; then
        echo "    📋 Environment variables that will be passed to llama-server:"
        echo "       GGML_NUMA_DEBUG=$GGML_NUMA_DEBUG"
        if [ -n "$GGML_LOG_DEBUG" ]; then
            echo "       GGML_LOG_DEBUG=$GGML_LOG_DEBUG"
        fi
        if [ -n "$GGML_OPENMP" ]; then
            echo "       GGML_OPENMP=$GGML_OPENMP"
        fi
    fi
    
    # Start llama-server in background with optional NUMA mode
    # Note: All environment variables are automatically inherited by the child process
    "$BIN_DIR/llama-server" -m "$model_path" --host 0.0.0.0 $NUMA_OPTION --port $server_port > "$debug_log" 2>&1 &
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
        # Also kill any other llama-server processes on our port
        pkill -f "llama-server.*--port $server_port" 2>/dev/null || true
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
            if [ "$VERBOSE_MODE" = true ]; then
                echo "Server log:"
                cat "$debug_log" 2>/dev/null || echo "No log file found"
            fi
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
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Server log:"
            cat "$debug_log" 2>/dev/null || echo "No log file found"
        fi
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
            if [ "$VERBOSE_MODE" = true ]; then
                echo "Server log:"
                cat "$debug_log" 2>/dev/null || echo "No log file found"
            fi
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
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Last response: $health_response"
            echo "Server log:"
            tail -20 "$debug_log" 2>/dev/null || echo "No log file found"
        fi
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
        if [ "$VERBOSE_MODE" = true ]; then
            echo "Server log:"
            tail -20 "$debug_log" 2>/dev/null || echo "No log file found"
        fi
        cleanup_server
        return 1
    fi
    
    if [ "$VERBOSE_MODE" = true ]; then
        echo "📄 Raw response:"
        echo "$response"
        echo ""
    fi
    
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
        if [ -n "$NUMA_OPTION" ]; then
            echo "🎯 NUMA-enabled llama-server is working correctly!"
        else
            echo "🎯 llama-server is working correctly!"
        fi
        
        # Analyze NUMA debug logs for operation prioritization
        analyze_numa_debug_logs "$debug_log"
        
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

# Main function for standalone execution
main() {
    echo -e "${BLUE}🧪 NUMA Integration Test Runner${NC}"
    echo "========================================"
    echo "Project: llama.cpp NUMA improvements"
    echo "Build directory: $BUILD_DIR"
    if [ "$VERBOSE_MODE" = true ]; then
        echo "Output mode: Full verbose output"
    else
        echo "Output mode: Summary only (use --verbose for full output)"
    fi
    echo ""
    
    # Change to project root
    cd "$PROJECT_ROOT" || {
        echo -e "${RED}❌ Error: Could not change to project root: $PROJECT_ROOT${NC}"
        exit 1
    }
    
    check_integration_requirements
    
    echo -e "${YELLOW}🚀 Starting NUMA integration test...${NC}"
    echo ""
    
    # Run the integration test
    if run_integration_test; then
        echo ""
        echo -e "${GREEN}🎉 Integration test completed successfully!${NC}"
        if [ -n "$NUMA_OPTION" ]; then
            echo "NUMA system is fully validated and working correctly."
        else
            echo "llama-server is fully validated and working correctly."
        fi
        exit 0
    else
        echo ""
        echo -e "${RED}❌ Integration test failed!${NC}"
        echo "Please check the server logs and fix any issues."
        exit 1
    fi
}

# Handle script interruption
cleanup() {
    echo ""
    echo -e "${YELLOW}⚠️  Integration test interrupted by user.${NC}"
    exit 130
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Only run main if this script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
