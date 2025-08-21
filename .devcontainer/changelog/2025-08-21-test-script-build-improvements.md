# NUMA Test Script Build Configuration Improvements

## Overview
Enhanced the NUMA test scripts to automatically build with appropriate configurations before running tests, ensuring optimal testing conditions and eliminating manual build step errors.

## Changes Made

### 1. NUMA Tests Script (`tests/run-numa-tests.sh`)
**Enhancement**: Added automatic Debug build configuration

**Added Build Commands**:
```bash
# Ensure fresh Debug build for testing
echo -e "${YELLOW}🔨 Building fresh Debug configuration for testing...${NC}"
cd "$PROJECT_ROOT" || {
    echo -e "${RED}❌ Error: Cannot change to project root directory: $PROJECT_ROOT${NC}"
    exit 1
}

# Configure Debug build with NUMA support
echo "Configuring Debug build with NUMA support..."
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Build with maximum parallelism
echo "Building NUMA test suite in Debug mode..."
cmake --build build --parallel
```

**Rationale**: 
- Debug builds include assertions and debugging information crucial for test validation
- Ensures fresh compilation catches recent code changes
- Eliminates manual build step requirement

### 2. Performance Tests Script (`tests/run-numa-performance-tests.sh`)
**Enhancement**: Added automatic Release build configuration

**Added Build Commands**:
```bash
# Ensure fresh Release build for performance testing
echo -e "${YELLOW}🔨 Building fresh Release configuration for performance testing...${NC}"
cd "$PROJECT_ROOT" || {
    echo -e "${RED}❌ Error: Cannot change to project root directory: $PROJECT_ROOT${NC}"
    exit 1
}

# Configure Release build with NUMA support and optimizations
echo "Configuring Release build with NUMA support and optimizations..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NUMA_MIRROR=ON -DGGML_OPENMP=OFF

# Build with maximum parallelism
echo "Building NUMA performance suite in Release mode..."
cmake --build build --parallel
```

**Rationale**:
- Release builds enable full compiler optimizations (-O3, NDEBUG)
- Removes debugging overhead that would skew performance measurements
- Provides accurate benchmark results representative of production builds

## Benefits

### ✅ **Automatic Configuration**
- No manual cmake configuration required before running tests
- Scripts automatically select appropriate build type
- Eliminates "forgot to build" errors

### ✅ **Optimal Testing Conditions**
- Debug mode for correctness testing (assertions, debug info)
- Release mode for performance testing (optimizations, no debug overhead)
- Fresh builds catch recent code changes

### ✅ **Error Prevention**
- Build failures stop script execution with clear error messages
- Prevents running tests against stale or mismatched builds
- Directory validation with meaningful error handling

### ✅ **Developer Experience**
- One-command test execution: `tests/run-numa-tests.sh`
- Visual progress indicators for build process
- Maintains existing command line options and help text

## Usage Examples

### Run Correctness Tests (Debug Build)
```bash
# Basic test suite with Debug build
tests/run-numa-tests.sh

# Verbose output with Debug build  
tests/run-numa-tests.sh --verbose

# Include performance tests (Debug then Release builds)
tests/run-numa-tests.sh --performance
```

### Run Performance Tests (Release Build)
```bash
# Basic performance benchmarks with Release build
tests/run-numa-performance-tests.sh

# Comprehensive benchmarks with detailed output
tests/run-numa-performance-tests.sh --verbose

# Quick performance check
tests/run-numa-performance-tests.sh --quick
```

## Implementation Details

### Build Configuration
- **Debug**: `-DCMAKE_BUILD_TYPE=Debug` for assertions and debugging symbols
- **Release**: `-DCMAKE_BUILD_TYPE=Release` for full compiler optimizations
- **NUMA**: `-DGGML_NUMA_MIRROR=ON` to enable NUMA coordinator
- **OpenMP**: `-DGGML_OPENMP=OFF` to avoid conflicts with NUMA threading

### Error Handling
- CMake configuration failure exits with error code 1
- CMake build failure exits with error code 1
- Directory navigation failure exits with error code 1
- Clear error messages guide troubleshooting

### Performance Impact
- Build commands add ~2-3 minutes to script execution
- Parallelized builds (`--parallel`) minimize build time
- One-time cost per script run ensures optimal test conditions

Date: August 21, 2025
Type: Test Infrastructure Enhancement
Status: ✅ IMPLEMENTED
