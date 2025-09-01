#!/bin/bash
echo "=== NUMA Operation Analysis ==="
echo

# Extract operation types and their NUMA support status
echo "🔍 Analyzing NUMA vs Fallback operations..."
echo

# Create temporary files for analysis
operations_file=$(mktemp)
support_file=$(mktemp)

# Extract operations and their support status
grep -n "Starting execution for" numa_server_debug.log | sed 's/.*Starting execution for \([^,]*\),.*/\1/' > $operations_file
grep -n "Query result - supported=" numa_server_debug.log | sed 's/.*supported=\([^,]*\),.*/\1/' > $support_file

# Combine operations with their support status
paste $operations_file $support_file | sort | uniq -c | sort -nr > operation_support_counts.txt

echo "📊 Operation counts with NUMA support status:"
echo "Format: COUNT OPERATION NUMA_SUPPORTED"
echo "==============================================="
cat operation_support_counts.txt

# Summary by NUMA support
echo
echo "📈 Summary by NUMA Support:"
echo "============================"
echo "✅ NUMA-Supported Operations:"
grep -E " true$" operation_support_counts.txt | awk '{print "   " $1 "x " $2 " (" $3 ")"}'

echo
 Fallback Operations:"echo "
grep -E " false$" operation_support_counts.txt | awk '{print "   " $1 "x " $2 " (" $3 ")"}'

# Clean up
rm $operations_file $support_file operation_support_counts.txt
