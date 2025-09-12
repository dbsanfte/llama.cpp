#!/bin/bash

# Mass fix tensor->data patterns to tensor_data() function calls
# This script converts all remaining tensor->data accesses to use the new NUMA-compatible function

echo "🔧 Mass fixing tensor->data patterns..."

# Create backup directory
mkdir -p tensor_data_backup

# Files that need tensor->data fixes (from build failures)
files_to_fix=(
    "tests/test-rope.cpp"
    "tests/test-gguf.cpp"
    "examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp"
    "examples/eval-callback/eval-callback.cpp"
    "tools/quantize/quantize.cpp"
    "tools/imatrix/imatrix.cpp"
    "tools/cvector-generator/cvector-generator.cpp"
    "tools/cvector-generator/pca.hpp"
)

# Function to fix tensor->data patterns 
fix_tensor_data() {
    local file="$1"
    echo "  📝 Fixing $file..."
    
    # Backup original
    cp "$file" "tensor_data_backup/$(basename $file).bak"
    
    # Multiple sed patterns to handle different cases
    sed -i 's/result->data/tensor_data(result)/g' "$file"
    sed -i 's/tensor->data/tensor_data(tensor)/g' "$file"
    sed -i 's/t_input->data/tensor_data(t_input)/g' "$file"
    sed -i 's/output->data/tensor_data(output)/g' "$file"
    sed -i 's/t_layer->data/tensor_data(t_layer)/g' "$file"
    sed -i 's/t->data/tensor_data(t)/g' "$file"
    sed -i 's/ptr->data/tensor_data(ptr)/g' "$file"
    sed -i 's/cur->data/tensor_data(cur)/g' "$file"
    sed -i 's/p0->data/tensor_data(p0)/g' "$file"
    sed -i 's/p1->data/tensor_data(p1)/g' "$file"
    sed -i 's/p2->data/tensor_data(p2)/g' "$file"
    sed -i 's/r1->data/tensor_data(r1)/g' "$file"
    sed -i 's/r2->data/tensor_data(r2)/g' "$file"
    sed -i 's/counts->data/tensor_data(counts)/g' "$file"
    sed -i 's/sums->data/tensor_data(sums)/g' "$file"
    sed -i 's/src1->data/tensor_data(src1)/g' "$file"
    sed -i 's/in_sum2->data/tensor_data(in_sum2)/g' "$file"
    sed -i 's/diff->data/tensor_data(diff)/g' "$file"
    sed -i 's/diff_filtered->data/tensor_data(diff_filtered)/g' "$file"
    sed -i 's/last_eigenvector->data/tensor_data(last_eigenvector)/g' "$file"
    sed -i 's/dev_input->data/tensor_data(dev_input)/g' "$file"
    sed -i 's/t_read->data/tensor_data(t_read)/g' "$file"
    
    # Handle assignment patterns that need tensor_set_data instead
    sed -i 's/tensor_data(\([^)]*\)) = /tensor_set_data(\1, /g' "$file"
    sed -i 's/tensor_data(\([^)]*\))\s*=/tensor_set_data(\1, /g' "$file"
}

# Fix all files
for file in "${files_to_fix[@]}"; do
    if [[ -f "$file" ]]; then
        fix_tensor_data "$file"
        echo "    ✅ Fixed $file"
    else
        echo "    ⚠️  File not found: $file"
    fi
done

echo "🎉 Mass tensor->data fix complete!"
echo "📁 Backups stored in tensor_data_backup/"
echo "🔨 Running build test..."