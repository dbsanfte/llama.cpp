/**
 * @file mul_mat_id.c
 * @brief NUMA-aware expert-based matrix multiplication kernel implementation
 * @author David Sanftenberg
 * 
 * This implementation provides NUMA-optimized expert-based matrix multiplication
 * using the shared macro patterns for safe pointer arithmetic and boundary checking.
 */

#include "mul_mat_id.h"
#include "numa-kernels.h"
#include "ggml-numa-shared.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Matrix row mapping structure (from reference implementation)
struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

// Macro for matrix row mapping access (from reference implementation)
#define MMID_MATRIX_ROW(ids_tensor, matrix_rows, row_id, i1) matrix_rows[(row_id)*(ids_tensor)->ne[0]*(ids_tensor)->ne[1] + (i1)]

/**
 * @brief Utility function for aligned pointer increment
 * 
 * Helper function to advance work buffer pointer with proper alignment.
 * This matches the reference implementation pattern.
 */
static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

/**
 * @brief Process one chunk of expert-based matrix multiplication
 * 
 * This function handles the core mathematical computation for a specific chunk
 * of the expert matrix multiplication. Exactly replicates reference implementation.
 */
static void ggml_numa_kernel_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    // Get tensor dimensions and computation parameters (exactly like reference)
    const int64_t ne00 = src0->ne[0];
    const int64_t ne11 = src1->ne[1];
    const size_t nb01 = src0->nb[1];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];

    const enum ggml_type type = src0->type;
    ggml_vec_dot_t    const vec_dot      = ggml_get_type_traits_cpu(type)->vec_dot;
    enum ggml_type    const vec_dot_type = ggml_get_type_traits_cpu(type)->vec_dot_type;

    // Block processing parameters (from reference implementation)
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    NUMA_LOG_TRACE("one_chunk: cur_a=%ld, ir0=[%ld,%ld), ir1=[%ld,%ld), src1_cont=%d\n", 
                   cur_a, ir0_start, ir0_end, ir1_start, ir1_end, src1_cont);

    // Exact replication of reference implementation triple loop structure
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(ids, matrix_rows, cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                NUMA_LOG_TRACE("Row mapping: _i12=%ld -> id=%d, i11=%ld, i12=%ld, i1=%ld, i2=%ld\n", 
                               _i12, id, i11, i12, i1, i2);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) tensor_data(dst) + (i1*nb1 + i2*nb2));

                NUMA_LOG_TRACE("Pointers: src1_col offset=%ld, dst_col offset=%ld, wdata=%p, src1_cont=%d, type_convert=%d\n",
                               (const char*)src1_col - (const char*)wdata,
                               (char*)dst_col - (char*)tensor_data(dst),
                               wdata, src1_cont, (src1->type != vec_dot_type));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    NUMA_LOG_TRACE("vec_dot: ne00=%ld, src0_cur offset=%ld, src1_col ready\n", 
                                   ne00, ir0*nb01);
                    
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                    
                    NUMA_LOG_TRACE("vec_dot result: tmp[%ld] = %f\n", ir0 - iir0, tmp[ir0 - iir0]);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
                
                NUMA_LOG_TRACE("Copied %ld results to dst_col[%ld]\n", 
                               MIN(iir0 + blck_0, ir0_end) - iir0, iir0);
            }
        }
    }
}

/**
 * @brief Main execution function for MUL_MAT_ID NUMA kernel
 * 
 * Expert-based matrix multiplication with ID-based expert selection.
 * This implementation uses proper NUMA patterns with shared data access
 * and clear phase separation (type conversion + computation).
 */
enum ggml_status ggml_numa_kernel_mul_mat_id_execute(void * work_context, struct ggml_compute_params * params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *)work_context;
    
    // Input validation
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "src0 tensor cannot be null");
    NUMA_ASSERT(tensor->src[1] != NULL, "src1 tensor cannot be null");
    NUMA_ASSERT(tensor->src[2] != NULL, "ids tensor cannot be null");
    NUMA_ASSERT(params != NULL, "Compute params cannot be null");
    
    // Extract tensors
    const struct ggml_tensor * dst = tensor;
    const struct ggml_tensor * src0 = tensor->src[0];  // Expert weights [K, M, n_expert, 1]
    const struct ggml_tensor * src1 = tensor->src[1];  // Input data    [K, N, batch2, 1] 
    const struct ggml_tensor * ids = tensor->src[2];   // Expert ids    [n_expert_used, batch2]

    // Extract dimensions using GGML macro pattern
    GGML_TENSOR_BINARY_OP_LOCALS;

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;
    const bool src1_cont = ggml_is_contiguous(src1);

    const struct ggml_type_traits_cpu * type_traits = ggml_get_type_traits_cpu(type);
    enum ggml_type    const vec_dot_type    = type_traits->vec_dot_type;
    ggml_from_float_t const from_float      = type_traits->from_float;
    ggml_vec_dot_t    const vec_dot         = type_traits->vec_dot;

    // Critical assertions from reference implementation
    NUMA_ASSERT(nb00 == ggml_type_size(type), "src0 must not be permuted");
    NUMA_ASSERT(nb10 == ggml_type_size(src1->type), "src1 must not be permuted");
    NUMA_ASSERT(nb0 == sizeof(float), "dst must be F32");
    NUMA_ASSERT(vec_dot != NULL, "vec_dot function must be available");

    // Row groups (from reference implementation)
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    // Get shared result tensor data for direct writes
    float * dst_data;
    NUMA_GET_SHARED_DATA(tensor, dst_data, float);

    // PHASE 1: MULTITHREADED TYPE CONVERSION (PER NUMA NODE)
    // Each NUMA node converts full src1 tensor using multiple threads for speed
    char * wdata = NULL;
    if (src1->type != vec_dot_type) {
        wdata = (char *)params->wdata;
        NUMA_ASSERT(wdata != NULL, "Work buffer required for type conversion");
        
        ggml_from_float_t const from_float = ggml_get_type_traits_cpu(vec_dot_type)->from_float;
        NUMA_ASSERT(from_float != NULL, "from_float function not available");
        
        // Block-based conversion matching reference implementation exactly
        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;
        
        NUMA_LOG_TRACE("Type conversion: src1_type=%d, vec_dot_type=%d, ne10=%ld, ne11=%ld, ne12=%ld, ne13=%ld", 
                       (int)src1->type, (int)vec_dot_type, ne10, ne11, ne12, ne13);
        NUMA_LOG_TRACE("Type conversion strides: nbw0=%zu, nbw1=%zu, nbw2=%zu, nbw3=%zu", nbw0, nbw1, nbw2, nbw3);
        NUMA_LOG_TRACE("Type conversion thread: ith=%d, nth=%d", ith, nth);
        
        // MULTITHREADED conversion following reference implementation exactly
        // Key: preserve complete matrix structure, only slice innermost dimension (ne10)
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
        
        // BARRIER: All threads on this NUMA node must complete conversion before proceeding
        NUMA_OPENMP_BARRIER();
    } else {
        // No conversion needed - use original data
        wdata = (char *)tensor_data(src1);
    }

    // Get work data buffer and set up structures using remaining buffer space
    void * wdata_cur = (char *)params->wdata + (src1->type != vec_dot_type ? ggml_row_size(vec_dot_type, ggml_nelements(src1)) : 0);
    NUMA_ASSERT(wdata_cur != NULL, "Work data buffer cannot be null");

    // Matrix row counts and mappings (matching reference exactly)
    int64_t * matrix_row_counts = 
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = 
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    // Phase 2: Initialize matrix row mappings (critical for expert-based multiplication)
    if (ith == 0) {
        // Initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // Group rows by src0 matrix (expert mapping from reference implementation)
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) tensor_data(ids) + iid1*ids->nb[1] + id*ids->nb[0]);

                NUMA_ASSERT(i02 >= 0 && i02 < n_as, "Expert index out of bounds");

                MMID_MATRIX_ROW(ids, matrix_rows, i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // Barrier after matrix row mapping initialization
    #pragma omp barrier

    NUMA_LOG_DEBUG("MUL_MAT_ID Execute: dst=[%lld,%lld,%lld,%lld], n_experts=%d, n_ids=%d",
                   (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3, n_as, n_ids);

    // Phase 3: Process each expert matrix using NUMA-aware distribution
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;  // Skip experts with no assigned rows
        }

        // Calculate src0 pointer for this expert
        const char * src0_cur = (const char *)tensor_data(src0) + cur_a * nb02;
        const size_t row_size = src1_cont || src1->type != vec_dot_type ? 
                              ggml_row_size(vec_dot_type, ne10) : nb11;

        // Expert matrix dimensions
        const int64_t nr0 = ne01;  // Expert matrix rows
        const int64_t nr1 = cne1;  // Number of rows assigned to this expert
        
        // Simple NUMA-aware thread distribution
        const int64_t dr0 = (nr0 + nth - 1) / nth;
        
        const int64_t ir0_start = dr0 * ith;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);
        const int64_t ir1_start = 0;
        const int64_t ir1_end = nr1;
        
        // Skip if this thread has no work
        if (ir0_start >= ir0_end) continue;
        
        // Process this chunk - use corrected contiguous flag for converted data
        const bool wdata_is_contiguous = (src1->type == vec_dot_type) ? src1_cont : true; // Converted data is always contiguous
        ggml_numa_kernel_mul_mat_id_one_chunk(
            tensor, src0, src1, ids,
            cur_a, ir0_start, ir0_end, ir1_start, ir1_end,
            src0_cur, matrix_rows, row_size, wdata_is_contiguous, 
            wdata); // Use NUMA-converted data
    }
    
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Calculate work buffer size for MUL_MAT_ID operation
 * 
 * MUL_MAT_ID requires extensive work buffers for matrix row mapping,
 * type conversion, and thread coordination structures.
 */
size_t ggml_numa_kernel_mul_mat_id_work_buffer_calc(const struct ggml_tensor * tensor, int total_numa_nodes, int total_threads) {
    NUMA_ASSERT(tensor != NULL, "Tensor cannot be null");
    NUMA_ASSERT(tensor->src[0] != NULL, "src0 cannot be null");  
    NUMA_ASSERT(tensor->src[1] != NULL, "src1 cannot be null");
    NUMA_ASSERT(tensor->src[2] != NULL, "ids cannot be null");

    // Unused parameters (required by function signature)
    (void)total_numa_nodes;
    (void)total_threads;

    const struct ggml_tensor * src0 = tensor->src[0];
    const struct ggml_tensor * src1 = tensor->src[1];
    const struct ggml_tensor * ids = tensor->src[2];
    
    // Get type traits for src0 to determine required vec_dot_type
    const struct ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(src0->type);
    const enum ggml_type vec_dot_type = traits->vec_dot_type;
    const int n_as = src0->ne[2];  // Number of expert matrices
    
    size_t work_size = 0;
    
    // 1. Type conversion buffer for src1 (if needed)
    if (src1->type != vec_dot_type) {
        work_size += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
    }
    
    // 2. Matrix row mapping structures
    work_size += n_as * sizeof(int64_t) + sizeof(int64_t);  // matrix_row_counts
    work_size += n_as * ids->ne[0] * ids->ne[1] * sizeof(struct mmid_row_mapping) + sizeof(int64_t);  // matrix_rows
    
    NUMA_LOG_DEBUG("MUL_MAT_ID work buffer: src0_type=%d, src1_type=%d, vec_dot_type=%d, n_experts=%d, size=%zu bytes", 
                   (int)src0->type, (int)src1->type, (int)vec_dot_type, n_as, work_size);
    
    return work_size;
}

// Generate query and registration functions
NUMA_KERNEL_QUERY_FUNCTION(
    mul_mat_id, 
    4096, 
    65536
)

NUMA_KERNEL_REGISTRATION_FUNCTION_NO_AGG(
    mul_mat_id, 
    GGML_OP_MUL_MAT_ID, 
    "NUMA Expert Matrix Multiplication Kernel", 
    4096, 
    65536, 
    ggml_numa_kernel_mul_mat_id_execute
)
