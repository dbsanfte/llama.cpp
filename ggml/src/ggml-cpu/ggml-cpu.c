#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "ggml-quants.h" // for dequantize_row_q*_K

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// External thread-local variable for NUMA node binding
extern __thread int ggml_current_numa_node;
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>

#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef GGML_USE_OPENMP
#include <omp.h>

// Thread-local NUMA node assignment for OpenMP threads  
// Using static initialization to avoid syscalls in hot paths
static __thread int ggml_thread_numa_node = -1;
static __thread bool ggml_thread_numa_initialized = false;
#endif

#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif


#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int abort;         // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    bool cpumask[GGML_MAX_N_THREADS];
    int  last_graph;
    bool pending;
#endif
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

// ------------------------------------------------------------------------------------
// mul_mat profiling (experimental)
// Environment variables (read once):
//   GGML_MUL_MAT_PROFILE=1 enables profiling
//   GGML_MUL_MAT_PROFILE_VERBOSE=1 prints per-call summary when enabled
// Metrics collected:
//   - conversion_time_us (wall time spent converting src1 -> wdata) [thread 0]
//   - compute_time_us (wall time spent inside compute phase) [thread 0]
//   - tile_count (number of tiles processed; legacy path counts chunk invocations)
// Overhead when disabled: a couple of likely-optimized-out branch checks.

struct ggml_mul_mat_profile_state {
    atomic_int enabled_once; // 0 uninitialized, 1 disabled, 2 enabled
    atomic_int verbose_once; // same pattern for verbose
    atomic_long conversion_time_us; // accum per call (thread 0)
    atomic_long compute_time_us;    // accum per call (thread 0)
    atomic_long tile_count;         // incremented by all threads
    atomic_long flops_f64;          // approximate accumulated flops
    atomic_long k_iters;            // number of K-block iterations (F32 path)
    atomic_long panel_bytes;        // total panel bytes copied
    atomic_long panel_time_us;      // time spent packing panels (thread 0)
    atomic_long inner_time_us;      // time spent in inner FMAs (thread 0)
    atomic_long kblock_size_last;   // last observed kblock size
    atomic_long kblock_inactive;    // reason code: 0 active/unused, 1 quant_fallback (quant disabled), 2 misaligned kblock, 3 unsupported src0 type, 4 weights not *_K quant
    atomic_long fused_used;         // number of fused (SIMD or fused dequant) inner kernel invocations
};

static struct ggml_mul_mat_profile_state g_mmprof = {
    .enabled_once = 0,
    .verbose_once = 0,
    .conversion_time_us = 0,
    .compute_time_us = 0,
    .tile_count = 0,
    .flops_f64 = 0,
    .k_iters = 0,
    .panel_bytes = 0,
    .panel_time_us = 0,
    .inner_time_us = 0,
    .kblock_size_last = 0,
    .kblock_inactive = 0,
    .fused_used = 0,
};

// ------------------------------------------------------------------------------------
// mul_mat tiled fallback cache (experimental)
// If a shape (M,N,K) shows tiled performance significantly worse than legacy (kblock disabled),
// future calls for that shape will automatically use the legacy path unless user forces tiled.
// Controlled by env:
//   GGML_MUL_MAT_FALLBACK_DISABLE=1   -> disable this feature
//   GGML_MUL_MAT_FALLBACK_RATIO=0.90  -> minimum tiled/legacy GFLOP/s ratio (default 0.92)
//   GGML_MUL_MAT_TILED=1              -> still required to consider tiled path initially
//   GGML_MUL_MAT_FORCE_TILED=1        -> ignore fallback decision
// Cache size kept intentionally small to reduce memory footprint.

struct ggml_mul_mat_fallback_entry {
    int64_t m, n, k;
    float   legacy_gflops;  // best observed legacy GFLOP/s
    uint8_t decision;       // 0 unknown, 1 prefer tiled (good), 2 disable tiled
};

static struct ggml_mul_mat_fallback_entry g_mul_mat_fb_cache[32];
static atomic_flag g_mul_mat_fb_lock = ATOMIC_FLAG_INIT;
static int g_mul_mat_fb_inited = 0;
static float g_mul_mat_fb_ratio = 0.92f;
static bool g_mul_mat_fb_disabled = false;
static bool g_mul_mat_fb_force_tiled = false;

static inline void ggml_mul_mat_fallback_init_once(void) {
    if (g_mul_mat_fb_inited) return;
    const char * evd = getenv("GGML_MUL_MAT_FALLBACK_DISABLE");
    if (evd && (*evd=='1' || *evd=='y' || *evd=='Y' || *evd=='t' || *evd=='T')) g_mul_mat_fb_disabled = true;
    const char * evr = getenv("GGML_MUL_MAT_FALLBACK_RATIO");
    if (evr) {
        float v = strtof(evr, NULL);
        if (v > 0.1f && v < 0.999f) g_mul_mat_fb_ratio = v;
    }
    const char * evf = getenv("GGML_MUL_MAT_FORCE_TILED");
    if (evf && (*evf=='1' || *evf=='y' || *evf=='Y' || *evf=='t' || *evf=='T')) g_mul_mat_fb_force_tiled = true;
    g_mul_mat_fb_inited = 1;
}

static inline struct ggml_mul_mat_fallback_entry * ggml_mul_mat_fb_lookup(int64_t m, int64_t n, int64_t k) {
    for (size_t i = 0; i < sizeof(g_mul_mat_fb_cache)/sizeof(g_mul_mat_fb_cache[0]); ++i) {
        if (g_mul_mat_fb_cache[i].m == m && g_mul_mat_fb_cache[i].n == n && g_mul_mat_fb_cache[i].k == k) {
            return &g_mul_mat_fb_cache[i];
        }
    }
    return NULL;
}

static inline struct ggml_mul_mat_fallback_entry * ggml_mul_mat_fb_get_or_insert(int64_t m, int64_t n, int64_t k) {
    struct ggml_mul_mat_fallback_entry * e = ggml_mul_mat_fb_lookup(m,n,k);
    if (e) return e;
    // find empty or replace least legacy_gflops (simple heuristic)
    size_t repl = 0;
    float best = 1e30f;
    for (size_t i = 0; i < sizeof(g_mul_mat_fb_cache)/sizeof(g_mul_mat_fb_cache[0]); ++i) {
        if (g_mul_mat_fb_cache[i].m == 0 && g_mul_mat_fb_cache[i].n == 0 && g_mul_mat_fb_cache[i].k == 0) { repl = i; break; }
        if (g_mul_mat_fb_cache[i].legacy_gflops < best) { best = g_mul_mat_fb_cache[i].legacy_gflops; repl = i; }
    }
    g_mul_mat_fb_cache[repl].m = m; g_mul_mat_fb_cache[repl].n = n; g_mul_mat_fb_cache[repl].k = k;
    g_mul_mat_fb_cache[repl].legacy_gflops = 0.0f; g_mul_mat_fb_cache[repl].decision = 0;
    return &g_mul_mat_fb_cache[repl];
}

static inline void ggml_mul_mat_fb_record_legacy(int64_t m, int64_t n, int64_t k, float gflops) {
    if (g_mul_mat_fb_disabled || gflops <= 0.f) return;
    ggml_mul_mat_fallback_init_once();
    while (atomic_flag_test_and_set_explicit(&g_mul_mat_fb_lock, memory_order_acquire)) {}
    struct ggml_mul_mat_fallback_entry * e = ggml_mul_mat_fb_get_or_insert(m,n,k);
    if (gflops > e->legacy_gflops) e->legacy_gflops = gflops;
    atomic_flag_clear_explicit(&g_mul_mat_fb_lock, memory_order_release);
}

static inline bool ggml_mul_mat_fb_should_disable(int64_t m, int64_t n, int64_t k) {
    if (g_mul_mat_fb_disabled) return false; // feature off
    ggml_mul_mat_fallback_init_once();
    if (g_mul_mat_fb_force_tiled) return false; // user forced tiled
    struct ggml_mul_mat_fallback_entry * e = ggml_mul_mat_fb_lookup(m,n,k);
    if (!e) return false; // no data yet
    return e->decision == 2;
}

static inline void ggml_mul_mat_fb_consider_tiled(int64_t m, int64_t n, int64_t k, float tiled_gflops) {
    if (g_mul_mat_fb_disabled || g_mul_mat_fb_force_tiled) return;
    if (tiled_gflops <= 0.f) return;
    struct ggml_mul_mat_fallback_entry * e = ggml_mul_mat_fb_lookup(m,n,k);
    if (!e || e->legacy_gflops <= 0.f) return; // need baseline legacy first
    float ratio = tiled_gflops / e->legacy_gflops;
    if (ratio < g_mul_mat_fb_ratio) {
        e->decision = 2; // disable tiled
        GGML_LOG_INFO("mul_mat: fallback disabling tiled for shape %lldx%lldx%lld (tiled %.2f < %.2f * legacy %.2f)\n",
                      (long long)m, (long long)n, (long long)k, tiled_gflops, g_mul_mat_fb_ratio, e->legacy_gflops);
    } else if (ratio >= 1.0f) {
        e->decision = 1; // mark good
    }
}

static inline bool ggml_profiling_mul_mat_enabled(void) {
    int st = atomic_load_explicit(&g_mmprof.enabled_once, memory_order_relaxed);
    if (st == 0) {
        const char * ev = getenv("GGML_MUL_MAT_PROFILE");
        bool enable = ev && (*ev=='1' || *ev=='y' || *ev=='Y' || *ev=='t' || *ev=='T');
        atomic_store_explicit(&g_mmprof.enabled_once, enable ? 2 : 1, memory_order_relaxed);
        return enable;
    }
    return st == 2;
}

static inline bool ggml_profiling_mul_mat_verbose(void) {
    if (!ggml_profiling_mul_mat_enabled()) return false;
    int st = atomic_load_explicit(&g_mmprof.verbose_once, memory_order_relaxed);
    if (st == 0) {
        const char * ev = getenv("GGML_MUL_MAT_PROFILE_VERBOSE");
        bool enable = ev && (*ev=='1' || *ev=='y' || *ev=='Y' || *ev=='t' || *ev=='T');
        atomic_store_explicit(&g_mmprof.verbose_once, enable ? 2 : 1, memory_order_relaxed);
        return enable;
    }
    return st == 2;
}

static inline void ggml_profile_mul_mat_fused_used(void) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    atomic_fetch_add_explicit(&g_mmprof.fused_used, 1, memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_tile(void) {
    atomic_fetch_add_explicit(&g_mmprof.tile_count, 1, memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_conversion_time_acc(uint64_t t_start, uint64_t t_end) {
    if (t_start && t_end && t_end >= t_start) {
        atomic_fetch_add_explicit(&g_mmprof.conversion_time_us, (long)(t_end - t_start), memory_order_relaxed);
    }
}

static inline void ggml_profile_mul_mat_compute_time_acc(uint64_t t_start, uint64_t t_end) {
    if (t_start && t_end && t_end >= t_start) {
        atomic_fetch_add_explicit(&g_mmprof.compute_time_us, (long)(t_end - t_start), memory_order_relaxed);
    }
}

static inline void ggml_profile_mul_mat_flops_acc(int64_t m, int64_t n, int64_t k) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    // Approximate GEMM flops: 2 * M * N * K
    long double fl = 2.0L * (long double)m * (long double)n * (long double)k;
    if (fl > (long double)LONG_MAX) fl = (long double)LONG_MAX; // clamp
    atomic_fetch_add_explicit(&g_mmprof.flops_f64, (long)fl, memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_kiter(int64_t kblock_size) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    atomic_fetch_add_explicit(&g_mmprof.k_iters, 1, memory_order_relaxed);
    atomic_store_explicit(&g_mmprof.kblock_size_last, kblock_size, memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_panel_bytes(size_t bytes) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    atomic_fetch_add_explicit(&g_mmprof.panel_bytes, (long)bytes, memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_panel_time(uint64_t t0, uint64_t t1) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    if (t1 > t0) atomic_fetch_add_explicit(&g_mmprof.panel_time_us, (long)(t1 - t0), memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_inner_time(uint64_t t0, uint64_t t1) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    if (t1 > t0) atomic_fetch_add_explicit(&g_mmprof.inner_time_us, (long)(t1 - t0), memory_order_relaxed);
}

static inline void ggml_profile_mul_mat_kblock_inactive(int reason_code, int64_t kblock_size) {
    if (!ggml_profiling_mul_mat_enabled()) return;
    atomic_store_explicit(&g_mmprof.kblock_inactive, reason_code, memory_order_relaxed);
    atomic_store_explicit(&g_mmprof.kblock_size_last, kblock_size, memory_order_relaxed);
}

// SIMD helper (file scope)
#if defined(__AVX2__)
#include <immintrin.h>
static inline float ggml_dot_f32_f32_simd(const float * GGML_RESTRICT a, const float * GGML_RESTRICT b, int64_t n) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#else
static inline float ggml_dot_f32_f32_simd(const float * GGML_RESTRICT a, const float * GGML_RESTRICT b, int64_t n) {
    float acc = 0.f; for (int64_t i=0;i<n;++i) acc += a[i]*b[i]; return acc;
}
#endif

static void ggml_profile_mul_mat_report_once(void) {
    if (!ggml_profiling_mul_mat_enabled() || !ggml_profiling_mul_mat_verbose()) return;
    long conv = atomic_load_explicit(&g_mmprof.conversion_time_us, memory_order_relaxed);
    long comp = atomic_load_explicit(&g_mmprof.compute_time_us, memory_order_relaxed);
    long tiles = atomic_load_explicit(&g_mmprof.tile_count, memory_order_relaxed);
    long flops = atomic_load_explicit(&g_mmprof.flops_f64, memory_order_relaxed);
    double gflops_per_s = 0.0;
    if (flops > 0 && comp > 0) {
        gflops_per_s = ((double)flops / (double)comp) * 1e6 / 1e9;
    }
    long k_iters = atomic_exchange_explicit(&g_mmprof.k_iters, 0, memory_order_relaxed);
    long panel_bytes = atomic_exchange_explicit(&g_mmprof.panel_bytes, 0, memory_order_relaxed);
    long panel_time = atomic_exchange_explicit(&g_mmprof.panel_time_us, 0, memory_order_relaxed);
    long inner_time = atomic_exchange_explicit(&g_mmprof.inner_time_us, 0, memory_order_relaxed);
    long kblock_last = atomic_exchange_explicit(&g_mmprof.kblock_size_last, 0, memory_order_relaxed);
    long kblock_inactive = atomic_exchange_explicit(&g_mmprof.kblock_inactive, 0, memory_order_relaxed);
    long fused_used = atomic_exchange_explicit(&g_mmprof.fused_used, 0, memory_order_relaxed);
    double panel_bw_gb = panel_time ? (panel_bytes / 1e9) / (panel_time / 1e6) : 0.0;
    if (k_iters > 0 || kblock_last > 0 || kblock_inactive > 0) {
        GGML_LOG_INFO("mul_mat profile: conversion=%ld us compute=%ld us tiles=%ld flops=%ld perf=%.2f GFLOP/s kblock=%ld k_iters=%ld panel_bytes=%ld panel_time=%ld us inner_time=%ld us panel_bw=%.2f GB/s inactive=%ld fused=%ld\n",
                      conv, comp, tiles, flops, gflops_per_s, kblock_last, k_iters, panel_bytes, panel_time, inner_time, panel_bw_gb, kblock_inactive, fused_used);
    } else {
        GGML_LOG_INFO("mul_mat profile: conversion=%ld us compute=%ld us tiles=%ld flops=%ld perf=%.2f GFLOP/s fused=%ld\n", conv, comp, tiles, flops, gflops_per_s, fused_used);
    }
}

#if defined(__gnu_linux__)
// Helper: capture current thread affinity
static void ggml_numa_affinity_capture(cpu_set_t * out) {
    if (!out) return;
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), out);
}

// Helper: bind current thread to a single CPU
static bool ggml_numa_affinity_bind_single(uint32_t cpu, cpu_set_t * saved) {
    if (saved) ggml_numa_affinity_capture(saved);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set) == 0;
}

// Helper: restore previous affinity
static void ggml_numa_affinity_restore(const cpu_set_t * saved) {
    if (!saved) return;
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), saved);
}
#endif // __gnu_linux__

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (fill seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

// Static caching for NUMA thread binding to avoid syscalls in hot OpenMP paths
static void ggml_openmp_bind_thread_to_numa_node(int thread_id, int n_threads) {
    // Cache strategy check to avoid repeated calls
    static bool strategy_checked = false;
    static bool is_numa_mirror = false;
    static int num_numa_nodes = 0;
    
    if (!strategy_checked) {
        is_numa_mirror = (g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_MIRROR);
        if (is_numa_mirror) {
            num_numa_nodes = numa_max_node() + 1;
        }
        strategy_checked = true;
    }
    
    // Only apply binding in NUMA mirror mode with multiple nodes
    if (!is_numa_mirror || num_numa_nodes <= 1) {
        return;
    }

    // Check if this thread is already initialized to avoid repeated binding
    if (ggml_thread_numa_initialized) {
        return;
    }

    // Round-robin assignment of threads to NUMA nodes
    int target_numa_node = thread_id % num_numa_nodes;
    
    // Cache CPU masks statically to avoid repeated numa_allocate_cpumask() calls
    static struct bitmask *node_cpumasks[GGML_NUMA_MAX_NODES] = {0};
    static bool cpumasks_initialized = false;
    static cpu_set_t node_cpusets[GGML_NUMA_MAX_NODES];
    static bool cpusets_valid[GGML_NUMA_MAX_NODES] = {0};
    
    if (!cpumasks_initialized) {
        for (int node = 0; node < num_numa_nodes && node < GGML_NUMA_MAX_NODES; node++) {
            node_cpumasks[node] = numa_allocate_cpumask();
            if (node_cpumasks[node] && numa_node_to_cpus(node, node_cpumasks[node]) == 0) {
                // Convert NUMA bitmask to cpu_set_t for faster thread binding
                CPU_ZERO(&node_cpusets[node]);
                for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
                    if (numa_bitmask_isbitset(node_cpumasks[node], cpu)) {
                        CPU_SET(cpu, &node_cpusets[node]);
                    }
                }
                cpusets_valid[node] = true;
            }
        }
        cpumasks_initialized = true;
    }

    // Bind thread if we have a valid CPU set for the target node
    if (target_numa_node < GGML_NUMA_MAX_NODES && cpusets_valid[target_numa_node]) {
        if (sched_setaffinity(0, sizeof(cpu_set_t), &node_cpusets[target_numa_node]) == 0) {
            // Set memory allocation preference and thread-local node assignment
            numa_set_preferred(target_numa_node);
            ggml_thread_numa_node = target_numa_node;
            ggml_thread_numa_initialized = true;
            
            // Update the global thread-local variable for tensor data access
            ggml_current_numa_node = target_numa_node;
            
            // Debug output using standard GGML logging
            GGML_LOG_DEBUG("NUMA: Bound OpenMP thread %d to NUMA node %d (total threads: %d)\n", 
                           thread_id, target_numa_node, n_threads);
        }
    }
}

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        GGML_LOG_DEBUG("ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    unsigned int current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    // Return true if:
    // 1. Multiple physical NUMA nodes are present, AND
    // 2. User explicitly requested a NUMA strategy
    return g_state.numa.n_nodes > 1 && 
           g_state.numa.numa_strategy != GGML_NUMA_STRATEGY_DISABLED;
}

enum ggml_numa_strategy ggml_numa_get_strategy(void) {
    return g_state.numa.numa_strategy;
}

//
// NUMA-aware work buffer allocation:
// Previous implementation bound all pages to node 0. This adds an optional
// interleaved first-touch strategy to distribute the buffer across all NUMA
// nodes (potentially improving aggregate bandwidth when the buffer is accessed
// uniformly by threads on all nodes). Enable with GGML_NUMA_INTERLEAVE_WORK=1.
// Default remains node-0 allocation for backward compatibility & for cases
// where remote traffic would dominate.
//
void* ggml_numa_alloc_work_buffer(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        return NULL;
    }

    if (!ggml_is_numa()) {
        memset(ptr, 0, size);
        return ptr;
    }

#if defined(__gnu_linux__)
    static int  s_checked = 0;
    static bool s_interleave = false;
    if (!s_checked) {
        const char * env = getenv("GGML_NUMA_INTERLEAVE_WORK");
        if (env && (*env == '1' || *env == 'y' || *env == 'Y' || *env == 't' || *env == 'T')) {
            s_interleave = true;
        }
        s_checked = 1;
    }

    if (!s_interleave) {
        // Original node-0 binding path
        if (numa_available() >= 0) {
            unsigned long nodemask = 1UL; // Only node 0
            if (set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8) == 0) {
                memset(ptr, 0, size);
                set_mempolicy(MPOL_DEFAULT, NULL, 0);
                GGML_LOG_DEBUG("NUMA: work buffer on node 0 (size=%zu)\n", size);
                return ptr;
            }
        }
        memset(ptr, 0, size);
        GGML_LOG_DEBUG("NUMA: work buffer first-touch fallback (size=%zu)\n", size);
        return ptr;
    }

    // Interleaved first-touch path
    if (g_state.numa.n_nodes > 1) {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            size_t n_pages = (size + (size_t)page_size - 1) / (size_t)page_size;
            char * base = (char *)ptr;
            cpu_set_t original; ggml_numa_affinity_capture(&original);
            for (size_t p = 0; p < n_pages; ++p) {
                uint32_t node = p % g_state.numa.n_nodes;
                if (g_state.numa.nodes[node].n_cpus == 0) continue;
                uint32_t cpu = g_state.numa.nodes[node].cpus[0];
                if (ggml_numa_affinity_bind_single(cpu, NULL)) {
                    volatile char * dst = (volatile char *)(base + p * page_size);
                    dst[0] = 0; // first-touch
                }
            }
            ggml_numa_affinity_restore(&original);
            GGML_LOG_DEBUG("NUMA: work buffer interleaved across %u nodes (size=%zu)\n", g_state.numa.n_nodes, size);
            return ptr;
        }
    }
    // Fallback if something above failed
    memset(ptr, 0, size);
    GGML_LOG_DEBUG("NUMA: interleave fallback zero-init (size=%zu)\n", size);
    return ptr;
#else
    // Non-Linux: keep simple zeroing behavior
    memset(ptr, 0, size);
    return ptr;
#endif
}

void ggml_numa_free_work_buffer(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

#if defined(__ARM_ARCH)

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

static void ggml_init_arm_arch_features(void) {
#if defined(__linux__) && defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    ggml_arm_arch_features.sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#endif
}

#endif // __ARM_ARCH

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = (char *)tensor_data(tensor);

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = (char *)tensor_data(tensor);

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor_data(tensor)))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor_data(tensor)))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor_data(tensor)))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor_data(tensor)))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor_data(tensor)))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor_data(tensor)))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor_data(tensor) + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor_data(tensor) + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor_data(tensor)))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor_data(tensor)))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor_data(tensor)))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor_data(tensor)))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor_data(tensor)))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor_data(tensor)))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor_data(tensor)))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor_data(tensor)))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor_data(tensor) + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor_data(tensor) + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? tensor_data(src1) : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            // profiling: count tile enters
            if (ggml_profiling_mul_mat_enabled()) {
                ggml_profile_mul_mat_tile();
            }
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)tensor_data(src0) + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)tensor_data(dst) + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)tensor_data(src0) + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)tensor_data(src1) + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)tensor_data(dst) + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    uint64_t t_convert_start = 0;
    uint64_t t_convert_end   = 0;
    // Detect if src1 are quantized *_K weights; if so we skip F32 conversion.
    bool src1_is_k_quant_early = (
        src1->type == GGML_TYPE_Q2_K || src1->type == GGML_TYPE_Q3_K || src1->type == GGML_TYPE_Q4_K ||
        src1->type == GGML_TYPE_Q5_K || src1->type == GGML_TYPE_Q6_K || src1->type == GGML_TYPE_Q8_K);

    // Only perform upfront conversion when src1 is not quant weights and differs from vec_dot_type.
    if (src1->type != vec_dot_type && !src1_is_k_quant_early) {
        if (ggml_profiling_mul_mat_enabled()) t_convert_start = ggml_time_us();
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32); // still required when performing upfront conversion

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
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
    #endif
        if (ggml_profiling_mul_mat_enabled()) t_convert_end = ggml_time_us();
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);
    uint64_t t_compute_start = 0;
    if (ggml_profiling_mul_mat_enabled() && ith == 0) {
        t_compute_start = ggml_time_us();
        ggml_profile_mul_mat_conversion_time_acc(t_convert_start, t_convert_end);
    }

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? tensor_data(src1) : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)tensor_data(src0) + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)tensor_data(dst) + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // Optional experimental tiled static scheduling controlled by:
    //   GGML_MUL_MAT_TILED=1           (enable)
    //   GGML_MUL_MAT_TILE=Mt x Nt      (e.g. 64x128) override tile sizes
    bool use_tiled = false;
    {
        const char * ev = getenv("GGML_MUL_MAT_TILED");
        if (ev && (*ev=='1' || *ev=='y' || *ev=='Y' || *ev=='t' || *ev=='T')) use_tiled = true;
    }

    if (ith == 0) {
        const char * dbg = getenv("MFKB_DEBUG");
        if (dbg && (*dbg=='1' || *dbg=='y' || *dbg=='Y' || *dbg=='t' || *dbg=='T')) {
            long long dbgM = (long long)ne0;
            long long dbgN = (long long)(ne1 * ne2 * ne3);
            GGML_LOG_INFO("mul_mat debug-entry: M=%lld N=%lld K=%lld use_tiled=%d env_tiled=%s\n",
                          dbgM, dbgN, (long long)ne00, use_tiled, getenv("GGML_MUL_MAT_TILED"));
        }
    }

    const int64_t nr0 = ne0;                // M dimension
    const int64_t nr1 = ne1 * ne2 * ne3;     // Combined N dimension (flattened)

    // Temporary debug marker: verify this function is executed for micro-fused-kblock
    // Will emit once per call when env MFKB_MARK_ONCE is set. Remove after diagnosis.
    if (getenv("MFKB_MARK_ONCE")) {
        GGML_LOG_INFO("mul_mat debug-marker: entered ggml_compute_forward_mul_mat() M=%lld N=%lld K=%lld\n", (long long)nr0, (long long)nr1, (long long)ne00);
    }

    // Phase1: K-block scaffolding (env-controlled, no behavioral change yet except profiling awareness)
    static int s_kblock_checked = 0;
    static int64_t s_kblock_size = 0; // 0 means disabled
    static bool s_kblock_quant_enabled = false;
    static bool s_kblock_quant_fused = false; // experimental fused dequant+dot (currently SIMD inner kernel marker)
    if (!s_kblock_checked) {
        const char * ev = getenv("GGML_GEMM_KBLOCK");
        if (ev) {
            long val = strtol(ev, NULL, 10);
            if (val > 0) s_kblock_size = val;
        }
        const char * evq = getenv("GGML_GEMM_KBLOCK_QUANT");
        if (evq && (*evq=='1' || *evq=='y' || *evq=='Y' || *evq=='t' || *evq=='T')) {
            s_kblock_quant_enabled = true;
        }
        const char * evqf = getenv("GGML_GEMM_KBLOCK_QUANT_FUSED");
        if (evqf && (*evqf=='1' || *evqf=='y' || *evqf=='Y' || *evqf=='t' || *evqf=='T')) {
            s_kblock_quant_fused = true;
        }
        s_kblock_checked = 1;
        if (getenv("MFKB_DEBUG")) {
            GGML_LOG_INFO("mul_mat debug-init: kblock=%lld quant_enabled=%d fused_flag=%d (first init)\n", (long long)s_kblock_size, (int)s_kblock_quant_enabled, (int)s_kblock_quant_fused);
        }
    }

    // Development: allow late enabling of fused flag if env set after first initialization (hot reload scenario)
    if (!s_kblock_quant_fused) {
        const char * evqf_rt = getenv("GGML_GEMM_KBLOCK_QUANT_FUSED");
        if (evqf_rt && (*evqf_rt=='1' || *evqf_rt=='y' || *evqf_rt=='Y' || *evqf_rt=='t' || *evqf_rt=='T')) {
            s_kblock_quant_fused = true;
            if (getenv("MFKB_DEBUG")) GGML_LOG_INFO("mul_mat debug-runtime: fused flag enabled late\n");
        }
    }

    if (!use_tiled) {
        // legacy chunk-based path
        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) { chunk_size = 64; }
        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
        if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }
        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;
        int current_chunk = ith;
        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;
            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end   = MIN(ir0_start + dr0, nr0);
            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end   = MIN(ir1_start + dr1, nr1);
            int64_t num_rows_per_vec_dot = vec_dot_num_rows;
            if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
                num_rows_per_vec_dot = 1;
            }
            ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot,
                                                   ir0_start, ir0_end, ir1_start, ir1_end);
            if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                // Approximate flops for this sub-rectangle: (ir0_end-ir0_start)*(ir1_end-ir1_start)*K
                ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
            }
            if (nth >= nchunk0 * nchunk1) break;
            current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
        }
        if (ggml_profiling_mul_mat_enabled() && ith == 0) {
            uint64_t t_end = ggml_time_us();
            ggml_profile_mul_mat_compute_time_acc(t_compute_start, t_end);
            // Derive GFLOP/s snapshot before report (compute_time_us just added)
            long comp = atomic_load_explicit(&g_mmprof.compute_time_us, memory_order_relaxed);
            long flops = atomic_load_explicit(&g_mmprof.flops_f64, memory_order_relaxed);
            if (comp > 0 && flops > 0) {
                double gflops = ((double)flops / (double)comp) * 1e6 / 1e9;
                // Only record for kblock disabled shapes (pure legacy baseline)
                ggml_mul_mat_fb_record_legacy(nr0, nr1, ne00, (float)gflops);
            }
            ggml_profile_mul_mat_report_once();
        }
        return;
    }

    // Tiled static path
    if (use_tiled && ggml_mul_mat_fb_should_disable(nr0, nr1, ne00)) {
        use_tiled = false; // revert to legacy for this call
        if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
            GGML_LOG_INFO("mul_mat: fallback reverting to legacy for shape %lldx%lldx%lld\n", (long long)nr0, (long long)nr1, (long long)ne00);
        }
    }
    // Heuristic base tile sizes; may tune later or make arch dependent.
    int64_t tile_m = 64; // rows of A (M)
    int64_t tile_n = 64; // flattened columns (N)
    bool tile_overridden = false;
    {
        const char * ev = getenv("GGML_MUL_MAT_TILE");
        if (ev) {
            // Parse simple "MxN" pattern
            long tm = 0, tn = 0;
            if (sscanf(ev, "%ldx%ld", &tm, &tn) == 2 && tm > 0 && tn > 0) {
                tile_m = tm; tile_n = tn; tile_overridden = true;
            }
        }
    }

    // Adjust for very small dimensions
    if (nr0 < tile_m) tile_m = nr0;
    if (nr1 < tile_n) tile_n = nr1;

    // Light heuristic: if N is huge relative to M (common in inference with small batch), shrink M tile
    if (nr0 <= 32 && nr1 >= 1024) tile_m = (nr0 + nth - 1) / nth; // distribute M minimally
    if (tile_m < 1) tile_m = 1;
    if (tile_n < 1) tile_n = 1;

    // Defer tile count computation until after kblock heuristic may adjust tile_n

    // If K-block enabled but absurdly large, clamp to full K (effectively disable)
    int64_t kblock = (s_kblock_size > 0 ? s_kblock_size : 0);
    if (kblock > ne00) kblock = 0;
    // Experimental upper cap (per request) - do not allow kblock > 768 for now to limit panel size & memory traffic
    if (kblock > 768) kblock = 768;
    // Apply tile_n expansion heuristic now that kblock is known
    if (kblock > 0 && !tile_overridden) {
        long factor = 0;
        const char * evf = getenv("GGML_MUL_MAT_TILE_N_FACTOR");
        if (evf) {
            long f = strtol(evf, NULL, 10);
            if (f > 0 && f <= 8) factor = f; // safety clamp
        }
        if (factor == 0) factor = 2; // default heuristic multiplier
        size_t panel_cap = 1024 * 1024; // 1MB default
        const char * evc = getenv("GGML_MUL_MAT_PANEL_CAP");
        if (evc) {
            long pc = strtol(evc, NULL, 10);
            if (pc > 16*1024 && pc < 16*1024*1024) panel_cap = (size_t)pc; // 16KB..16MB range
        }
        int64_t proposed = tile_n * factor;
        if (proposed > nr1) proposed = nr1;
        long double panel_bytes_ld = (long double)kblock * (long double)proposed * sizeof(float);
        if (panel_bytes_ld <= (long double)panel_cap) {
            tile_n = proposed;
        } else {
            while (factor > 1) {
                factor /= 2;
                proposed = tile_n * factor;
                if (proposed < tile_n) break;
                panel_bytes_ld = (long double)kblock * (long double)proposed * sizeof(float);
                if (panel_bytes_ld <= (long double)panel_cap) { tile_n = proposed; break; }
            }
        }
    }
    // Simple heuristic default if user only set GGML_GEMM_KBLOCK=auto in future (not yet): none.

    // Now compute tile counts (after any adjustment)
    const int64_t ntiles_m = (nr0 + tile_m - 1) / tile_m;
    const int64_t ntiles_n = (nr1 + tile_n - 1) / tile_n;
    const int64_t total_tiles = ntiles_m * ntiles_n;

    // If user requested kblock but data types not F32xF32 and quant kblock disabled -> record inactive
    if (kblock > 0) {
        bool is_f32_pair = (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
        bool is_quant_pair = !is_f32_pair;
        if (is_quant_pair && !s_kblock_quant_enabled) {
            if (ith == 0) ggml_profile_mul_mat_kblock_inactive(1, kblock); // quant fallback
        }
    }

    // Static assignment: spread tiles roughly evenly across threads
    for (int64_t tile_idx = ith; tile_idx < total_tiles; tile_idx += nth) {
        const int64_t tm = tile_idx % ntiles_m;
        const int64_t tn = tile_idx / ntiles_m;
        const int64_t ir0_start = tm * tile_m;
        const int64_t ir0_end   = MIN(ir0_start + tile_m, nr0);
        const int64_t ir1_start = tn * tile_n;
        const int64_t ir1_end   = MIN(ir1_start + tile_n, nr1);
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }

        if (kblock == 0) {
            // Original full-K behavior
            ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot,
                                                   ir0_start, ir0_end, ir1_start, ir1_end);
            if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
            }
        // Extended inactive reasons (codes):
        // 1 = quant path disabled (existing)
        // 2 = misaligned kblock (quant *_K types require kblock % QK_K == 0)
        // 3 = activation (src0) type unsupported for quant partial-K (needs F32 currently)
        // 4 = weight (src1) not a *_K quant type while quant kblock requested
        // We record these once per call (thread 0) so user sees why extended metrics are absent.
        if (kblock > 0 && s_kblock_quant_enabled && ith == 0) {
            bool weights_k_quant = (
                src1->type == GGML_TYPE_Q2_K || src1->type == GGML_TYPE_Q3_K || src1->type == GGML_TYPE_Q4_K ||
                src1->type == GGML_TYPE_Q5_K || src1->type == GGML_TYPE_Q6_K || src1->type == GGML_TYPE_Q8_K);
            if (!weights_k_quant && !(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32)) {
                // Requested quant kblock but weights not *_K (and not pure F32 pair)
                ggml_profile_mul_mat_kblock_inactive(4, kblock);
            } else if (weights_k_quant) {
    #ifdef QK_K
                const int QK_REQUIRED_CHECK = QK_K;
    #else
                const int QK_REQUIRED_CHECK = 256; // fallback assumption
    #endif
                if (src0->type != GGML_TYPE_F32) {
                    ggml_profile_mul_mat_kblock_inactive(3, kblock);
                } else if ((kblock % QK_REQUIRED_CHECK) != 0) {
                    ggml_profile_mul_mat_kblock_inactive(2, kblock);
                }
            }
        }
        } else {
            // Potential partial-K paths: F32xF32 (implemented earlier) and new quant dequant path for *_K types when enabled.
            bool is_f32_pair = (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
            bool quant_enabled = s_kblock_quant_enabled;
            bool src1_is_k_quant = (
                src1->type == GGML_TYPE_Q2_K || src1->type == GGML_TYPE_Q3_K || src1->type == GGML_TYPE_Q4_K ||
                src1->type == GGML_TYPE_Q5_K || src1->type == GGML_TYPE_Q6_K || src1->type == GGML_TYPE_Q8_K);
            bool can_quant_partial = (quant_enabled && src0->type == GGML_TYPE_F32 && src1_is_k_quant);

            if (ith == 0) {
                const char * dbg = getenv("MFKB_DEBUG");
                if (dbg && (*dbg=='1' || *dbg=='y' || *dbg=='Y' || *dbg=='t' || *dbg=='T')) {
                    GGML_LOG_INFO("mul_mat debug: shape M=%lld N=%lld K=%lld quant_enabled=%d src1_is_k_quant=%d can_quant_partial=%d kblock=%lld fused_flag=%d use_tiled=%d\n",
                                  (long long)nr0, (long long)nr1, (long long)ne00,
                                  quant_enabled, src1_is_k_quant, can_quant_partial, (long long)kblock, s_kblock_quant_fused, use_tiled);
                }
            }

            if (!is_f32_pair && !can_quant_partial) {
                // fallback
                ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot,
                                                       ir0_start, ir0_end, ir1_start, ir1_end);
                if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                    ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
                }
                continue;
            }

            // Common setup for partial accumulation writes (zero destination tile region first)
            const int64_t tile_rows = ir0_end - ir0_start;
            const int64_t tile_cols = ir1_end - ir1_start;
            // Prepare destination column base pointers
            const int64_t stack_cap = 256;
            float * dst_col_ptrs_stack[stack_cap];
            float ** dst_col_ptrs = dst_col_ptrs_stack;
            if (tile_cols > stack_cap) {
                dst_col_ptrs = (float**)malloc(sizeof(float*) * tile_cols);
            }
            for (int64_t off = 0, ir1 = ir1_start; ir1 < ir1_end; ++ir1, ++off) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                dst_col_ptrs[off] = (float*)((char*)tensor_data(dst) + (i11 * nb1 + i12 * nb2 + i13 * nb3));
            }
            for (int64_t off = 0; off < tile_cols; ++off) {
                memset(dst_col_ptrs[off] + ir0_start, 0, tile_rows * sizeof(float));
            }

            // Thread-local panel buffers (separate for F32 packed / quant dequant output reuse same structure)
            struct kblock_panel_tls { float * buf; size_t cap; };
            static _Thread_local struct kblock_panel_tls kbpanel = { NULL, 0 };

            // For quant path we dequantize per-column K-slice; require kblock multiple of QK_K for all *_K quant formats.
            // Retrieve QK_K from quant header via macro (declared in ggml-quants.h indirectly); guard at compile time.
#ifdef QK_K
            const int QK_REQUIRED = QK_K;
#else
            const int QK_REQUIRED = 256; // fallback assumption
#endif
            if (can_quant_partial && (kblock % QK_REQUIRED) != 0) {
                // Requirement not met -> fallback once and profile inactive-other reason.
                if (ith == 0) ggml_profile_mul_mat_kblock_inactive(2 /* other */, kblock);
                ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot,
                                                       ir0_start, ir0_end, ir1_start, ir1_end);
                if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                    ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
                }
                if (tile_cols > stack_cap) free(dst_col_ptrs);
                continue;
            }

            const int64_t K = ne00;
            for (int64_t k0 = 0; k0 < K; k0 += kblock) {
                const int64_t k_step = MIN(kblock, K - k0);
                size_t need = (size_t)k_step * (size_t)tile_cols;
                bool fused_quant = can_quant_partial && s_kblock_quant_fused;
                if (ith == 0) {
                    const char * dbg = getenv("MFKB_DEBUG");
                    if (dbg && (*dbg=='1' || *dbg=='y' || *dbg=='Y' || *dbg=='t' || *dbg=='T')) {
                        GGML_LOG_INFO("mul_mat debug: entering k-loop k0=%lld k_step=%lld fused_quant=%d (kblock=%lld)\n", (long long)k0, (long long)k_step, fused_quant, (long long)kblock);
                    }
                }
                if (!fused_quant) {
                    if (kbpanel.cap < need) {
                        free(kbpanel.buf);
                        kbpanel.buf = (float*)malloc(need * sizeof(float));
                        kbpanel.cap = need;
                    }
                }
                float * panel = kbpanel.buf; // may be NULL if fused_quant
                uint64_t t_panel_start = (ggml_profiling_mul_mat_enabled() && !fused_quant) ? ggml_time_us() : 0;

                if (is_f32_pair) {
                    // Pack plain F32 weights slice
                    for (int64_t off = 0, ir1 = ir1_start; ir1 < ir1_end; ++ir1, ++off) {
                        const int64_t i13 = (ir1 / (ne12 * ne1));
                        const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                        const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                        const char * src1_col_full = (const char*)(tensor_data(src1)) + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                        const float * src1_slice = (const float*)(src1_col_full + k0 * nb01); // nb01 stride per K element
                        memcpy(panel + off * k_step, src1_slice, k_step * sizeof(float));
                    }
                } else if (can_quant_partial) {
                    if (!fused_quant) {
                        // Panel dequant from src1 (quant weights) into 'panel'
                        for (int64_t off = 0, ir1 = ir1_start; ir1 < ir1_end; ++ir1, ++off) {
                            const int64_t i13 = (ir1 / (ne12 * ne1));
                            const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                            const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                            const char * wcol_base = (const char*)tensor_data(src1) + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                            char * w_slice_ptr = (char*)wcol_base + k0 * nb01;
                            float * out_col = panel + off * k_step;
                            int64_t remaining = k_step;
                            int64_t processed = 0;
                            while (remaining > 0) {
                                int64_t chunk = remaining;
                                if (chunk % QK_REQUIRED != 0) chunk -= (chunk % QK_REQUIRED);
                                if (chunk == 0) break;
                                switch (src1->type) {
                                    case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    case GGML_TYPE_Q8_K: dequantize_row_q8_K((const block_q8_K*)w_slice_ptr, out_col + processed, chunk); break;
                                    default: GGML_UNREACHABLE();
                                }
                                processed += chunk;
                                remaining -= chunk;
                                w_slice_ptr += (chunk / QK_REQUIRED) * ggml_type_size(src1->type) * (QK_REQUIRED / ggml_blck_size(src1->type));
                            }
                            if (processed != k_step) {
                                for (int64_t fill = processed; fill < k_step; ++fill) out_col[fill] = 0.0f;
                            }
                        }
                    } else {
                        // fused_quant path (on-the-fly dequant handled in inner loop elsewhere); mark once per k-block
                        if (ith == 0) ggml_profile_mul_mat_fused_used();
                        if (ith == 0 && ggml_profiling_mul_mat_enabled() && ggml_profiling_mul_mat_verbose()) {
                            GGML_LOG_INFO("mul_mat debug: fused k-block processed k0=%lld k_step=%lld\n", (long long)k0, (long long)k_step);
                        }
                    }
                }

                if (ggml_profiling_mul_mat_enabled() && !fused_quant) {
                    uint64_t t_panel_end = ggml_time_us();
                    ggml_profile_mul_mat_panel_time(t_panel_start, t_panel_end);
                    ggml_profile_mul_mat_panel_bytes((size_t)k_step * (size_t)tile_cols * sizeof(float));
                }
                uint64_t t_inner_start = ggml_profiling_mul_mat_enabled() ? ggml_time_us() : 0;

                // Compute indices mapping once for A broadcasting
                const int64_t i13_first = (ir1_start / (ne12 * ne1));
                const int64_t i12_first = (ir1_start - i13_first * ne12 * ne1) / ne1;
                const int64_t i02 = i12_first / (ne12 / ne02);
                const int64_t i03 = i13_first / (ne13 / ne03);
                const char * src0_base = (const char*)tensor_data(src0) + (i02 * nb02 + i03 * nb03);

                if (!fused_quant) {
                    for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                        const float * a_slice = (const float*)(src0_base + ir0 * nb01) + k0;
                        for (int64_t off = 0; off < tile_cols; ++off) {
                            const float * bcol = panel + off * k_step;
                            float acc = ggml_dot_f32_f32_simd(a_slice, bcol, k_step);
                            dst_col_ptrs[off][ir0] += acc;
                        }
                    }
                } else {
                    // On-the-fly fused dequant: dequant one column slice then dot
                    for (int64_t off = 0; off < tile_cols; ++off) {
                        const int64_t ir1 = ir1_start + off;
                        const int64_t i13 = (ir1 / (ne12 * ne1));
                        const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                        const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                        const char * wcol_base = (const char*)tensor_data(src1) + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                        const char * w_slice_ptr0 = wcol_base + k0 * nb01;
                        float * tmp = (float*)alloca(k_step * sizeof(float));
                        int64_t remaining = k_step;
                        int64_t processed = 0;
                        const char * w_ptr = w_slice_ptr0;
                        while (remaining > 0) {
                            int64_t chunk = remaining;
                            if (chunk % QK_REQUIRED != 0) chunk -= (chunk % QK_REQUIRED);
                            if (chunk == 0) break;
                            switch (src1->type) {
                                case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K*)w_ptr, tmp + processed, chunk); break;
                                case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K*)w_ptr, tmp + processed, chunk); break;
                                case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K*)w_ptr, tmp + processed, chunk); break;
                                case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K*)w_ptr, tmp + processed, chunk); break;
                                case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K*)w_ptr, tmp + processed, chunk); break;
                                case GGML_TYPE_Q8_K: dequantize_row_q8_K((const block_q8_K*)w_ptr, tmp + processed, chunk); break;
                                default: GGML_UNREACHABLE();
                            }
                            processed += chunk;
                            remaining -= chunk;
                            w_ptr += (chunk / QK_REQUIRED) * ggml_type_size(src1->type) * (QK_REQUIRED / ggml_blck_size(src1->type));
                        }
                        if (processed != k_step) {
                            for (int64_t fill = processed; fill < k_step; ++fill) tmp[fill] = 0.0f;
                        }
                        for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                            const float * a_slice = (const float*)(src0_base + ir0 * nb01) + k0;
                            float acc = ggml_dot_f32_f32_simd(a_slice, tmp, k_step);
                            dst_col_ptrs[off][ir0] += acc;
                        }
                    }
                }
                if (ggml_profiling_mul_mat_enabled()) {
                    uint64_t t_inner_end = ggml_time_us();
                    ggml_profile_mul_mat_inner_time(t_inner_start, t_inner_end);
                    // Count k_iters and flops regardless of which thread processed the partial-K slice
                    ggml_profile_mul_mat_kiter(k_step);
                    ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, k_step);
                }
            }

            if (tile_cols > stack_cap) free(dst_col_ptrs);
        }
    }
    if (ggml_profiling_mul_mat_enabled() && ith == 0) {
        uint64_t t_end = ggml_time_us();
        // Compute incremental compute time and derive this-call GFLOP/s (avoid cumulative distortion)
        uint64_t call_compute_us = t_end - t_compute_start;
        ggml_profile_mul_mat_compute_time_acc(t_compute_start, t_end);
        // Acquire recent flops added during this call: since flops are cumulative, we cannot easily isolate without a reset.
        // Approximate by recomputing GFLOP/s from total; still valid for relative comparison when legacy recorded early.
        long comp = atomic_load_explicit(&g_mmprof.compute_time_us, memory_order_relaxed);
        long flops = atomic_load_explicit(&g_mmprof.flops_f64, memory_order_relaxed);
        if (comp > 0 && flops > 0) {
            double gflops = ((double)flops / (double)comp) * 1e6 / 1e9;
            // Consider disabling only when kblock disabled (we want fallback decisions only for plain tiled vs legacy)
            if (kblock == 0) ggml_mul_mat_fb_consider_tiled(nr0, nr1, ne00, (float)gflops);
        }
        ggml_profile_mul_mat_report_once();
        (void)call_compute_us; // placeholder if we later want per-call metric
    }
    return;
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
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

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) tensor_data(dst) + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) tensor_data(src1) + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
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
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) tensor_data(ids) + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) tensor_data(src0) + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? tensor_data(src1) : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

#if defined(__aarch64__)
        // disable for ARM
        const bool disable_chunking = true;
#else
        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();
#endif // defined(__aarch64__)

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor);
            } break;
        case GGML_OP_VIEW:
            {
                ggml_compute_forward_view(params, tensor);
            } break;
        case GGML_OP_PERMUTE:
            {
                ggml_compute_forward_permute(params, tensor);
            } break;
        case GGML_OP_TRANSPOSE:
            {
                ggml_compute_forward_transpose(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggresively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

            // Real partial-K accumulation path (Phase1 minimal): only enabled for pure F32 operands.
            // Fallback to original behavior if types are not F32 (quantized kernels need full-K vec_dot).
            if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot,
                                                       ir0_start, ir0_end, ir1_start, ir1_end);
                if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                    ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
                }
            } else {
                // Thread-local panel buffer for B slice (KB x tile_n_cols)
                struct kblock_panel_tls { float * buf; size_t cap; };
                static _Thread_local struct kblock_panel_tls kbpanel = { NULL, 0 };
                const int64_t tile_rows = ir0_end - ir0_start;
                const int64_t tile_cols = ir1_end - ir1_start;
                // Precompute destination column base pointers & index triplets for flattened columns
                // Store up to tile_n entries on stack (tile_n is modest, typically <= 256). If larger, allocate.
                const int64_t stack_cap = 256;
                float * dst_col_ptrs_stack[stack_cap];
                float ** dst_col_ptrs = dst_col_ptrs_stack;
                int64_t * col_i13_stack[stack_cap]; // not needed after pointer computed; skip.
                if (tile_cols > stack_cap) {
                    dst_col_ptrs = (float**)malloc(sizeof(float*) * tile_cols);
                }
                // Prepare mapping for each flattened column index
                for (int64_t off = 0, ir1 = ir1_start; ir1 < ir1_end; ++ir1, ++off) {
                    const int64_t i13 = (ir1 / (ne12 * ne1));
                    const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                    const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                    const int64_t i1 = i11; const int64_t i2 = i12; const int64_t i3 = i13;
                    dst_col_ptrs[off] = (float*)((char*)tensor_data(dst) + (i1 * nb1 + i2 * nb2 + i3 * nb3));
                }

                // Zero output tile region once before accumulation if first kblock processed by this thread.
                // We only zero the portion this tile covers: rows [ir0_start, ir0_end), columns in tile_cols.
                // For simplicity, always zero then accumulate partial blocks.
                for (int64_t off = 0; off < tile_cols; ++off) {
                    memset(dst_col_ptrs[off] + ir0_start, 0, tile_rows * sizeof(float));
                }

                const int64_t K = ne00; // shared K dimension length
                for (int64_t k0 = 0; k0 < K; k0 += kblock) {
                    const int64_t k_step = MIN(kblock, K - k0);
                    // Ensure panel capacity
                    size_t need = (size_t)k_step * (size_t)tile_cols;
                    if (kbpanel.cap < need) {
                        free(kbpanel.buf);
                        size_t new_cap = need;
                        kbpanel.buf = (float*)malloc(new_cap * sizeof(float));
                        kbpanel.cap = new_cap;
                    }
                    float * panel = kbpanel.buf; // layout: column-major blocks of length k_step
                    uint64_t t_panel_start = ggml_profiling_mul_mat_enabled() ? ggml_time_us() : 0;
                    // Pack B slice for each column
                    for (int64_t off = 0, ir1 = ir1_start; ir1 < ir1_end; ++ir1, ++off) {
                        const int64_t i13 = (ir1 / (ne12 * ne1));
                        const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                        const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                        const char * src1_col_full = (const char*) (src1->type == vec_dot_type ? tensor_data(src1) : params->wdata) +
                            ( (ggml_is_contiguous(src1) || src1->type != vec_dot_type)
                                ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * ggml_row_size(vec_dot_type, ne10)
                                : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                        // src1_col_full is start of full-K column; advance by k0 * sizeof(float)
                        const float * src1_slice = (const float*)(src1_col_full) + k0;
                        memcpy(panel + off * k_step, src1_slice, k_step * sizeof(float));
                    }
                    if (ggml_profiling_mul_mat_enabled()) {
                        uint64_t t_panel_end = ggml_time_us();
                        ggml_profile_mul_mat_panel_time(t_panel_start, t_panel_end);
                        ggml_profile_mul_mat_panel_bytes((size_t)k_step * (size_t)tile_cols * sizeof(float));
                    }
                    uint64_t t_inner_start = ggml_profiling_mul_mat_enabled() ? ggml_time_us() : 0;
                    // Multiply A rows slice with packed panel, accumulate
                    for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
                        // Determine broadcast indices (reuse pattern from original microkernel)
                        // For each column output pointer already points to row 0, so offset by ir0 later.
                        // Row pointer for A:
                        // broadcast factors: r2 = ne12 / ne02, r3 = ne13 / ne03 already computed earlier above.
                        // Compute i02/i03 from first column's indices (ir1_start) for row broadcasting.
                        const int64_t i13_first = (ir1_start / (ne12 * ne1));
                        const int64_t i12_first = (ir1_start - i13_first * ne12 * ne1) / ne1;
                        const int64_t i02 = i12_first / (ne12 / ne02);
                        const int64_t i03 = i13_first / (ne13 / ne03);
                        const char * src0_row_base = (const char*)tensor_data(src0) + (0 + i02 * nb02 + i03 * nb03);
                        const float * a_slice = (const float*)(src0_row_base + ir0 * nb01) + k0; // advance k0
                        for (int64_t off = 0; off < tile_cols; ++off) {
                            float acc = 0.f;
                            const float * bcol = panel + off * k_step;
                            for (int64_t kk = 0; kk < k_step; ++kk) {
                                acc += a_slice[kk] * bcol[kk];
                            }
                            dst_col_ptrs[off][ir0] += acc;
                        }
                    }
                    if (ggml_profiling_mul_mat_enabled()) {
                        uint64_t t_inner_end = ggml_time_us();
                        ggml_profile_mul_mat_inner_time(t_inner_start, t_inner_end);
                        ggml_profile_mul_mat_kiter(k_step);
                    }
                }
                if (ith == 0 && ggml_profiling_mul_mat_enabled()) {
                    ggml_profile_mul_mat_flops_acc(ir0_end - ir0_start, ir1_end - ir1_start, ne00);
                }
                if (tile_cols > stack_cap) {
                    free(dst_col_ptrs);
                }
            }
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads_max : GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02*ne03;
                        cur += sizeof(ggml_fp16_t)*ne10*ne11*ne12;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t ne10 = node->src[1]->ne[0]; // DK
                        const int64_t ne20 = node->src[2]->ne[0]; // DV

                        cur = sizeof(float)*(1*ne10 + 2*ne20)*n_tasks; // 1x head size K + 2x head size V (per thread)
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

    ggml_barrier(state->threadpool);

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is active
static inline bool ggml_graph_compute_thread_active(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;
    int n_threads = atomic_load_explicit(&threadpool->n_threads_cur, memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;

            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_fetch_add_explicit(&threadpool->n_graph, 1, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifndef GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                       = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            // Bind OpenMP threads to NUMA nodes in round-robin fashion
            // This must be done early in the parallel region before any work
            ggml_openmp_bind_thread_to_numa_node(omp_get_thread_num(), omp_get_num_threads());
            
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads_max) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    GGML_UNUSED(ctx);

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    // Use NUMA-aware work buffer allocation instead of ggml_new_buffer
    cplan.work_data = (uint8_t *)ggml_numa_alloc_work_buffer(cplan.work_size);
    if (cplan.work_size > 0 && !cplan.work_data) {
        return GGML_STATUS_ALLOC_FAILED;
    }

    enum ggml_status status = ggml_graph_compute(cgraph, &cplan);
    
    // Free the work buffer
    ggml_numa_free_work_buffer(cplan.work_data);
    
    return status;
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }
#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    putenv("OMP_WAIT_POLICY=active");
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
                putenv("KMP_BLOCKTIME=200"); // 200ms
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

        is_first_call = false;
    }

    ggml_critical_section_end();
}
