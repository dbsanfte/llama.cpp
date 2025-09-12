/**
 * Simplified Work Item Implementation
 * 
 * Implements the simple function pointer + payload pattern for clean
 * separation between dispatcher and coordinator.
 */

#include "ggml-work-item.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>

// Global atomic counters for unique IDs
static atomic_int next_work_id = ATOMIC_VAR_INIT(1);
static atomic_int next_batch_id = ATOMIC_VAR_INIT(1);
static atomic_int next_barrier_id = ATOMIC_VAR_INIT(1);

// =============================================================================
// Work Item Creation and Management
// =============================================================================

ggml_work_item_t * ggml_work_item_new(ggml_task_t func, void * data, size_t data_size, int target_numa_node) {
    if (!func) {
        return NULL;
    }
    
    ggml_work_item_t * item = malloc(sizeof(ggml_work_item_t));
    if (!item) {
        return NULL;
    }
    
    item->func = func;
    item->data_size = data_size;
    item->work_id = ggml_work_item_get_next_id();
    item->target_numa_node = target_numa_node;
    
    // Set default values for enhanced fields
    item->strategy = GGML_PARALLEL_ELEMENT;
    item->sync_req = GGML_SYNC_NONE;
    item->chunk_start = 0;
    item->chunk_end = 0;
    item->total_chunks = 1;
    item->priority = 5; // Medium priority
    item->next = NULL;
    
    // Handle payload data
    if (data_size > 0 && data) {
        // Copy the data if size is specified
        item->data = malloc(data_size);
        if (!item->data) {
            free(item);
            return NULL;
        }
        memcpy(item->data, data, data_size);
    } else {
        // Just store the pointer if no size specified
        item->data = data;
    }
    
    return item;
}

ggml_work_item_t * ggml_work_item_new_enhanced(
    ggml_task_t func, 
    void * data, 
    size_t data_size, 
    int target_numa_node,
    ggml_parallel_strategy_t strategy,
    ggml_sync_requirement_t sync_req,
    int chunk_start,
    int chunk_end,
    int total_chunks,
    int priority
) {
    if (!func) {
        return NULL;
    }
    
    ggml_work_item_t * item = malloc(sizeof(ggml_work_item_t));
    if (!item) {
        return NULL;
    }
    
    item->func = func;
    item->data_size = data_size;
    item->work_id = ggml_work_item_get_next_id();
    item->target_numa_node = target_numa_node;
    
    // Set enhanced metadata
    item->strategy = strategy;
    item->sync_req = sync_req;
    item->chunk_start = chunk_start;
    item->chunk_end = chunk_end;
    item->total_chunks = total_chunks;
    item->priority = priority;
    item->next = NULL;
    
    // Handle payload data
    if (data_size > 0 && data) {
        // Copy the data if size is specified
        item->data = malloc(data_size);
        if (!item->data) {
            free(item);
            return NULL;
        }
        memcpy(item->data, data, data_size);
    } else {
        // Just store the pointer if no size specified
        item->data = data;
    }
    
    return item;
}

void ggml_work_item_free(ggml_work_item_t * item) {
    if (!item) {
        return;
    }
    
    // Free copied payload data if it was allocated
    if (item->data_size > 0 && item->data) {
        free(item->data);
    }
    
    free(item);
}

ggml_work_batch_t * ggml_work_batch_new(ggml_work_item_t * items, int num_items, bool use_barrier) {
    if (!items || num_items <= 0) {
        return NULL;
    }
    
    ggml_work_batch_t * batch = malloc(sizeof(ggml_work_batch_t));
    if (!batch) {
        return NULL;
    }
    
    batch->items = items;
    batch->num_items = num_items;
    batch->batch_id = ggml_work_batch_get_next_id();
    
    // Create barrier if requested
    if (use_barrier) {
        batch->barrier = ggml_work_barrier_new(num_items);
        if (!batch->barrier) {
            free(batch);
            return NULL;
        }
    } else {
        batch->barrier = NULL;
    }
    
    return batch;
}

void ggml_work_batch_free(ggml_work_batch_t * batch) {
    if (!batch) {
        return;
    }
    
    // Free all work items in the batch
    for (int i = 0; i < batch->num_items; i++) {
        ggml_work_item_free(&batch->items[i]);
    }
    free(batch->items);
    
    // Free barrier if it exists
    if (batch->barrier) {
        ggml_work_barrier_free(batch->barrier);
    }
    
    free(batch);
}

// =============================================================================
// Synchronization Functions
// =============================================================================

ggml_work_barrier_t * ggml_work_barrier_new(int num_work_items) {
    if (num_work_items <= 0) {
        return NULL;
    }
    
    ggml_work_barrier_t * barrier = malloc(sizeof(ggml_work_barrier_t));
    if (!barrier) {
        return NULL;
    }
    
    atomic_init(&barrier->remaining_work, num_work_items);
    atomic_init(&barrier->completed, false);
    barrier->barrier_id = ggml_work_barrier_get_next_id();
    
    return barrier;
}

void ggml_work_barrier_signal(ggml_work_barrier_t * barrier) {
    if (!barrier) {
        return;
    }
    
    int remaining = atomic_fetch_sub(&barrier->remaining_work, 1) - 1;
    if (remaining == 0) {
        atomic_store(&barrier->completed, true);
    }
}

bool ggml_work_barrier_wait(ggml_work_barrier_t * barrier, int timeout_ms) {
    if (!barrier) {
        return false;
    }
    
    // If already completed, return immediately
    if (atomic_load(&barrier->completed)) {
        return true;
    }
    
    // Handle no timeout case (wait indefinitely)
    if (timeout_ms == 0) {
        while (!atomic_load(&barrier->completed)) {
            // Busy wait with small sleep to reduce CPU usage
            struct timespec sleep_time = { .tv_sec = 0, .tv_nsec = 1000000 }; // 1ms
            nanosleep(&sleep_time, NULL);
        }
        return true;
    }
    
    // Handle timeout case
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    while (!atomic_load(&barrier->completed)) {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        
        long elapsed_ms = (current_time.tv_sec - start_time.tv_sec) * 1000 +
                         (current_time.tv_nsec - start_time.tv_nsec) / 1000000;
        
        if (elapsed_ms >= timeout_ms) {
            return false; // Timeout reached
        }
        
        // Small sleep to reduce CPU usage
        struct timespec sleep_time = { .tv_sec = 0, .tv_nsec = 1000000 }; // 1ms
        nanosleep(&sleep_time, NULL);
    }
    
    return true;
}

bool ggml_work_barrier_is_completed(ggml_work_barrier_t * barrier) {
    if (!barrier) {
        return false;
    }
    
    return atomic_load(&barrier->completed);
}

void ggml_work_barrier_free(ggml_work_barrier_t * barrier) {
    if (!barrier) {
        return;
    }
    
    free(barrier);
}

// =============================================================================
// Utility Functions
// =============================================================================

int ggml_work_item_get_next_id(void) {
    return atomic_fetch_add(&next_work_id, 1);
}

int ggml_work_batch_get_next_id(void) {
    return atomic_fetch_add(&next_batch_id, 1);
}

int ggml_work_barrier_get_next_id(void) {
    return atomic_fetch_add(&next_barrier_id, 1);
}
