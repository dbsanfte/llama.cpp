2025-08-28 - CRITICAL BUG IDENTIFIED: NUMA Node 1 Memory Isolation Issue

## Problem Analysis
- Node 1 threads ARE computing correct values (-3.22441244, -3.12334442, etc.)
- Node 1 threads ARE executing vec_dot calls successfully
- Node 1 computed values are NOT appearing in final result tensor (showing zeros)

## Root Cause  
NUMA memory isolation: Node 1 writes to its local tensor copy instead of shared result.

## Evidence
- Node 0 dst_ptr: 0x7905f4694800 (high addresses)
- Post-vec_dot debug shows Node 1 computing: ir0=64→after=-3.22441244 
- Final test result shows: Element[64]=0.00000000 (zeros, not computed values)

## Technical Details
dst_col calculation: tensor_data(dst) + (... + row_start * nb0)
- Node 0: tensor_data(dst) + (... + 0 * nb0) 
- Node 1: tensor_data(dst) + (... + 64 * nb0)

Issue: tensor_data(dst) returns different base addresses per NUMA node in MIRROR mode.

## Solution Required
Ensure all nodes write to shared result tensor, not local NUMA copies.
