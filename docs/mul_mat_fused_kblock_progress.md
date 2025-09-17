# mul_mat Fused K-Block Exploration Progress

_Last updated: 2025-09-17_

## Executive Summary
We introduced k-block (partial-K) scaffolding, profiling instrumentation, and an experimental fused quantized matmul path (on-the-fly dequant + dot) in the CPU backend (`ggml_compute_forward_mul_mat`). Despite implementing the fused logic and building a synthetic microbenchmark (`micro-fused-kblock`), the fused path has **not yet activated** (profiling counter `fused=0`, no debug prints). Root cause: current execution for (F32 activations × Q*_K weights) appears to bypass the modified generic path, likely using a specialized quant matmul kernel.

## Implemented Features
- Tiled matmul path with adaptive fallback heuristic (env controlled)
- K-block partial accumulation (panel-based) for F32 and quant candidates
- Extended profiling counters:
  - conversion_time_us, compute_time_us, inner_time_us
  - tile_count, flops_f64, k_iters, kblock_size_last
  - panel_bytes, panel_time_us, panel_bw
  - kblock_inactive (reason codes 1–4)
  - fused_used (counter for fused path usage)
- Environment variable controls:
  - `GGML_MUL_MAT_PROFILE{,_VERBOSE}`
  - `GGML_MUL_MAT_TILED`, `GGML_MUL_MAT_TILE`, `GGML_MUL_MAT_TILE_N_FACTOR`, `GGML_MUL_MAT_PANEL_CAP`
  - `GGML_GEMM_KBLOCK`, `GGML_GEMM_KBLOCK_QUANT`, `GGML_GEMM_KBLOCK_QUANT_FUSED`
  - Fallback: `GGML_MUL_MAT_FALLBACK_DISABLE`, `GGML_MUL_MAT_FALLBACK_RATIO`, `GGML_MUL_MAT_FORCE_TILED`
  - Debug: `MFKB_DEBUG`, and temporary `MFKB_MARK_ONCE`
- AVX2 SIMD dot helper for inner accumulation
- Fused on-the-fly dequant logic (per K-slice, per column) with stack buffer
- Microbenchmark tool: `tools/micro-fused-kblock` (activations F32, weights quant)

## Key Code Touchpoints
- `ggml/src/ggml-cpu/ggml-cpu.c`
  - Added k-block scheduling & panel packing/dequant
  - Inserted fused quant branch (on-the-fly dequant) guarded by env flag
  - Runtime re-check for `GGML_GEMM_KBLOCK_QUANT_FUSED` after first initialization
  - Profiling & debug logging hooks
- `tools/micro-fused-kblock/micro-fused-kblock.cpp`
  - Synthetic harness to stress a single GEMM: A (F32 MxK) * B (Q*_K KxN)
  - Environment-driven shape override: `MFKB_SHAPE=MxNxK`

## Observations
| Aspect | Status |
|--------|--------|
| Panel (non-fused) partial-K path | Executes (timings captured) |
| Fused path instrumentation | Present but never triggers |
| `fused_used` counter | Always 0 |
| Debug logs (`MFKB_DEBUG=1`) | Not emitted for microbenchmark runs |
| Profiling line (`mul_mat profile:`) | Not printed during microbenchmark |
| Generic matmul function marker (`MFKB_MARK_ONCE`) | Not printed |

This strongly indicates the microbenchmark’s matmul op is dispatched via a specialized quantized path that bypasses the modified `ggml_compute_forward_mul_mat` implementation.

## Likely Root Cause
A dedicated quant matmul kernel for *_K formats short-circuits before reaching the generic tiled/k-block code we modified. Our fused logic lives only in the generic path; therefore setting fused env flags has no effect.

## Inactive / Fallback Reason Codes (implemented)
- 1: Quant path disabled (weights quant but `GGML_GEMM_KBLOCK_QUANT` not enabled)
- 2: Misaligned kblock (`kblock % QK_K != 0` requirement)
- 3: Unsupported activation type (needs src0 F32 currently)
- 4: Weights not *_K quant type when quant k-block requested

## Remaining Gaps
1. No integration of fused logic into the specialized quant kernel path.
2. No dispatch-side hook to force fallback to generic path for experimentation.
3. Missing high-level dispatch logging (which branch chosen) for `GGML_OP_MUL_MAT`.
4. No performance comparison yet between panel vs fused (fused inactive).
5. Microbenchmark doesn’t validate correctness of dequant (weights zeroed for now).

## Proposed Next Steps
1. Locate dispatch site for quant matmul (search for calls selecting quant-specific kernels for Q*_K types).
2. Add a debug dispatch log: record chosen path (legacy, tiled, kblock-panel, kblock-fused, quant-specialized).
3. Introduce env override (e.g., `GGML_FORCE_GENERIC_MATMUL=1`) to bypass specialized quant path and confirm generic fused code runs (expect debug marker + profile line + fused_used>0).
4. Once verified, refactor fused logic into a shared helper so specialized path can optionally call the k-block fused variant (or move scheduling into a common function).
5. Expand microbenchmark to:
   - Run multiple iterations (warm + timed loops) and report averages
   - Optionally populate quant weight buffer with valid packed blocks (reuse existing quant pack code) for a realistic test
6. Collect performance metrics for panel vs fused vs specialized baseline.
7. Add tests (or at least an assertion) that fused path numerical output matches panel path within tolerance.
8. Remove temporary debug markers (`MFKB_MARK_ONCE`) and gate extra logs strictly behind `MFKB_DEBUG`.

## Metrics to Capture (Future)
- GFLOP/s fused vs panel vs specialized
- Panel bytes saved (expect near-zero panel_bytes when fused active)
- k_iters distribution vs K/kblock
- Dequant bandwidth reduction if any (time shift from panel_time to inner_time)

## Risk & Considerations
- Duplicating logic between specialized and generic paths risks drift; prefer a unified partial-K scheduler.
- Fused on-the-fly dequant may increase instruction overhead if memory reuse from panel buffers provided cache benefits (needs empirical confirmation).
- Stack `alloca` for each column per k-block: verify no large k_step values risk stack overflow (guard or switch to thread-local heap above a threshold).
- Need to ensure thread safety of added static runtime env toggles (currently safe due to simple bool but re-check before making dynamic).

## Environment Variables Reference (New / Modified)
```
GGML_GEMM_KBLOCK               (int >0) partial-K size
GGML_GEMM_KBLOCK_QUANT         (bool) enable quant k-block for *_K formats
GGML_GEMM_KBLOCK_QUANT_FUSED   (bool) enable fused on-the-fly dequant path
GGML_MUL_MAT_TILED             (bool) enable tiled path candidate
GGML_MUL_MAT_FORCE_TILED       (bool) override fallback disable
GGML_MUL_MAT_FALLBACK_RATIO    (float) min perf ratio to keep tiled
GGML_MUL_MAT_PROFILE{,_VERBOSE}(bool) profiling output
MFKB_DEBUG                     (bool) verbose debug
MFKB_SHAPE                     (MxNxK) microbenchmark override
MFKB_MARK_ONCE                 (bool) temporary marker for path hit check
```

## Open Questions
- What exact file/function performs the quant *_K dispatch? (Needs confirmation.)
- Should fused logic live only in k-block mode, or also support full-K (kblock=K) as a degenerate case?
- Do we expand fused path to support more activation dtypes (e.g., f16/bf16) for broader applicability?

## Cleanup To Do (Deferred)
- Remove or consolidate duplicate debug prints.
- Document fused path design choices in developer docs (`docs/architecture`?).
- Add command-line or programmatic knob instead of env-only control.

## Current Microbenchmark Command Examples
(Executed but did not activate fused path due to dispatch bypass.)
```
GGML_GEMM_KBLOCK=256 \
GGML_GEMM_KBLOCK_QUANT=1 \
GGML_GEMM_KBLOCK_QUANT_FUSED=1 \
GGML_MUL_MAT_PROFILE=1 \
GGML_MUL_MAT_PROFILE_VERBOSE=1 \
MFKB_DEBUG=1 ./build/bin/micro-fused-kblock
```

## Definition of Done (Planned)
- fused_used > 0 with realistic Q*_K weights
- Numerical parity with existing panel dequant path
- Performance characterization (win/regression documented)
- Minimal dispatch logging retained (gated) + docs updated

---
_This document captures the state up to the point where fused activation debugging is pending. Resume by implementing dispatch forcing and integrating fused logic into the actually used quant path._
