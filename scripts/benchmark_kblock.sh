#!/usr/bin/env bash
# benchmark_kblock.sh
# Automated benchmarking for tiled + kblock experiments (F32 & quant *_K) in llama.cpp
# Produces a CSV summary plus raw logs for each scenario.
#
# Usage:
#   scripts/benchmark_kblock.sh -m path/to/model.gguf [options]
#
# Options:
#   -m MODEL         (required) model path
#   -p PROMPT_TOKENS default 32
#   -n GEN_TOKENS    default 32
#   -t THREADS       override threads (default: auto detect via nproc)
#   -o OUTDIR        output directory (default: benchmark-results-<timestamp>)
#   --kblocks "list" comma/space separated kblock sizes to test (default: 256,512)
#   --tiles  "list" tile spec list (e.g. 64x64,64x128) default: 64x64,64x128
#   --no-quant       skip quant partial-k scenarios (only baseline/tiled)
#   --quick          fewer scenarios (baseline + one kblock)
#   --repeat N       repeat each scenario N times (default 1)
#
# Environment variables honored:
#   EXTRA_BENCH_ARGS  appended to llama-bench invocation
#   KEEP_LOGS=0|1     keep raw bench logs (default 1)
#
# Output:
#   <outdir>/summary.csv with columns:
#     scenario,model,threads,prompt_tokens,gen_tokens,tile,kblock,quant_enabled,inactive,k_iters,flops,perf_gflops,
#     panel_bytes,panel_time_us,inner_time_us,compute_time_us,conversion_time_us,tokens_per_s_pp,tokens_per_s_tg,notes
#
# Dependencies: awk, grep, sed, date. Assumes llama-bench built at ./build/bin/llama-bench
set -euo pipefail

MODEL=""
PROMPT_TOKENS=32
GEN_TOKENS=32
THREADS=$(nproc)
OUTDIR=""
KBLOCKS=(256 512 768)
TILES=(64x64 64x128)
DO_QUANT=1
QUICK=0
REPEAT=1

log() { echo "[bench] $*"; }
err() { echo "[bench][error] $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2;;
    -p) PROMPT_TOKENS="$2"; shift 2;;
    -n) GEN_TOKENS="$2"; shift 2;;
    -t) THREADS="$2"; shift 2;;
    -o) OUTDIR="$2"; shift 2;;
    --kblocks) IFS=', ' read -r -a KBLOCKS <<< "$2"; shift 2;;
    --tiles) IFS=', ' read -r -a TILES <<< "$2"; shift 2;;
    --no-quant) DO_QUANT=0; shift;;
    --quick) QUICK=1; shift;;
    --repeat) REPEAT="$2"; shift 2;;
    *) err "Unknown arg: $1";;
  esac
done

[[ -f "$MODEL" ]] || err "Model not found: $MODEL"
[[ -x ./build/bin/llama-bench ]] || err "llama-bench binary missing (expected ./build/bin/llama-bench). Build first."

TS=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTDIR" ]]; then
  OUTDIR="benchmark-results-$TS"
fi
mkdir -p "$OUTDIR/logs"

KEEP_LOGS=${KEEP_LOGS:-1}
SUMMARY="$OUTDIR/summary.csv"
BASELINE_TIME=0
TIMEOUT_FACTOR=${TIMEOUT_FACTOR:-2}
HAVE_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then HAVE_TIMEOUT=1; fi

# CSV header
cat > "$SUMMARY" <<EOF
scenario,model,threads,prompt_tokens,gen_tokens,tile,kblock,quant_enabled,inactive,k_iters,flops,perf_gflops,panel_bytes,panel_time_us,inner_time_us,compute_time_us,conversion_time_us,tokens_per_s_pp,tokens_per_s_tg,notes,elapsed_s,timed_out
EOF

# helper: extract mul_mat profile last line from a log
extract_profile() {
  # prints the last mul_mat profile line
  grep 'mul_mat profile:' "$1" | tail -n 1 || true
}
# helper: extract t/s lines (pp & tg) from llama-bench summary table (first occurrence)
extract_tokens_per_s() {
  local logf="$1"
  local pp=$(grep '| .*pp' "$logf" | awk -F '|' '{gsub(/ /,"",$5); gsub(/ /,"",$7); if($5 ~ /pp[0-9]+/){gsub(/±.*/,"",$7); print $7; exit}}')
  local tg=$(grep '| .*tg' "$logf" | awk -F '|' '{gsub(/ /,"",$5); gsub(/ /,"",$7); if($5 ~ /tg[0-9]+/){gsub(/±.*/,"",$7); print $7; exit}}')
  echo "$pp,$tg"
}
# parse profile fields
parse_profile() {
  local line="$1"
  # default empties
  local compute="" conv="" flops="" perf="" kblock="" k_iters="" panel_bytes="" panel_time="" inner_time="" inactive="" bw=""
  # general pattern tokens separated by spaces; we use sed/awk to isolate numbers
  # Example:
  # mul_mat profile: conversion=15442 us compute=867779 us tiles=303920 flops=24175968256 perf=27.86 GFLOP/s kblock=256 k_iters=64 panel_bytes=123456 panel_time=789 us inner_time=4567 us panel_bw=1.23 GB/s inactive=0
  [[ -z "$line" ]] && { echo ',,,,,,,,,,'; return; }
  compute=$(echo "$line" | sed -n 's/.* compute=\([0-9]*\) us.*/\1/p')
  conv=$(echo "$line" | sed -n 's/.* conversion=\([0-9]*\) us.*/\1/p')
  flops=$(echo "$line" | sed -n 's/.* flops=\([0-9]*\) perf=.*/\1/p')
  perf=$(echo "$line" | sed -n 's/.* perf=\([0-9.]*\) GFLOP.*/\1/p')
  kblock=$(echo "$line" | sed -n 's/.* kblock=\([0-9]*\).*/\1/p')
  k_iters=$(echo "$line" | sed -n 's/.* k_iters=\([0-9]*\).*/\1/p')
  panel_bytes=$(echo "$line" | sed -n 's/.* panel_bytes=\([0-9]*\).*/\1/p')
  panel_time=$(echo "$line" | sed -n 's/.* panel_time=\([0-9]*\) us.*/\1/p')
  inner_time=$(echo "$line" | sed -n 's/.* inner_time=\([0-9]*\) us.*/\1/p')
  inactive=$(echo "$line" | sed -n 's/.* inactive=\([0-9]*\).*/\1/p')
  echo "$flops,$perf,$kblock,$k_iters,$panel_bytes,$panel_time,$inner_time,$compute,$conv,$inactive"
}

run_case() {
  local scenario="$1"; shift
  local tile="$1"; shift
  local kblock="$1"; shift
  local quant_flag="$1"; shift
  local rep="$1"; shift
  local tile_m=${tile%x*}; local tile_n=${tile#*x}
  local logf="$OUTDIR/logs/${scenario}_tile-${tile}_kb-${kblock}_q-${quant_flag}_r${rep}.log"
  log "Running scenario=$scenario tile=$tile kblock=$kblock quant=$quant_flag rep=$rep"
  # env setup
  unset GGML_MUL_MAT_TILED GGML_GEMM_KBLOCK GGML_GEMM_KBLOCK_QUANT GGML_MUL_MAT_TILE
  export GGML_MUL_MAT_PROFILE=1
  export GGML_MUL_MAT_PROFILE_VERBOSE=1
  if [[ "$tile" != "legacy" ]]; then
    export GGML_MUL_MAT_TILED=1
    export GGML_MUL_MAT_TILE="${tile_m}x${tile_n}"
  fi
  if [[ "$kblock" != "0" ]]; then
    export GGML_GEMM_KBLOCK="$kblock"
  fi
  if [[ "$quant_flag" == "1" ]]; then
    export GGML_GEMM_KBLOCK_QUANT=1
  fi
  local start_ts=$(date +%s)
  local rc=0
  local timed_out=0
  set +e
  if [[ $scenario != baseline && $BASELINE_TIME -gt 0 ]]; then
    local limit=$(python3 - <<PY 2>/dev/null || echo $((BASELINE_TIME*TIMEOUT_FACTOR)) )
import math,os
bt=int(os.environ.get('BASELINE_TIME','0'))
tf=float(os.environ.get('TIMEOUT_FACTOR','2'))
print(int(math.ceil(bt*tf)))
PY
    if [[ $limit -le 0 ]]; then limit=$((BASELINE_TIME*TIMEOUT_FACTOR)); fi
    if [[ $HAVE_TIMEOUT -eq 1 ]]; then
      timeout ${limit}s ./build/bin/llama-bench -m "$MODEL" --numa mirror -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$THREADS" --verbose ${EXTRA_BENCH_ARGS:-} >"$logf" 2>&1
      rc=$?
      if [[ $rc -eq 124 ]]; then timed_out=1; fi
    else
      # Manual watchdog
      ./build/bin/llama-bench -m "$MODEL" --numa mirror -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$THREADS" --verbose ${EXTRA_BENCH_ARGS:-} >"$logf" 2>&1 &
      local pid=$!
      local waited=0
      while kill -0 $pid 2>/dev/null; do
        sleep 1
        waited=$((waited+1))
        if [[ $waited -ge $limit ]]; then
          timed_out=1
          kill -TERM $pid 2>/dev/null
          sleep 1
          kill -KILL $pid 2>/dev/null
          wait $pid 2>/dev/null
          rc=124
          break
        fi
      done
      if [[ $timed_out -eq 0 ]]; then
        wait $pid
        rc=$?
      fi
    fi
  else
    ./build/bin/llama-bench -m "$MODEL" --numa mirror -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$THREADS" --verbose ${EXTRA_BENCH_ARGS:-} >"$logf" 2>&1
    rc=$?
  fi
  set -e
  local end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))
  if [[ $scenario == baseline ]]; then
    # Use the first baseline run as reference (fastest or simplest). If baseline repeats, keep the minimum.
    if [[ $BASELINE_TIME -eq 0 || $elapsed -lt $BASELINE_TIME ]]; then
      BASELINE_TIME=$elapsed
      export BASELINE_TIME
      log "Baseline reference time set to ${BASELINE_TIME}s (limit per test: $((BASELINE_TIME*TIMEOUT_FACTOR))s)"
    fi
  fi
  if [[ $rc -ne 0 ]]; then
    log "WARN: llama-bench exited rc=$rc (scenario=$scenario)"
  fi
  local profile_line=$(extract_profile "$logf")
  local pp_tg=$(extract_tokens_per_s "$logf")
  local pp_rate=$(echo "$pp_tg" | cut -d',' -f1)
  local tg_rate=$(echo "$pp_tg" | cut -d',' -f2)
  local parsed=$(parse_profile "$profile_line")
  IFS=',' read -r flops perf kblock_p k_iters panel_bytes panel_time inner_time compute conv inactive <<< "$parsed"
  local note="rc=${rc}"
  if [[ $timed_out -eq 1 ]]; then note="${note};timeout"; fi
  echo "${scenario},${MODEL},${THREADS},${PROMPT_TOKENS},${GEN_TOKENS},${tile},${kblock_p:-$kblock},${quant_flag},${inactive},${k_iters},${flops},${perf},${panel_bytes},${panel_time},${inner_time},${compute},${conv},${pp_rate},${tg_rate},${note},${elapsed},${timed_out}" >> "$SUMMARY"
  if [[ ${KEEP_LOGS} -ne 1 ]]; then rm -f "$logf"; fi
}

# Scenario matrix
SCENARIOS=()
if [[ $QUICK -eq 1 ]]; then
  # Quick matrix still broad enough to exercise each path once
  SCENARIOS+=(baseline tiled tiled-kb tiled-kb-quant misalign-quant)
  # Narrow tiles for speed but keep a larger one for variety
  TILES=(legacy 64x64 64x128)
  if [[ $DO_QUANT -eq 1 ]]; then
    KBLOCKS=(256 768)
  else
    KBLOCKS=(256 768)
  fi
else
  SCENARIOS+=(baseline tiled tiled-kb tiled-kb-quant misalign-quant)
fi

for rep in $(seq 1 $REPEAT); do
  for scen in "${SCENARIOS[@]}"; do
    case $scen in
      baseline)
        run_case baseline legacy 0 0 $rep ;;
      tiled)
        for tile in "${TILES[@]}"; do
          [[ $tile == legacy ]] && continue
          run_case tiled "$tile" 0 0 $rep
        done ;;
      tiled-kb)
        for tile in "${TILES[@]}"; do
          [[ $tile == legacy ]] && continue
          for kb in "${KBLOCKS[@]}"; do
            run_case tiled-kb "$tile" "$kb" 0 $rep
          done
        done ;;
      tiled-kb-quant)
        [[ $DO_QUANT -eq 0 ]] && continue
        for tile in "${TILES[@]}"; do
          [[ $tile == legacy ]] && continue
          for kb in "${KBLOCKS[@]}"; do
            run_case tiled-kb-quant "$tile" "$kb" 1 $rep
          done
        done ;;
      misalign-quant)
        [[ $DO_QUANT -eq 0 ]] && continue
        # Intentionally pick first KBLOCK minus half if divisible
        KB_MIS=128
        run_case misalign-quant 64x64 "$KB_MIS" 1 $rep
        ;;
      *) err "Unknown scenario template: $scen";;
    esac
  done
done

log "Summary written: $SUMMARY"
# Post-process to add baseline speedup columns (prompt/gen) and simple perf ratio vs baseline flops per compute time
if command -v awk >/dev/null 2>&1; then
  TMPCSV="$OUTDIR/summary_with_speedup.csv"
  awk -F',' 'NR==1 {print $0",pp_speedup,tg_speedup"; next} NR>1 {print $0",,"}' "$SUMMARY" > "$TMPCSV" || true
  # Compute speedups using first baseline row as reference
  # tokens_per_s_pp is still at NF-4 after adding speedup placeholders (original CSV unchanged before speedup stage)
  base_pp=$(awk -F',' 'NR>1 && $1=="baseline" {print $(NF-6); exit}' "$SUMMARY")
  base_tg=$(awk -F',' 'NR>1 && $1=="baseline" {print $(NF-5); exit}' "$SUMMARY")
  if [[ -n "$base_pp" && -n "$base_tg" ]]; then
    awk -F',' -v bpp="$base_pp" -v btg="$base_tg" 'BEGIN{OFS=","} NR==1 {print $0} NR>1 {pp=$(NF-6); tg=$(NF-5); spp=(pp>0?pp/bpp:""); stg=(tg>0?tg/btg:""); print $0,spp,stg}' "$SUMMARY" > "$TMPCSV.tmp" && mv "$TMPCSV.tmp" "$TMPCSV"
  fi
  log "Augmented summary with speedups: $TMPCSV"
fi
log "Done." 
