#!/usr/bin/env bash
# ab_llama_stock.sh [OUT_DIR]   (2026-10-31, run in WSL, GPU must be idle)
# A-B-C-A speed check of stock llama.cpp master against our branch, for the llama.cpp PR (only optional / GPU-specific changes of ours go into it):
#   STOCK1  ~/llama-stock/build-stock  upstream/master 36a0e2ffc, no fused attention (it does not have it)
#   OURS0   ~/llama-up/build-up        our branch (merge of upstream + our 73 commits), GGML_CUDA_FATTN_TILE_Q4=0 (the default: our optional kernel off)
#   OURS2   ~/llama-up/build-up        same build, GGML_CUDA_FATTN_TILE_Q4=2 (fused q4_0 / q8_0 tile attention on)
#   STOCK2  ~/llama-stock/build-stock  again, to see drift of the GPU
# Same server flags for all (only options stock knows): UDflat, q4_0 KV, -c 131072, -fa on, -np 1, MTP head of Unsloth with draft length 3 (SPEC=mtp) or no speculation (SPEC=none).
# Pass 1 warms, pass 2 counts: pi16 / pi32 / pi113 prompts of bench/prompts, 256 greedy tokens, prefill and decode tok/s from probe_n.py.
set -u
OUT="${1:-$HOME/ab_stock_$(date +%Y%m%d_%H%M)}"
SPECS="${SPECS:-mtp none}"
R=/mnt/f/programming/llm/qwen3.8-27b_rx7900xtx_inference
M=$HOME/models
STOCK=$HOME/llama-stock/build-stock/bin
OURS=$HOME/llama-up/build-up/bin
PORT=8080
mkdir -p "$OUT"
source $R/rocmenv.sh >/dev/null 2>&1
stop() { pkill -x llama-server; for i in $(seq 1 40); do pgrep -x llama-server >/dev/null || return 0; sleep 1; done; pkill -9 -x llama-server; sleep 2; }

run_one() {   # label bindir fused spec
  local L="$1" B="$2" F="$3" S="$4"
  echo "=== $L ($S)  $B  fused=$F  $(date +%T)" | tee -a "$OUT/summary.txt"
  stop
  local ARGS=(-m $M/Qwen3.8-27B-UDflat.gguf -ngl 99 -c 131072 -fa on --jinja --no-warmup -np 1 --no-kv-unified -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 --cache-ram 16384 --host 127.0.0.1 --port $PORT --temp 0 --top-k 1)
  [ "$S" = mtp ] && ARGS+=(--spec-type draft-mtp --spec-draft-n-max 3 -md $M/mtp-Qwen3.8-27B-Q4_0_dynamicV3.0.gguf)
  ( export LLAMA_ATTN_ROT_DISABLE=1; [ "$F" != none ] && export GGML_CUDA_FATTN_TILE_Q4=$F; nohup "$B/llama-server" "${ARGS[@]}" > "$OUT/$L.$S.server.out" 2>&1 < /dev/null & )
  local up=0
  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }
    pgrep -x llama-server >/dev/null || break
    sleep 2
  done
  [ $up = 1 ] || { echo "server $L ($S) did not come up" | tee -a "$OUT/summary.txt"; tail -8 "$OUT/$L.$S.server.out" | tee -a "$OUT/summary.txt"; stop; return 1; }
  # one short request so the first probe does not pay lazy initialisation
  curl -s http://127.0.0.1:$PORT/completion -d '{"prompt":"Hello","n_predict":8,"temperature":0}' >/dev/null 2>&1
  "$B/llama-server" --version 2>&1 | grep -E "version|commit" | head -1 | tee -a "$OUT/summary.txt"
  for pass in 1 2; do
    for f in pi16 pi32 pi113; do
      r=$(/usr/bin/python3 $R/bench/prompts/probe_n.py $PORT $R/bench/prompts/$f.txt 256 2>&1 | tr '\n' ' ')
      echo "$L $S pass$pass $f $r" | tee -a "$OUT/summary.txt"
    done
  done
  grep -E "draft acceptance|accept" "$OUT/$L.$S.server.out" | tail -1 >> "$OUT/summary.txt"
  stop
}

for S in $SPECS; do
  run_one STOCK1 "$STOCK" none "$S"
  run_one OURS0  "$OURS"  0    "$S"
  run_one OURS2  "$OURS"  2    "$S"
  run_one STOCK2 "$STOCK" none "$S"
done
echo "=== done $(date +%T)" | tee -a "$OUT/summary.txt"
