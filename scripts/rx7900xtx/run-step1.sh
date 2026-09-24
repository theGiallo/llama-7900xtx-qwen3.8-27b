#!/usr/bin/env bash
# Step 1 of CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md: confirm where decode time goes.
#
#   1. FLASH_ATTN_EXT bandwidth for the 27B attention shape (decode + verify batches),
#      with the kernel choice logged (GGML_CUDA_FATTN_LOG=1).
#   2. llama-bench decode at several context depths per KV type, fitted into
#      fixed ms/token + ms per 1k tokens of context.
#
# Usage:
#   scripts/rx7900xtx/run-step1.sh [-B build-dir] [-o out-dir] [-d depths] [-k kv-types] model.gguf [model2.gguf ...]
#
# Environment is passed through, e.g. LLAMA_ATTN_ROT_DISABLE=1 to match earlier runs.
# The depth sweep prefills every depth once per repetition; at ~500 t/s prefill the
# default depths take roughly 10 minutes per model and KV type.

set -euo pipefail

BUILD=build
OUT="step1-$(date +%Y%m%d-%H%M%S)"
DEPTHS="0,32768,65536,131072"
KV_TYPES="q4_0,q8_0,f16"
BACKEND="${BACKEND:-ROCm0}"

while getopts "B:o:d:k:" opt; do
    case $opt in
        B) BUILD=$OPTARG ;;
        o) OUT=$OPTARG ;;
        d) DEPTHS=$OPTARG ;;
        k) KV_TYPES=$OPTARG ;;
        *) sed -n '2,16p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -lt 1 ]; then
    sed -n '2,16p' "$0"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

{
    echo "# step 1 run $(date -Is)"
    echo
    echo "- git: $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- build: $BUILD"
    echo "- LLAMA_ATTN_ROT_DISABLE=${LLAMA_ATTN_ROT_DISABLE:-<unset>}"
    echo "- models: $*"
    echo
} > "$OUT/README.md"

echo "== 1/2 FLASH_ATTN_EXT bandwidth ($BACKEND)"
python3 "$HERE/fattn_bw.py" --bin "$BUILD/bin/test-backend-ops" --backend "$BACKEND" \
    --save "$OUT/fattn.sql" | tee "$OUT/fattn_bw.md"

echo "== 2/2 decode vs depth"
MODEL_ARGS=()
for m in "$@"; do
    MODEL_ARGS+=(-m "$m")
done
python3 "$HERE/decode_depth_fit.py" --bench "$BUILD/bin/llama-bench" "${MODEL_ARGS[@]}" \
    --ctk "$KV_TYPES" --depths "$DEPTHS" --save-dir "$OUT/bench" | tee "$OUT/decode_depth_fit.md"

echo
echo "results in $OUT/ (fattn_bw.md, decode_depth_fit.md, raw sql/jsonl/logs)"
