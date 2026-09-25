#!/usr/bin/env bash
#
# quality-kld.sh — Phase Q: measure candidate model quality vs a high-precision
# reference via KL divergence, top-1 (same-top-p) agreement and PPL ratio,
# using the merged build's `llama-perplexity --kl-divergence-base` mechanism.
#
# Usage:
#   quality-kld.sh [--ref GGUF] [--corpus FILE ...] [--cand TAG|GGUF|EXTRA] [opts]
#
# The reference run is done once per corpus and its per-token logits are stored
# in a .kld binary. Each candidate ("TAG|gguf[:extra-args]") is then compared
# against that reference on the same corpus.
#
# Environment / defaults:
#   BIN        path to llama-perplexity  (default: build/bin/llama-perplexity)
#   MODELS_DIR default model directory   (default: /home/thegiallo/models)
#   PPL_CTX    prompt context            (default: 8192)
#   PPL_NGL    gpu layers for the REFERENCE run (leave empty for all+auto)
#   OUTDIR     output dir                (default: results/quality/<date>)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$ROOT/build/bin/llama-perplexity}"
MODELS_DIR="${MODELS_DIR:-/home/thegiallo/models}"
OUTDIR="${OUTDIR:-$ROOT/results/quality/$(date +%Y-%m-%d)}"
CTX="${PPL_CTX:-8192}"
NGL_REF="${PPL_NGL:-}"

REF=""
CORPORA=()
CANDIDATES=()
SAVE_ONLY=0

usage() {
    sed -n '2,16p' "${BASH_SOURCE[0]}"
    echo
    echo "Options:"
    echo "  --ref GGUF        reference model (default: \$MODELS_DIR/Qwen3.8-27B-Q8_0.gguf)"
    echo "  --corpus FILE     text corpus (repeatable; default: wikitext-2 test raw)"
    echo "  --cand TAG|MODEL[:EXTRA]  candidate (repeatable); EXTRA = extra args, e.g."
    echo "                    -ctk\ q4_0\ -ctv\ q4_0; numeric EXTRA is a -ngl override"
    echo "  --save-only       only (re)generate the reference .kld file(s) and exit"
    echo "  --outdir DIR      output directory"
    echo "  --ngl-ref N       gpu layers for the reference run"
    echo "  -h, --help        this message"
}

# ---------------------------------------------------------------- candidates
cand_tag()   { echo "${1%%|*}"; }
cand_model() { local b="${1#*|}"; echo "${b%%|*}"; }
cand_extra() { local b="${1#*|}"; if [[ "$b" == *"|"* ]]; then echo "${b#*|}"; fi; }

# ---------------------------------------------------------------- corpus io
corpus_name() {
    local f="$1"
    python3 -c "import sys,os; print(os.path.splitext(os.path.basename(sys.argv[1]))[0])" "$f"
}

# ---------------------------------------------------------------- default corpus
DEFAULT_CORPUS="$MODELS_DIR/wikitext-2-raw/wiki.test.raw"

if [[ ! -f "$DEFAULT_CORPUS" ]]; then
    # try the repo downloader (writes ./wikitext-2-raw + zip in cwd)
    ( cd "$MODELS_DIR" && bash "$ROOT/scripts/get-wikitext-2.sh" >/dev/null 2>&1 ) || true
fi

# ---------------------------------------------------------------- parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)        REF="${2:?--ref needs a path}"; shift 2;;
        --corpus)     CORPORA+=("${2:?--corpus needs a path}"); shift 2;;
        --cand)       CANDIDATES+=("${2:?--cand needs TAG|MODEL[:EXTRA]}"); shift 2;;
        --save-only)  SAVE_ONLY=1; shift;;
        --outdir)     OUTDIR="${2:?--outdir needs a path}"; shift 2;;
        --ngl-ref)    NGL_REF="$2"; shift 2;;
        -h|--help)    usage; exit 0;;
        *)            echo "unknown arg: $1" >&2; usage; exit 2;;
    esac
done

[[ -n "$REF" ]] || REF="$MODELS_DIR/Qwen3.8-27B-Q8_0.gguf"
if [[ ${#CORPORA[@]} -eq 0 ]] && [[ -f "$DEFAULT_CORPUS" ]]; then
    CORPORA+=("$DEFAULT_CORPUS")
fi
if [[ ${#CORPORA[@]} -eq 0 ]]; then
    echo "error: no corpus found; pass --corpus FILE (default wikitext missing)" >&2
    exit 1
fi

[[ -x "$BIN" ]] || { echo "error: $BIN missing (build target llama-perplexity)" >&2; exit 1; }
[[ -f "$REF" ]] || { echo "error: reference model not found: $REF" >&2; exit 1; }

mkdir -p "$OUTDIR"

echo "== quality-kld.sh =="
echo "  ref    : $REF"
echo "  bin    : $BIN"
echo "  outdir : $OUTDIR"
echo "  ctx    : $CTX"
echo "  corpora: ${#CORPORA[@]}"

# ---------------------------------------------------------------- run one ppl
run_ppl() { # tag model extra_args log_prefix
    local tag="$1" model="$2" extra="$3" log="$4"
    # a numeric extra means "override --gpu-layers N"; anything else is appended verbatim
    local ngl=()
    if [[ "$extra" =~ ^[0-9]+$ ]]; then
        ngl=(--gpu-layers "$extra")
        extra=""
    fi
    echo "  [run] $tag  ($model)"
    "$BIN" -m "$model" -c "$CTX" -f "$CORPUS" \
        "${ngl[@]}" $extra \
        --kl-divergence-base "$KLD_FILE" --kl-divergence > "$log" 2>&1 \
        || { echo "  !! $tag failed; see $log" >&2; return 1; }
}

# ---------------------------------------------------------------- the matrix
SUMMARY="$OUTDIR/summary.csv"
: > "$SUMMARY"
echo "corpus,candidate,mean_kld,kld_99.0,kld_99.9,same_top_p,mean_ppl_q,mean_ppl_base,ppl_ratio" >> "$SUMMARY"

for CORPUS in "${CORPORA[@]}"; do
    [[ -f "$CORPUS" ]] || { echo "error: corpus not found: $CORPUS" >&2; exit 1; }
    CNAME="$(corpus_name "$CORPUS")"
    KLD_FILE="$OUTDIR/$CNAME.ref.kld"
    REF_LOG="$OUTDIR/$CNAME.ref.log"
    KVTAG=""
    if [[ ${KV_REF_TYPE:-} ]]; then KVTAG=" -ctk $KV_REF_TYPE -ctv $KV_REF_TYPE"; fi

    if [[ ! -f "$KLD_FILE" ]] || [[ $SAVE_ONLY -eq 1 ]]; then
        echo "== base logits: $CNAME (reference $REF)"
        echo "  (this is the slow one-time run)"
        "$BIN" -m "$REF" -c "$CTX" -f "$CORPUS" \
            ${NGL_REF:+"--gpu-layers" "$NGL_REF"} \
            --save-all-logits "$KLD_FILE" > "$REF_LOG" 2>&1 \
        || { echo "  !! reference run failed; see $REF_LOG" >&2; exit 1; }
        [[ $SAVE_ONLY -eq 1 ]] && continue
    else
        echo "== using existing base logits: $KLD_FILE"
    fi

    [[ ${#CANDIDATES[@]} -gt 0 ]] || { echo "  (no candidates; base done)"; continue; }

    echo "== comparing on corpus $CNAME"
    for C in "${CANDIDATES[@]}"; do
        TAG="$(cand_tag "$C")"
        MODEL="$(cand_model "$C")"
        EXTRA="$(cand_extra "$C")"
        LOG="$OUTDIR/$CNAME.$(echo "$TAG" | tr '/:' '__').log"
        if run_ppl "$TAG" "$MODEL" "$EXTRA" "$LOG"; then
            # parse the printed summary lines
            mean_kld="$(  sed -nE "s/^Mean    KLD:[[:space:]]*([-0-9.]+).*/\1/p"        "$LOG")"
            kld99="$(     sed -nE "s/^99.0%   KLD:[[:space:]]*([0-9.]+).*/\1/p"   "$LOG")"
            kld999="$(    sed -nE "s/^99.9%   KLD:[[:space:]]*([0-9.]+).*/\1/p"   "$LOG")"
            same_top="$(  sed -nE "s/^Same top p:[[:space:]]*([0-9.]+).*/\1/p"    "$LOG")"
            ppl_q="$(     sed -nE "s/^Mean PPL\\(Q\\)[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p" "$LOG")"
            ppl_base="$(  sed -nE "s/^Mean PPL\\(base\\)[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p" "$LOG")"
            ppl_ratio="$( sed -nE "s/^Mean PPL\\(Q\\)\\/PPL\\(base\\)[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p" "$LOG")"
            echo "$CNAME,$TAG,$mean_kld,$kld99,$kld999,$same_top,$ppl_q,$ppl_base,$ppl_ratio" >> "$SUMMARY"
            echo "  $TAG: mean_kld=$mean_kld  kld99=$kld99  kld99.9=$kld999  same_top=$same_top"
        fi
    done
done

echo
echo "summary: $SUMMARY"
cat "$SUMMARY"