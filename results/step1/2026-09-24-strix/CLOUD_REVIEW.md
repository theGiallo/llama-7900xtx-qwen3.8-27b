# Cloud review of the step-1 report (2026-09-24, STRIX)

Review of `STEP1_REPORT.md`, `fattn_bw.md` and `decode_depth_fit.md` by the cloud session,
with the code checked against each claim.

## What holds

- **Diagnosis confirmed.** q4_0 decode attention runs the VEC kernel at 9–12 % of peak; f16 runs
  TILE at 71–82 %. Quantized-KV batches ≥ 3 always take TILE with whole-cache f16 conversion.
- **Attention cost per token (16 layers, nb=1):** q4_0 23.0 ms @113k and 53.1 ms @262k;
  q8_0 12.9 / 37.6 ms; f16 10.3 / 24.2 ms. At peak bandwidth q4_0 would be 2.2 / 5.0 ms.
- **Fixed part ≈ 26.5–28 ms/token** for 14.75 GB of weights ⇒ ~530 GB/s (55 % of peak).

## Corrections

### 1. The re-read is absorbed by caches; the VEC kernel is instruction-bound

The numbers rule out "6× DRAM traffic" as the cause:

- q8_0 @113k, nb=1: 247 MB per layer in 0.808 ms. If each of the 6 query heads re-read it from
  DRAM, that would be 6 × 247 MB / 0.808 ms ≈ **1.8 TB/s, almost 2× the card's peak**. So the
  re-reads are served by L2 (6 MiB) or the Infinity Cache (96 MiB), as predicted in
  `RX7900XTX_KERNEL_STRATEGIES.md` §2.2.
- q4_0 moves **half** the bytes of q8_0 but takes **1.8× longer** (1.44 vs 0.81 ms). The q4_0
  VEC path is bound by instructions (nibble unpacking plus the redundant per-head dequantize
  and dot), not by memory.

The fix is still the GQA-grouped kernel, now with a sharper reason: it removes 5/6 of the
dequantization and dot work. Unpack cost per byte also matters, so the kernel should decode
q4_0 with as few instructions as possible (`v_perm`/shift tricks, one pass per 16 bytes).

### 2. Even the f16 TILE path reads each KV block 3×

`fattn-tile.cuh` (`launch_fattn_tile_switch_ncols2`) packs query heads only in powers of two
(`gqa_ratio % 8/4/2`). With GQA 6 it packs 2 heads per block, so each KV block is processed 3
times. The WMMA FlashAttention path also never engages for this model's verify batches: its
GQA factor is the largest power of two dividing 6, i.e. 2, and it needs `n_q × 2 > 16`. That
explains why f16 verify still costs ~3× at nb=6. The new kernel should pack **all 6 heads**
(padded to 8 if needed), for quantized and f16 KV alike.

### 3. `GGML_CUDA_FATTN_LOG` was on the executor path; its output was filtered

The hook is in `ggml_cuda_flash_attn_ext()` (`fattn.cu`), called from the op dispatcher
(`ggml-cuda.cu:2372`), not only from `get_alloc_size`. It printed via `GGML_LOG_INFO`, and:

- `common` maps ggml INFO logs to TRACE (`common/log.cpp:532`), hidden by default in
  `llama-server` and `llama-cli`;
- `llama-bench` installs a null log callback unless `-v` (`llama-bench.cpp:2247`).

**Fixed:** it now writes directly to stderr, so it shows up in every binary. Rebuild, then run
`llama-server` with MTP and `GGML_CUDA_FATTN_LOG=1` to see the live verify shapes.

### 4. The "merged-build regression" is most likely the Hadamard rotation, not the build

This run used `LLAMA_ATTN_ROT_DISABLE=1`. The earlier ~35 ms fixed-part estimate came from
merged rows 57/59 in `RESULTS_tok_per_sec.md`, which ran with rotation **on** (the merged
default). Row 62 vs 58 (MTP @113k: 27.1 t/s rotation off vs 20.6 on) points the same way.
So the rotation costs several ms/token in this build. That matters because 4-bit KV *should* run
with rotation on for quality. Worth one depth-fit run with rotation on
(`scripts/rx7900xtx/run-step1.sh -k q4_0`, no `LLAMA_ATTN_ROT_DISABLE`) to size it.

### 5. The fixed part is still ~2× the bandwidth floor

28 ms for 14.75 GB is 55 % of peak; the floor is ~15.4 ms. The matrix-vector findings in
`RX7900XTX_KERNEL_STRATEGIES.md` §2.1 still apply: ROCmFP4 gets **1 warp per block** on RDNA3
where Q4_0/IQ4_NL get 8, plus misaligned 4-byte loads. That is ~12 ms/token of headroom,
comparable to the attention win at 113k.

## Immediate, no-code recommendation

**Use q8_0 KV instead of q4_0 up to ~160k context.** It is faster today (24.2 vs 20.8 t/s at
131072 in this run; attention 12.9 vs 23.0 ms @113k) and higher quality. Memory: q8_0 KV is
~4.6 GB at 131072 (vs ~2.4 GB for q4_0), which fits next to 14.75 GB of weights. At 262144 it
would be ~9.1 GB, too tight on a 24 GB card that also drives the desktop. The f16 depth-131k
row (0.95 t/s) shows what running out of VRAM does.

## Next steps (proposed; the user decides)

1. **E0** — add ROCmFP4 cases to `test-backend-ops` (there are none on this branch).
2. **E1/E2** — ROCmFP4 MMVQ warps (1 → 2/4/8) and aligned loads, measured with E0's perf cases.
3. **Phase C / E7** — GQA-grouped decode FlashAttention: all 6 heads per block, one dequant per
   KV block, sequence split for parallelism, batch 1–8 without f16 conversion.
4. Size the rotation cost (item 4 above).
