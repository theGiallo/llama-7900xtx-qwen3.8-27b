# Cloud review: Phase C (VEC_GQA) and PLAN_STATUS.md

Reviewed `408d4b0`, `07c5bca`, `b18e69d`: `fattn-vec-gqa.{cu,cuh}`,
`VEC_GQA_phaseC_report.md`, `PLAN_STATUS.md`.

## Verdict

The measured regression is real, and keeping VEC_GQA opt-in is the right call. But it shows that
**this implementation** doesn't pay. It does not falsify grouping the query heads. The kernel
keeps the part of the reference VEC design that makes q4_0 decode instruction-bound, the
per-row warp reduction, un-amortized across the 6 heads.

## Why this kernel can't win

`flash_attn_ext_vec_gqa`, KQ pass:

```cpp
for (int jG = 0; jG < GQA; ++jG) {
    float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[jG], Q_i32[jG], Q_ds[jG]);
    sum = warp_reduce_sum<nthreads_KQ>(sum);
    ...
```

- The layout is inherited from VEC: the 32 lanes of a warp split **one KV row** (D=256 ⇒ 8
  elements per lane), then combine with `warp_reduce_sum` (5 cross-lane steps). That reduction
  runs once per **(KV row, head)**: 6 per KV row, exactly as many as 6 separate VEC blocks do.
  For q4_0 each lane only does ~2 `dp4a` per row, so the reduction plus max/exp bookkeeping is a
  large share of the instructions. **That share is not amortized at all.**
- The K load and nibble unpack are shared only if LLVM merges six identical inlined
  `vec_dot_KQ` calls (possible with `__restrict__` and no intervening stores, but not
  guaranteed). The comment says "loaded and dequantized once"; the code doesn't enforce it.
  Check the ISA (`--save-temps`) for 6× `global_load` per row.
- The V side *is* amortized (dequantize once, FMA into 6 accumulators), which is why the kernel
  isn't 6× slower.
- Per-head state (`VKQ[6][…]`, `Q_i32[6][…]`, `Q_ds[6][…]`, `KQ_max/sum[6]`) with
  `__launch_bounds__(128, 1)` likely pushes VGPRs past ~120, i.e. ≤ 12 waves/SIMD (table in
  `RX7900XTX_KERNEL_STRATEGIES.md` §1). That lowers `launch_fattn`'s `max_blocks_per_sm` and
  therefore `parallel_blocks`. Please report VGPRs/occupancy with
  `-Rpass-analysis=kernel-resource-usage` (E5).

Net: roughly the reference's instruction count plus register pressure. A 1.2–2× regression is
the expected outcome, not a surprise.

## A shape that does amortize (for the next attempt, if the user wants one)

Goal: per KV row, do the K load, unpack and **all reductions** once for all 6 heads.

**Option 1 — thread-per-KV-row (no cross-lane reductions in KQ).**
- Each lane owns one KV row of the chunk (32 rows per warp). Q for the 6 heads is quantized
  once to q8_1 and kept in **LDS** (6 × 256 B + scales), read via broadcast (same address for
  all lanes ⇒ no bank conflicts).
- The lane loads its q4_0 K row (144 B = 9 × `global_load_b128`, 16-byte aligned because the
  head stride is 144 B and the token stride is 576 B), unpacks it **once** in registers and does
  6 × 64 `dp4a` against the 6 Q vectors. The dot product is complete in-lane: no warp reduction.
- Softmax per head is then lane-local plus one max/sum reduction per warp per 32 rows.
- V: keep the current (already amortized) scheme, or transpose so lanes own output dims.
- Coalescing: each lane reads a separate 144 B run (32 separate cache lines per instruction),
  but every byte is used once and it all goes through L0/L1. Worth measuring against the
  cooperative load.

**Option 2 — WMMA iu8 for KQ.** Q (6 heads, padded to 16 rows) as A, 16 K rows as B,
16×16×16 iu8 tiles over D=256 (16 WMMAs per 16 KV rows). No reductions, and it scales with n_q
for verify (6 × n_q rows ⇒ 6/16 → 48/48 utilisation at n_q = 8). RDNA3 WMMA runs on the DOT
units (ISA §7.9), so padding costs dot throughput, but it removes the reduction overhead. The
best fit for verify batches of 3–8.

Both should be judged by the same `fattn_bw.py` tables. The target is the f16 TILE line (75 %
of peak), not the reference VEC.

## Corrections to PLAN_STATUS.md

- Phase C: "premise falsified by measurement" → **"first implementation regresses; it keeps
  per-(row, head) warp reductions, so the head grouping only amortizes the V side"**. The
  premise (one K/V pass for all 6 heads) hasn't been tested yet.
- Phase D: agreed that D1 is closed (no fixed-cost regression). For D2 the proposed alternative
  "MMA_F16 … for the verify batch" still requires the whole-cache f16 conversion with quantized
  KV (`need_f16_K/V` for TILE and MMA_F16). The verify path needs a kernel that reads quantized
  KV in-kernel, which is Phase C again. D2's acceptance test is unchanged: merged + fixed verify
  path vs the fork's 36.3 t/s @113k and 24.8 t/s @258k with MTP.
- Phase B2 is partly done: step 1 already measured q8_0 KV at 131072: 24.23 vs 20.82 t/s for
  q4_0, attention 12.9 vs 23.0 ms @113k (`results/step1/2026-09-24-strix/`). q8_0 at 262144
  (~9.1 GB KV) is the open part.
- E0/E1/E2: see `CLOUD_REVIEW.md`. The MUL_MAT perf runs were cache-hot, so the E1 "warps are not
  the lever" conclusion still needs a DRAM-bound measurement (`llama-bench -p 0 -n 128 -d 0`).

## Agreed priorities

Phase Q next is right: it gives every later change an acceptance gate. A second Phase C attempt
(Option 1 or 2 above) is the user's call.
