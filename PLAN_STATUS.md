# Plan phase status — Qwen3.8-27B on RX 7900 XTX (gfx1100, ROCm 7)

Status snapshot against `CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md` (Phase A/B/Q/C/D/E).
Machine: RX 7900 XTX (24.5 GB), WSL2, rocmdev SDK, `F/optimization` branch.
HEAD: a1d5d2e71. Decode fixed base ~26.5-28.1 ms/tok; peak BW assumed 960 GB/s.
Corrections from cloud review: results/e0/2026-09-25/CLOUD_REVIEW_PHASE_C.md (pulled in a1d5d2e71).

## Phase A — confirm the diagnosis: DONE
- **A1/A2 kernel-per-shape log**: DONE. `GGML_CUDA_FATTN_LOG=1` prints kernel, n_q, gqa,
  K/V types, f16-conversion flag (fattn.cu). Decode batch 1: f16→TILE, q8_0→VEC,
  q4_0→VEC. Verify batch≥3 quantized: TILE + whole-cache f16 conversion.
  Files: results/step1/2026-09-24-strix/fattn_bw.md, results/mtp/2026-09-25-fattn-probe/REPORT.md.
- **A3 bandwidth table**: DONE (scripts/rx7900xtx/fattn_bw.py + perf cases in
  test-backend-ops.cpp). q4_0 decode @113408 = 9% of peak (23.03 ms/tok), q8_0 = 32% (12.93),
  f16 TILE = 75% (10.33). Verify nb=3→q4_0 1.57x, nb=6→2.13x vs nb=1.
- **A4 depth fit**: DONE (decode_depth_fit.py). q4_0 fixed 28.06 ms + 0.151 ms/1k ctx.
- **A5 run-step1**: DONE on STRIX.
- **Decision gate**: CONFIRMED q4_0 decode attention far below peak → Phase C the main job.
  (Note the projected 23 ms→2.3 ms did NOT materialize — see Phase C.)

## Phase B — no-code wins: PARTIAL
- **B1** IQ3_XXS + f16 KV @131072: NOT STARTED.
- **B2** q8_0 KV @131072/262144: PARTLY DONE. Step 1 already measured q8_0 @131072: 24.23 vs
  20.82 t/s for q4_0 (attention 12.9 vs 23.0 ms @113k); q8_0 @262144 (~9.1 GB KV) is the open
  part. See results/step1/2026-09-24-strix/.
- **B3** rotation-ON cost: DONE — rotation is ~free (27.07 vs 28.06 ms fixed, not the
  suspected ~8 ms regression). results/step1/rot-q4f16-strix/ROTATION_FIT.md, commit a5cf12505.
- Also: f16 KV is VRAM-impossible at ≥113k on 24.5 GB (weights 14.75 GB + f16 KV thrash),
  so B1/B2 would need a lighter weight quant to fit.

## Phase Q — make "95% quality" measurable: NOT STARTED
- No scripts/rx7900xtx/quality-kld.sh, no KLD/agreement baselines, no Q3 matrix run.

## Phase C — GQA-grouped quantized-KV decode attention: FIRST ATTEMPT DONE, measured REGRESSION
- **C1 kernel** `flash_attn_ext_vec_gqa`: DONE (new fattn-vec-gqa.{cu,cuh}, commit 64f6a36eb).
  One block per KV head, all GQA heads share the dequantized KV; D=256, K==V∈{q4_0,q8_0},
  gqa∈{2,4,6,8}. Original NaN root cause: epilogue readback read the slot region with a
  float-typed 256-element stride while slots are stored 128 float-positions apart → warps
  2..3 read never-written shared memory → stale NaN. Fixed by reading via a half view
  (matches the reference layout where KQ is half[]).
- **C2 tests**: DONE. 79-case gqa6 suite (kv 16k..262k, nb 1..8, f16/q8_0/q4_0) + extra
  gqa-2/4/8 and prefill nb=512 cases. Full GPU suite 16505/16527.
- **C3 benches**: DONE — NEGATIVE for THIS implementation. VEC_GQA is 1.19-2.2x SLOWER than
  incumbents everywhere: decode nb=1 +21% (q4_0) to +105% (q8_0) vs VEC at 113k; verify
  nb≥3 +114..223% vs TILE+f16conv at 113k/262k. Pinned ~1.5-1.9 TFLOPS.
- **Why this kernel can't win (cloud review, confirmed by resource report)**: the KQ pass
  keeps the reference VEC layout — 32 lanes split ONE KV row (D=256), then `warp_reduce_sum`
  (5 cross-lane steps) runs once per (KV row, head) = 6 per KV row, as many as 6 separate VEC
  blocks; that share is NOT amortized. K load/nibble unpack sharing across the 6 heads is only
  as good as LLVM's inliner (not enforced in code). Only the V side is amortized (dequantize
  once, FMA into 6 accumulators) — hence not 6× slower. Register pressure from six per-head
  states (`VKQ[6]`, `Q_i32[6]`, `Q_ds[6]`, `KQ_max/sum[6]`) on gfx1100 (`-Rpass-analysis=
  kernel-resource-usage`):

  | GQA | VGPRs | Occupancy (waves/SIMD) |
  |-----|-------|------------------------|
  | 2   | 116   | 12                     |
  | 4   | 211   | 7                      |
  | 6   | 256   | 5                      |
  | 8   | 210-256 | 5-7                  |

  GQA=6 (our model) sits at the 256-VGPR clamp → ~5 waves/SIMD, i.e. exactly the pinned
  1.5-1.9 TFLOPS ceiling. Net: reference instruction count + register pressure; 1.2-2.2×
  regression is the expected outcome.
- **Resolution**: VEC_GQA now **opt-in** (`GGML_CUDA_FATTN_VEC_GQA=1`, default off); default
  dispatch = reference behavior → production decode/verify perf unchanged (commit d25f0a254).
  VEC_GQA is more precise at the extreme (kv=262144 nb 6/8 quantized pass on it, marginal
  on reference VEC). Full details: results/e0/2026-09-25/VEC_GQA_phaseC_report.md.
- **Corrected framing**: the premise ("one K/V pass amortizing everything across 6 heads")
  has NOT been falsified — the specific shape here keeps per-(row,head) warp reductions and
  only amortizes the V side. See the two alternative shapes below.

### Second Phase C attempt (user's call) — shapes that WOULD amortize KQ
Goal: per KV row, do the K load, unpack and ALL reductions once for all 6 heads. Both are
judged by the same `fattn_bw.py` tables; target = the f16 TILE line (75% of peak), not the
reference VEC. Design detail: results/e0/2026-09-25/CLOUD_REVIEW_PHASE_C.md.
- **Option 1 — thread-per-KV-row**: each lane owns one KV row of the chunk (32 rows/warp);
  Q→q8_1 for 6 heads kept in LDS (read via broadcast ⇒ no bank conflicts); lane loads its
  q4_0 K row (144 B = 9 × global_load_b128, aligned since head stride 144 B / token stride
  576 B), unpacks once, 6 × 64 dp4a in-lane — the dot product completes WITHOUT any warp
  reduction. Softmax becomes lane-local + one max/sum reduction per warp per 32 rows. V: keep
  the current (already-amortized) scheme. Coalescing: 32 separate 144 B runs per instruction
  (L0/L1-backed) — worth measuring vs the cooperative load.
- **Option 2 — WMMA iu8 for KQ**: Q (6 heads padded to 16 rows) as A, 16 K rows as B,
  16×16×16 iu8 tiles over D=256 (16 WMMAs per 16 KV rows). No reductions; scales with n_q
  (verify: n_q 3..8 ⇒ 6*n_q/16 → 18/48 .. 48/48 utilisation). RDNA3 WMMA runs on the DOT
  units, so zero-padding costs dot throughput, but it removes the reduction overhead. Best
  fit for verify batches 3-8.

## Phase D — recover merged-build regressions / fused MTP: PARTIAL (analysis only)
- **D1 graph diff / ~8 ms fixed cost**: effectively DONE under Step 1 — no fixed-cost
  regression found (fixed base 26.5-28.1 ms vs fork ~27 ms; the earlier "~35 ms" not
  reproduced). The remaining topic is not a merge regression.
- **D2 fused MTP port design**: analysis DONE (MTP trace: verify = TILE + whole-cache f16
  conversion, the collapse driver; results/mtp/). Porting DESIGN/PLAN for the merged build:
  NOT STARTED. NOTE: the earlier D2 alternative "MMA_F16 (or a Q-rows≥2 vector path) for the
  verify batch" — with quantized KV, TILE and MMA_F16 BOTH still require the whole-cache f16
  conversion (`need_f16_K/V`), so the verify path still needs a kernel that reads quantized KV
  in-kernel = Phase C again. D2's acceptance test is unchanged: merged + fixed verify path vs
  the fork's 36.3 t/s @113k and 24.8 t/s @258k with MTP.
- **D3 port** and **D4 validation**: NOT STARTED (design first, offload for review before
  coding).

## Phase E — fixed-cost polish: PARTIAL
- **E1 HIP graphs / per-token launch latency**: NOT DONE. The MUL_MAT perf runs were
  cache-hot, so the earlier "warps are not the lever" conclusion still needs a DRAM-bound
  measurement (`llama-bench -p 0 -n 128 -d 0`).
- **E2 ROCmFP4 MMVQ tuning**: DONE — single wide qs load etc. (3-8% FFN decode).
  results/e0/2026-09-25/E1E2.md, commit b4fba04ed.
- **E3 verify-batch cost curve nb=1..8**: DONE (fattn_bw.md verify-x tables; C3 re-measured
  under VEC_GQA for reference). E2/E3 review: results/e0/2026-09-25/CLOUD_REVIEW.md.

## Priority recommendations (next)
1. Close Phase Q (quality-KLD script + thresholds) so perf work has an acceptance gate —
   agreed with cloud review.
2. Phase D2/D3: design the fused-MTP port WITHOUT relying on VEC_GQA (quantized KV is read
   in-kernel in any case); offload a design for review before coding.
3. Second Phase C attempt (Option 1 thread-per-row or Option 2 WMMA iu8) — the USER'S CALL;
   the first attempt regresses for a structural reason (per-(row,head) warp reductions +
   256-VGPR clamp at GQA=6), not because grouping is impossible.
4. Optionally B2 (q8_0 @262144, ~9.1 GB KV) and B1/B2 to confirm the quality/latency trade for
   a lighter weight quant + f16 KV.