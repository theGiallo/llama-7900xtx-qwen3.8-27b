# Plan phase status — Qwen3.8-27B on RX 7900 XTX (gfx1100, ROCm 7)

Status snapshot against `CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md` (Phase A/B/Q/C/D/E).
Machine: RX 7900 XTX (24.5 GB), WSL2, rocmdev SDK, `F/optimization` branch.
HEAD: d25f0a254. Decode fixed base ~26.5-28.1 ms/tok; peak BW assumed 960 GB/s.

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
- **B2** q8_0 KV @131072/262144: NOT STARTED.
- **B3** rotation-ON cost: DONE — rotation is ~free (27.07 vs 28.06 ms fixed, not the
  suspected ~8 ms regression). results/step1/rot-q4f16-strix/ROTATION_FIT.md, commit a5cf12505.
- Also: f16 KV is VRAM-impossible at ≥113k on 24.5 GB (weights 14.75 GB + f16 KV thrash),
  so B1/B2 would need a lighter weight quant to fit.

## Phase Q — make "95% quality" measurable: NOT STARTED
- No scripts/rx7900xtx/quality-kld.sh, no KLD/agreement baselines, no Q3 matrix run.

## Phase C — GQA-grouped quantized-KV decode attention: DONE, but a measured REGRESSION
- **C1 kernel** `flash_attn_ext_vec_gqa`: DONE (new fattn-vec-gqa.{cu,cuh}, commit 64f6a36eb).
  One block per KV head, all GQA heads share the dequantized KV; D=256, K==V∈{q4_0,q8_0},
  gqa∈{2,4,6,8}. Original NaN root cause: epilogue readback read the slot region with a
  float-typed 256-element stride while slots are stored 128 float-positions apart → warps
  2..3 read never-written shared memory → stale NaN. Fixed by reading via a half view
  (matches the reference layout where KQ is half[]).
- **C2 tests**: DONE. 79-case gqa6 suite (kv 16k..262k, nb 1..8, f16/q8_0/q4_0) + extra
  gqa-2/4/8 and prefill nb=512 cases. Full GPU suite 16505/16527.
- **C3 benches**: DONE — NEGATIVE. VEC_GQA is 1.19-2.2x SLOWER than incumbents everywhere:
  decode nb=1 +21% (q4_0) to +105% (q8_0) vs VEC at 113k; verify nb≥3 +114..223% vs
  TILE+f16conv at 113k/262k. Pinned ~1.5-1.9 TFLOPS → compute/latency bound: at nb≤8 only
  n_q×4 warps launch (4 warps at nb=1) → severe under-occupancy; per-head softmax+VKQ
  accumulate ×6 in a register-heavy kernel. n_rows=512 prefill is taken by MMA_F16 upstream.
- **Resolution**: VEC_GQA now **opt-in** (`GGML_CUDA_FATTN_VEC_GQA=1`, default off); default
  dispatch = reference behavior → production decode/verify perf unchanged (commit d25f0a254).
  VEC_GQA is more precise at the extreme (kv=262144 nb 6/8 quantized pass on it, marginal
  on reference VEC). Full details: results/e0/2026-09-25/VEC_GQA_phaseC_report.md.

## Phase D — recover merged-build regressions / fused MTP: PARTIAL (analysis only)
- **D1 graph diff / ~8 ms fixed cost**: effectively DONE under Step 1 — no fixed-cost
  regression found (fixed base 26.5-28.1 ms vs fork ~27 ms; the earlier "~35 ms" not
  reproduced). The remaining topic is not a merge regression.
- **D2 fused MTP port design**: analysis DONE (MTP trace: verify = TILE + whole-cache f16
  conversion, the collapse driver; results/mtp/). Porting DESIGN/PLAN for the merged build:
  NOT STARTED.
- **D3 port** and **D4 validation**: NOT STARTED. NOTE: Phase C's premise (a GQA-grouped
  kernel removes the verify f16-conversion cheaply) is now falsified by measurement, so the
  D2/D3 design should pick a different lever (e.g. MMA_F16 or a Q-rows≥2 vector path for the
  verify batch, or a fused MTP draft that reuses one attention pass).

## Phase E — fixed-cost polish: PARTIAL
- **E1 HIP graphs / per-token launch latency**: NOT DONE.
- **E2 ROCmFP4 MMVQ tuning**: DONE — single wide qs load etc. (3-8% FFN decode).
  results/e0/2026-09-25/E1E2.md, commit b4fba04ed.
- **E3 verify-batch cost curve nb=1..8**: DONE (fattn_bw.md verify-x tables; C3 re-measured
  under VEC_GQA for reference).

## Priority recommendations (next)
1. Close Phase Q (quality-KLD script + thresholds) so perf work has an acceptance gate.
2. Phase D2/D3: design the fused-MTP port WITHOUT relying on VEC_GQA (use the measured data
   in VEC_GQA_phaseC_report.md); offload a design for review before coding.
3. Only revisit VEC_GQA if an occupancy-first redesign is explicitly desired (coarser blocks
   / lower 6-head register footprint) — evidence suggests it is the wrong shape for nb≤8.
4. Optionally B1/B2 to confirm the quality/latency trade for a lighter weight quant + f16 KV.