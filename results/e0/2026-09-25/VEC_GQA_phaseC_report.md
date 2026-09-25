# Phase C: VEC_GQA grouped-query FlashAttention — RDNA3 report (RX 7900 XTX, gfx1100)

Machine-measured on the RX 7900 XTX / ROCm 7 rocmdev SDK build, commit 64f6a36eb (kernel)
and the follow-up opt-in dispatch commit (see git log). Peak BW assumed 960 GB/s.
Kernel choice verified with GGML_CUDA_FATTN_LOG=1; times from `test-backend-ops perf`
(576-2000 iterations per case, low variance).

## C1+C2 — Correctness (committed 64f6a36eb)

`flash_attn_ext_vec_gqa`: one block handles all GQA query heads sharing a KV head,
dequantizing q4_0/q8_0 KV once per KV row instead of once per query head.
Dispatched for D=256, K==V in {q4_0, q8_0}, gqa in {2,4,6,8}.

Root cause of the NaN it started from:
the epilogue readback read `VKQ_scratch` through a **float** pointer with stride
`w*V_cols_per_iter*D` (= w*256 floats), but each warp slot was **written** at
`w*(D/2)` half2 units (= 128 float-positions apart). Reads swept 2x ahead of the
written slots, so warps w=2..3 read shared memory this block never wrote (stale
data from previous blocks -> NaN, index varied run to run). Fix: read the slots
through a `half` view so the stride (256 halves = one 512-byte slot) matches the
write layout, mirroring the reference where `KQ` is `half[]`.

Validation: full GPU `test-backend-ops` 16505/16523 at commit time; the previously
failing q8_0 gqa6 kv=16384 case was 1/1 PASS. The remaining 18 failures are the
pre-existing f16/f16 gqa6 long-kv tolerance cases (ERR 0.001-0.006 vs 5e-4) that
fail identically on the reference VEC kernel (verified gate-off) - f16 is not
dispatched to VEC_GQA.

## C3 — Performance (all per-call us, model shape D=256 gqa6, x16 full-attn layers -> ms/token)

f16 control (TILE, unchanged path): 577.6 us vs 645.5 us baseline -> GPU in good
clock state, so the comparisons below are trustworthy.

Decode (nb=1), VEC_GQA vs baseline VEC:

| kv   | KV   | VEC @4109 (us) | VEC_GQA (us) | delta |
|------|------|----------------|--------------|-------|
|113408| q4_0 | 1439.3         | 1736.5       | +21%  |
|113408| q8_0 |  807.8         | 1653.8       | +105% |
|262144| q4_0 | 3318.0         | 3953.9       | +19%  |
|262144| q8_0 | 2352 (est)     | 3697.1       | +57%  |

Verify (nb>=3), VEC_GQA vs baseline TILE+f16conv:

| kv   | nb | q4_0 TILE (us) | VEC_GQA (us) | delta |
|------|----|----------------|--------------|-------|
|113408| 3  | 2263.0         | 4851.3       | +114% |
|113408| 6  | 3063.8         | 8973.6       | +193% |
|262144| 3  | 5143.9         | 12544.9      | +144% |
|262144| 6  | ~6700 (est)    | 20760.0      | +210% |

Verdict: VEC_GQA is correctness-valid but a **uniform performance regression**
(1.19x-2.2x) at every shape it can dispatch (nb<=8). Its throughput is pinned at
~1.5-1.9 TFLOPS regardless of KV size/type -> it is compute/latency bound, not
bandwidth bound. Diagnosed cause: at nb<=8 only n_q * 4 warps are launched
(e.g. 4 errors for nb=1, 12-32 for nb<=8) -> severe under-occupancy, and each
block repeats the per-head softmax + VKQ accumulate for all 6 heads in a
register-heavy kernel. The 6x KV-read/dequant saving is worthless when KV reads
are only 9-32% of peak and the launch leaves the GPU mostly idle.

Prefill-shaped n_rows=512 was checked: the dispatcher now routes that shape to
MMA_F16 (WMMA + f16 conversion) before VEC_GQA is considered, so VEC_GQA is
irrelevant there. VEC_GQA does NOT take over prefill.

## Resolution

Because it regresses every shape it would affect, VEC_GQA is now **opt-in only**
via `GGML_CUDA_FATTN_VEC_GQA=1` (default off, kernels still built+tested). The
default dispatch matches the reference VEC/TILE behavior, so production decode/
verify perf is unchanged from the pre-VEC_GQA build.

Follow-up idea (not done): the occupancy gap could be closed by coarsening each
block over MORE KV rows (larger gridDim.y per block) or more query columns, but
the measured fixed overhead (~1700 us at nb=1) suggests register pressure in the
6-head Q/KQ state dominates; a from-scratch occupancy-first design would be
needed for it to beat VEC/TILE.

## Test suite notes (final build, default dispatch)

16505/16527. Failing:
- 18 f16/f16 gqa6 long-kv (kv 65536/113408/262144, nb 1-8): reference-path
  precision edge (ERR 0.001-0.006, identical on reference), pre-existing.
- 4 q4_0/q8_0 kv=262144 nb=6,8: reference VEC at the extreme (ERR 0.00066-0.00070
  vs 5e-4). These PASS under VEC_GQA (env=1) - VEC_GQA is more precise here,
  another incidental argument to revisit it once occupancy is fixed.
- 4 new prefill D=256 quantized nb=512 cases (MMA_F16): PASS.