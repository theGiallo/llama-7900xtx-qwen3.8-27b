# E0 — ROCmFP4 correctness + MUL_MAT FFN baseline (gfx1100, 2026-09-25)

Branch HEAD: `5efe75c30`. ROCm: rocmdev SDK (Clang 23 / GGML 0.25.1). Backend `ROCm0`.
`LLAMA_ATTN_ROT_DISABLE` not relevant (no model workload). 4-type FFN decode baseline
for E1/E2.

## 1. Correctness — ROCmFP4 coverage

`test-backend-ops -b ROCm0 -p "rocmfp4"` → **42/42 tests passed** on ROCm0 vs CPU
reference (`2/2 backends passed`). First FP4 validation on this branch (the fork's
"12967/12967 pass" was measured elsewhere; here it is now covered by `test-backend-ops`
cases added in `d5a1e9fde`).

## 2. MUL_MAT perf baseline (us/run at n=1 and scale with n)

`test-backend-ops perf -b ROCm0 -o MUL_MAT -p "type_a=(q4_0_rocmfp4|q4_0_rocmfp4_fast|q4_0|iq4_nl),type_b=f32,m=(17408|5120),n=[1-8],k=(17408|5120),"`
(raw log in `mmvq_ffn_perf.log`).

| type_a             | m / k         | n=1 us/run | n=2   | n=4   | TFLOPS@n=1 |
|--------------------|---------------|-----------|-------|-------|------------|
| q4_0_rocmfp4       | 17408 / 5120  | **35.80** | 41.60 | 62.23 | 4.98 |
| q4_0_rocmfp4_fast  | 17408 / 5120  | 40.19     | 47.84 | 67.43 | 4.43 |
| q4_0               | 17408 / 5120  | 54.87     | 41.91 | 64.94 | 3.25 |
| iq4_nl             | 17408 / 5120  | 58.48     | 43.75 | 66.18 | 3.05 |
| q4_0_rocmfp4       | 5120 / 17408  | **35.07** | 44.54 | 66.80 | 5.08 |
| q4_0_rocmfp4_fast  | 5120 / 17408  | 41.54     | 52.86 | 75.08 | 4.29 |
| q4_0               | 5120 / 17408  | 35.36     | 42.77 | 67.72 | 5.04 |
| iq4_nl             | 5120 / 17408  | 38.65     | 48.62 | 74.06 | 4.61 |

## 3. Reading

- **No n=1 regression to blame on 1 warp**: at the decode shape (n=1) ROCmFP4 is the
  fastest type on the 17408x5120 FFN (35.8 vs 54.9 us for Q4_0) and tied on the
  5120x17408 direction (35.1 vs 35.4). `q4_0_rocmfp4_fast` is consistently ~4-6 us
  SLOWER than plain `q4_0_rocmfp4` at n=1 here.
- n=1 is ~35 us in every type (floor looks latency/fixed-cost, not bandwidth): the two
  5120x17408 and 17408x5120 shapes cost the same. At 4.98-5.08 TFLOPS ROCmFP4 exceeds
  Q4_0 (3.2-5.0).
- This floor (~35 us at n=1 per FFN) is the "fixed part" ingredient: 32-36 FFN/decode
  step would be ~1.1-1.3 ms/decode of the ~28 ms fixed part — attention and KV dominate.

## 4. Follow-ups

- E1/E2 kernel variants (warps 1→{2,4,8}, aligned 128-bit loads) should be measured
  against this n=1 floor; if the ~35 us holds, the win is elsewhere (instruction count,
  not parallelism).
- `q4_0_rocmfp4_fast` being slower than `q4_0_rocmfp4` at n=1 is worth checking first.