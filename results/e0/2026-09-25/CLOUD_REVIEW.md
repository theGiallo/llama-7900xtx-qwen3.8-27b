# Cloud review of the 2026-09-25 machine results (E0, E1/E2, MTP probe, rotation fit)

## 0. Phase C is not on any pushed branch

The user reports that Phase C (the GQA-grouped quantized-KV FlashAttention kernel) is done, and
that its outcome invalidates Phase D. No such kernel is on `F/optimization`,
`merge-upstream-head`, `feat/benchmark`, `main`, or the GitLab wrapper repo. **Please commit and
push it** (kernel diff, `test-backend-ops -o FLASH_ATTN_EXT` result, `fattn_bw.md` before/after,
and the end-to-end numbers that support the Phase D conclusion) so it can be reviewed.

Until then, the Phase D verdict below is based only on what is pushed.

## 1. What holds

- **E0: 42/42 ROCmFP4 cases pass on ROCm0.** FP4 is now covered by `test-backend-ops`.
- **Rotation is essentially free** (27.07 vs 28.06 ms fixed, slope 0.132 vs 0.151 ms/1k). My
  hypothesis that it explained the old ~35 ms estimate was wrong. **Run 4-bit KV with rotation
  on.**
- **E2 kernel change (`b4fba04`) is correct.** Word mapping matches the old loops, the `_fast`
  8-byte load stays inside the block for `iqs ∈ {0, 2}`, and 42/42 pass. Keep it.
  - Small follow-up: in `vec_dot_q4_0_rocmfp4_q8_1`, `q4[iqs + l]` indexes the `uint4` with the
    runtime `iqs`. It is always 0 there (VDR 4 = QI), which is presumably why it didn't spill.
    Writing `q4[l]` with a comment makes that explicit and robust.
- **MTP probe:** verify (n_q=4) with quantized KV = TILE + whole-cache f16 conversion (203 MiB per
  layer at 52k). Prompt fill with quantized KV also converts (MMA_F16 + conversion per chunk).
  Good trace.

## 2. Corrections

### 2.1 The MUL_MAT perf numbers are cache-hot; E0 §3 and the E1 conclusion don't carry over to decode

- ROCmFP4 17408×5120 = 17408 × 5120 × 18/32 B = **50.1 MB** of weights in **35.8 µs** ⇒
  **1.40 TB/s**, ~1.5× the card's 960 GB/s. That is only possible from the **96 MiB Infinity
  Cache**: `test-backend-ops perf` repeats the same matmul, and a 50 MB matrix stays resident.
- Real decode streams ~14.75 GB of weights per token from VRAM (64 layers × gate/up/down plus
  attention and DeltaNet projections), so nothing stays cached between uses.
- Therefore:
  - "FFN is ~1.1–1.3 ms of the 28 ms fixed part" is wrong. The FFN weights alone are
    64 × 3 × 50.1 MB ≈ 9.6 GB per token ⇒ ≥ 10 ms at peak bandwidth, and ~17–18 ms at the
    measured ~55 % efficiency. The FFN is most of the fixed part.
  - "Warps are not the lever" is not established. More warps mainly help hide **VRAM latency**
    (more loads in flight), which a cache-hot benchmark never exercises. The k=5120 idle-thread
    point is valid, but it argues for rows-per-block > 1 on RDNA (as `should_use_small_k` does
    elsewhere), not against more parallelism.
  - The E2 gain (3–8 %) was also measured cache-hot. Instruction-count savings usually survive
    in the DRAM-bound case, but re-measure.
- **How to measure DRAM-bound MMVQ:** use a batch of matrices bigger than the Infinity Cache,
  e.g. `test_mul_mat(type, F32, 17408, n, 5120, {8, 1}, {1, 1})` (8 × 50 MB = 400 MB). Or, simpler
  and the real metric: `llama-bench -p 0 -n 128 -d 0` tokens/s on the model, before and after.
  Judge E1/E2 by that.

### 2.2 The whole-cache conversion does not explain ~104 t/s fill

A 512-token chunk at 104 t/s takes ~4.9 s. The per-chunk conversion at 52k is
16 layers × (≈57 MB read + ≈201 MB write) ≈ 4.1 GB ⇒ **~5–10 ms**, i.e. < 0.2 % of the chunk.
Something else makes fill slow in that server config: candidates are MTP draft-context prompt
processing, `--ctx-checkpoints` snapshots, or the 262144 `-c` with q4_0 draft KV. The same model
filled at 512 t/s at 113k in the step-1 `llama-bench` runs without MTP. Worth one A/B: the same
server command with and without `--spec-type draft-mtp`.

## 3. Does Phase C invalidate Phase D?

Phase D in `CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md` has two parts:

- **D1 (recover the merged build's fixed-cost regression): invalidated.** Agreed, but by the
  step-1 and rotation fits, not by Phase C. The fixed part is 27–28 ms on this branch with
  rotation on or off, matching the fork's ~27 ms. There is no regression to recover. (The ~8 ms
  headroom of the fixed part vs the bandwidth floor remains, but that is Phase E / MMVQ work.)
- **D2/D3 (port the fork's fused MTP draft): not yet shown to be unnecessary.** Phase C removes
  the verify-time f16 conversion and makes quantized-KV attention cheap for the target and the
  MTP draft head, which plausibly explains most of the MTP collapse. But the fork reached
  **36.3 t/s @113k and 24.8 t/s @258k** with MTP using the *same slow attention kernels*, so its
  fused draft path avoided some cost the merged build still pays. D2 is invalidated only if the
  merged build **with Phase C** matches or beats the fork on the same MTP workloads:

  | workload (STRIX, MTP n5) | fork | merged before C | merged + C (needed) |
  |---|---|---|---|
  | agentic 113k fill, q4_0 KV | 36.3 t/s, acc .63/4.1 | 20.6 (rot on) / 27.1 (rot off) | ? |
  | network 258k fill, q4_0 KV | 24.8 t/s | 2.77–2.86 | ? |

  If "merged + C" reaches the fork's numbers with similar acceptance, D2 is indeed moot. If it
  stays below, the remaining gap is the fused-draft path and D2 stands.

## 4. Requests

1. Push the Phase C kernel and its evidence (§0), including the two MTP rows in §3.
2. Re-judge E1/E2 with `llama-bench -p 0 -n 128 -d 0` tokens/s (or the 400 MB batched perf case).
3. One A/B of the ~104 t/s fill with and without `--spec-type draft-mtp` (§2.2).
