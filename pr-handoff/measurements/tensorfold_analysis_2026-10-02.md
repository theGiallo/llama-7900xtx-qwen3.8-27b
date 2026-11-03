# TensorFold ideas, re-ranked with measurements on this machine (2026-10-02, night session)

Earlier analysis: `TENSORFOLD_ANALYSIS.md` (tree 4, commit `179c496cc`): seven candidate techniques ranked from the TensorFold docs only. This note
replaces the ranking with what was measured on the RX 7900 XTX (gfx1100, WSL2, ROCm), tree 4 `~/llama-rocmfp4-merge`, UD-Q4_K_S dyn + MTP, `-c 131072`,
113,182-token agentic prompt, q4_0 KV, 128 generated tokens, idle GPU.

## 1. Where one round goes (measured)

| quantity | value |
|---|---|
| serial decode at 113k fill (no draft) | 23.13 tok/s = **43.2 ms/token** |
| MTP n_max 3 round (`SERVER_SPEC_PROF`, avg 4 verify columns) | 77-80 ms: verify decode 66-69, draft 10, other ~1; yields 3.34 tokens (42.1-42.9 tok/s) |
| verify of 4 columns vs one serial token | 66 ms vs 43 ms = **1.5x** (TensorFold reports 1.25x for 12 rows) |
| verify graph, eager per-op profile (`GGML_CUDA_OP_PROF`, graphs off, 4 columns, total 119 ms with launch overhead) | `FLASH_ATTN_EXT` 27.8 ms (23%), RMS_NORM 12.8 (305 launches, mostly eager overhead), CPY 6.9 (192 snapshot copies), all MUL_MAT n4 ~35 |
| attention kernels at 113k, one layer, `test-backend-ops perf` (hs 256, 4 KV heads, GQA 6, q4_0) | decode (1 col, vector kernel) **1172 us**; 4 columns (tile kernel + f16 conversion of the whole K/V) **2054 us**; the data is 130 MB per layer = ~140 us at DRAM speed |

So the verify cost is dominated by attention at long context (about 28 of 66 ms), not by the weights matmuls (n4 vs n1 is almost flat, as TensorFold's
row-invariant verify promises) and not by the recurrent-state snapshots: 4 snapshots x 3 MiB x 48 layers = ~1.2 GB of traffic per round = ~1.3 ms (2%).
That moves the ranking:

| TensorFold idea | earlier estimate | after measuring |
|---|---|---|
| T3 commit by replay / direct-write GDN snapshots | medium, "3-6% of a round" | **~2% (1.3 ms of 66)**: not worth a prototype |
| T6 draft vocabulary (row view of the drafter's lm_head) | small | MTP draft step is 3.3 ms of which the q3_K lm_head matvec 0.75 ms; restricting it to 98k rows saves ~0.4 ms/step, ~1.2 ms/round (1.5%) |
| T5 cheaper drafter | measure drafter ms/round | 10 ms/round (13%); MTP head is one layer, nothing to fuse cheaply |
| T1 row-invariant verify matmul | medium | MUL_MAT n4 costs ~1.0-1.2x n1 already; the gap is attention |
| T4 draft trees | large | the round is already 1.5x a serial token; trees need a tree mask in attention, which is the expensive part here |
| **"Phase C": attention for 3-16 query rows without the f16 conversion of the whole cache** | listed as part of T1 | **the biggest measured item; prototyped, see below** |
| `--spec-draft-n-max` | tuning only | the biggest single config lever: n_max 3 = 42.1 tok/s vs n_max 5 = 31.6 (README Part 12) |

## 2. Negative results on the way (flags kept, off by default, branch `exp/fattn-vec-cols4`)

- Vector kernel for 3-4 columns (`GGML_CUDA_FATTN_VEC_COLS4=1`): 29.1 tok/s vs 42.95 (-32%).
- WMMA kernel for 4 columns (`GGML_CUDA_FATTN_WMMA_MIN_ROWS=4/8`): 30.9 / 31.2 tok/s (-27%).
- The existing opt-in GQA vector kernel was already known to be slower (comment in `fattn.cu`).

## 3. The prototype: fused q4_0 dequantization in the tile kernel (branch `exp/fattn-tile-q4` @ `ac08925ea`, flag `GGML_CUDA_FATTN_TILE_Q4`)

Why: with q4_0 KV the tile kernel (the one used for 3-16 query columns) first converts the *whole visible K and V* to f16 in global memory
(463 MB written + read back per layer at 113k; the same scratch is part of the compute buffer) and only then runs. The new variant dequantizes
K/V tiles straight from the q4_0 cache into shared memory (one thread per 32-element block, same arithmetic as `dequantize_q4_0` + float->half) and
never allocates the f16 scratch for those batches. Head size 256 and q4_0/q4_0 only; `=1` for 3-16 columns (verify), `=2` also for 1-2 columns
(decode, replacing the vector kernel). Prefill (more than 16 columns) is unchanged.

| measurement | default | fused | change |
|---|---|---|---|
| attention, 1 column, 113k, per layer (`test-backend-ops perf`) | 1172 us | **680 us** (mode 2) | -42% |
| attention, 4 columns, 113k, per layer | 2054 us | **1369 us** | -33% |
| UD-Q4_K_S, no speculation, 113k fill | 23.13 tok/s | **26.97** (mode 2) | **+16.6%** |
| UD-Q4_K_S + MTP n_max 3, 113k fill | 42.89 (42.1-42.95 over 4 runs) | **46.05** (mode 1), 45.95 (mode 2) | +7% |
| STRIX fp4 + MTP n_max 5, 113k fill | 42.98 | **47.43** (mode 2) | **+10.4%** |
| STRIX fp4 + MTP n_max 3 + footprint flags, 262144 ctx, 258,801-token fill | 34.42 | **41.07** (mode 2) | **+19.3%**, identical output hash |

Exactness: STRIX run has an identical output hash and acceptance (.685 / 4.38) with and without the flag. UD-Q4_K_S + MTP gives a different
continuation at 113k (acceptance .779 vs .795) but the same text at a 27k prompt (acceptance .857 vs .824), i.e. tiny numerical differences that flip
near-ties; the cause of the non-identity was not found (the loader uses the same arithmetic as the converter). Quality, measured with
`llama-perplexity` (wiki_q, `-c 8192 -b 8 -ub 8` so that every token goes through the fused path, 2 chunks, q4_0 KV): PPL 4.8810 (default) vs
4.8769 (fused), standard error +-0.12. `test-backend-ops` FLASH_ATTN_EXT, hs 256 / GQA 6 / q4_0, kv up to 262144, 1-8 columns: the same two
pre-existing tolerance failures (kv=262144, nb=6 and 8) with and without the flag at essentially the same error (0.00067), and the same crash in the
softcap=2 case with and without it.

Why not more: GQA is 6 here, so the tile kernel can group only 2 query heads per block and each KV head is read 3 times (3 x 130 MB per layer =
~430 us at DRAM speed against 680 us measured). Reading each KV head once needs a kernel that groups 6 (or 8, padded) query heads; that is the next step,
not attempted tonight. Raising the occupancy hint of the tile config made no difference (tested, reverted).

## 4. What is left, ranked
1. Make the fused path the default for q4_0 KV at head size 256 after a longer quality run (KLD vs the Q8_0 reference with `-ub 8`), and extend to q8_0 KV.
2. Attention for 3-16 columns on matrix cores (WMMA) without the f16 conversion (padding the 6 query heads to one group of 8 only helps single-token decode, see section 5).
3. `--spec-draft-n-max 3` as the documented default for these models (config only).
4. Snapshots by direct write and a draft vocabulary: 1-2% each, only worth doing after 1-2.

## 5. Addendum: GQA-grouping sensitivity of the fused kernel (`test-backend-ops perf`, hs 256, 4 KV heads, kv 113408, q4_0, per layer)

| GQA ratio (query heads per KV head) | query groups per KV head (ncols2) | 1 column | 4 columns |
|---|---|---|---|
| 1 | 1 (1) | 259 us | 413 us |
| 2 | 1 (2) | 256 us | 483 us |
| 4 | 1 (4) | 399 us | 744 us |
| **6 (Qwen3.8-27B)** | **3 (2)** | **675 us** | **1361 us** |
| 8 | 1 (8) | 480 us | 1901 us |

Time grows with the number of query rows in the block, not with how often K/V are read: a one-group kernel at ratio 8 is no faster at 4 columns than the
three-group one at ratio 6. So padding the 6 heads to one group of 8 would save ~200 us per layer for single-token decode (~3 ms of 43 ms) and lose
for 4-column verify. The tile kernel is bound by per-row vector work on the K/V tiles (~57 us per row and layer at 113k), not by DRAM. Dropped as the next
step; a matrix-core (WMMA) path that does not need the f16 conversion would be the real next kernel, and the earlier WMMA test (with conversion) lost.

## 6. Addendum: weight-matmul scaling with the number of columns (T1 "row-invariant verify"), measured

`test-backend-ops perf`, m = 17408, k = 5120, batched 8x (weights do not fit the Infinity Cache), us per matmul, branch `diag/mmvq-ncols-scan`:

| type | n=1 | n=2 | n=4 | n=8 | n=4 / n=1 |
|---|---|---|---|---|---|
| q4_0 | 587 | 498 | 496 | 897 | 0.85 |
| iq4_nl | 491 | 461 | 499 | 875 | 1.02 |
| **iq4_xs** | 419 | 433 | 459 | 773 | **1.10** |
| iq3_s | 427 | 473 | 642 | 921 | 1.50 |
| iq3_xxs | 385 | 449 | 581 | 904 | 1.51 |
| q6_K | 667 | 714 | 1021 | 1717 | 1.53 |
| q5_K | 543 | 584 | 918 | 1609 | 1.69 |
| q3_K | 507 | 646 | 937 | 1559 | 1.85 |
| q4_K | 461 | 534 | 864 | 1556 | **1.87** |

So the K-quants (the bulk of UD-Q4_K_S / UD-Q3_K_XL) are compute-bound in the MMVQ kernel at 4 columns (6-7 TFLOPS of int8 work against 12-14 for iq4_xs / q4_0 / iq4_nl)
and a 4-column verify pays about +0.9x a serial token for them, while q4_0 / iq4-class weights are flat. This is consistent with STRIX (fp4-in-q4_0-like layout) gaining more from
speculation than the K-quant models do, and means the quant choice of a model matters for speculative decoding as much as for plain decode.

Tuning attempt (negative): `calc_nwarps` / `calc_rows_per_block` for RDNA3 at 2-8 columns on Q3_K..Q6_K. Warps per block 1 / 2 / 4 and rows per block 1 / 2 give, for q4_K at n=4,
871 / 907 / 1061 us (1 row) and 872 / 905 us (2 rows): the current setting (1 warp, 1 row) is the best; more warps are worse. The inefficiency is in the per-column scale/min
arithmetic of the K-quant dot product, not in the launch shape; a dedicated kernel that handles all columns of a block of weights together is the next step (not attempted).

## 7. Addendum: KL divergence of the fused path against the Q8_0 reference (2026-10-02)

STRIX fp4 (ROCMFP4-STRIX), `-c 4096`, q4_0 KV, `-b 8 -ub 8` (so every position is decoded through the small-batch attention path), the stored Q8_0 reference logits
(`~/models/quality/matrix_q8/*.ref.kld`), `llama-perplexity --kl-divergence`, `build-exp` of tree 4 (`exp/fattn-tile-q4`), flag off vs `GGML_CUDA_FATTN_TILE_Q4=2`:

| corpus | flag | mean KLD | same top-p | PPL(Q) |
|---|---|---|---|---|
| coding_q | off | 0.046585 +- 0.001352 | 87.621 +- 0.326 % | 3.0954 +- 0.0599 |
| coding_q | **on** | 0.046979 +- 0.001396 | 87.787 +- 0.324 % | 3.0956 +- 0.0599 |
| wiki_q | off | 0.090210 +- 0.006990 | 89.301 +- 0.342 % | 5.6070 +- 0.1492 |
| wiki_q | **on** | 0.095316 +- 0.007675 | 89.179 +- 0.343 % | 5.5633 +- 0.1471 |

All differences are within one standard error. Together with the UD-Q4_K_S perplexity check (section 3) there is no measurable quality change; the flag could be made the
default for q4_0 KV at head size 256 on gfx1100 (not done: the decision is the user's). Raw logs: `~/bench/dram/kld_{coding_q,wiki_q}_f{0,2}.log` in WSL.

## 8. Addendum 2026-10-03: the same KL-divergence check on UD-Q4_K_S (B3)

UD-Q4_K_S dynamicV3.0, `-c 4096`, q4_0 KV, `-b 8 -ub 8` (every position through the small-batch attention path), stored Q8_0 reference logits, `llama-perplexity --kl-divergence`, tree 4 `exp/fattn-tile-q4`, flag off vs `GGML_CUDA_FATTN_TILE_Q4=2`
(GPU clocks capped at 2850 MHz, driver healthy):

| corpus | flag | mean KLD | same top-p | PPL(Q) |
|---|---|---|---|---|
| coding_q | off | 0.017697 +- 0.000831 | 93.972 +- 0.235 % | 3.0231 +- 0.0578 |
| coding_q | **on** | 0.017075 +- 0.000816 | 94.148 +- 0.232 % | 3.0175 +- 0.0576 |
| wiki_q | off | 0.042987 +- 0.006380 | 94.529 +- 0.251 % | 5.8018 +- 0.1636 |
| wiki_q | **on** | 0.038395 +- 0.005946 | 94.871 +- 0.244 % | 5.8041 +- 0.1639 |

All differences are within one standard error (the fused values are, if anything, slightly better). With the STRIX result (section 7) two models on two corpora show no measurable quality change from the fused path, so making it the default for
q4_0 KV at head size 256 on gfx1100 is supported by the evidence; the decision stays with the user (it is still a flag). Raw logs: `~/bench/dram/kldu4_*.log` in WSL.

## 9. Addendum 2026-10-03: K-quant multi-column MMVQ (B5): ceiling test, dropped

`test-backend-ops perf`, q4_K, m = 17408, k = 5120, batched 8x, us per matmul (tree 4 `exp/fattn-tile-q4`):

| | n=1 | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| as is | 456 | 543 | 866 | 1563 |
| ceiling test: the sum-of-activations dot product (`dot2`, 4 of the 8 dp4a per column and pure y-side work repeated for every weight row) removed (wrong numerics, timing only) | 456 | 495 | **792** | **1417** |

Even deleting that term entirely gives only -8.6% at 4 columns and -9% at 8, so it is not where the cost of a K-quant verify matmul sits (n=4 stays 1.74x n=1). The rest is the per-column activation loads and scale
arithmetic, which a launch-shape change (section 6) or a smaller edit does not remove; forcing the int8-matrix MMQ path for verify batches was already negative (decode_investigation README part 4). A dedicated multi-column kernel would need a
different algorithm and its ceiling is a few percent of a round (the matmul share), so it was not built.

## 10. Addendum 2026-10-03: prefill attention on the matrix cores (B4): configuration tuning, negative

At 258k tokens of fill the prefill is dominated by attention (quadratic), which runs on the WMMA (MMA_F16) kernel at 35-39 TFLOPS against a ~123 TFLOPS fp16 peak. `test-backend-ops perf`, hs 256, 4 KV heads, GQA 6,
512 query columns, kv 65536 (f16 KV; the q4_0 row includes the f16 conversion), configuration of the 64-column case in `ggml_cuda_fattn_mma_get_config_rdna` (nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, stages, Q_in_reg):

| configuration | f16 KV | q4_0 KV |
|---|---|---|
| **as is** (256, 2, 64, 128, 128, 64, 1, true) | 35.1-38.6 TFLOPS | 39.4 TFLOPS |
| occupancy 1 | 35.2-35.7 | 39.5 |
| 128 threads | 16.2-19.8 | 22.3 |
| nbatch_fa 32 | 27.2-27.5 | 36.6 |
| nbatch_fa 128, 512 threads, Q_in_reg false | do not build / run | do not build / run |

The shipped configuration is the best of the variants that run; there is no cheap gain. A better prefill at 258k would need a restructured kernel (software pipelining without `cp.async`, larger tiles), not parameters; the f16 conversion of the K/V
cache is only ~3% of a ubatch at that depth. Not attempted.
