# Step 1 report — where decode time goes on RX 7900 XTX (gfx1100)

Machine run of the Claude Opus 5.5 handoff (`HANDOFF_STEP1_FOR_MACHINE_AGENT.md`),
RDNA3 single 7900 XTX, ROCm 7.x SDK (rocmdev), WSL2.
Fresh HIP build of `F/optimization` @ `410992dfc`, compiled for gfx1100
(`test-backend-ops -o FLASH_ATTN_EXT`: 2/2 backends passed, ROCm0 + CPU).
Runtime: `LLAMA_ATTN_ROT_DISABLE=1`.

Model: `Qwen3.8-27B-ROCMFP4-STRIX.gguf` (14.75 GB weights, 24 query heads / 4 KV heads,
D=256, gqa=6; user script reported "16 full-attention layers" from GGUF reads).
Peak bandwidth assumed 960 GB/s.

Inputs:
- `fattn_bw.md` — `test-backend-ops` perf for the model attention shape (D=256, gqa 6,
  n_kv 16384/65536/113408/262144, batch 1/2/3/4/6/8, KV f16/q8_0/q4_0), kernels from
  `GGML_CUDA_FATTN_LOG=1`.
- `decode_depth_fit.md` — `llama-bench -p 0 -n 64 -d` sweep, fit to fixed ms/token +
  ms per 1k tokens, KV types q4_0/q8_0/f16.

## Headline numbers

### 1) Decode attention @113408, batch = 1 (16 full-attn layers, per token)

| KV     | kernel          | effective GB/s | % of peak | ms/token |
|--------|-----------------|----------------|-----------|----------|
| f16    | TILE            | 720            | 75%       | 10.33    |
| q8_0   | VEC             | 306            | 32%       | 12.93    |
| q4_0   | VEC             | 91             | 9%        | 23.03    |

At 262144: f16 24.15 ms, q8_0 37.64 ms, q4_0 53.09 ms per token.

### 2) Speculative verify (batch >= 3) @113408, q4_0 KV

All batch >= 3 quantized-KV cases run **TILE + full-cache f16 conversion**.
verify x = t(nb) / t(1): nb=3 -> 1.57x, nb=6 -> 2.13x (q8_0: 2.88x / 3.81x).
At 262144 a nb=6 q4_0 verify costs ~6.9 ms/layer x 16 ≈ 110 ms per verify step.

### 3) Fixed part and KV slope (depth fit)

| KV   | fixed ms/token | weight BW   | per-context slope | fit r2 |
|------|----------------|-------------|-------------------|--------|
| q4_0 | 28.06          | 526 GB/s (55%) | 0.151 ms/1k | 0.998 |
| q8_0 | 26.50          | 557 GB/s (58%) | 0.105 ms/1k | 0.925 |

f16 fit is garbage (negative intercept, r2 0.774): VRAM thrash at depth 131072
(0.95 t/s, 1056 ms/tok — weights + f16 KV exceed 24.5 GB, swaps).
Attention-only per-1k slopes (from fontSize table): f16 0.071 < q8_0 0.096 < q4_0 0.22 ms/1k.

## Verdict

Decode-attention hypothesis **CONFIRMED**:

- q4_0 decode (batch 1) uses **VEC** at ~9% of peak; f16 uses **TILE** at ~75%.
  Each KV block is streamed/dequantized once per query head sharing it (gqa=6).
- Quantized-KV speculative verify (batch >= 3) always falls to **TILE+f16conv**:
  the whole visible K/V is converted to f16 on every verify — the MTP collapse driver.
- Per-token cost grows ~linearly with context; per-context slope is KV-bound
  (q4_0 slope > q8_0 > f16 even though f16 moves 3.5x the bytes).
- **No fixed-cost regression in the merged build**: fixed base decode is 26.5-28.1
  ms/token vs the fork's ~27 ms estimate. The earlier "~35 ms merged" figure was not
  reproduced (may have been MTP/old-binary overhead).

## Consequences

- f16 KV decode attention is fastest, but **VRAM-impossible at >=113k** on 24.5 GB
  (weights 14.75 GB + f16 KV thrashes). So "use f16 KV" is not a free win at long context.
- q8_0 KV is the best achievable KV type today, yet still loses to f16.
- **Phase C is the main job**: a GQA-grouped quantized-KV decode path that reads+dequants
  each KV block once would take q4_0 decode attention from 23 ms to ~2.3 ms at 113k
  (~2.2 ms at 100% peak), and batch 1-8 support removes the verify f16 conversion.
- Phase B insight: rotation on/off + 4-bit KV only matters once Phase C exists.

## Tooling bug found (send back to cloud session)

`GGML_CUDA_FATTN_LOG` (A2) only fires in the eager / `test-backend-ops` path: the hook is
placed at the end of `ggml_cuda_flash_attn_ext_get_alloc_size`, which the llama runtime
(server/cli) never reaches — 0 lines even with `GGML_CUDA_DISABLE_GRAPHS=1`.
The log must also be emitted from the op executor (`ggml_cuda_op_flash_attn_ext`) so a
live MTP/speculative run can dump its real shapes.

## Artifacts

- `fattn_bw.md`, `decode_depth_fit.md` — raw reports (committed 0d7be902e).
- build dir `build/` @ 410992dfc, env: rocmdev, GGML_HIP=ON, gfx1100, OpenMP on.