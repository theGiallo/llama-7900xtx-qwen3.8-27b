# Handoff: step 1 (find where decode time goes) — for the agent on the RX 7900 XTX machine

From: the Claude Code cloud session (no GPU, no `hipcc`).
To: the agent running on the physical machine (Windows 11 + WSL2, ROCm 7.x, gfx1100).
Branch: `F/optimization`. Full reasoning is in `CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md`.

## 0. Update: hardware research findings (read `RX7900XTX_KERNEL_STRATEGIES.md`)

Three findings from the AMD/LLVM documentation pass affect this step:

1. **ROCmFP4 has no `test-backend-ops` coverage on this branch.** The port (`b783d79`) touched no
   tests, so the "12967/12967 pass" in `docs/rocmfp4.md` was measured on the fork, not here.
   Passing `test-backend-ops` does **not** validate the FP4 kernels until FP4 cases are added
   (strategies doc, experiment E0).
2. **ROCmFP4 matrix-vector decode runs with 1 warp per block on RDNA3.** Q4_0 and IQ4_NL get 8
   (`calc_nwarps()`, `mmvq.cu:511`). This is a likely part of the merged build's ~8 ms/token
   regression. Please also report what the fork's `feat/benchmark` has there:
   `git show feat/benchmark:ggml/src/ggml-cuda/mmvq.cu | grep -n -A20 "RDNA3_0) {"`.
   Do not change it before the user agrees; the A/B needs E0's perf cases.
3. The attention re-read may be partly absorbed by L2 and the 96 MiB Infinity Cache, because the
   6 query heads sharing a KV head run together. A low `fattn_bw.md` number then points at
   redundant instructions and dequantization rather than DRAM traffic. Either way the fix is
   the same.

## 1. Why this step exists

Your own benchmark results (`RESULTS_tok_per_sec.md` in the wrapper repo) show that at
real context lengths **decode is not bound by weight size**. At 113k and at 258k fill, every
model decodes at about the same speed, from 10.1 GB (IQ3_XXS) to 16.1 GB (Q4_K_S).

Fitting the fork's STRIX base decode (21.4 t/s @113k, 13.79 t/s @258k):

- fixed part ≈ **27 ms/token** (weights + everything else); the merged build is ≈ 35 ms, so it
  lost about 8 ms/token somewhere outside attention;
- per-context part ≈ **0.18 ms per 1k tokens**, i.e. q4_0 KV read at **~110 GB/s, about 12 % of
  960 GB/s**. That is ~20 of 47 ms at 113k and ~45 of 72 ms at 258k.
- **f16 KV was faster than q4_0 KV** (29.5 vs 21.4 t/s @113k) despite 3.5× the bytes.

Hypothesis, from reading `ggml/src/ggml-cuda/fattn.cu`:

1. On RDNA3, batch-1 decode with a **quantized** KV cache uses the **VEC** kernel. It runs one
   block per *query* head and reads K/V at `head / gqa_ratio` (`fattn-vec.cuh:108-111`), so
   each KV head is streamed **once per query head sharing it**. That is gqa ≈ 6 for this model
   (24 query heads, 4 KV heads, D=256; inferred, not yet verified).
2. f16 KV takes the **TILE** kernel, which groups the query heads and reads each KV block once.
3. Speculative verify batches (**more than 2 tokens**) with quantized KV take TILE, which first
   **converts the whole visible K/V of each layer to f16**. That is probably part of the MTP
   collapse at 258k.

Step 1 confirms or rejects this before anyone writes a new kernel.

## 2. What the cloud session added (commit `2092519`)

| item | where |
|---|---|
| `GGML_CUDA_FATTN_LOG=1`: one log line per distinct FlashAttention setup (kernel, D, `n_q`, heads, gqa, K/V types, n_kv, whether K/V are converted to f16). Opt-in; no behaviour change when unset. | `ggml/src/ggml-cuda/fattn.cu` |
| Perf cases for this model's attention shape: D=256, 4 KV heads, gqa 6, n_kv 16384/65536/113408/262144, batch 1/2/3/4/6/8, KV f16/q8_0/q4_0 | `tests/test-backend-ops.cpp` (`make_test_cases_perf`) |
| `fattn_bw.py`: runs those cases and prints effective GB/s (K+V counted once), % of peak, verify cost `t(nb)/t(1)`, and the kernel per case | `scripts/rx7900xtx/` |
| `decode_depth_fit.py`: `llama-bench -p 0 -n 64 -d <depths>` per model and KV type, fitted into fixed ms/token and ms per 1k tokens of context, converted to GB/s using the attention shapes read from the GGUF (MTP/nextn layers excluded) | `scripts/rx7900xtx/` |
| `run-step1.sh`: runs both and saves everything in one folder | `scripts/rx7900xtx/` |

Validation done in the cloud:

- `fattn.cu` passes a syntax-only compile through clang's CUDA frontend: no errors and no
  new warnings.
- The perf cases build and run on a CPU build, and `fattn_bw.py` parses their SQL output.
- `decode_depth_fit.py` was checked on a synthetic GGUF and JSONL; it reproduces the
  ~104 GB/s hand estimate.
- **Nothing has been compiled with `hipcc` or run on the GPU yet.** If the HIP build fails
  on `fattn.cu`, fix it minimally and note what you changed.

The branch also contains your `cc7076b` MTP diagnostics commit, merged in with a normal
merge commit.

## 3. What to do

```bash
git fetch origin && git checkout F/optimization && git pull
pip install numpy   # gguf-py needs it for the KV-shape read; otherwise that line is skipped

cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target test-backend-ops llama-bench

# 1) correctness is unchanged (the patch only adds logging, but check anyway)
build/bin/test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT

# 2) step 1 on the STRIX model (same env as the earlier battery)
LLAMA_ATTN_ROT_DISABLE=1 scripts/rx7900xtx/run-step1.sh \
    -o step1-strix /path/to/Qwen3.8-27B-ROCMFP4-STRIX.gguf
```

- The default depths are `0,32768,65536,131072` and the default KV types are
  `q4_0,q8_0,f16`, so expect roughly 30 minutes. To shorten it use `-k q4_0,f16`; to add
  256k use `-d 0,65536,131072,262144`.
- A depth that runs out of memory (e.g. f16 KV at 131072 with a big model) is reported and
  skipped. The run continues.
- If the backend is not called `ROCm0`, set `BACKEND=<name>`. The first lines of any
  `test-backend-ops` run print the backend names.
- Optional, very informative: run `llama-server` once with MTP or DFlash2 and
  `GGML_CUDA_FATTN_LOG=1`, then grep `fattn:` in its log. That shows what the real verify
  batches use.

## 4. How to read the results

The hypothesis is **confirmed** if all of these hold:

- in `fattn_bw.md`, q4_0 at `nb=1` sits far below peak (roughly < 30 %) and f16 gets clearly
  higher;
- q4_0 rows with `nb >= 3` show `TILE+f16conv`, and their `verify x` grows steeply with nb;
- in `decode_depth_fit.md`, the q4_0 "ms per 1k tokens" slope is not smaller than f16's,
  even though f16 moves 3.5× the bytes.

It is **rejected**, or needs rethinking, if q4_0 decode attention already runs at more than
~60 % of peak. In that case the per-context cost comes from somewhere else, such as DeltaNet
state or the MTP draft KV.

Also note from `decode_depth_fit.md` the **fixed ms/token** per model. It tells whether the
merged build's extra ~8 ms/token is real (compare against the fork build if convenient).

## 5. What to send back

Commit the two reports, and nothing else from the output folder, to `F/optimization`:

```
results/step1/<date>-<model>/fattn_bw.md
results/step1/<date>-<model>/decode_depth_fit.md
```

Include in the commit message: the git SHA used, the ROCm version, whether
`LLAMA_ATTN_ROT_DISABLE` was set, and anything you had to change to build. Use a normal
merge if the branch moved, **never force-push**. Then tell the user the headline numbers:

- q4_0 vs f16 GB/s at nb=1 for 113k;
- the verify `x` at nb=6 for 113k q4_0;
- the fixed ms/token and the KV GB/s from the depth fit.

## 6. Next steps depending on the result (for context; do not start without the user)

- **Confirmed** → Phase C of the plan: a decode FlashAttention path for quantized K/V that
  handles all query heads of a KV head in one block (each KV block read and dequantized
  once), splits long sequences across blocks, and supports batch 1–8 so verify never
  converts the cache to f16. Validate with `test-backend-ops` plus before/after `run-step1.sh`.
- **Cheap wins to try in parallel (no code)**: IQ3_XXS GSQ-RCO with f16 KV at 131072 and
  DFlash2; q8_0 KV at 131072/262144; 4-bit KV with the Hadamard rotation **on**.
- **Quality**: before trusting any config, measure KL divergence against a Q8_0/BF16 reference
  on your own agentic text (`llama-perplexity --kl-divergence-base`). Most retention numbers
  in the results file are proxies from other models.

## 7. Rules for the optimization loop

- Rank kernels by **achieved bandwidth vs 960 GB/s**, not by time spent. The biggest kernel may
  already be efficient.
- Every kernel change must pass `test-backend-ops`. Once Phase Q exists, it must also pass the
  KLD check. ROCmFP4 bugs are "silently-wrong-but-plausible" (`docs/rocmfp4.md`).
- Fixed content type, fixed fill level, several repeats, report the median. WSL2 is noisy.
- Keep changes additive and HIP/RDNA3-guarded so upstream merges stay cheap. Follow
  `AGENTS.md`: the original fork checkout stays read-only.
