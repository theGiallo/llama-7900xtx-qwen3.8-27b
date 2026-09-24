# Qwen3.8-27B on RX 7900 XTX — analysis and optimization plan

Goal: the fastest Qwen3.8-27B on a single RX 7900 XTX (gfx1100, 24 GB, ~960 GB/s),
keeping **≥ 95 % of the full model's quality**.

Sources: `RESULTS_tok_per_sec.md` in the wrapper repo
(gitlab.com/theGiallo/qwen-3.8-27b_rx7900xtx), `docs/rocmfp4.md`, and the flash-attention
dispatch code in this branch (`ggml/src/ggml-cuda/fattn.cu`, `fattn-vec.cuh`,
`fattn-common.cuh`).

Legend for the plan:
- **[NOW]** — can be done from the cloud session right away (no GPU needed to write it;
  the result needs a GPU run on the real machine to validate).
- **[NOW, analysis only]** — can be investigated/written here, but the conclusion needs data
  from the real machine.
- **[MACHINE]** — needs the physical RX 7900 XTX (WSL2 + ROCm 7.x). No GPU and no `hipcc`
  are available in the cloud container.

---

> **Update:** hardware facts from the AMD/LLVM docs, and the kernel-level findings they led to
> (ROCmFP4 matrix-vector runs at 1 warp per block on RDNA3; ROCmFP4 has no `test-backend-ops`
> coverage on this branch; unaligned FP4 loads), are in `RX7900XTX_KERNEL_STRATEGIES.md`. Its
> experiment list (E0–E9) complements the phases below.

## 1. What the benchmark data says

### 1.1 Decode is not bound by weight size at real context lengths

| fill | IQ3_XXS (10.1 GB) | UD-Q3_K_XL (13.1 GB) | STRIX fp4 (13.8 GB) | Q4_K_S (16.1 GB) |
|---|---|---|---|---|
| 113k tok | 22.8 | 21.9 | 21.4 (fork) | 20.8 |
| 258k tok | 13.68 | 13.26 | 13.79 (fork) | 13.18 |

A 37 % smaller file changes decode speed by ~4–10 %. If reading weights were the limit,
IQ3_XXS would be ~60 % faster than Q4_K_S.

### 1.2 Per-token time grows linearly with context — attention over the KV cache

Linear fit of the fork's STRIX base decode (21.4 t/s @113k, 13.79 t/s @258k):

| component | cost |
|---|---|
| per 1k tokens of filled context | ≈ 0.18 ms / token |
| fixed part (weights + everything else) | ≈ 27 ms / token |

About 19 MB of q4_0 KV per 1k tokens (4.7 GiB / 262144 tokens), read in 0.18 ms, works
out to **≈ 110 GB/s effective, about 12 % of the card's bandwidth**. At 113k that is
~20 of 47 ms per token; at 258k it is ~45 of 72 ms.

The same fit on the merged build (18.57 @113k, 12.82 @258k) gives the same slope but a
**~35 ms** fixed part: the merged build lost ~8 ms/token outside attention (see §3.4).

### 1.3 f16 KV beats q4_0 KV even though it is 3.5× bigger

STRIX @113k: **29.5 t/s with f16 KV vs 21.4 with q4_0**. The dispatch code explains why:

- On RDNA3 (no Turing/Volta MMA, no MFMA), single-token decode with a **quantized** KV
  cache selects `BEST_FATTN_KERNEL_VEC` (last branch of `ggml_cuda_get_best_fattn_kernel`).
- The vector kernel runs one block per **query** head and indexes K/V by
  `head / gqa_ratio` (`fattn-vec.cuh:108-111`). Every KV head is therefore streamed from
  VRAM once per query head that shares it. A per-head KV slice at 100k+ tokens is far
  larger than the 96 MB Infinity Cache, so there is little reuse.
- The 4.7 GiB @256K / 16 layers figure implies ~1024 K elements per layer per token,
  i.e. probably 4 KV heads × 256 dims. With 24 query heads that is **GQA ratio 6 → the
  q4_0 cache is effectively read ~6×**. *(Inferred — verify `head_count` /
  `head_count_kv` in the GGUF metadata.)*
- With **f16** KV and `gqa_opt_applies`, the tile kernel is chosen instead, which groups
  the query heads of a KV head and reads each KV block once.

Hand check at 113k: q4_0 ≈ 2.0 GB × 6 ≈ 12 GB of traffic per token, vs f16 ≈ 7.2 GB
read once. That matches f16 being faster.

### 1.4 Speculative verify batches likely take a slow path

For batches of **more than 2 tokens** (every MTP / DFlash2 verify step) with a quantized
KV cache, dispatch goes to the tile kernel. That sets `need_f16_K/V = true`, which
**converts the whole K and V of the layer to f16 on every call**
(`ggml_cuda_flash_attn_ext_get_f16_extra_data`, `fattn-common.cuh`). At 258k fill that is
gigabytes of conversion traffic per layer per verify. This is a probable contributor to
the merged-build MTP collapse at 258k (2.77 t/s). *(Not yet confirmed — see task A1.)*

### 1.5 Where the fork's advantage actually comes from

- **Not FP4.** RDNA3 has no FP4 hardware; FP4 is only a smaller storage format, decoded
  to int8/fp16 in the kernel. Decode isn't bound by weight size at real contexts (§1.1),
  so a smaller weight format buys little there.
- **The fused MTP draft** (fork 36.3 t/s @113k vs merged 20.6) is the real asset.
- STRIX has the **lowest quality estimate** in the table (~95–97 %, proxy — right at the
  95 % limit). IQ3_XXS GSQ-RCO is smaller, decodes as fast, and its publisher claims it is
  task-lossless on this exact model.

### 1.6 Quality budget

- **Speculative decoding costs no quality at temp 0.** The target verifies every drafted
  token, so MTP, DFlash2 and ngram produce the target's own output (up to floating-point
  noise). The quality budget is spent only on **weight quantization + KV quantization**.
- **q4_0 KV with `LLAMA_ATTN_ROT_DISABLE=1` is probably the biggest current quality risk.**
  The Hadamard rotation is what makes 4-bit KV acceptable; disabling it removes that.
- Most retention numbers in the results file are **proxies from other models**, and the
  GSQ "100.6 %" is the publisher's own claim. None has been measured on this setup yet.

---

## 2. Projection (to be confirmed by measurement)

If §1.3 is right and a quantized-KV decode kernel reads each KV head once at a realistic
~70 % of peak bandwidth:

| fill | today (fork STRIX base) | projected base | notes |
|---|---|---|---|
| 113k | 21.4 t/s (46.7 ms) | **~33 t/s** (~30 ms) | attention ~20 ms → ~3–4 ms |
| 258k | 13.8 t/s (72.5 ms) | **~25 t/s** (~40 ms) | attention ~45 ms → ~8 ms |

Speculative decoding multiplies on top (MTP was ×1.7 at 113k on the fork). Recovering the
merged build's ~8 ms fixed-cost regression (§3.4) adds further headroom.

---

## 3. Plan

Ordered by expected gain per unit of effort.

### Phase A — confirm the diagnosis (cheap, decides everything below)

- **A1 [MACHINE]** Log which flash-attention kernel runs for decode (batch 1) and for
  verify (batch 2–6), for q4_0 / q8_0 / f16 KV. Grep server logs for
  `converting K and V to f16`.
- **A2 [DONE]** `GGML_CUDA_FATTN_LOG=1` prints the chosen kernel, Q batch, GQA ratio, K/V
  types and whether f16 conversion runs, once per distinct shape (`ggml/src/ggml-cuda/fattn.cu`).
- **A3 [DONE]** `scripts/rx7900xtx/fattn_bw.py` + new perf cases in `tests/test-backend-ops.cpp`
  (D=256, 4 KV heads, gqa 6, n_kv 16k/64k/113k/256k, batch 1–8, f16/q8_0/q4_0). Reports
  **GB/s and % of 960 GB/s**, verify cost ratio and kernel choice.
- **A4 [DONE, different approach]** `scripts/rx7900xtx/decode_depth_fit.py`: llama-bench decode
  at several depths per model and KV type, fitted into fixed ms/token (→ weight GB/s) and
  ms per 1k tokens of context (→ KV GB/s, using the attention shapes read from the GGUF).
  This measures the real model instead of guessed matrix shapes. A per-`MUL_MAT` breakdown
  is only needed if the fixed part turns out to be the problem.
- **A5 [MACHINE]** Run `scripts/rx7900xtx/run-step1.sh <model.gguf>` (does A1, A3, A4 in one
  go; see `scripts/rx7900xtx/README.md`) and share `fattn_bw.md` + `decode_depth_fit.md`.

Decision gate: if q4_0 decode attention sits far below peak and f16 is much faster in A3,
§1.3 is confirmed → Phase C is the main job.

### Phase B — no-code wins to try immediately

- **B1 [MACHINE]** IQ3_XXS GSQ-RCO + **f16 KV** @131072 (~10.1 GB weights + ~8.4 GB KV)
  with DFlash2 n5. Expected faster *and* higher quality than q4_0 KV.
- **B2 [MACHINE]** Same with **q8_0 KV** @131072 and @262144.
- **B3 [MACHINE]** Keep the Hadamard rotation **on** for any 4-bit KV run and measure its
  cost separately, instead of disabling it by default.

### Phase Q — make "95 % quality" measurable

- **Q1 [NOW]** Write `scripts/rx7900xtx/quality-kld.sh`:
  1. one-time reference logits with `llama-perplexity --kl-divergence-base` from Q8_0
     (or BF16), partially offloaded — slow but done once;
  2. for every candidate (weight quant × KV type × rotation on/off): mean KLD, 99th-pct
     KLD, top-1 token agreement, PPL ratio;
  3. on **your own** text: the agentic/coding prompt, not only wikitext.
- **Q2 [NOW]** Propose the pass threshold, e.g. same-top-token ≥ 95 % and mean KLD below
  an agreed value, plus a small task spot-check (a handful of real coding tasks with
  pass/fail).
- **Q3 [MACHINE]** Run Q1 for IQ3_XXS, UD-Q3_K_XL, STRIX, Q4_K_S × {q4_0, q4_0+rot, q8_0,
  f16} KV.

### Phase C — the kernel that matters: GQA-grouped quantized-KV decode attention

- **C1 [NOW]** Implement a decode flash-attention path for **quantized K/V** (q4_0, q8_0)
  on HIP/RDNA3 that:
  - processes **all query heads of one KV head in one block**, so each KV block is read
    and dequantized **once**;
  - splits the sequence across blocks (split-K / parallel blocks) with a small reduce
    kernel, to fill 96 CUs at long context;
  - supports Q batch **1–8**, so speculative verify never falls back to whole-cache f16
    conversion (§1.4);
  - uses wave32, 128-bit loads, and the existing q4_0/q8_0 dequant helpers.

  Preferred shape: extend the vector kernel to handle `ncols2 = gqa_ratio` (the concept
  already exists for the tile/MMA kernels) rather than adding a new kernel family. It is
  generic HIP/CUDA code and could go upstream later.
- **C2 [NOW]** Add `test-backend-ops` cases for the real shapes (D=256, GQA 6 or the
  actual ratio, KV 4k–256k, batch 1–8, q4_0/q8_0) so correctness is checked against CPU.
  The CPU reference path can be built and run in the cloud container.
- **C3 [MACHINE]** Compile for gfx1100, run `test-backend-ops` (must stay all-pass), run A3
  before/after, then the real 113k / 258k server benchmarks.

### Phase D — recover the merged build's regressions

- **D1 [NOW, analysis only]** Diff the per-token graph between the fork (`89e531a` lineage)
  and the merged build: rotation ops, extra copies, fusions that stopped applying, HIP
  graph capture. Target: the ~8 ms/token extra fixed cost (§1.2).
- **D2 [NOW, analysis only]** Study the fork's fused MTP draft (`common/speculative.cpp`
  and related code at the fork base, plus the `feat/benchmark` branch on the machine)
  against upstream's `draft-mtp`. Write a porting design for the merged build.
- **D3 [NOW]** Implement the port once D2's design is agreed.
- **D4 [MACHINE]** Validate: MTP @113k back to ≥ 36 t/s; @258k no collapse; acceptance
  back to ~.63 / 4.1.

### Phase E — fixed-cost polish (after C and D)

- **E1 [MACHINE]** Check HIP graphs are actually active under WSL2 and count kernel
  launches per token. WSL2 adds per-launch latency, and at ~1000+ launches per token
  that can be several ms.
- **E2 [NOW]** Tune the FP4 matrix-vector kernel (`vec_dot_q4_0_rocmfp4_q8_1`): 128-bit
  loads, lookup-table UE4M3 scale decode, split-nibble handling without extra shifts,
  VDR and rows-per-block for wave32. Only worth it if STRIX survives Phase Q.
- **E3 [MACHINE]** Verify-batch cost curve: time of an n-token verify ÷ time of 1 token,
  for n = 1..8. That ratio decides the best `--spec-draft-n-max`.

---

## 4. What I can start on right now (cloud session, no GPU)

| # | item | output |
|---|---|---|
| A2 | FA dispatch logging env var | **done**: `GGML_CUDA_FATTN_LOG=1` in `fattn.cu` |
| A3, A4 | bandwidth benchmark scripts | **done**: `scripts/rx7900xtx/` (`run-step1.sh`) |
| Q1, Q2 | KLD quality harness + threshold proposal | `scripts/rx7900xtx/quality-kld.sh` |
| C1, C2 | GQA-grouped quantized-KV decode FA + tests | kernel patch + `test-backend-ops` cases |
| D1, D2 | fork-vs-merged decode and fused-MTP analysis | design notes |
| E2 | FP4 matrix-vector tuning | kernel patch |

Limits: the container has **no GPU and no `hipcc`**. Anything touching kernels is written
and CPU-tested here, but must be compiled for gfx1100 and validated on the real card
(`test-backend-ops` all-pass + before/after benchmarks) before it is trusted.

Suggested order: **A2 + A3 → (machine run) → C1/C2 if confirmed**, with Q1 and D2 in
parallel.

## 5. Rules for the optimization loop (opencode or any agent)

- Rank kernels by **achieved bandwidth vs 960 GB/s**, not by time spent.
- Every kernel change must pass `test-backend-ops` **and** the Phase Q KLD check. Bugs in
  this format produce "silently-wrong-but-plausible" output (`docs/rocmfp4.md`).
- Benchmark with a fixed content type, fixed fill level, several repeats, and report the
  median. WSL2 numbers are noisy.
- Keep changes **additive** (new files, HIP/RDNA3-guarded dispatch) so future upstream
  merges stay cheap.

## 6. Side notes found while reading

- `docs/rocmfp4.md` says the measurement used an "fp4 KV cache" but the flags it lists are
  `-ctk q4_0 -ctv q4_0`. Worth checking which one was actually used.
- The results file says the FP4 speed comes from an "RDNA3 WMMA fp4 path". RDNA3 WMMA has
  no FP4 input type; FP4 values are decoded to int8/fp16 before any WMMA/dot instruction.
