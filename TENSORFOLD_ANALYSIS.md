# TensorFold → llama.cpp on RX 7900 XTX: what's portable

Source: https://github.com/ashhart/TensorFold @ `3283e59` (0.3.0, 2026-09-26), MIT license
(Apache-2.0 for vendored parts). Read: `README.md`, `docs/recipes/qwen3.8-27b.md`,
`src/tensorfold/families/qwen3_5/cuda/*`.

## 1. What TensorFold is

A Python inference server (MLX on Apple Silicon; PyTorch + Triton + small CUDA extensions on
NVIDIA) with hand-written kernels per model family. **Qwen3.8-27B is a first-class target,
drafting with the same `z-lab/Qwen3.8-27B-DFlash2` drafter we use.** No AMD/ROCm support.

Its headline for our model (DGX Spark, ~240 GB/s measured read, one stream):

| | tok/s |
|---|---|
| serial decode (no drafts) | 13.1 |
| vLLM + MTP=3 (NVFP4) | 15.0–17.7 |
| **TensorFold + DFlash2 (tree)** | **45.8–49.6** |

That's 3.5–3.8× serial on a card with a quarter of the 7900 XTX's bandwidth. A 12-row verify
round costs ~95 ms vs ~76 ms for one serial token (1.25×) and yields ~4.7 tokens.

## 2. Why it matters for us: our verify rounds are expensive

From our own results: UD-Q3_K_XL + DFlash2 @113k = 37.9 t/s, acceptance .60, mean draft 4.0
⇒ ≈ 3.4 tokens per round ⇒ **≈ 90 ms per round vs ≈ 45 ms per serial token (2×)**. TensorFold's
round costs 1.25× a serial token *and* yields more tokens (trees). Both halves are where the
speed is:

1. **Make a 5–16-row verify cost about one row** (weights read once; attention without the
   whole-cache f16 conversion).
2. **Get more tokens per round** (draft trees instead of chains).

Rough ceiling if we matched their round efficiency on the 7900 XTX (not a promise; their
acceptance is on their prompts):

| context | serial ms/token (UD-Q3_K_XL, q8_0 KV) | round at 1.25× | ~4.7 tokens/round |
|---|---|---|---|
| short | 25.3 | ~32 ms | **~145 t/s** |
| 113k | ~40 | ~50 ms | **~95 t/s**, only if attention for 12–16 rows is also ~flat (Phase C) |

## 3. The techniques, ranked for porting into llama.cpp

| # | technique (TensorFold) | their measured effect | llama.cpp today | port effort |
|---|---|---|---|---|
| **T1** | **Row-invariant verify matmul**: one kernel for 1–128 rows; rows up to 16 cost ≈ one row (weights read once, tensor-core dot per 64-group) | whole-model forward 46.8 ms @1 row vs 53.7 @16 (Mac); verify windows exact and cheap | MMVQ handles 1–8 columns; our (cache-hot) E0 numbers show n=4 ≈ 1.7× n=1; >8 columns go to MMQ | medium: measure first (below), then tune MMVQ/MMQ for 2–16 columns |
| **T2** | **Weights regrouped once at load** into contiguous blocks per (column tile, 64-input group), scales group-major | **107–130 → 200–220 GB/s** on Spark (~1.8×); one-row forward 123 → 75 ms | ~530 GB/s (55 % of peak) on our fixed part; blocks are AoS, 18 B/17 B stride for FP4 | large (new repacked layout + reorder-aware MMVQ/MMQ/dequant); = our E8, now with outside evidence; ggml-sycl's Q4_0 "reorder" is the in-tree precedent |
| **T3** | **Commit by replay**: keep only the round-start DeltaNet state; after verify, replay the accepted path into the state, all 48 GDN layers in one launch; KV rows written in place | no cache copies at long context; commit < 1 ms/round | verify writes **one full recurrent-state snapshot per drafted token** (`gated_delta_net.cu:146–150`, `n_rs_seq` = draft size): ~144 MB of fp32 state per snapshot for 48 layers ⇒ 0.7–1.7 GB written per 5–12-token round, plus that much VRAM | medium: a replay op for GDN + change the rollback in `llama-memory-recurrent`/server spec path |
| **T4** | **Draft trees**: best-first over DFlash2's candidates, 4 children/node, 12–15 nodes, verified in one forward; tree attention mask; GDN tree kernel (each node from its parent's state) | code 76.5 → 99.1 t/s, story 37 → 50 (Mac, sampled); calibrated scores +6.5 % tokens | chains only | large: tree batches (positions + custom mask), GDN tree op, tree selection in `common/speculative` |
| **T5** | **Cheaper drafter**: 4-bit DFlash2 projections, fused kernels (918 → 288 launches), context K/V cached per layer | drafting 19.5 → 7.4 ms/round (Spark) | DFlash2 runs as a separate llama context; kernel count and launch overhead not measured (WSL2 launch latency makes this matter) | small–medium: measure drafter ms/round first |
| **T6** | **Draft vocabulary**: drafter's LM head reads only token ids < 98,304 (99.64 % of committed tokens; 40 % of the head) | cheaper drafting; verification still full-vocab | full 248,320-row head in the drafter | **small**: a row view of the drafter's output weight + mapping; no quality risk (target verifies) |
| **T7** | Copy rule: ≥ 8-token verbatim match replaces the tree | +whole-file edits 239 → 293 t/s | `ngram-map-k4v` stacked with DFlash2 already does this | none |
| **T8** | 12-row windows as the sweet spot (vs 16/32/64) | ~3 ms/round saved at same acceptance | `--spec-draft-n-max` | tuning only |
| — | Byte-exact drafted = serial decoding | correctness guarantee, **no speed gain** | not needed for our goal | skip |

## 4. Alternative: run TensorFold itself on the 7900 XTX

The CUDA engine is PyTorch + Triton + two small `.cu` extensions (`gdn.cu`/`gdn_tree.cu`). In
principle ROCm PyTorch (which supports gfx1100, including under WSL2) builds `.cu` extensions
through hipify, and Triton has an AMD backend. Realistically:

- **Unknowns:** Triton's RDNA3 (gfx1100) codegen quality for these `tl.dot` shapes (the kernels
  are tuned for NVIDIA); whether the extensions hipify cleanly; ROCm PyTorch under WSL2
  stability.
- **Model format:** MLX 4-bit affine g64 checkpoints from Hugging Face (16.1 GB + drafter), not our
  GGUFs. Quality needs its own Phase Q run (MLX 4-bit g64 is roughly Q4_1-class).
- **Memory:** bf16/fp16 KV without quantization ⇒ ~3.7 GB at 113k for the 16 attention layers on
  top of ~16 GB of weights and the drafter. 113k fits; 256k (~8.6 GB of KV) doesn't.
- **Value:** a few hours on the machine would tell us whether the verify-cost and tree ideas deliver
  on this GPU *before* porting them into llama.cpp. As a benchmark it is worth it even if we
  don't adopt it.

## 5. Recommended order

1. **Measure our verify cost curve (cheap, decides T1):** DRAM-bound MMVQ/MMQ at 1–16 columns.
   Batch the FFN matrix 8× so it doesn't fit the Infinity Cache
   (`test_mul_mat(type, F32, 17408, n, 5120, {8, 1}, {1, 1})`), plus `llama-bench` or server
   timing of one verify round vs one serial token at 4k and 113k context.
2. **T6 draft vocabulary** (small, self-contained) and **T5 drafter timing** (is drafting 5 ms or 20 ms
   of our ~90 ms round?).
3. **T3 commit by replay**: removes up to 1.7 GB of state writes per round and the snapshot VRAM.
4. **T1 + Phase C** together: flat verify for weights *and* attention. Without Phase C, long-context
   verify still pays the whole-cache f16 conversion.
5. **T4 trees** once rounds are cheap (trees multiply the value of a cheap round).
6. **T2 repacked weights** as the big fixed-part win (also lifts serial decode).
7. Optional, in parallel: try TensorFold on ROCm (§4) as a reference point.

## 6. Licensing

MIT (core) + Apache-2.0 (vendored z-lab DFlash code). Porting code or ideas into llama.cpp
(MIT) is compatible; keep attribution in commit messages / file headers for anything translated
closely (e.g. the GDN tree/replay kernels).
