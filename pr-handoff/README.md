# Handoff: llama.cpp pull request(s) from the RX 7900 XTX fork (for Claude cloud), 2026-10-31

You (Claude cloud) have no GPU and no access to the WSL machine. This note and `notes/llama_pr/*.patch` (in this repository) are everything you get. Prepare branches and PR texts; **do not push to `ggml-org/llama.cpp` and do not open a PR without the user's explicit OK**. The user submits.

## What the PR may contain
Only changes that are **optional** (off by default, behind an env variable or a new option whose default keeps today's behaviour) or **default-for-this-GPU** (selected only for AMD RDNA3 / gfx1100). Nothing that changes results or speed on other hardware by default. Do NOT include: the ROCmFP4 quant types and kernels, DFlash2 speculative decoding, any `docs`/`results`/`quality`/`plan`/`handoff` commit of the fork, the negative-result experiments (`GGML_CUDA_FATTN_VEC_GQA`, `GGML_CUDA_FATTN_VEC_COLS4`, `GGML_CUDA_FATTN_WMMA_MIN_ROWS`, `fattn-vec-gqa.*`), diagnostics (`GGML_CUDA_OP_PROF`, `SPEC_DFT_*`, `SPEC_DFT_SKIP`). Follow the repository's CONTRIBUTING / AI-usage policy: disclose that the code was written with AI assistance and was tested by a human on the stated hardware.

## Base and where the code is
- **Everything is on GitHub in `theGiallo/llama-7900xtx-qwen3.8-27b` (this repository)**: the fork branches `exp/fattn-tile-q4` (0001, 0006, 0007), `exp/fattn-tile-q8` (0002), `exp/mtp-share-io` (0004), `opt/vram-footprint` (0005), `main`, and the merge branch `merge-upstream-2026-10-30` (pushed with this handoff, so you CAN fetch it). This note, the patches, and the measurements are in `pr-handoff/` on branch `pr-handoff`.
- Upstream base: `ggml-org/llama.cpp` master `36a0e2ffc` (29 Oct 2026, build 12003). Our fork = upstream + 73 commits; the merge with that base is `5b0c0af3b` + `e3f26c928`, branch `merge-upstream-2026-10-30` on the user's machine (pushed to this repository).
- The patches are `git format-patch` output of the original commits (they apply to the *fork's* base, not to current master). A trial `git cherry-pick` of each onto `36a0e2ffc` gave:

| patch | commit | what | applies to master? |
|---|---|---|---|
| 0001 | ac08925ea | fused q4_0 dequantization in the tile flash-attention kernel (opt-in) | conflict in `ggml-cuda/fattn.cu` |
| 0002 | 39f459c26 | same for q8_0 K/V | conflict in `fattn-tile.cu`, `fattn-tile.cuh` (builds on 0001) |
| 0003 | 8a4f661e9 | one CUDA graph per small batch size | clean |
| 0004 | 484a3c9aa | MTP head without token_embd/output borrows them from the target | clean |
| 0005 | a35a77184 | `--ctx-reserve`, `--spec-draft-ctx` | conflict in `src/llama-kv-cache.h` |
| 0006, 0007 | 24a5ec096, 67a36ae9f | test-backend-ops perf cases for flash attention | conflict in `tests/test-backend-ops.cpp` |

Resolve conflicts by re-implementing on current master (the fork's merged versions of `fattn-tile.cuh` and `fattn.cu` also hold the excluded VEC_GQA / WMMA experiments: take only the fused-dequant parts: `ggml_cuda_fattn_tile_q4_mode()`, `ggml_cuda_fattn_tile_q4_fused()` and the tile-kernel changes).

## Candidate 1 (the main one): fused tile dequantization for quantized KV
Opt-in env `GGML_CUDA_FATTN_TILE_Q4` (head size 256, q4_0/q4_0 and q8_0/q8_0 K/V): `1` for 3..16 query columns, `2` also for 1..2 columns. K/V tiles are dequantized from the quantized cache straight into shared memory (same arithmetic as `dequantize_*` + float to half) instead of converting the whole visible K/V to f16 in global memory first; the f16 scratch is then not allocated. Off by default; not bit-identical to the default path (at 111k tokens the 256 greedy tokens differ from the default path's; perplexity/KLD were measured later, on the fork base: with q4_0 KV the flag-on and flag-off values are within one standard error on STRIX and UD-Q4_K_S, with q8_0 KV they are identical to every digit; see `notes/tensorfold_analysis_2026-10-02.md` and RESULTS 13.3-13.6, copied here as `pr-handoff/`. Not re-measured on the rebased PR branch: the user must repeat it there).

Measured (RX 7900 XTX, gfx1100, ROCm 7, Qwen3.8-27B UDflat with MTP head, draft length 3, q4_0 KV, `-c 131072`, greedy, 256 tokens, the prompts' pass 2; stock = upstream `36a0e2ffc`, "ours" = our branch):

| prompt | stock before | ours, flag 0 | ours, flag 2 | stock after |
|---|---|---|---|---|
| 11k tokens: prefill / decode tok/s | 956 / 85.4 | 945 / 84.8 | 971 / 84.9 | 967 / 85.2 |
| 28k | 927 / 68.0 | 918 / 68.0 | 917 / 68.6 | 913 / 69.0 |
| 111k | 658 / 45.7 | 658 / 46.3 | 657 / **51.3** (+11 %) | 658 / 46.1 |

Without speculation (same prompts, same flags, no MTP head; prefill / decode tok/s):

| prompt | stock before | ours, flag 0 | ours, flag 2 | stock after |
|---|---|---|---|---|
| 11k | 1033 / 36.3 | 1028 / 36.0 | 1041 / 36.9 (+2 %) | 1035 / 36.2 |
| 28k | 968 / 33.9 | 971 / 33.9 | 965 / 35.2 (+4 %) | 969 / 33.9 |
| 111k | 695 / 24.9 | 692 / 24.9 | 693 / **27.5** (+10 %) | 696 / 25.0 |

With the MTP head, as above, the same pattern: **no effect up to ~28k tokens, +11 % decode at 111k** (the longer the KV, the larger the gain). Acceptance of the MTP draft is the same (0.752 vs 0.763 on the last request). Earlier micro-benchmarks (test-backend-ops, hs=256, 6 query heads per KV head, kv=113408, q4_0): 1 column 1172 -> 680 us, 4 columns 2054 -> 1369 us, q8_0 4 columns 2053 -> 1433 us. Source of the table: `bench/decode_investigation/ab_llama_stock.sh`, raw output copied into this repository as `bench/decode_investigation/ab_stock_20261030_summary.txt`.

What the reviewers will ask, and what to do: (1) why env and not a heuristic: the gain exists only for long KV with quantized cache and was measured on one GPU; keep it opt-in, propose the auto-select for gfx1100 as a follow-up. (2) correctness: ask the user to run `test-backend-ops -o FLASH_ATTN_EXT` with the flag on and off on their machine (same pass/fail pattern as default was seen); you cannot run it. (3) maintainability: the change should stay inside `fattn-tile.cuh` and a few lines of dispatch in `fattn.cu`.

## Candidate 2: CUDA graph per small batch size (only if it still helps)
Speculative verify batches change size each round; the graph cache key was the first node only, so each size change reset the warm-up and the round ran eagerly (about 80 % of verify passes never replayed a graph with DFlash2 on this model). The patch keys batches of 2..16 tokens by `(first node, n_tokens)`; `GGML_CUDA_GRAPH_KEY_BY_BATCH=0` restores the old key; outputs are bit-identical. Measured on the fork's old base with DFlash2: 6k fill 55.5-56.3 -> 62.8-63.0 tok/s, 27.5k 50.1-50.6 -> 58.9.
**Warning from the A-B run above:** with the MTP head on current master, "stock" and "ours, flag 0" (which contains this change) are identical within noise at all three prompt sizes. Either upstream already fixed the effect, or MTP verify batches do not hit it. Do not submit it unless the user can show a gain with `--spec-type draft-dflash` on current master; ask.

## Candidate 3: long-context memory options
`--ctx-reserve N` (caps the KV span used to size the compute buffers; the worst case f16 conversion workspace + mask scales with n_ctx) and `--spec-draft-ctx N` (the draft context holds N positions per sequence, older ones are dropped before each draft prefill). Both default to 0 = unchanged. They are what lets `-c 196608` and `262144` fit in 24 GiB with a q4_0 KV and an MTP draft. Not part of the A-B run (it ran at `-c 131072`). Cost of `--spec-draft-ctx 32768`: MTP acceptance 0.489 -> 0.379 in an earlier test at long context, so describe it as a memory/speed trade-off, not a free win.

## Candidate 4 (only if the maintainers ask for it): MTP head without embeddings
`llama_model_share_io`: an MTP head file that carries no `token_embd`/`output` tensors borrows them from the target model (new function in `include/llama.h`, 20 lines). Format-specific and an API addition: lowest priority.

## What the user must run (you cannot)
Build with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`, `-DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"` (our build flags), then: `test-backend-ops -o FLASH_ATTN_EXT` with and without the flag; `llama-perplexity` on wikitext-2 with q4_0 KV and the flag on/off; the A-B script `bench/decode_investigation/ab_llama_stock.sh` against the PR branch build. Report these numbers in the PR text only after the user supplies them. Run flags used in our measurements: `LLAMA_ATTN_ROT_DISABLE=1 -fa on -np 1 --no-kv-unified -b 2048 -ub 512 -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3`.

## Deliverables
One branch per candidate (start with Candidate 1 and its test cases), each rebased on current `ggml-org/llama.cpp` master, one logical commit, a PR description with the table above (clearly "one GPU, one model"), the opt-in flag documented in the code comment and, if there is a suitable place, in `docs/`. A short summary for the user of what you changed relative to the patches and what still needs their GPU.
