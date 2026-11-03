# llama.cpp upgrade test, 2026-10-30

**What:** tree 4 (`~/llama-rocmfp4-merge`, branch `exp/iq2-signs` @ `1fd86c390`, 71 commits of ours on top of the mainline of 2026-09-24) merged with `upstream/master` @ `36a0e2ffc` (29 Oct, **846 new upstream commits**) in a separate worktree `~/llama-up`, branch `merge-upstream-2026-10-30`, build dir `build-up` (not pushed anywhere; production still runs `build-exp` @ `1fd86c390`).

## Merge
- Merge commit `5b0c0af3b` + fixes `e3f26c928`. 14 files conflicted, 22 hunks. Most were "both sides add" (our ROCmFP4 types 100/101 next to upstream's new `GGML_TYPE_F8_E4M3 = 43`; `GGML_TYPE_COUNT` stays 102).
- Real ports: the MMQ refactor of upstream (`prec_src1`, `x_gate` in the tile loaders) needed `ggml_cuda_mmq_load_tiles_q4_0_rocmfp4` and its dp4a/mma entries updated, `mul_mat_q_case<..., GGML_PREC_Q8>`; the MTP-only trunk flag of `qwen35.cpp` now comes from upstream's `nextn_flags`; `common_batch` (tokens[].pos[0], `.view.size()`) in `speculative.cpp` and `server-context.cpp`; upstream's `llama_decode(ctx, batch)` became `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())` (our `SPEC_DFT_*` profiling hooks kept); `n_pad` is gone upstream (our `n_kv_reserve_cap` stays); `VDR_Q2_K_Q8_1_MMVQ` takes upstream's 2; the sparse-mask flash-attention path is now enabled on HIP too (upstream dropped the HIP exclusion).
- Fused MoE gate/up in MMQ (upstream #31188) is limited to Q4_K/Q5_K/Q6_K/Q8_0/IQ4_XS by `ggml_cuda_mmq_fusion_supported`, so the ROCmFP4 types do not take it.
- Build: `bench/decode_investigation/build_llama_worktree.sh ~/llama-up build-up 12` (HIP gfx1100, UI off; the default build tries to download a UI from huggingface).

## Checks (GPU, same day, UDflat, `run_model.sh udflat`: MTP n_max 3, q4_0 KV, fused q4_0 attention, -c 196608, pre-warm)
`bench/decode_investigation/ab_llama_upgrade.sh OLD_BIN NEW_BIN` (OLD, NEW, OLD; pass 2 counted; pi-session prompts of `bench/prompts`, 256 greedy tokens):

| prompt (tokens) | OLD1 pp / tg | NEW pp / tg | OLD2 pp / tg | NEW vs OLD mean |
|---|---|---|---|---|
| pi16 (11.4k) | 930 / 80.6 | 980 / 86.1 | 914 / 80.7 | pp +6.2 %, tg +6.8 % |
| pi32 (27.7k) | 871 / 66.6 | 909 / 68.8 | 870 / 66.0 | pp +4.4 %, tg +3.7 % |
| pi113 (111k) | 639 / 49.4 | 658 / 49.4 | 636 / 49.1 | pp +3.3 %, tg +0.3 % |

OLD1 and OLD2 agree within 2 %, so the GPU did not drift. Greedy outputs of NEW differ from OLD at 256 tokens (different SHA, kernels changed), so quality was checked:
- Perplexity UDflat, wikitext-2, `-c 4096 --chunks 20`, q8_0 KV: OLD 6.0106 +/- 0.073, NEW 6.0061 +/- 0.073 (no difference).
- `test-backend-ops` on the new build: MUL_MAT rocmfp4 34/34, MUL_MAT_ID rocmfp4 2/2, MUL_MAT_ID q4_K 84/84 (the fused gate/up path), all on ROCm0.

## Not done
Dense 27B only. The MoE models (GLM-5.3 Flash, Qwen3.8 Flash Next, DeepSeek-V4.1 Flash) were not run: upstream has since added DSV4.1 flash (#31086), GLM5-next QSA/DSA (#31009), MoE gate/up fusion in MMQ (#31188), top-k fusion for 384 experts (#31288), grouped experts top-k fusion (#29181), an SLRU expert cache (#30637), `MUL_MAT_ID -1 index skips the compute` (#26631). No quality run beyond perplexity, no KLD against Q8_0, no long-context (170k+) cell, no DeepSWE run. The merge is not pushed; an upstream PR would take only the optional parts (env flags / GPU-specific defaults).
