# RX 7900 XTX (gfx1100): hardware facts and kernel optimization strategies

Companion to `CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md` and `HANDOFF_STEP1_FOR_MACHINE_AGENT.md`.
It covers what the AMD and LLVM documentation says about this GPU, and what that means for
the kernels Qwen3.8-27B runs in this branch.

## Sources

Written first from AMD documentation and source that AMD publishes on GitHub, and from the
LLVM AMDGPU backend, which is the compiler `hipcc` uses. gpuopen.com and rocm.docs.amd.com were
later allowed and cross-checked (the ISA PDF on www.amd.com was still blocked):

| source | used for |
|---|---|
| `ROCm/ROCm` `docs/reference/gpu-specs.rst` | CU count, caches, LDS, register file of the RX 7900 XTX |
| `ROCm/ROCm` `docs/reference/precision-support.rst` | data types supported by RDNA3 ALUs and matrix cores |
| `ROCm/HIP` `docs/understand/hardware_implementation.rst`, `performance_optimization.rst`, `docs/how-to/performance_guidelines.rst` | RDNA WGP/CU structure, LDS banking, coalescing, occupancy, latency hiding |
| `llvm/llvm-project` `llvm/docs/AMDGPUUsage.rst` | gfx11 memory model, cache hierarchy, WGP vs CU mode, non-temporal loads, VGPR encoding |
| `llvm/llvm-project` `llvm/lib/Target/AMDGPU/AMDGPU.td`, `Utils/AMDGPUBaseInfo.cpp`, `IntrinsicsAMDGPU.td` | gfx1100 feature bits: VGPR budget and granule, max waves, dot and WMMA instructions |
| GPUOpen "RDNA performance guide" (gpuopen.com/learn/rdna-performance-guide) | LDS banking, wave32, SoA layouts |
| `ROCm/rocm-blogs` register-pressure, LDS bank conflict, matrix-cores posts | compiler resource reports, `__launch_bounds__`, spill avoidance |
| this repo: `ggml/src/ggml-cuda/*` | what the current kernels actually do |

Items marked *(inferred)* are my conclusions, not statements from the sources.

## 1. Hardware facts that matter

| property | RX 7900 XTX / gfx1100 | source |
|---|---|---|
| compute | 96 CUs = 48 WGPs (2 CUs per WGP), 2 × SIMD32 per CU | gpu-specs, HIP hardware doc |
| wave size | wave32 native, wave64 supported | gpu-specs, AMDGPUUsage |
| VGPRs | **1536 per SIMD lane in wave32** (192 KiB per SIMD, 768 KiB per WGP); max 256 per wave; **allocated in blocks of 24** | AMDGPU.td `Feature1536VGPRs` on 11.0.0, `getVGPRAllocGranule` |
| max waves | **16 per SIMD** | AMDGPU.td `FeatureMaxWavesPerEU16` in `FeatureGFX11` |
| LDS | **128 KiB per WGP**, one LDS shared by both CUs; **32 banks × 4 B** (GPUOpen RDNA performance guide and LLVM agree; the HIP doc's "64 banks" doesn't apply to single-workgroup access patterns) | gpu-specs, AMDGPUUsage, GPUOpen RDNA performance guide, AMDGPU.td `FeatureLDSBankCount32` |
| caches | vector L0 32 KiB per CU; scalar L0 16 KiB per WGP; L1 256 KiB per shader array; L2 6 MiB; **Infinity Cache (MALL) 96 MiB**; 128-byte lines | gpu-specs, AMDGPUUsage, HIP doc |
| DRAM | 24 GiB GDDR6, ~960 GB/s (384-bit at 20 Gbps; public spec, not in the docs above) | — |
| matrix cores (WMMA) | 16×16×16: f16 and bf16 in (f32 or f16 accumulate), iu8 and iu4 in (i32 accumulate). **No FP8, FP6 or FP4, neither in matrix cores nor in regular ALUs.** On RDNA3, the A and B operands must be duplicated in both half-waves. | precision-support, IntrinsicsAMDGPU.td, `mma.cuh:81` (`DATA_LAYOUT_I_MAJOR_MIRRORED`) |
| dot instructions | `v_dot4_i32_iu8` (what `ggml_cuda_dp4a` uses on RDNA3), `v_dot8_i32_iu4`, `v_dot2_f32_f16`, `v_dot2_f32_bf16`, `v_dot2_f16_f16` | AMDGPU.td Dot7/8/9/10/12, `common.cuh:720` |
| execution mode | default **WGP mode**: a workgroup's waves can run on both CUs of the WGP, which have separate L0 caches. `-mcumode` keeps a workgroup on one CU. | AMDGPUUsage "Memory Model GFX10-GFX11" |
| non-temporal | `__builtin_nontemporal_load` compiles to `slc=1 dlc=1` on gfx11 (streaming hint) | AMDGPUUsage code-sequence table |

Occupancy by VGPR count (waves per SIMD = min(16, 1536 / round_up(VGPRs, 24))):

| VGPRs per thread | ≤ 96 | 104–120 | 128–144 | 168 | 192 | 216 | 256 |
|---|---|---|---|---|---|---|---|
| waves per SIMD | 16 | 12 | 10 | 9 | 8 | 7 | 5 |

The steps are coarse. For example, going from 96 to 97 VGPRs drops from 16 to 12 waves, so
tuning register usage around these edges is worth doing for memory-bound kernels.

### What this means in general

- **FP4 is a storage format only.** Every FP4 weight must be expanded to int8, f16 or bf16
  before any dot or WMMA instruction. The FP4 gain is bytes moved, never math throughput.
- **Memory-bound kernels need many loads in flight** (Little's Law, HIP performance doc).
  Occupancy, or several independent loads per thread, is what reaches the 70–90 % of peak
  that the HIP docs quote for coalesced streaming.
- **Wide, aligned loads matter.** 128-byte lines; a wave32 dword load covers exactly one line.
  `global_load_b128` covers four lines per instruction and cuts instruction count 4×.
- **Streaming data should not evict reused data.** Weights (~14 GB/token) and KV (GBs) stream
  through once. The DeltaNet state (~144 MiB, see §2.4), activations and partial results are
  reused and could benefit from staying in the 6 MiB L2 or the 96 MiB MALL *(inferred)*.

## 2. The kernels, one by one

Per-token work of Qwen3.8-27B decode on this branch, in order of measured cost at 113k fill:

1. weight matrix-vector products (the ~27 ms fork / ~35 ms merged fixed part);
2. full-attention FlashAttention over the KV cache (~20 ms at 113k, ~45 ms at 258k);
3. DeltaNet recurrent update (48 layers) and everything else;
4. launch overhead between all of the above.

### 2.1 Weight matrix-vector (MMVQ) for ROCmFP4 — two concrete findings

**Finding A: ROCmFP4 decode runs with 1 warp per block on RDNA3.**
`calc_nwarps()` in `ggml/src/ggml-cuda/mmvq.cu:511` has an RDNA3 table, measured upstream on a
W7900 (same gfx1100 chip). For `ncols_dst == 1` it gives **8 warps** to Q4_0/Q4_1/Q5_x/Q8_0
and IQ4_NL, and **1 warp to every other type**, including `Q4_0_ROCMFP4` and
`Q4_0_ROCMFP4_FAST`. IQ4_NL is the closest analogue to ROCmFP4: a 16-entry codebook looked up
with `v_perm`, then `dp4a`. With 1 warp, each wave walks the whole K=5120 row alone and there
is no split of K across the block.

This is a strong candidate for part of the merged build's ~8 ms/token regression. It is a
two-line experiment:

```cpp
// mmvq.cu, calc_nwarps(), MMVQ_PARAMETERS_RDNA3_0, ncols_dst == 1
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_Q4_0_ROCMFP4:        // try 8 (then 4, 2)
                case GGML_TYPE_Q4_0_ROCMFP4_FAST:
                    return 8;
```

The fork's setting is not visible from this repo. Check it on the machine with
`git show feat/benchmark:ggml/src/ggml-cuda/mmvq.cu | grep -n -A20 "RDNA3_0) {"`.

**Finding B: FP4 quant bytes are read with 4-byte loads from 2- or 1-byte-aligned addresses.**
`block_rocmfp4` is 18 bytes and `block_rocmfp4_fast` is 17. So `qs` of block *k* is only 2-byte
(18 B) or 1-byte (17 B) aligned. `vec_dot_q4_0_rocmfp4[_fast]_q8_1` (`vecdotq.cuh:373-413`)
reads them with `get_int_b4`, which assumes 4-byte alignment. Upstream uses `get_int_b2` for the
18-byte Q4_0 and `get_int_b1` for the 17-byte MXFP4 for this reason. It works because the
hardware accepts unaligned dword loads (results are correct), but misaligned dwords can straddle
dword and cache-line boundaries. Two options:

- **quick:** switch to `get_int_b2` (18 B) and `get_int_b1` (17 B) and A/B the timing — cheap,
  but more instructions per byte;
- **real fix, a repacked layout** *(inferred, best payoff)*: at load time, rearrange each weight
  row into separate aligned arrays: all `qs` nibbles contiguous and 16-byte aligned, and the
  scale bytes in their own array. Then a thread loads 16 B of nibbles (32 weights) with one
  `global_load_b128` and the scales with one small load per few blocks. ggml already has this
  pattern: the **SYCL backend "reorders" Q4_0-family weights** into this layout for Intel GPUs
  (`ggml/src/ggml-sycl/`, `*_reorder`). The same trick for HIP needs a buffer type or a
  load-time transform plus reorder-aware MMVQ/MMQ/dequant kernels. That is more work, but it is
  the main lever for moving the ~27–35 ms fixed part towards the ~14 ms bandwidth floor.

**Other MMVQ items**

- `v_perm`-based codebook lookup (`get_int_from_table_16`) is already the right instruction
  choice. Codebook values ±{0..10} don't fit `iu4` (range −8..7), so they must stay int8 and
  `v_dot4_i32_iu8`.
- Check that the fused gate/up path (`has_fusion` in `mmvq.cu`) is active for the FFN of this
  model with ROCmFP4. It reads both weight matrices in one pass and saves a launch plus an
  activation round trip.
- Verify batches (`ncols_dst` 2–8): the RDNA3 table returns 1 warp for every type when
  `ncols_dst > 1`. Measure 1/2/4 warps for ROCmFP4 at 2–6 columns; that is the speculative
  verify cost.

### 2.2 Decode FlashAttention over a quantized KV cache

What the code does now (`fattn.cu`, `fattn-vec.cuh`, `fattn-common.cuh`):

- batch 1–2 with q4_0/q8_0 KV → **VEC kernel**; batch ≥ 3 → **TILE kernel after converting
  the whole visible K/V to f16** (`need_f16_K/V`);
- VEC grid: `x` = query tile, `y` = `parallel_blocks` (the KV sequence split), `z` = **query
  heads**. Each block computes one query head, reading K/V at `head / gqa_ratio`;
- block `y` walks KV rows `y·nthreads, (y + gridDim.y)·nthreads, …`.

**Step-1 result (`results/step1/2026-09-24-strix/CLOUD_REVIEW.md`):** the re-reads are served
by caches (q8_0 would need ~1.8 TB/s otherwise) and q4_0 VEC is instruction-bound. Also, the TILE
kernel packs query heads only in powers of two, so with GQA 6 it groups 2 heads and processes
each KV block 3×, even for f16. The new kernel should pack all 6 heads.

**Refinement of the earlier diagnosis** *(inferred, now confirmed)*: the 6 query heads that share a KV head
run in the same dispatch wave with the same `y`, so they read **the same KV addresses at about
the same time**. Much of the 6× re-read may therefore hit L2 or MALL instead of DRAM. What
surely remains is 6× the load instructions, 6× the dequantization ALU work and 6× the L2
traffic. Either way the fix is the same, but step 1 should also test **how time scales with the
GQA ratio at a fixed KV size** (see §3, E3). That tells DRAM-bound from instruction-bound.

Design for the replacement kernel, sized for this GPU *(inferred from the facts above)*:

- **One workgroup per (KV head, KV chunk)** holding all `gqa × n_q` query rows: 6 for decode,
  up to 48 for a verify batch of 8. Each K/V row is loaded and dequantized **once**.
- **Loads:** a q4_0 K row of D=256 is 8 blocks × 18 B = 144 B, and 144 is a multiple of 16. So
  each row starts 16-byte aligned (if the view's row stride keeps that) and can be read with 9 ×
  `global_load_b128`, then unpacked in registers or LDS. No 2-byte loads.
- **Math:** quantize Q to q8_1 once (as VEC already does) and use `v_dot4_i32_iu8` for
  `gqa × n_q ≤ 8` rows. For verify batches (24–48 rows), WMMA iu8 16×16×16 with the K block as
  the B operand becomes worth it; remember A and B are duplicated per half-wave on RDNA3.
- **Parallelism:** only 4 KV heads per layer, so the sequence split must supply the
  parallelism. 48 WGPs × a few workgroups each ⇒ split each KV head into ≳ 48–96 chunks at long
  context, then combine (the existing `flash_attn_combine_results` pattern).
- **Registers:** stay ≤ 96 VGPRs for 16 waves, or accept 120 (12 waves) if it saves reloads.
  Check with the resource report (§3, E5).
- **Non-temporal K/V loads** (`slc dlc`) so streamed KV doesn't evict the DeltaNet state and
  partial results from L2/MALL. Worth an A/B test, not a guaranteed win.

### 2.3 Prefill (MMQ with WMMA)

Prefill is compute-bound, so WMMA matters, but decode is the priority.

- The RDNA3 WMMA A/B duplication halves the useful register payload per instruction.
  `mma.cuh` already handles the layout (`DATA_LAYOUT_I_MAJOR_MIRRORED`).
- FP4 tiles must be expanded to int8 in the tile loader (`mmq-load-tiles.cuh`), so the same
  aligned-repack idea from §2.1 speeds up tile loads.
- LDS has 32 banks of 4 B. Pad tile rows, or prefer struct-of-arrays, so a wave's accesses
  spread across banks (GPUOpen RDNA performance guide, rocm-blogs LDS bank-conflict post).
- Prefill already improved +30–70 % on the merged build; lower priority than decode.

### 2.4 DeltaNet (`gated_delta_net.cu`, 48 layers)

The state is f32, roughly 48 value heads × 128 × 128 × 4 B ≈ 3 MiB per layer ≈ 144 MiB per
token, read and written *(inferred from the Qwen3.5/3.8-27B config)*. That is ~0.3 ms at peak
bandwidth: small, but bigger than the MALL, so it streams from DRAM. Watch that speculative
decoding checkpoints (`--ctx-checkpoints`) and rollbacks don't copy it more than needed. Low
priority until step 1 says otherwise.

### 2.5 Launch overhead and HIP graphs

`GGML_HIP_GRAPHS` defaults to ON (`ggml/CMakeLists.txt:216`), but whether graphs are actually
used at runtime under WSL2 has not been checked. With ~1000+ kernels per token, per-launch
latency adds up to milliseconds. Confirm with a debug log or trace that graph capture/replay
happens during decode, and compare decode tok/s with graphs forced off.

### 2.6 Loading and data flow outside the kernels

- Keep GGUF files on the WSL2 **ext4** filesystem, not `/mnt/c`. Windows-drive access goes
  through a 9P bridge and is much slower for mmap and loading *(general WSL2 behaviour, not from
  the AMD docs)*.
- The GPU is shared with the Windows desktop under WSL2. Your results show browsers/Discord
  pushing VRAM to saturation and collapsing speed. Close GPU-heavy apps for benchmarks, and keep
  ~1 GiB of headroom.
- Keep `-ub` (physical batch) sizes steady across runs. Compute-buffer size changes shift VRAM
  headroom and can change kernel choices.

## 3. Experiments, cheapest first

| # | experiment | why | effort |
|---|---|---|---|
| **E0** | **Restore ROCmFP4 test coverage**: add `GGML_TYPE_Q4_0_ROCMFP4` and `_FAST` to `test-backend-ops` MUL_MAT (real shapes k=5120/6144/17408, batch 1–8, plus MMQ sizes) and GET_ROWS | the port has **no** ROCmFP4 tests on this branch; nothing below can be judged safely without them | small |
| E1 | MMVQ nwarps for ROCmFP4 on RDNA3: 1 (now) vs 2/4/8, `ncols_dst` 1 and 2–6, via `test-backend-ops perf -o MUL_MAT` | Finding A; likely part of the regression | tiny |
| E2 | `get_int_b4` → `get_int_b2`/`get_int_b1` in the ROCmFP4 vec dot | Finding B, quick variant | tiny |
| E3 | FA perf with the KV size fixed and GQA varied (`nr23` = 1, 2, 3, 6 with 4 KV heads) for q4_0 and f16 | time ∝ gqa ⇒ redundant work dominates; flat ⇒ DRAM-bound | small (perf cases) |
| E4 | HIP graphs on vs off during decode; count launches per token | launch overhead under WSL2 | small |
| E5 | Build with `-Rpass-analysis=kernel-resource-usage` (add to `CMAKE_HIP_FLAGS`) and record VGPRs, spills, LDS and occupancy for MMVQ-ROCmFP4, FA-VEC q4_0 D=256, and `gated_delta_net` | shows occupancy cliffs from the table in §1 | small |
| E6 | Non-temporal loads for weights/KV in MMVQ and FA | protect L2/MALL for reused data | small–medium |
| E7 | GQA-grouped quantized-KV decode FA (§2.2) | the big attention win | large |
| E8 | Aligned repacked ROCmFP4 layout + b128 loads (§2.1) | the big weight-path win | large |
| E9 | `-mcumode` build vs default WGP mode, on the kernels above | L0 locality for workgroup-shared data; could go either way | small |

E0 comes first: it is the correctness gate for everything else. E1, E2 and E5 can be batched
into one machine run together with step 1.

## 4. What is not known yet

- Whether the fork's `feat/benchmark` branch set different MMVQ warps for ROCmFP4 (check on the
  machine).
- The actual GQA ratio and head counts of the GGUF (inferred 24 query / 4 KV heads, D=256).
- Whether HIP graphs work under WSL2 with this ROCm version.
- Whether `rocprofv3` hardware counters work under WSL2. If not, bandwidth has to be inferred
  from timings, as the step-1 scripts do.
