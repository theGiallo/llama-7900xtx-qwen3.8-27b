# Output tok/s recap - Qwen3.8-27B on RX 7900 XTX (gfx1100, 24 GB)

What we are up against, gathering every measurement we have so far.

## 1. Your own measurements (Windows native, ROCm 6.1, stock llama.cpp)

Source: `F:\programming\llm\llama.cpp_win\report_test_qwen3.8-27B.md` + run logs.
All "tg" = generation token/s reported by llama-server.

### Context 32768 (clean machine)
| Config | Model | tg t/s | pp t/s | Accept |
|--------|-------|--------|--------|--------|
| Base | Q4_K_S | **33.2** | 820 | - |
| Base | UD-Q4_K_S dynV3.0 | **36.9** | 745 | - |
| MTP-nat | Q4_K_S | 37.2 | 720 | 55% |
| MTP-ext | UD-v3 | **40.4** | 751 | 43% |
| DFlash2 | Q4_K_S | 12.2 | 186 | 31% |

### Context 65536 (clean machine)
| Config | Model | tg t/s | pp t/s | Accept |
|--------|-------|--------|--------|--------|
| Base | Q4_K_S | **23.7** | 817 | - |
| Base | UD-v3 | **34.1** | 842 | - |
| MTP-nat | Q4_K_S | 23.7 | 702 | 52% |
| MTP-ext | UD-v3 | 22.9 | 704 | 43% |
| DFlash2 | Q4_K_S | 10.9 | 187 | 30% |

### Context 65536 (VRAM saturated, ~22.5/24 GB - browsers/Discord open)
| Config | tg t/s |
|--------|--------|
| Base Q4KS | 23.7 |
| MTP-ext UD-v3 | **9-11** |
| DFlash2 | 11-14 |

### Your daily driver (Q3_K_XL @131072, f16 KV, no quant, no spec)
- `UD-Q3_K_XL` at 131072 ctx, context 40-70% full: **20-26 tok/s** (your recent coding sessions).

Takeaway from those: already-observed ceiling at 128k with normal quants is ~20-34 t/s
depending on model size and KV fullness; spec decoding hurts at 64k+ (memory-bound).

## 2. My measurements (WSL, ROCm 7.14, QingYis ROCmFP4 fork build)

Same GPU, but fresh build; these numbers should beat anything native-Windows
since the fork's kernels are the things under audit.

- llama-bench, `Q4_K_S`, 131072 ctx, 99 ngl, q4_0 KV, fa on (unified KV):
  - `pp131072` = **360.2 t/s**
  - `tg32` = **30.4 t/s**
- llama-server, same model, non-unified KV, q4_0 KV, 67k-token agentic/coding
  prompt (real llama.cpp source wrapped in XML, ~19 MB), no speculation:
  - **tg = 23.9 t/s**, pp (67k) = 509 t/s
- llama-server, IDENTICAL workload + stacked spec
  (`--spec-type ngram-map-k4v,draft-dflash`, `-md` DFlash2 draft loaded,
  acceptance 0.477, mean draft len 3.3):
  - **tg = 23.8 t/s** -> zero gain on code content (spec overhead == draft win)

Conclusion: on **Q4_K_S / Q3_K_XL class quants**, the fork gives us the same
23-30 t/s wall we already had. Its big claims (41 / 68 / 76 / 98 t/s) are
tied to the FP4 model format, NOT to the plain quants we are running.

## 3. What the fork's docs claim, and what it is tied to

Source: `llama-rocmfp4/docs/rocmfp4.md`. Measurement context: Qwen3.8-27B,
**256K ctx, fp4 KV cache, FP4 model (type 100/101 = Q4_0_ROCMFP4[_FAST])**,
RDNA3 wave32.

| Config | token/s |
|--------|---------|
| fp4 baseline, no speculation | **41** |
| + DFlash2 (n5) | reasoning 68-76, prose 30-35 |
| + DFlash2 + ngram-map-k4v stacked | repetitive **98** |

The speedup is intrinsic to the FP4 tensors (18/17 bytes per 32 values, RDNA3
WMMA fp4 path) - tiny KV and model footprint vs Q4_K. The DFlash2/ngram spec
stack layers on top of that. On our current GGUFs there is no fp4 path to exploit.

## 4. Are we using FP4? Do we need an FP4 model?

- **No, none of our models is FP4.** All are standard quants:
  - `Qwen3.8-27B-Q4_K_S.gguf` (16.1 GB)
  - `Qwen3.8-27B-UD-Q4_K_S_dynamicV3.0.gguf` (15.4 GB)
  - `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13.1 GB)
  - `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf` (10.1 GB) + mtp
  - DFlash2 draft (1.1 GB), MTP drafts, mmproj
- The fork's whole point IS the fp4 format. To hit 60-90 tok/s we must run a
  **Q4_0_ROCMFP4 (type 100/101)** gguf. Candidates found earlier:
  - `pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN`
  - `kingjones777/...` (ROCmFP4 STRIX builds)
  - `jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF`
  - fork script defaults to `Qwen3.8-27B-heretic-ara.ROCmFP4-STRIX.gguf`
- Recommendation: yes, phase-3 sourcing = acquire such a Q4_0_ROCMFP4 Qwen3.8-27B
  gguf (or quantize from BF16 with `llama-quantize --type 100`). Without it we
  are stuck at the ~23-30 t/s of Section 2 no matter how well the kernels run.

## 5. Open items / next steps

1. Verify KPIs on an actual FP4 gguf (Phase 3): pp >500 t/s?, tg 40-70+.
2. Confirm DFlash2/ngram benefit on the fp4 base (author claims +1.5-2.4x).
3. Note: fork `serve-dflash2.sh` does NOT pass `-md` (draft model), it relies
   on `-ngld` only - likely a script bug; we confirmed the stack needs `-md`.

## 6. STRIX FP4 measured (fits the 41/68/98 claims IF ctx is right)

STRIX = `Qwen3.8-27B-ROCMFP4-STRIX.gguf` (13.75 GiB, fp4 type100/101), fork build
gfx1100, 512 gen tokens, temp0/k1. "ctx" = -c value. VRAM = dedicated usage live
(GPU counter) right after the 113k prefill (or as measured during repetitive run).
q4_0 KV + `LLAMA_ATTN_ROT_DISABLE=1` unless marked f16.

### Coding/reasoning prompt (agentic90k_plain.txt, 113,182 tokens) — full matrix
| engine/quant | spec | ctx | pp | tg | acc/len | VRAM |
|---|---|---|---|---|---|---|
| FORK STRIX | none | 131072 | 358 | 21.4 | - | 17.5 |
| FORK STRIX | none f16KV | 131072 | 408 | 29.5 | - | - |
| FORK STRIX | stack ngram+dflash | 131072 | 343 | 27.1 | .48/3.8 | 19.5 |
| FORK STRIX | MTP n5 | 131072 | 339 | **36.3** | .63/4.1 | 19.8 |
| FORK STRIX | MTP n5 | 262144 | 338 | **36.6** | .63/4.1 | 19.8 |
| STOCK Q4_K_S | none f16KV | 131072 | 126 | 10.8 | - | (spills to RAM) |
| STOCK Q4_K_S | none | 131072 | 579 | 20.8 | - | - |
| STOCK Q4_K_S | MTP n5 | 131072 | 177 | 34.0 | .70/4.5 | - |
| STOCK IQ3_XXS | none | 131072 | 577 | 22.8 | - | - |
| STOCK IQ3_XXS | DFlash2 | 131072 | 538 | 31.6 | .46/3.3 | - |
| STOCK UD-Q3_XL | none | 131072 | 586 | 21.9 | - | - |
| STOCK UD-Q3_XL | DFlash2 | 131072 | 539 | **37.9** | .60/4.0 | - |
| FORK UD-Q3_XL | DFlash2 | 131072 | 351 | 32.2 | .65/4.0 | 19.3 |
| STOCK UD-Q4 dyn | none | 131072 | 592 | 20.9 | - | - |
| STOCK UD-Q4 dyn | MTP | 131072 | 407 | 31.3 | .53/3.6 | - |

### Repetitive boilerplate mirror (60 json blocks + "print block 61", 14,409 tok)
| engine/quant | spec | ctx | pp | tg | acc/len | VRAM |
|---|---|---|---|---|---|---|
| FORK STRIX | none | 131072 | 679 | 32.7 | - | - |
| FORK STRIX | ngram-only | 131072 | 674 | 80.4 | .96/39.9 | - |
| FORK STRIX | ngram-only | 262144 | 678 | 78.8 | .96/39.9 | - |
| FORK STRIX | stack ngram+dflash | 131072 | 574 | **111.9** | .97/17.0 | - |
| FORK STRIX | stack | 262144 | 122 | 35.0 | .97/17.0 | 21.5 |
| FORK STRIX | MTP n5 (n=60) | 262144 | - | 68.3 | - | - |
| STOCK Q4_K_S | MTP | 131072 | 493 | 64.4 | .95/5.8 | - |
| STOCK Q4_K_S | none | 131072 | 826 | 30.7 | - | - |

### Decode scales with ctx (MTP, 262144): 5tok->60.4, 4.5k->43.7, 113k->36.6 t/s
t/s is attention+KV-bandwidth bound -> the docs' 41/68/76/98 are valid only at the
same ctx/fill level. At 70-113k (our real weekly load) MTP delivers ~37 on STRIX.

### Burst sweep (2026-09-27, FORK build-wsl q4_0 KV rot off, `-np 1`, 64-tok burst, cold):
| fill | STRIX+MTP n5 | UD-Q3_K_XL+DFlash2 n5 p0.4 |
|---|---:|---:|
| ~5 tok | **73.6** | 51.5 |
| ~4.5k | **49.0** | 35.6-36.2 |
| ~16k | **57.1** | 34.2-41.4 (warm) |
| ~32k | **44.5** | 23.6-32.0 |
| ~64k | 38.7 | **43.9** (acc 0.91/5.25) |

STRIX+MTP wins every <=16k burst point; UD-Q3_K_XL+DFlash2 takes over at 64k+ and
matches history at 113k (37.9 vs 36.6). The earlier 4.27-6.83 t/s server "collapse"
and the 32k STRIX "cliff" were n_slots=4 VRAM overflow (4 x ctx q4_0 KV); `-np 1`
restores full speed everywhere. Root cause of all big-KV slowdowns this week:
default `n_slots=4` on a 24 GB card, not model/build/GPU throttling.

### Q4_K_S burst ladder (2026-09-28, MERGED build a3d2c2b52, `-np 1`, 128-tok decode, q4_0 KV)
Fills the gap: the 2026-09-27 burst table had no Q4_K_S column. Q4_K_S has no
embedded MTP, so it needs `-md mtp-Qwen3.8-27B-Q4_0_dynamicV3.0.gguf`.
| fill | pp | MTP n5 tg | MTP off tg | note |
|---|---:|---:|---:|---|
| 8 tok | 14.0 | **50.41** | - | prefill n/a at 8 tok |
| 5,152 | 642.7 | **37.11** | - | |
| 20,651 | 710.9 | 30.79 | - | |
| 51,952 | 646.5 | 32.92 | - | |
| 116,841 | 508.4 | **25.91** | 19.36 | MTP +34% decode, -5% prefill |

MTP is a clear decode win at every fill (+34% at 117k) and costs ~5% of prefill,
matching the Qwopus prompt-length curve. Q4_K_S+MTP 25.9 t/s at 117k is
comparable to STRIX+MTP 36.6 but STRIX still leads decode at long ctx.

### IQ3_XXS on a3d2c2b52 (2026-09-28, MERGED build, 113182 fill, q4_0 KV, `-np 1`)
Fills the gap: section 7's IQ3_XXS rows were stock-build only and section 11
re-tested FP4 models only, so there was no merged-build IQ3_XXS row.
| spec | pp | tg | note |
|---|---:|---:|---|
| DFlash2 n5 p0.4 | 541.5 | **26.45** | -md DFlash2-Q4_K_M |
| none (control) | 583.4 | 23.02 | |

DFlash2 costs ~7% of prefill and buys +15% decode on IQ3_XXS. This model also has
`nextn_predict_layers=1` (n_layer_all=65), so it is a candidate for the
Qwopus-style MTP path, but its own 10.4 GB MTP head was previously
observed to hang and is recorded in section 7 as "not a drafter" - untested on
the current build.

### Qwopus small-ctx ladder with VRAM (2026-09-28, a3d2c2b52, 12368 fill, MTP n5, q4_0 KV, `-np 1`)
VRAM is captured from `-lv 5` allocation logs (`vram_of.sh`), not from
`amd-smi`/`rocm-smi`, which return nothing usable on this WSL GPU-PV path.
| n_ctx | pp | tg | main KV | MTP draft KV | recurrent | compute |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | - | - | 144 MiB | 32 MiB | 897.75 MiB | 28.02 MiB |
| 32,768 | 786.20 | 48.48 | 576 MiB | 128 MiB | 897.75 MiB | 52.02 MiB |
| 65,536 | 787.57 | 50.98 | 1,152 MiB | 256 MiB | 897.75 MiB | 84.02 MiB |
| 131,072 | 780.74 | 49.81 | 2,304 MiB | 512 MiB | 897.75 MiB | 148.02 MiB |

c=8192 is not runnable: the 12368-token probe prompt exceeds it and the
server returns HTTP 400. Its allocation figures are still valid.

**This settles the architecture question.** Qwopus is a hybrid: 16 attention
layers (the only part whose KV grows with context) plus 64 recurrent
DeltaNet layers whose state is a single fixed 897.75 MiB cell, independent of
n_ctx. That is why prefill stays ~780-790 t/s from 32k to 131k: only 16 of
80 layers do quadratic attention, and at 262k the projected total is 22.3 GiB
of 24 GiB.

Note the MTP draft KV is **f16 regardless of `-ctk/-ctv q4_0`**, which apply to
the main cache only. Draft KV is 1/4 of main KV here. Forcing
`-ctkd q4_0 -ctvd q4_0` should halve that; untested on the current build.


### UD-Q4_K_S dynamic on a3d2c2b52 (2026-09-28, MERGED build, q4_0 KV, `-np 1`)
Section 7 had UD-Q4_K_S rows from the stock build only. Draft model is
`mtp-Qwen3.8-27B-Q4_0_dynamicV3.0.gguf` (1.37 GB).
| fill | spec | pp | tg |
|---|---|---:|---:|
| 113,182 | MTP n5 | 563.6 | **29.63** |
| 113,182 | none (control) | 594.7 | 21.29 |
| 116,841 | MTP n5 | 554.2 | 29.30 |

MTP is +39% decode for ~5% prefill, again matching the Qwopus and Q4_K_S
results. This is the largest *relative* MTP decode win measured so far.

**Corrected for a wrong claim.** UD-Q4_K_S+MTP at 113k (29.63 t/s) is *not*
the fastest non-FP4 configuration on the merged build. Qwopus Q4_K_M+MTP
reaches 33.46 t/s at the same 113182 fill (FO3_113k_mtp), 13% faster. UD-Q4_K_S
leads only within the two Q4_K_S-family models.

### IQ3_XXS small-ctx + 256k with VRAM (2026-09-28, a3d2c2b52, q4_0 KV, `-np 1`)
| case | fill | pp | tg | main KV | draft KV | recurrent | compute | weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c=32768 DFlash2 | 12,368 | 774.1 | 32.77 | 576 MiB | 50 MiB | 897.75 MiB | 52.02 MiB | 10.17 GiB |
| c=262144 DFlash2 | 199,626 | 470.2 | 24.65 | 4,608 MiB | 50 MiB | 897.75 MiB | 276.02 MiB | 10.17 GiB |
| c=262144 none | 199,626 | 502.3 | 18.32 | 4,608 MiB | - | 149.62 MiB | 276.02 MiB | 10.17 GiB |

**IQ3_XXS is the best long-context model tested.** At 262144 it decodes at
24.65 t/s, versus 3.47 for Qwopus and 22.30 for STRIX at the same context. Its
10.17 GiB weights leave enough VRAM that the whole 4.5 GiB q4_0 KV fits
comfortably, which the 15.65 GiB Qwopus does not.

### What speculative decoding actually costs in VRAM (hybrid models)
The recurrent state is *not* simply context-independent. It is 149.62 MiB with
`0 rs_seq` (speculation off) and 897.75 MiB with `5 rs_seq` (DFlash2 or MTP on)
- a **+748 MiB** cost to maintain 5 parallel recurrent state sequences. That
also corrects the note above: the 897.75 MiB figure is speculation-dependent,
not fixed.

The draft cache itself is negligible and context-independent: DFlash2 allocates
50 MiB (2560 cells, 5 layers, f16) at both 32768 and 262144. By contrast the
Qwopus MTP head allocates 1 f16 layer that *does* scale (32 MiB at 8k to
512 MiB at 131k), so MTP is the more expensive of the two drafters on these
models.

### Draft-KV quantisation fixes the 262k MTP decode collapse (2026-09-28, a3d2c2b52, Qwopus, 199626 fill, q4_0 main KV, `-np 1`)
`-ctk/-ctv q4_0` apply to the **main** cache only; the MTP draft cache stays f16
and scales with context. Forcing `-ctkd q4_0 -ctvd q4_0` cuts draft KV from
**1024 MiB to 288 MiB** at c=262144 (measured in the `-lv 5` allocation log).

| draft KV | pp | tg | wall |
|---|---:|---:|---:|
| f16 (default) | 429.78 | 3.47 | 501 s |
| q4_0 (forced) | 214.25 | **8.53** | 947 s |

**Quantising the draft cache more than doubles decode (3.47 -> 8.53 t/s, +146%)
and halves prefill (429.78 -> 214.25, -50%).** The 262144 MTP collapse was
substantially a draft-cache precision/bandwidth problem, not an inherent
context limit. The VRAM it frees (736 MiB) is real, but the prefill cost is not
worth paying for short generations.

Break-even is a fill-vs-generate tradeoff, not a straight win. Equating
`N/429.78 + M/3.47` against `N/214.25 + M/8.53` gives **M ~ 0.0137 N**: after a
199626-token prefill, forced q4_0 draft KV only pays off past roughly 2730
generated tokens. Our 128-token probe measures the wrong side of that line,
which is why wall time nearly doubles here. Real workloads decode far more.

### Harness constraint discovered: do NOT use -lv 5 on long prompts
`run_one_vram.sh` runs the server at `-lv 5` to scrape allocations. On Qwopus at
c=262144 with a 199626-token prompt that makes the server log at debug level
per slot event and it **loops in `process_sing: decoding, decline task` without
ever prefilling** (388 declines, no progress, `FO21`/`FO22`). The identical
configuration on the plain `run_one.sh` harness runs with 0 declines. The
q4_0 arm in `FO23` was therefore collected with `run_one.sh`, which is why it
has no `[vram: ...]` suffix. Allocation figures for that arm come from the
`-lv 5` log of the run that did allocate before it wedged. Use `-lv 5` for short
prompts only.

### The 262144 decode cliff is a step, not a slope (Qwopus, fixed fill 129945, MTP, f16 draft KV)
Adding c=196608 bisects the 131072->262144 gap the earlier data only bracketed.
Every row is the same fill, so `-c` is the only variable.

| n_ctx | pp | tg |
|---:|---:|---:|
| 131,072 | 494.87 | 27.64 |
| 196,608 | 494.14 | 27.81 |
| 262,144 | 517.45 | **6.41** |

Decode is flat at ~27.7 t/s through 196608, then falls off a cliff at 262144.
There is no gradual decay to explain, and prefill is unaffected (494-517 t/s at
all three). `262144` is also the model's declared `qwen35.context_length`, so the
boundary looks like a hard cap in the model or its rope config rather than a
VRAM or attention-cost effect. This is consistent with `-ctkd q4_0` recovering
6.41 -> 8.53 t/s: something about operating exactly at the cap makes the f16
draft cache pathological, and quantising it mitigates but does not remove it.

### q4_0 draft KV only helps AT 262144, and is neutral-to-negative below
| case | draft KV | pp | tg |
|---|---|---:|---:|
| fill 113182, c=131072 | f16 | 559.88 | 33.46 |
| fill 113182, c=131072 | q4_0 | 523.93 | 33.18 |
| fill 129945, c=196608 | f16 | 494.14 | 27.81 |
| fill 129945, c=196608 | q4_0 | 499.33 | 26.82 |
| fill 199626, c=262144 | f16 | 429.78 | 3.47 |
| fill 199626, c=262144 | q4_0 | 214.25 | **8.53** |

This sharpens the earlier conclusion. Forced q4_0 draft KV is not a general
improvement: below the boundary it changes nothing measurable (33.18 vs 33.46,
26.82 vs 27.81) and costs prefill everywhere. Its only use is as a targeted
mitigation at c=262144, where it more than doubles decode and frees 736 MiB.

### Context ladder at fixed fill 129945, all 4 models (2026-09-28, a3d2c2b52, q4_0 KV, `-np 1`)
Same fill for every row, so `-c` is the only variable. This is the table that
shows where each model actually falls off.

| n_ctx | Qwopus + MTP | UD-Q4_K_S + MTP | IQ3_XXS + DFlash2 | STRIX + MTP |
|---:|---:|---:|---:|---:|
| 131,072 | 27.64 | 23.60 | 28.09 | 27.94 |
| 196,608 | 27.81 | 23.73 | 27.49 | 27.96 |
| 262,144 | **6.41** | **24.83** | **24.65** | 22.30 |

**Only Qwopus falls off a cliff at 262144.** The other three are flat through
196608 and stay healthy at 262144 (24.83, 24.65, 22.30 vs Qwopus's 6.41). All
four are within 23.6-28.1 t/s of each other at both 131072 and 196608, so the
cliff is specific to the Qwopus finetune at the boundary, not a property of the
Qwen3.8-27B architecture, the draft scheme, or this GPU. Qwopus is the only one
of the four carrying its MTP head inside the main model rather than a separate
`-md` file, and it is also the only one whose declared `context_length` is
exactly 262144 - the two facts that would need explaining to anyone chasing the
root cause.

### UD-Q4_K_S 262144 control (closes the last unpaired 256k run)
| case | fill | pp | tg |
|---|---:|---:|---:|
| c=262144 MTP | 199,626 | 130.06 | 24.83 |
| c=262144 none | 199,626 | 429.97 | 15.37 |

MTP still wins decode (+62%) but costs 70% of prefill at this context, far worse
than the ~5% prefill penalty MTP carries at 131072. Long-context MTP is
prefill-expensive on this model in a way it is not at moderate context.

### Complete 4-model x 5-context ladder, with input/output speed and full VRAM (2026-09-28, a3d2c2b52)
Fill is 6022 tokens for all 20 cells, so `-c` is the only variable. Every row's
`n_ctx` is asserted against the server log by `assert_ctx.sh`.
`pp` = input/prefill t/s, `tg` = output/decode t/s. All VRAM columns are GiB and
are the sum of the allocations the server reports at `-lv 5`; TOTAL is their sum.
Qwopus has no separate draft file (its MTP head lives inside the main GGUF), which
is why its draft-weight column is n/a.

| model | n_ctx | input pp | output tg | weights | draft w | main KV | draft KV | recurrent | compute | TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwopus Q4_K_M+MTP | 8,192 | 612.1 | 41.56 | 15.65 | n/a | 0.14 | 0.031 | 0.88 | 0.03 | **16.73** |
| Qwopus Q4_K_M+MTP | 32,768 | 683.5 | 43.09 | 15.65 | n/a | 0.56 | 0.125 | 0.88 | 0.05 | **17.27** |
| Qwopus Q4_K_M+MTP | 65,536 | 678.8 | 41.98 | 15.65 | n/a | 1.12 | 0.250 | 0.88 | 0.08 | **17.98** |
| Qwopus Q4_K_M+MTP | 131,072 | 679.4 | 42.17 | 15.65 | n/a | 2.25 | 0.500 | 0.88 | 0.14 | **19.42** |
| Qwopus Q4_K_M+MTP | 196,608 | 704.5 | 44.58 | 15.65 | n/a | 3.38 | 0.750 | 0.88 | 0.21 | **20.86** |
| STRIX FP4+MTP | 8,192 | 633.6 | 61.10 | 13.74 | 1.84 | 0.14 | 0.031 | 0.88 | 0.03 | **16.66** |
| STRIX FP4+MTP | 32,768 | 641.1 | 63.98 | 13.74 | 1.84 | 0.56 | 0.125 | 0.88 | 0.05 | **17.20** |
| STRIX FP4+MTP | 65,536 | 629.2 | 59.52 | 13.74 | 1.84 | 1.12 | 0.250 | 0.88 | 0.08 | **17.91** |
| STRIX FP4+MTP | 131,072 | 636.3 | 60.05 | 13.74 | 1.84 | 2.25 | 0.500 | 0.88 | 0.14 | **19.35** |
| STRIX FP4+MTP | 196,608 | 641.3 | 63.35 | 13.74 | 1.84 | 3.38 | 0.750 | 0.88 | 0.21 | **20.79** |
| UD-Q4_K_S+MTP | 8,192 | 572.1 | 52.11 | 14.29 | 1.27 | 0.14 | 0.031 | 0.88 | 0.03 | **16.64** |
| UD-Q4_K_S+MTP | 32,768 | 709.5 | 54.81 | 14.29 | 1.27 | 0.56 | 0.125 | 0.88 | 0.05 | **17.18** |
| UD-Q4_K_S+MTP | 65,536 | 694.6 | 53.44 | 14.29 | 1.27 | 1.12 | 0.250 | 0.88 | 0.08 | **17.89** |
| UD-Q4_K_S+MTP | 131,072 | 672.2 | 51.29 | 14.29 | 1.27 | 2.25 | 0.500 | 0.88 | 0.14 | **19.33** |
| UD-Q4_K_S+MTP | 196,608 | 677.4 | 51.93 | 14.29 | 1.27 | 3.38 | 0.750 | 0.88 | 0.21 | **20.77** |
| IQ3_XXS+DFlash2 | 8,192 | 635.9 | 33.04 | 10.17 | 1.05 | 0.14 | 0.049 | 0.88 | 0.03 | **12.31** |
| IQ3_XXS+DFlash2 | 32,768 | 650.2 | 33.58 | 10.17 | 1.05 | 0.56 | 0.125 | 0.88 | 0.05 | **12.76** |
| IQ3_XXS+DFlash2 | 65,536 | 640.7 | 32.64 | 10.17 | 1.05 | 1.12 | 0.250 | 0.88 | 0.08 | **13.35** |
| IQ3_XXS+DFlash2 | 131,072 | 656.5 | 32.91 | 10.17 | 1.05 | 2.25 | 0.500 | 0.88 | 0.14 | **14.54** |
| IQ3_XXS+DFlash2 | 196,608 | 643.9 | 30.99 | 10.17 | 1.05 | 3.38 | 0.750 | 0.88 | 0.21 | **15.73** |

**VRAM grows almost entirely through the KV caches, and only ~4.1 GiB separates
8192 from 196608.** 8k -> 196608 costs +4.13 GiB for every model, and the split is
the same everywhere: main KV +3.23 GiB (q4_0, scales with `-c`) and MTP draft KV
+0.72 GiB, against a constant 0.88 GiB of recurrent state and <=0.21 GiB of
compute. Nothing else moves. So context costs VRAM predictably and cheaply on
this GPU; it is the *decode speed* at 262144 that is anomalous, not the memory.

**STRIX is the best decode/VRAM trade by a wide margin**: 59-64 t/s output at
every context, essentially tied with UD-Q4_K_S on total VRAM, and it needs 1 GiB
less than Qwopus at every single point. Its 196608 point is the fastest decode
measured in this whole study at 63.35 t/s.

**IQ3_XXS is ~4 GiB cheaper than everything else at every context** (12.31 ->
15.73 GiB) because its weights are 5.5 GiB smaller, but it pays for that with the
slowest decode by a factor of two (31-34 t/s, less than half STRIX).



Fill is 6022 tokens for every row, so the only variable is the allocated `-c`.
Every row's `n_ctx` is asserted against the server log by `assert_ctx.sh`; two

first-attempt rows are excluded because they ran at the wrong context and are
listed under "rejected" below.

Decode-only view of the same 20 cells (all from the table above, which has VRAM):

| n_ctx | Qwopus + MTP | UD-Q4_K_S + MTP | IQ3_XXS + DFlash2 | STRIX + MTP |
|---:|---:|---:|---:|---:|
| 8,192 | 41.56 | 52.11 | 33.04 | 61.10 |
| 32,768 | 43.09 | 54.81 | 33.58 | 63.98 |
| 65,536 | 41.98 | 53.44 | 32.64 | 59.52 |
| 131,072 | 42.17 | 51.29 | 32.91 | 60.05 |
| 196,608 | 44.58 | 51.93 | 30.99 | 63.35 |

Output speed is flat in context for all four models, within run-to-run noise
(~+-3 t/s, larger than any real trend). This supersedes the earlier partial
ladder, which had no STRIX column and used an older VRAM parser.

**A third VRAM parser bug, found while filling this table.** The parser
identified the speculative draft cache as the one with `"  1 layers"`. That is true
for MTP drafts but false for DFlash2, whose draft cache is 5 layers, so IQ3_XXS
rows were reporting the 50 MiB *draft* as `main_kv` and `draft_kv=NA`. Fixed by
picking the cache with the **most** layers as main and any other non-zero cache as
the draft, which handles 1-layer MTP and 5-layer DFlash2 without hardcoding a
model. A fourth issue: `draft_w` took the second `file size` line, but Qwopus has
no `-md` file so all of its `print_info` lines repeat the same 15.65 GiB, which
inflated its total to 32.38 GiB. Now the draft weight is the first file size that
*differs* from the main weights, else `NA`. The whole 20-cell ladder was re-run
under the fixed parser (`FO28`/`FO29`) rather than patched in place; only the
Qwopus `draft_w` fields of the affected rows were corrected after the fact, and
the `pp`/`tg` measurements were never touched.


**Decode is essentially flat in context** below 131k for all three models, and
prefill is too (~700 t/s Qwopus/UD, ~625 t/s IQ3). Speculative decoding is
reading a small working set, not a large one, so there is no cliff to find here.

`c=8192` is now genuinely measured, not skipped: `short6k.txt` was cut from
`short8k.txt` and its token count verified as exactly 6022 against the server's
own tokenizer via `/tokenize` (2170 tokens of headroom). The earlier c=8192
failures were a 12368-token prompt exceeding the context, not a model limit.

**Rejected rows.** `FO19_iq_c65536_df` and `FO19_iq_c131072_df` were both
tagged for one context but actually ran at `-c 262144` because run_one.sh takes
the *last* `-c` and I copy-pasted the wrong value. Their numbers (32.25 and
33.03 t/s at fill 6022) are real measurements of c=262144, not of the tagged
context, and must not be read as ladder points. `FO19_u4_c131072_mtp` and
`FO20_u4_c131072_mtp` both hit a transient `cudaMalloc` OOM at a context that a
*larger* 262144 run had already succeeded at, from the previous server not
having released its VRAM; the third attempt, on an idle GPU, gave 53.20 t/s.
Back-to-back server loads need an idle-GPU precondition, not just a fixed sleep.

### UD-Q4_K_S small-ctx + 256k with VRAM (2026-09-28, a3d2c2b52, q4_0 KV, `-np 1`)
| case | fill | pp | tg | main KV | draft KV | recurrent | compute | weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c=32768 MTP | 12,368 | 818.3 | 58.74 | 576 MiB | 128 MiB | 897.75 MiB | 52.02 MiB | 14.29 GiB |
| c=262144 MTP | 199,626 | 130.1 | 24.83 | 4,608 MiB | 1,024 MiB | 897.75 MiB | 276.02 MiB | 14.29 GiB |

At 256k fill the KV allocation is identical to the 199626-token Qwopus run
(2304 MiB per 131072 cells), confirming the 16-attention-layer structure is a
Qwen3.8-27B property, not specific to the Qwopus finetune. Unlike Qwopus,
UD-Q4_K_S holds up at 256k: 24.83 t/s decode versus Qwopus's 3.47 t/s at the
same c, though prefill is much worse (130.1 vs 497.2 t/s) because the 64
recurrent layers that dominate a long prefill are now competing for a larger
attention KV. c=8192 is untestable with the 12368-token prompt (HTTP 400).



### Retry of UD-Q3_XL + DFlash2 (earlier "hang" = transient GPU issue, not model):
STOCK build: tg=37.9 acc .60 (72k prompt OK). FORK build: tg=32.2 acc .65. Both work.

### Working vs broken prompts (ROOT-CAUSED)
- XML-tag template (`<system>/<file>/<task>` ending `<report>\n`) -> deterministic
  1-token `</report>`+EOS at EVERY length 530..113k. Plain-prose framing of the same
  task works and yields real multi-KB output. NEVER use the report-tag trailer.
- Repetitive-JSON prompt + any instruction tail -> 1-token EOS at >=5k tok.
  Identical JSON body without a tail generates normally. ngram lever + "print next
  block" mirror = the reliable repetitive scenario (matches docs' 98-style case).
- KV type (q4_0 vs f16) and LLAMA_ATTN_ROT_DISABLE do NOT cause the above.

### Bottom line
Daily driver (70-113k ctx, coding): **STRIX + MTP @131072 = 37 t/s**, 22 t/s base.
Stretch (256k): **36.6 t/s**, fits 24 GB with q4_0 KV. Repetitive (ngram scenario):
**80-112 t/s**. Stock engine is within ~10% on MTP (34 vs 37). NOTE (corrected):
ngram-map-k4v IS available in BOTH builds (verified via `strings` on llama-server —
both stock and fork expose draft-dflash/draft-mtp/ngram-map-k4v/ngram-map-k/ngram-mod/
ngram-cache); see Section 10 for the fork-vs-original-HEAD A/B.

## 7. STOCK cross-model MTP/DFlash gap-fill (113k agentic prompt, 131072 ctx, q4_0 KV)
| STOCK model | spec | pp | tg | acc/len | note |
|---|---|---|---|---|---|
| IQ3_XXS | none | 577 | 22.75 | - | |
| IQ3_XXS | DFlash2 n5 | 538 | 31.62 | .46/3.3 | +39% |
| IQ3_XXS | its own mtp (10.4G file) | 139 | HANG | - | full-size MTP model not a drafter |
| UD-Q4_K_S dyn | none | 592 | 20.89 | - | |
| UD-Q4_K_S dyn | MTP (mtp-Q4_0_dynV3.0, 1.37G) | 407 | 31.26 | .53/3.6 | +50% |
| UD-Q3_K_XL | none | 586 | 21.90 | - | |
| UD-Q3_K_XL | MTP (dynV3.0) | 555 | 9.38 | 0.00/1.0 | WRONG DRAFTER family (0 accept) |
| UD-Q3_K_XL | DFlash2 (stock) | 539 | **37.93** | .60/4.0 | RETRY OK - earlier hang was transient GPU issue |
| UD-Q3_K_XL | DFlash2 (fork) | 351 | 32.16 | .65/4.0 | VRAM 19.3G |
| Q4_K_S | MTP (dynV3.0 compatible) | 177 | 34.0 | .70/4.5 | |

Notes: UD-Q3_K_XL's only spec-decoder that works = DFlash2 (MTP mismatched).

## 8. MAX-CTX FIT TEST: every model at 262144, context FILLED to 258,801 tokens (q4_0 KV)
Repetitive/network-config boilerplate prompt, n_predict=32, temp 0. ALL FIT.
"VRAM" = Dedicated GPU memory in use shortly after the request (steady state; STRIX rows
verified with a concurrent 2s-interval peak sampler during the whole request).

| Model | engine | spec | ctx | fill(tok) | pp | tg | VRAM(GB) |
|---|---|---|---|---|---|---|---|
| STRIX (fp4 hybrid) | fork | none | 262144 | 258801 | 209.7 | 13.79 | 23.37 |
| STRIX | fork | MTP n5 | 262144 | 258801 | 198.4 | **24.80** | 23.32 |
| STRIX | fork | ngram-k4v+DFlash2 stack | 262144 | (129k aborted) | ~40t/s prefill | - | 21.5 |
| Qwen3.8-27B-Q4_K_S | stock | none | 262144 | 258801 | 392.2 | 13.18 | 22.31 |
| UD-Q3_K_XL | stock | none | 262144 | 258801 | 395.9 | 13.26 | 19.79 |
| UD-Q3_K_XL | fork | DFlash2 n5 | 262144 | 258801 | 207.5 | 20.92 | 21.9 |
| UD-Q4_K_S dyn | stock | none | 262144 | 258801 | 397.6 | 13.22 | 21.85 |
| IQ3_XXS | stock | none | 262144 | 258801 | 391.1 | 13.68 | 17.4 |
| IQ3_XXS | stock | DFlash2 n5 | 262144 | 258801 | 370.0 | **21.82** | 19.61 |

Answer to "how can 128k and 256k use the same VRAM?": they don't once FILLED. Earlier
"19.8GB at both" was measured with the same 113k fill at both -c values. STRIX is a
hybrid (48 DeltaNet linear + 16 full-attn layers): only the 16 attn layers carry real KV
(and KV here is q4_0), so KV at 262k is only ~4.7GiB total and grows lazily per used
token; -c only caps it. Filled to 258k, STRIX+MTP peaks 23.32 GB. Everything SST fits in
24GB at FULL 262144. Stack is impractical at full fill (prefill collapses to ~40 t/s).

## 9. QUALITY: retention vs full BF16 (size on x, % full on y)
Plot: `quality_retention_plot.png` (repo root). Open markers = PROXY (eval on another
27B base: Gemma-3-27B MMLU, llama.cpp KLD gist, NVFP4 band); FILLED = MEASURED (in-model).

| our file | GB | bpw | retention | source |
|---|---|---|---|---|
| GSQ-RCO **IQ3_XXS** | 10.1 | 3.00 | **100.6%** (zs-avg; =base on AIME25, GPQA 88.89/89.90, LCB 84.57/85.71) | IN-MODEL, ISTA-DASLab GSQ page |
| UD-Q3_K_XL dynV3.0 | 13.1 | ~3.85 | ~99.1% | proxy (Unsloth Gemma-3-27B) |
| STRIX fp4 hybrid | 13.8 | 4.3 | ~95-97% | proxy (NVFP4 PTQ band) |
| UD-Q4_K_S dynV3.0 | 15.4 | ~4.5 | ~99.6% | proxy (Unsloth) |
| Q4_K_S imatrix | 16.1 | 4.57 | ~99.3% | proxy (llama.cpp KLD/linPPL) |
| Occamy-1.0 Q4_K_M | 21.2 | 4.5 | ~99% (vs its own BF16; DIFFERENT base, Qwen3.6-35B-A3B MoE) | proxy |

IQ3_XXS expectation CORRECTED: it's NOT a plain imatrix IQ3_XXS but the GSQ+RCO build
(arXiv 2604.18556 / 2605.00649); its page benchmarks it ON Qwen3.8-27B and it is
task-lossless (wp2 PPL 7.20 vs 7.05, zs-avg recovery 100.6%, AIME25 100.0 = base, GPQA-D
88.89/89.90, LCB 84.57/85.71). The earlier "~95.5%" gemma-proxy estimate was wrong.
Cross-source caveat: gemma/llama3 proxies still unverified on Qwen3.8; llama-perplexity
on wikitext2 with all six files would replace every proxy (offered, not run).

### Extracted vendor benchmark data (2026-09-25)

Full extraction + full-res plots live in the merge repo:
`llama-rocmfp4-merge/results/benchmarks/EXTRACTED_BENCHMARK_DATA.md` (images in
`.../images/`). These are the image-only numbers from the model web pages, pulled so
Claude Opus 5.5 can cross-check the charts. Text summary:

- **unsloth Dynamic 3.0 mean KLD vs BF16** (lower = closer): UD-Q4_K_XL 0.0237,
  UD-Q3_K_XL 0.0806, UD-Q2_K_XL 0.2209, UD-IQ2_M 0.2582, UD-IQ2_XXS 0.5210,
  UD-IQ1_M 0.8000, UD-IQ1_S 0.9729 (sizes 15.6 / 12.8 / 9.9 / 9.0 / 7.3 / 6.5 / 6.1 GB).
  Claims: >10% better top-1 vs next-best at same size; UD-Q2_K_XL +8% top-1;
  UD-IQ1_S ~72-77% top-1 agreement at 89% smaller; NVFP4 92-97% accuracy recovery.
- **ISTA-DASLab GSQ-RCO** (zs-avg over 5 tasks, recovery vs BF16): BF16 74.34 / 100.0%;
  GSQ-RCO IQ3_XXS (10.1 GB, 3.00 bpw) 74.81 / **100.6%**, AIME25 100.0 = base,
  wiki PPL 7.20 vs 7.05; GSQ-RCO IQ3_S (11.8 GB, 3.50 bpw) 74.47 / 100.2%,
  wiki PPL 7.07; unsloth UD-IQ3_S (12.0 GB) 75.49 / 101.5% but AIME25 96.67.
- Confronts our Phase Q: GSQ-RCO's own zs-avg calls its IQ3_XXS lossless, but our
  wiki_q gate (same-top 87.19) fails it - metric disagreement flagged in the matrix.

## 10. FORK vs ORIGINAL HEAD re-run (the non-FP4 rows that had run on the fork)
"Original HEAD" = `llama.cpp-stock/` clone of ggml-org/llama.cpp (commit e6ab7c1, 33
behind current upstream, pristine) -> `llama-stock-build` (/home/thegiallo). Same GPU,
same SDK env, same flags (-ctk/-ctv q4_0, -ngl 99, -np 1, temp 0/top_k 1, n_predict 32).

| workload | HEAD | FORK | delta |
|---|---|---|---|
| UD-Q3_K_XL + DFlash2 @262144, fill 258801 | pp 367.7 tg **20.07** acc23/23 len3.88 VRAM 22.6G | pp 207.5 tg 20.92 acc25/25 len6.0 VRAM 21.9G | HEAD prefill +77%, tg ~equal |
| Q4_K_S + ngram-map-k4v,draft-dflash @131072, fill 113182 (agentic90k_plain) | pp **580.7** tg **19.68** acc14/48 len1.82 VRAM 22.1G | pp 396.5 tg 16.62 acc14/48 len1.82 VRAM 22.1G | HEAD +46% pp, +18% tg |

Key findings:
- The 40 t/s prefill collapse (run7, STRIX ngram stack @262k) is FORK-SPECIFIC: on
  original HEAD the identical ngram-map-k4v+draft-dflash stack prefills at 580 t/s
  (= no-spec Q4_K_S rate), i.e. ngram does NOT serialize prefill upstream. The fork
  serializes prefill whenever ngram is active -> quadratic -> ~3.5h at 258k.
- Spec acceptance is engine-independent (same weights/draft/temp0 -> identical 14/48
  guarantees). tg/pp differences are pure engine throughput; the fork wins ONLY on the
  STRIX fp4 format (its fp4 kernels), plain quants run the same or better on HEAD.
- Both engines carry the full spec set (draft-dflash, draft-mtp, ngram-map-k4v, ...):
  Section 6's "k4v is fork-only" was wrong.
## 11. MERGED BUILD: upstream master + ROCmFP4 port, re-tested (FP4 only)
The fork (orphan import: llama.cpp @925e117 + DFlash2 + ROCmFP4) was merged against
upstream master (b11157 = 53ed051ce, the newest release, 11 commits after v0.5.0) via
`--allow-unrelated-histories -X theirs`, then the ~1100-line fp4 delta was hand-ported
onto the current tree (26 files: quant type, ggml-quants, mmq/mma/vecdot kernels,
llama loader/quant, rdna3 config resolved generically = fp4 mirrors NVFP4 config).
Brand "merge-upstream-head" in /home/thegiallo/llama-rocmfp4-merge, built from source
(SDK hipcc, gfx1100, MMQ_MFMA). Fork reference values from sections 6/8 (same GPU, same
prompts, same measurement method: temp0/top_k1, n_predict window, VRAM = steady state).

| workload (STRIX fp4 13.8G) | ctx/fill | MERGED pp | FORK pp | MERGED tg | FORK tg | MERGED acc | FORK acc | MERGED VRAM | FORK VRAM |
|---|---|---|---|---|---|---|---|---|---|
| base (none) | 131072 / 113182 | **512.1** | 358 | 18.57 | 21.4 | - | - | 17.92 | 17.5 |
| MTP n5 | 131072 / 113182 | **487.5** | 339 | 20.55 | **36.3** | .34/2.71 | .63/4.1 | 20.34 | 19.8 |
| base (none) | 262144 / 258801 | **357.9** | 209.7 | 12.82 | 13.79 | - | - | 20.8 | 23.37 |
| MTP n5 | 262144 / 258801 | **240.2** | 198.4 | **2.77** | **24.80** | .94/4.75 | .96/5.17 | 22.85 | 23.32 |
| ngram-k4v,draft-dflash | 131072 / 113182 | **474.3** | 343 | 16.32 | **27.1** | .39/2.71 | .48/3.8 | 20.44 | 19.5 |
| MTP n5 + q4_0 draft KV | 262144 / 199626 | 241.5 | - | **8.74** | - | .43/3.15 | - | - | - |

Root-cause follow-up (2026-09-24): the merged MTP collapse is the MTP head's serial
forward over the FULL draft KV. Verified by forcing the draft KV to q4_0 (-ctkd/-ctvd,
mtp head file has no nextn in main model, so -md is mandatory): decode @262144/fill199626
went 2.63 -> 8.74 t/s (3.3x) with unchanged acceptance (~.44/3.2), i.e. cost scales with
draft KV payload bytes per step. Still 3x under the fork (24.8 @258k, fused draft).
- **Prefill is much faster on the merged build** (+30..70%): base +43%, base@262 +71%,
  MTP +44%, stack +38%, MTP@262 +21%. The upstream scheduler/attention infra benefits the
  ported fp4 kernels; the fork only preserved a thin prefill edge on top of its old base.
- **Decode is slower on the merged build** for every config (-7%..-89%): base@131 -13%,
  base@262 -7%, stack -40%, MTP@131 -43%, MTP@262 -89% (2.77 t/s vs 24.80).
  The MTP regression is twofold: at 113k acceptance dropped .63->.34 (mean draft len
  4.1->2.71), so drafts land worse; at 258k acceptance was healthy (.94/4.75) yet the
  decode path still collapsed to 2.77 t/s - the merged build runs the MTP head's decode
  serially at full ctx (the fork's fused MTP did the same work at 24.8 t/s).

  **SUPERSEDED (2026-09-28, re-measured on a3d2c2b52, 11k commits newer):** the
  258k MTP decode collapse no longer reproduces. STRIX+MTP n5 @262144/fill 199,626,
  q4_0 KV, `-np 1`:
  | config | pp | tg |
  |---|---:|---:|
  | MTP n5, default draft KV | 367.7 | **22.30** |
  | MTP n5, forced q4_0 draft KV (-ctkd/-ctvd) | 379.5 | **23.75** |
  | MTP off (control) | 415.2 | 16.80 |
  So the "MTP head's serial forward over the full draft KV" collapse (2.63 t/s) is
  GONE on current upstream; MTP is now a +33% decode WIN at 262k (22.30 vs 16.80),
  not an 89% loss. Forcing q4_0 draft KV still helps a little (22.30 -> 23.75) but
  the gap it used to close (2.63 -> 8.74) no longer exists. Prefill remains
  ~10% worse with MTP at this fill. The root-cause note above is kept for history
  and is now wrong about the current build.
- VRAM: merged uses less at 258k (20.8/22.85 vs 23.37/23.32) but slightly more at 113k
  (17.92/20.34/20.44 vs 17.5/19.8/19.5). All comfortably inside 24G.
- Correctness: outputs match the fork on the same prompts (same reasoning/intros); the
  ported fp4 kernels load and run identically.
- Net takeaway: on current upstream the fp4 FORMAT prefills fast (and loads), but the
  everyday 70-113k daily-driver decode (base 18.6, MTP 20.6) no longer beats the fork
  (21.4 / 36.3), and extreme-ctx MTP collapses. A straight upstream rebuild of fp4 is not
  an upgrade for decode; the fork's FUSED MTP draft is the piece worth preserving.

## 12. OPEN WORK / TODO

### 12.1 Preserve and port the fork's FUSED MTP draft
- The fork (feat/benchmark, /home/thegiallo/llama-rocmfp4, build-wsl) stays exactly as-is
  (readonly per AGENTS.md) - it holds the working fused MTP draft (36.3 t/s @113k on STRIX
  fp4 vs merged-build 20.6). Its 925e117-era common/speculative.cpp + DFlash2 fuse is the
  thing that makes the draft-passthrough beat upstream's separate-draft-context design.
- TODO: port the fused MTP draft onto the merged build's spec engine so current-llama.cpp
  fp4 decode keeps ~36 t/s. The fork concrete: `git show feat/benchmark:common/speculative.cpp`
  vs upstream's current `common/speculative*` (which already has draft-mtp but runs the head
  serially at full ctx -> 2.77 t/s @258k). Also look at the llama.cpp ecosystem patches for
  "chained MTP" (e.g. NVIDIA forum: 24GB RTX PRO 4000 Blackwell, NVFP4+MTP, 55.4 t/s @256k
  with chained-MTP/verify patches; base 21.2, embedded MTP 59.5, n_max=8 cliff).

### 12.2 Investigate @0xSero "67-91 tok/s on 1x Arc B70" claim
Claim (x.com @0xSero): "1x Intel Arc B70 - Qwen3.8-27B in 4 bits - 262k context - 66-91
tok/s single stream". Status: VERIFIED as real, with caveats - repo github.com/0xSero/qwen38-b70
is fully documented (llama.cpp SYCL, tensor-parallel, Q4_K_M + f16 KV, custom fork:
mndodd/llama.cpp @4302fb5 + b70-optimization-lab dp4a2 + Q4K increment patches):
- It is a ONE-COMMAND **2x Arc Pro B70 TP2** server (48 GB); the headline numbers are the
  MTP "easy/deterministic" (counting) tasks: 84.3 t/s (2x TP2), 61.7 (1x B70 @64k),
  97.2% draft acceptance. "Hard/random" tasks: 49 (2x), 28.2 (1x).
- Base decode (no draft): 51 t/s (2x TP2, up to 160k ctx; 30.8 @245k); 33 t/s (1x B70,
  max ctx 131072, 27.5 @128k). Prefill drops with ctx: 955 -> 275 t/s (2x).
- Why it "beats" our card on headline claims: MTP-on-easy-tasks cherry-pick + TP2 (2
  cards) + f16 KV + the GDN hybrid (only 16/64 layers full attention -> decode stays near
  flat with ctx). NOT a fair single-card-comparable win: 1x B70 base is 33 t/s (Q4_K_M)
  vs our STRIX fp4 base 21.4 and Q4_K_S ~14-16 t/s on 7900XTX - but B70 also isn't ~2x
  slower despite ~half the memory bandwidth on paper, so worth a like-for-like check.
- TODO: settle it properly - run OUR 7900XTX build with the SAME recipe as his 1x/2x B70
  numbers: Qwen3.8-27B-Q4_K_M.gguf (ggml-org), f16 KV (-ctk/-ctv f16), same ctx, on both
  the merged build and fork; and ideally TP2 vs TP2 is impossible (single 7900XTX), so
  compare 1x-B70 (33 t/s base) vs 1x-7900XTX Q4_K_M-f16. Also confirm Arc Pro B70 memory
  bandwidth (G31 24GB) to sanity-check his 33 t/s against BW arithmetic (~16.5GB weights /
  33 t/s = 546 GB/s effective; RX 7900XTX does 960 GB/s, STRIX fp4 measures only ~290 GB/s
  effective at 21.4 t/s / 13.8GB - fp4 decode is under-BW, worth a second look).

### 12.3 PLOTS: quality-vs-speed and speed-vs-context (DONE - `plots/`)

Requested 2026-09-29. The gate was "every model x config x context cell exists
for 131072 / 196608 / 262144, both speed and quality". **Gate cleared 2026-09-29**
by FO46 (49 runs) plus the shipping-KV quality run: 54/54 unique cells, 0 holes
(`bench/sweep/audit_matrix.py`), 8/8 shipping-KV quality rows. Figures are in
`plots/` (21 SVG), built by `bench/sweep/make_plots.py` from
`bench/sweep/plot_data.csv`.

**Fill basis: 6022, and that is a change from the original request.** The brief
asked for fill 129945 everywhere. FO44 tried the 129945 series and was killed
after 90 min: UD-Q4_K_S MTP at ctx 262144 runs at ~18-30 pp t/s and takes hours
(section 13.2, FO45diag). The ladder was therefore completed at **fill 6022**,
which every one of the 54 cells supports. The 129945 rows that do exist stay in
section 13.1 as the long-prompt regime; they are **not** mixed into the figures,
because mixing fills in one scatter is exactly what the original constraint
forbade. **18 cells still lack a 129945 measurement at these contexts** - if the
figures are to be redrawn on a 129945 basis, say so and it is a multi-hour run.

Deliverables, all in `plots/`:

1. **Quality vs speed, one plot per quality metric, per context.** 15 figures,
   named `quality_<metric>_vs_speed_c<ctx>.svg` for the 5 metrics x 131072 /
   196608 / 262144. `kld_99.9` is excluded: not a stable gate metric
   (section 14.1), so it would mislead.
2. **Aggregate quality vs speed.** 3 figures, `quality_AGGREGATE_vs_speed_c*.svg`.
   - **Aggregate definition (written down before plotting, as required):** the
     mean of `same_top_p` across `wiki_q` and `coding_q`, per model. It is the
     shipping gate the brief is framed on and it is already a single 0-100 scale,
     so no arbitrary weighting is invented.
   - `ppl_ratio` was considered and **rejected** as the aggregate: all four
     candidates sit in 1.003-1.035, so it carries almost no discriminative
     signal. The five other metrics each get their own figure instead.
   - The aggregate is a property of the model, not of the config, so base and
     MTP/DFlash2 points of one model share a y value. That is correct, not a bug:
     the draft model changes sampling, not the logits.
3. **Context vs speed.** `speed_output_vs_ctx.svg`, `speed_prefill_vs_ctx.svg`
   and the bonus `vram_vs_ctx.svg`: every config of every model as its own series
   across the full 8192 -> 262144 range, base dashed. VRAM is included because
   FO46 captured it for all 54 cells and it is the axis that decides which config
   actually fits in 24 GiB.

Conventions the figures follow:

- Every caption states fill, KV type, the speed commit and the quality commit.
  Quality is at `-c 4096` and speed is at 131k-262k: that mismatch is called out
  in every caption rather than left implicit.
- **Whiskers** on STRIX MTP points are the measured replicate spread between
  FO29 and FO46, up to 16.2% at 196608 (62.3 vs 54.5 t/s). Without them the
  STRIX/UD speed gap looks tighter than it is; with them it is clearly real.
- The UD 262144 MTP prefill collapse (189 pp t/s vs 715-721 at every other ctx)
  is annotated in red on the prefill figure, not silently plotted.
- `kld_99` and `mean_kld` use a log y-axis; the spread across models is more than
  an order of magnitude and a linear axis would collapse the interesting models
  onto the baseline.

Known limitation, stated on the figures themselves: at fill 6022 the KV cache
holds only 6022 tokens, so `speed_output_vs_ctx.svg` shows how the *allocation*
scales with ctx, **not** how a real long prompt decodes. Qwopus MTP at 262144 is
37.3 t/s here and 6.41 t/s at fill 129945 (section 13.1). The ctx-vs-speed figure
is about headroom; the fill-129945 rows in section 13.1 are about reality.

**The figures are machine-verified, not eyeballed.** `check_plots.py` asserts
each SVG is well-formed, that no two labels overlap, that no label runs off the
canvas, that every marker is inside the plot box, and that the marker count
equals the row count of `plot_data.csv`. `check_values.py` then does the
non-circular check: it fits a line from data value to pixel through the markers
(asserting <1.5 px residual, which catches a swapped axis) and then asserts every
printed **tick label** agrees with that same fit, which catches a log10 value
printed where a raw value belongs. All 21 figures pass both. Neither script can
tell whether the *choice* of what to plot was right - that is 12.3's job, and the
aggregate definition is written down above.

### 12.4 Model inventory: dedup + no-single-point-of-loss (DONE 2026-09-30)

Decided with the user: every GGUF does **not** need a copy on F:. The goal is
(1) no duplicate copies anywhere, and (2) no data silently lost while cleaning
up. Both were checked on 2026-09-30; the state is recorded here so no future
pass re-verifies it.

**Deduplication done (hash-verified, then deleted):**

1. Four temporary benchmark copies in WSL `~/models` (Qwopus3.8-27B-Flash-MTP-
   Q4_K_M, Qwen3.8-27B-UD-Q4_K_S_dynamicV3.0, mtp-Qwen3.8-27B-Q4_0_dynamicV3.0,
   Qwen3.8-27B-DFlash2-Q4_K_M) were byte-identical to their `F:\...\models\`
   originals (SHA-256 prefix `2b9a335d`/`75bc9c8a`/`50d9ce5a`/`18a380ef` match
   on both drives) and were deleted from WSL. ~32 GiB freed on the vhdx.
2. `mtp-Qwen3.8-27B-ROCMFP4-STRIX.gguf` in the repo root on F: was a **truncated
   partial download**: 1,135,906,816 bytes vs the real 1,983,232,288, and its
   SHA-256 matched the first 1,135,906,816 bytes of the WSL file, proving it was
   a cut-off prefix, not a re-quant or a variant. It was deleted. No run ever
   loaded it (a truncated GGUF fails mid-tensor in llama.cpp); every STRIX MTP
   run (FO29, FO46_st, the section 14.2.2 quality rows) used the 1.85 GiB WSL
   file.

**Single-copy inventory (data is NOT lost, but lives in exactly one place).**
These exist only in the WSL vhdx and nowhere on F::

| file | size | why it matters |
|---|---|---|
| `Qwen3.8-27B-Q8_0.gguf` | 27.0 GiB | the reference that defines **every** KLD number in section 14; not re-fetchable on a metered connection |
| `Qwen3.8-27B-ROCMFP4-STRIX.gguf` | 13.7 GiB | STRIX fp4 candidate |
| `mtp-Qwen3.8-27B-ROCMFP4-STRIX.gguf` | 1.8 GiB | STRIX MTP draft (the F: copy was the corrupt one above) |
| `Qwen3.8-27B-UD-IQ3_XXS.gguf` | 10.1 GiB | IQ3_XXS UD candidate |
| `qwen3.6-35b-a3b.gguf` / `qwen3.6-35b-a3b-llama4.gguf` | 22.2 GiB each | not used by this project's matrix |

`F:\...\models\` was NOT full enough to host these (25-26 GB free against ~52.8
GiB) and the decision was to not force it. **Risk, accepted:** losing the WSL
vhdx loses all six irrecoverably on a metered connection. If a second drive ever
appears, these six are the priority list.

## 13. MASTER AGGREGATION TABLE (all measured cases, one per row)

Every performance case produced so far, grouped by engine. Units: pp/tg in tokens/s.
`acc/len` = draft acceptance / mean draft length. Blank cell = not measured.
Engine key:
- `win-stock`  = Windows-native llama.cpp, ROCm 6.1 (section 1)
- `main (e6ab7c1)` = llama.cpp original HEAD / stock clone, WSL ROCm 7.x (sections 2/6/7/8/10)
- `fork (925e117)` = QingYi ROCmFP4 + DFlash2 fork (sections 2/6/8/10)
- `merged (master+fp4)` = upstream master (b11157/53ed051ce) + hand-ported ROCmFP4 layer, build-merge (sections 11 + new)
- `LLAMA_ATTN_ROT_DISABLE=1` = Hadamard KV rotation disabled (fork battery set this; merged default is ON)

Prompts: `agentic(113182)` = agentic90k_plain.txt; `network(258801)` = fill255k.txt; `repetitive(14409)` = JSON boilerplate mirror; `67k-xml` = XML-wrapped agentic; `own-bench` = Windows test prompts.

| # | engine | model | spec | ctx | fill(tok) | KV | pp | tg | acc/len | VRAM(GB) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | win-stock | Q4_K_S | base | 32768 | | | 820 | 33.2 | | | |
| 2 | win-stock | UD-Q4_K_S dynV3.0 | base | 32768 | | | 745 | 36.9 | | | |
| 3 | win-stock | Q4_K_S | MTP | 32768 | | | 720 | 37.2 | .55 | | |
| 4 | win-stock | UD-Q4_K_S dynV3.0 | MTP | 32768 | | | 751 | 40.4 | .43 | | |
| 5 | win-stock | Q4_K_S | DFlash2 | 32768 | | | 186 | 12.2 | .31 | | |
| 6 | win-stock | Q4_K_S | base | 65536 | | | 817 | 23.7 | | | |
| 7 | win-stock | UD-Q4_K_S dynV3.0 | base | 65536 | | | 842 | 34.1 | | | |
| 8 | win-stock | Q4_K_S | MTP | 65536 | | | 702 | 23.7 | .52 | | |
| 9 | win-stock | UD-Q4_K_S dynV3.0 | MTP | 65536 | | | 704 | 22.9 | .43 | | |
| 10 | win-stock | Q4_K_S | DFlash2 | 65536 | | | 187 | 10.9 | .30 | | |
| 11 | win-stock | Q4_K_S | base | 65536 | | | | 23.7 | | | VRAM-sat ~22.5/24GB |
| 12 | win-stock | UD-Q4_K_S dynV3.0 | MTP | 65536 | | | | 9-11 | | | VRAM-sat |
| 13 | win-stock | Q4_K_S | DFlash2 | 65536 | | | | 11-14 | | | VRAM-sat |
| 14 | win-stock | UD-Q3_K_XL dyn | base | 131072 | | f16 | | 20-26 | | | daily driver range |
| 15 | fork (925e117) | Q4_K_S | base | 131072 | | q4_0 | 360.2 | 30.4 | | | llama-bench pp131072/tg32 |
| 16 | fork (925e117) | Q4_K_S | base | 131072 | 67k-xml | | 509 | 23.9 | | | |
| 17 | fork (925e117) | Q4_K_S | stack ngram-k4v,dflash | 131072 | 67k-xml | | | 23.8 | .477/3.3 | | spec overhead = draft win |
| 18 | main (e6ab7c1) | Q4_K_S | base | 131072 | agentic(113182) | q4_0 | 579 | 20.8 | | | |
| 19 | main (e6ab7c1) | Q4_K_S | base | 131072 | agentic(113182) | f16 | 126 | 10.8 | | | spill to RAM |
| 20 | main (e6ab7c1) | Q4_K_S | MTP n5 | 131072 | agentic(113182) | q4_0 | 177 | 34.0 | .70/4.5 | | |
| 21 | main (e6ab7c1) | IQ3_XXS GSQ-RCO | base | 131072 | agentic(113182) | q4_0 | 577 | 22.8 | | | |
| 22 | main (e6ab7c1) | IQ3_XXS GSQ-RCO | DFlash2 n5 | 131072 | agentic(113182) | q4_0 | 538 | 31.6 | .46/3.3 | | |
| 23 | main (e6ab7c1) | IQ3_XXS GSQ-RCO | own-mtp | 131072 | agentic(113182) | | 139 | HANG | | | full-size MTP is not a drafter |
| 24 | main (e6ab7c1) | UD-Q3_K_XL dyn | base | 131072 | agentic(113182) | q4_0 | 586 | 21.9 | | | |
| 25 | main (e6ab7c1) | UD-Q3_K_XL dyn | DFlash2 n5 | 131072 | agentic(113182) | q4_0 | 539 | 37.9 | .60/4.0 | | |
| 26 | main (e6ab7c1) | UD-Q3_K_XL dyn | MTP n5 (dynV3.0) | 131072 | agentic(113182) | q4_0 | 555 | 9.38 | 0.00/1.0 | | wrong drafter family |
| 27 | main (e6ab7c1) | UD-Q4_K_S dyn | base | 131072 | agentic(113182) | q4_0 | 592 | 20.9 | | | |
| 28 | main (e6ab7c1) | UD-Q4_K_S dyn | MTP n5 | 131072 | agentic(113182) | q4_0 | 407 | 31.3 | .53/3.6 | | |
| 29 | main (e6ab7c1) | Q4_K_S | base | 131072 | repetitive(14409) | q4_0 | 826 | 30.7 | | | |
| 30 | main (e6ab7c1) | Q4_K_S | MTP n5 | 131072 | repetitive(14409) | q4_0 | 493 | 64.4 | .95/5.8 | | |
| 31 | main (e6ab7c1) | Q4_K_S | base | 262144 | network(258801) | q4_0 | 392.2 | 13.18 | | 22.31 | |
| 32 | main (e6ab7c1) | UD-Q3_K_XL dyn | base | 262144 | network(258801) | q4_0 | 395.9 | 13.26 | | 19.79 | |
| 33 | main (e6ab7c1) | UD-Q4_K_S dyn | base | 262144 | network(258801) | q4_0 | 397.6 | 13.22 | | 21.85 | |
| 34 | main (e6ab7c1) | IQ3_XXS GSQ-RCO | base | 262144 | network(258801) | q4_0 | 391.1 | 13.68 | | 17.4 | |
| 35 | main (e6ab7c1) | IQ3_XXS GSQ-RCO | DFlash2 n5 | 262144 | network(258801) | q4_0 | 370.0 | 21.82 | | 19.61 | |
| 36 | main (e6ab7c1) | UD-Q3_K_XL dyn | DFlash2 n5 | 262144 | network(258801) | q4_0 | 367.7 | 20.07 | 23/23 len3.88 | 22.6 | HEAD vs fork A/B |
| 37 | main (e6ab7c1) | Q4_K_S | stack ngram-k4v,dflash | 131072 | agentic(113182) | q4_0 | 580.7 | 19.68 | 14/48 len1.82 | 22.1 | HEAD vs fork A/B |
| 38 | fork (925e117) | UD-Q3_K_XL dyn | DFlash2 n5 | 131072 | agentic(113182) | q4_0 | 351 | 32.2 | .65/4.0 | 19.3 | |
| 39 | fork (925e117) | STRIX fp4 | base | 131072 | agentic(113182) | q4_0 | 358 | 21.4 | | 17.5 | ROT off |
| 40 | fork (925e117) | STRIX fp4 | base | 131072 | agentic(113182) | f16 | 408 | 29.5 | | | |
| 41 | fork (925e117) | STRIX fp4 | stack ngram+dflash | 131072 | agentic(113182) | q4_0 | 343 | 27.1 | .48/3.8 | 19.5 | ROT off |
| 42 | fork (925e117) | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 339 | **36.3** | .63/4.1 | 19.8 | ROT off; the "fused MTP" case |
| 43 | fork (925e117) | STRIX fp4 | MTP n5 | 262144 | 113182 fill | q4_0 | 338 | 36.6 | .63/4.1 | 19.8 | ROT off |
| 44 | fork (925e117) | STRIX fp4 | MTP n5 | 262144 | ~5 tok fill | q4_0 | | 60.4 | | | |
| 45 | fork (925e117) | STRIX fp4 | MTP n5 | 262144 | ~4500 tok fill | q4_0 | | 43.7 | | | |
| 46 | fork (925e117) | STRIX fp4 | base | 131072 | repetitive(14409) | q4_0 | 679 | 32.7 | | | |
| 47 | fork (925e117) | STRIX fp4 | ngram-only | 131072 | repetitive(14409) | q4_0 | 674 | 80.4 | .96/39.9 | | |
| 48 | fork (925e117) | STRIX fp4 | ngram-only | 262144 | repetitive(14409) | q4_0 | 678 | 78.8 | .96/39.9 | | |
| 49 | fork (925e117) | STRIX fp4 | stack ngram+dflash | 131072 | repetitive(14409) | q4_0 | 574 | 111.9 | .97/17.0 | | |
| 50 | fork (925e117) | STRIX fp4 | stack ngram+dflash | 262144 | repetitive(14409) | q4_0 | 122 | 35.0 | .97/17.0 | 21.5 | prefill collapsed (fork bug) |
| 51 | fork (925e117) | STRIX fp4 | MTP n5 (n=60) | 262144 | repetitive(14409) | q4_0 | | 68.3 | | | |
| 52 | fork (925e117) | STRIX fp4 | base | 262144 | network(258801) | q4_0 | 209.7 | 13.79 | | 23.37 | |
| 53 | fork (925e117) | STRIX fp4 | MTP n5 | 262144 | network(258801) | q4_0 | 198.4 | 24.80 | | 23.32 | |
| 54 | fork (925e117) | STRIX fp4 | stack ngram+dflash | 262144 | 129k aborted | q4_0 | ~40 | | | 21.5 | fork prefill collapse |
| 55 | fork (925e117) | UD-Q3_K_XL dyn | DFlash2 n5 | 262144 | network(258801) | q4_0 | 207.5 | 20.92 | 25/25 len6.0 | 21.9 | |
| 56 | fork (925e117) | Q4_K_S | stack ngram-k4v,dflash | 131072 | agentic(113182) | q4_0 | 396.5 | 16.62 | 14/48 len1.82 | 22.1 | |
| 57 | merged (master+fp4) | STRIX fp4 | base | 131072 | agentic(113182) | q4_0 | 512.1 | 18.57 | | 17.92 | |
| 58 | merged (master+fp4) | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 487.5 | 20.55 | .34/2.71 | 20.34 | |
| 59 | merged (master+fp4) | STRIX fp4 | base | 262144 | network(258801) | q4_0 | 357.9 | 12.82 | | 20.8 | |
| 60 | merged (master+fp4) | STRIX fp4 | MTP n5 | 262144 | network(258801) | q4_0 | 240.2 | **2.77** | .94/4.75 | 22.85 | MTP decode collapse |
| 61 | merged (master+fp4) | STRIX fp4 | stack ngram-k4v,dflash | 131072 | agentic(113182) | q4_0 | 474.3 | 16.32 | .39/2.71 | 20.44 | |
| 62 | merged (master+fp4) | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 523.4 | 27.13 | .52/3.60 | | ROT disabled (LLAMA_ATTN_ROT_DISABLE=1) |
| 63 | merged (master+fp4) | STRIX fp4 | MTP n5 | 262144 | network(258801) | q4_0 | 243.8 | 2.86 | .94/4.75 | | ROT disabled; still collapses |
| 64 | vendor (unsloth) | UD-Q4_K_XL dynV3.0 | base (300 docs) | - | - | - | - | - | - | 15.64 | KLD vs BF16: mean 0.0237 (base 0.0249) |
| 65 | vendor (unsloth) | UD-Q3_K_XL dynV3.0 | base (300 docs) | - | - | - | - | - | - | 12.76 | KLD vs BF16: mean 0.0806 (base 0.0878) |
| 66 | vendor (unsloth) | UD-Q2_K_XL dynV3.0 | base (300 docs) | - | - | - | - | - | - | 9.95 | KLD vs BF16: mean 0.2209 (base 0.2297) |
| 67 | vendor (unsloth) | UD-IQ2_M dynV3.0 | base (300 docs) | - | - | - | - | - | - | 8.96 | KLD vs BF16: mean 0.2582 (base 0.2655) |
| 68 | vendor (unsloth) | UD-IQ2_XXS dynV3.0 | base (300 docs) | - | - | - | - | - | - | 7.31 | KLD vs BF16: mean 0.5210 (base 0.5358) |
| 69 | vendor (unsloth) | UD-IQ1_M dynV3.0 | base (300 docs) | - | - | - | - | - | - | 6.51 | KLD vs BF16: mean 0.8000 (base 0.8323) |
| 70 | vendor (unsloth) | UD-IQ1_S dynV3.0 | base (300 docs) | - | - | - | - | - | - | 6.06 | KLD vs BF16: mean 0.9729 (base 1.0357) |
| 71 | vendor (ISTA-DASLab) | GSQ-RCO IQ3_XXS | base (5-task ZS) | - | - | - | - | - | - | 10.1 | zs-avg 74.81 (recov 100.6%), AIME25 100.0 = base, wiki PPL 7.20 |
| 72 | vendor (ISTA-DASLab) | GSQ-RCO IQ3_S | base (5-task ZS) | - | - | - | - | - | - | 11.8 | zs-avg 74.47 (recov 100.2%), AIME25 100.0, wiki PPL 7.07 |
| 73 | vendor (ISTA-DASLab) | UD-IQ3_S (their comp row) | base (5-task ZS) | - | - | - | - | - | - | 12.0 | zs-avg 75.49 (recov 101.5%), AIME25 96.67 |
| 74 | vendor (ISTA-DASLab) | BF16 reference | base (5-task ZS) | - | - | - | - | - | - | 53.8 | zs-avg 74.34 (recov 100%), AIME25 100.0, GPQA-D 89.90, LCB 85.71 |

Rows 64-74 = vendor-PUBLISHED quality numbers (KLD vs BF16 / task recovery), NOT tok/s
measured on our GPU. They are here for the one-table view; see section 9 and
`llama-rocmfp4-merge/results/benchmarks/` for sources/images.

Reading the table:
- Prefill on merged always beats both others (upstream scheduler/attention): rows 57-63 vs 39-42/44-45 and 18-28.
- Daily-driver decode (113k fill): fork STRIX MTP 36.3 (#42) is still the top decode; merged fp4 base (#57) is fastest prefill, and ROT-off lifts merged MTP #58 -> #62 (20.6 -> 27.1 t/s, acc .34->.52).
- Extreme ctx MTP (258k fill): merged is unusable (2.8 t/s, rows 60/63); the fork's fused MTP keeps 24.8 (#53). Root-cause work (fused-MTP backport) tracked in section 12.1; the Hadamard-rotation flag did NOT fix the 258k collapse.
- QUALITY, not tok/s: the KLD/same-top/PPL quality matrix is
  `llama-rocmfp4-merge/results/e0/2026-09-25/PHASE_Q_QUALITY_MATRIX.md` (all rows measured
  on this GPU). Vendor-published quality numbers + full-res plots:
  `llama-rocmfp4-merge/results/benchmarks/`.


### 13.1 FO24-FO29 additions: the hole-free ladder (36 rows)

Added 2026-09-29. Every number below is scraped straight out of
`/home/thegiallo/bench/sweep/results.txt` by
`/home/thegiallo/bench/sweep/mk_master_rows.py`; none is hand-typed. VRAM is GiB
(weights + draft weights + main KV + draft KV + recurrent + compute); blank means
the run predates the corrected VRAM parser, which measured speed only.

Two distinct fills coexist in this table and must not be mixed when plotting:

| fill | what it is | which runs |
|---|---|---|
| 6022 | ~6k token prompt | the 4x5 ctx ladder (FO28/FO29), small-ctx STRIX |
| 113182 / 129945 / 149949 / 199626 | long prompts | the fill-129945 cliff bisect (FO24/FO26) and the 262144 rows |

`FO25_*` and `FO27_*` are deliberately NOT listed: they are the same
configurations re-measured after the VRAM-parser fix, kept as replicates rather
than as separate cases. Their spread is the honest noise estimate for the ladder,
and it is not small:

| ctx | config | replicate A (tg) | replicate B (tg) | spread |
|---|---|---|---|---|
| 196608 | Qwopus MTP | 44.21 (FO25) | 44.58 (FO28) | 0.8% |
| 196608 | STRIX MTP | 54.51 (FO25) | 63.35 (FO28) | 16.2% |
| 196608 | UD-Q4_K_S MTP | 51.69 (FO25) | 51.93 (FO28) | 0.5% |
| 196608 | IQ3_XXS DFlash2 | 32.69 (FO25) | 30.99 (FO29) | 5.2% |
| 131072 | STRIX MTP | 58.09 (FO27) | 60.05 (FO28) | 3.2% |

**STRIX's 16% replicate spread at 196608 is larger than most of the effects
claimed in this document.** Treat any single STRIX tg number as +/-8%.

| FO29_iq_c8192 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 8192 | 6022 | q4_0 | 635.9 | 33.04 | | 12.31 |  |
| FO28_qw_c8192 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 8192 | 6022 | q4_0 | 612.1 | 41.56 | | 16.73 |  |
| FO28_st_c8192 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 8192 | 6022 | q4_0 | 633.6 | 61.10 | | 16.66 |  |
| FO28_u4_c8192 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 8192 | 6022 | q4_0 | 572.1 | 52.11 | | 16.64 |  |
| FO29_iq_c32768 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 32768 | 6022 | q4_0 | 650.2 | 33.58 | | 12.76 |  |
| FO28_qw_c32768 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 32768 | 6022 | q4_0 | 683.5 | 43.09 | | 17.27 |  |
| FO28_st_c32768 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 32768 | 6022 | q4_0 | 641.1 | 63.98 | | 17.20 |  |
| FO28_u4_c32768 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 32768 | 6022 | q4_0 | 709.5 | 54.81 | | 17.18 |  |
| FO29_iq_c65536 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 65536 | 6022 | q4_0 | 640.7 | 32.64 | | 13.35 |  |
| FO28_qw_c65536 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 65536 | 6022 | q4_0 | 678.8 | 41.98 | | 17.98 |  |
| FO28_st_c65536 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 65536 | 6022 | q4_0 | 629.2 | 59.52 | | 17.91 |  |
| FO28_u4_c65536 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 65536 | 6022 | q4_0 | 694.6 | 53.44 | | 17.89 |  |
| FO26_iq_c131072_f130 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 131072 | 129945 | q4_0 | 480.4 | 28.09 | |  |  |
| FO29_iq_c131072 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 131072 | 6022 | q4_0 | 656.5 | 32.91 | | 14.54 |  |
| FO24_qw_113k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 131072 | 113182 | q4_0 | 523.9 | 33.18 | |  | draftKV q4_0 at 113k |
| FO24_qw_f130_c131072_mtp | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 131072 | 129945 | q4_0 | 494.9 | 27.64 | |  |  |
| FO28_qw_c131072 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 131072 | 6022 | q4_0 | 679.4 | 42.17 | | 19.42 |  |
| FO26_st_c131072_f130 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 131072 | 129945 | q4_0 | 449.8 | 27.94 | |  |  |
| FO28_st_c131072 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 131072 | 6022 | q4_0 | 636.3 | 60.05 | | 19.35 |  |
| FO26_u4_c131072_f130 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 131072 | 129945 | q4_0 | 498.8 | 23.60 | |  |  |
| FO28_u4_c131072 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 131072 | 6022 | q4_0 | 672.2 | 51.29 | | 19.33 |  |
| FO26_iq_c196608_f130 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 196608 | 129945 | q4_0 | 482.9 | 27.49 | |  |  |
| FO29_iq_c196608 | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 196608 | 6022 | q4_0 | 643.9 | 30.99 | | 15.73 |  |
| FO24_qw_f130_c196608_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 196608 | 129945 | q4_0 | 499.3 | 26.82 | |  | draftKV q4_0 at 196608 |
| FO24_qw_f130_c196608_mtp | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 196608 | 129945 | q4_0 | 494.1 | 27.81 | |  |  |
| FO28_qw_c196608 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 196608 | 6022 | q4_0 | 704.5 | 44.58 | | 20.86 |  |
| FO26_st_c196608_f130 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 196608 | 129945 | q4_0 | 448.4 | 27.96 | |  |  |
| FO28_st_c196608 | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 196608 | 6022 | q4_0 | 641.3 | 63.35 | | 20.79 |  |
| FO26_u4_c196608_f130 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 196608 | 129945 | q4_0 | 504.4 | 23.73 | |  |  |
| FO28_u4_c196608 | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 196608 | 6022 | q4_0 | 677.4 | 51.93 | | 20.77 |  |
| FO23_qw_262k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 262144 | 199626 | q4_0 | 214.2 | 8.53 | |  | 262k cliff, q4_0 draft KV (+146% tg) |
| FO7_262k_off | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 262144 | 199626 | q4_0 | 497.2 | 17.00 | |  | 262k cliff: no-spec control |
| FO7_262k_on | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 262144 | 199626 | q4_0 | 429.8 | 3.47 | |  | 262k cliff: MTP, f16 draft KV |
| FO9_262k_f130_on | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 262144 | 129945 | q4_0 | 517.5 | 6.41 | |  | 262k cliff, fixed fill 129945 |
| FO9_262k_f150_on | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 262144 | 149949 | q4_0 | 488.5 | 7.81 | |  | 262k cliff, fill 149949 |
| FO26_u4_262k_off | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 262144 | 199626 | q4_0 | 430.0 | 15.37 | |  | last unpaired 256k control |

### 13.2 FO45-FO46: the hole-free fill-6022 matrix (49 runs) + the 262144 prefill cliff

Added 2026-09-29. `driver46.sh` re-ran the full grid so that **every** cell
carries VRAM, not just speed. 37 minutes, 49/49 succeeded, `ALL_DONE46` in
`results.txt`. Combined with FO28/FO29 that gives **54 unique cells, 0 holes**
(`bench/sweep/audit_matrix.py`).

| tag | build | model | config | ctx | fill | KV | pp t/s | tg t/s | VRAM GiB | note |
|---|---|---|---|---|---|---|---:|---:|---:|---|
| FO45diag_u4_c262144_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 262144 | 6022 | q4_0 | 188.2 | 50.78 |  | diagnostic A/B, see 13.2 |
| FO45diag_u4_c262144_6k_uni | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 262144 | 6022 | q4_0 | 188.2 | 50.46 |  | diagnostic A/B, see 13.2 |
| FO45diag_u4_c262144_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 262144 | 6022 | q4_0 | 794.5 | 29.82 |  | diagnostic A/B, see 13.2 |
| FO46_iq_c8192_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 8192 | 6022 | q4_0 | 643.9 | 32.09 | 12.31 |  |
| FO46_iq_c32768_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 32768 | 6022 | q4_0 | 633.9 | 33.52 | 12.76 |  |
| FO46_iq_c65536_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 65536 | 6022 | q4_0 | 637.7 | 32.07 | 13.35 |  |
| FO46_iq_c131072_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 131072 | 6022 | q4_0 | 659.0 | 32.72 | 14.54 |  |
| FO46_iq_c196608_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 196608 | 6022 | q4_0 | 645.1 | 32.72 | 15.73 |  |
| FO46_iq_c262144_6k | merged (a3d2c2b52) | IQ3_XXS UD | DFlash2 n5 | 262144 | 6022 | q4_0 | 637.4 | 32.77 | 16.92 |  |
| FO46_iq_c8192_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 8192 | 6022 | q4_0 | 781.9 | 34.74 | 10.48 |  |
| FO46_iq_c32768_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 32768 | 6022 | q4_0 | 772.6 | 34.33 | 10.93 |  |
| FO46_iq_c65536_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 65536 | 6022 | q4_0 | 775.3 | 33.91 | 11.52 |  |
| FO46_iq_c131072_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 131072 | 6022 | q4_0 | 766.3 | 33.82 | 12.71 |  |
| FO46_iq_c196608_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 196608 | 6022 | q4_0 | 776.5 | 33.89 | 13.90 |  |
| FO46_iq_c262144_6k_base | merged (a3d2c2b52) | IQ3_XXS UD | base | 262144 | 6022 | q4_0 | 783.8 | 33.89 | 15.09 |  |
| FO46_qw_c262144_6k | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5 | 262144 | 6022 | q4_0 | 416.5 | 37.33 | 22.30 |  |
| FO46_qw_c8192_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 8192 | 6022 | q4_0 | 633.5 | 42.63 | 16.70 |  |
| FO46_qw_c32768_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 32768 | 6022 | q4_0 | 702.0 | 45.29 | 17.18 |  |
| FO46_qw_c65536_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 65536 | 6022 | q4_0 | 700.9 | 42.52 | 17.80 |  |
| FO46_qw_c131072_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 131072 | 6022 | q4_0 | 670.1 | 43.20 | 19.06 |  |
| FO46_qw_c196608_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 196608 | 6022 | q4_0 | 657.0 | 44.46 | 20.32 |  |
| FO46_qw_c262144_6k_dkq4 | merged (a3d2c2b52) | Qwopus Q4_K_M | MTP n5, draftKV q4_0 | 262144 | 6022 | q4_0 | 581.1 | 39.46 | 21.58 |  |
| FO46_qw_c8192_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 8192 | 6022 | q4_0 | 787.5 | 28.69 | 15.96 |  |
| FO46_qw_c32768_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 32768 | 6022 | q4_0 | 784.4 | 28.19 | 16.41 |  |
| FO46_qw_c65536_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 65536 | 6022 | q4_0 | 775.4 | 28.35 | 17.00 |  |
| FO46_qw_c131072_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 131072 | 6022 | q4_0 | 777.0 | 28.65 | 18.19 |  |
| FO46_qw_c196608_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 196608 | 6022 | q4_0 | 780.5 | 27.81 | 19.38 |  |
| FO46_qw_c262144_6k_base | merged (a3d2c2b52) | Qwopus Q4_K_M | base | 262144 | 6022 | q4_0 | 767.2 | 28.42 | 20.57 |  |
| FO46_st_c8192_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 8192 | 6022 | q4_0 | 628.8 | 60.23 | 16.66 |  |
| FO46_st_c32768_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 32768 | 6022 | q4_0 | 638.8 | 63.97 | 17.20 |  |
| FO46_st_c65536_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 65536 | 6022 | q4_0 | 637.8 | 61.78 | 17.91 |  |
| FO46_st_c131072_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 131072 | 6022 | q4_0 | 632.0 | 60.86 | 19.35 |  |
| FO46_st_c196608_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 196608 | 6022 | q4_0 | 633.2 | 60.31 | 20.79 |  |
| FO46_st_c262144_6k | merged (a3d2c2b52) | STRIX fp4 | MTP n5 | 262144 | 6022 | q4_0 | 629.8 | 62.28 | 22.23 | separate draft GGUF, NOT affected (control) |
| FO46_st_c8192_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 8192 | 6022 | q4_0 | 650.6 | 31.16 | 14.05 |  |
| FO46_st_c32768_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 32768 | 6022 | q4_0 | 683.0 | 31.15 | 14.50 |  |
| FO46_st_c65536_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 65536 | 6022 | q4_0 | 708.1 | 31.44 | 15.09 |  |
| FO46_st_c131072_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 131072 | 6022 | q4_0 | 692.2 | 30.18 | 16.28 |  |
| FO46_st_c196608_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 196608 | 6022 | q4_0 | 692.0 | 30.76 | 17.47 |  |
| FO46_st_c262144_6k_base | merged (a3d2c2b52) | STRIX fp4 | base | 262144 | 6022 | q4_0 | 705.4 | 31.33 | 18.66 |  |
| FO46_u4_c8192_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 8192 | 6022 | q4_0 | 721.2 | 53.33 | 16.64 |  |
| FO46_u4_c32768_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 32768 | 6022 | q4_0 | 718.8 | 54.03 | 17.18 |  |
| FO46_u4_c65536_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 65536 | 6022 | q4_0 | 717.4 | 53.65 | 17.89 |  |
| FO46_u4_c131072_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 131072 | 6022 | q4_0 | 715.5 | 51.23 | 19.33 |  |
| FO46_u4_c196608_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 196608 | 6022 | q4_0 | 715.6 | 53.03 | 20.77 |  |
| FO46_u4_c262144_6k | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | MTP n5 | 262144 | 6022 | q4_0 | 189.4 | 49.47 | 22.21 | **prefill cliff** 189 pp t/s, see 13.2 |
| FO46_u4_c8192_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 8192 | 6022 | q4_0 | 644.2 | 29.37 | 14.60 |  |
| FO46_u4_c32768_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 32768 | 6022 | q4_0 | 714.7 | 30.17 | 15.05 |  |
| FO46_u4_c65536_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 65536 | 6022 | q4_0 | 800.1 | 30.09 | 15.64 |  |
| FO46_u4_c131072_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 131072 | 6022 | q4_0 | 820.7 | 30.51 | 16.83 |  |
| FO46_u4_c196608_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 196608 | 6022 | q4_0 | 798.3 | 29.27 | 18.02 |  |
| FO46_u4_c262144_6k_base | merged (a3d2c2b52) | UD-Q4_K_S dynV3.0 | base | 262144 | 6022 | q4_0 | 801.1 | 30.19 | 19.21 |  |

**The 129945 attempt failed and that is why the ladder is at 6022.** FO44 set out
to fill the 18 remaining cells at fill 129945. The first run,
`FO44_u4_c262144_f130` (UD-Q4_K_S MTP), made no progress in 90 minutes and was
killed; `results.txt` holds only a `Traceback` for it, so it contributes no
number. FO45diag measured the same cell at fill 6022 to isolate the cause:

| run | config | pp t/s | tg t/s | reading |
|---|---|---:|---:|---|
| `FO45diag_u4_c262144_6k` | MTP + **separate** draft GGUF | 188.2 | 50.8 | the collapsing case |
| `FO45diag_u4_c262144_6k_uni` | MTP + unified (in-model head) | 188.2 | 50.5 | also collapses |
| `FO45diag_u4_c262144_6k_base` | base, no draft at all | 794.5 | 29.8 | **unaffected** |

That rules out the separate draft GGUF as the cause - the unified-head variant
collapses identically - and rules out speculation generally, since base at the
same context is fine. Base prefill sits at 794 pp t/s and MTP prefill at 188 pp
t/s, both at fill 6022 and ctx 262144. **The 2026-09-29 claim that this is "a
cliff at exactly 2^18" is RETRACTED - see the FO50-54 rebuttal below.** Decode is
unaffected (50.8 t/s, the fastest in the matrix at that context). Same-boot
controls show the effect is UD-specific: STRIX (630 pp t/s) and IQ (637 pp t/s)
at 262144 were fine while UD measured 189 pp t/s.

**The pre-existing "Qwopus 262144 cliff" is a different, unreplicated effect.**
FO46 Qwopus MTP at 262144 runs at 416 pp / 37.3 t/s at fill 6022, versus 6.41
t/s at fill 129945 (FO9). So Qwopus's collapse needs *both* the large context
*and* the long fill; UD's appeared to need only the large context.

#### 13.2.1 FO47-FO54: the "2^18 cliff" does not reproduce; it is a session/GPU-state effect

`driver47/48/49` (2026-09-30 03:0x-03:3x) first bisected the collapse and found
a sharp boundary. `driver50/51/52/53/54` (same day 04:1x+, same binary
`a3d2c2b52`, same models, same fill 6022) reran those exact cells and the
collapse was **gone** - and then the boundary reappeared at a *different* ctx.

All rows below are UD-Q4_K_S dynV3.0 + MTP n5, `-b 2048 -ub 512`, KV q4_0
unified-off, fill 6022, `n_ctx_slot` capped at 262144 = n_ctx_train (KV cells
track the raw `-c`):

| run (day) | ctx cells | pp t/s | reading |
|---|---:|---:|---|
| FO47_u4_c245760 (03:07) | 245760 | 757.4 | fast tier |
| FO47_u4_c260096 (03:09) | 260096 | 602.8 | fast tier |
| FO47_u4_c261632 (03:10) | 261632 | 206.1 | **collapsed** |
| FO47_u4_c262144 (03:11) | 262144 | 208.9 | **collapsed** |
| FO47_u4_c266240 (03:13) | 266240 | 431.8 | intermediate (anomaly) |
| FO47_u4_c274432 (03:14) | 274432 | 203.8 | **collapsed** |
| FO47_u4_c262144_np2 (03:19) | 2x131072 | 754.3 | fast tier (per-slot split) |
| FO48_u4_c260352 (03:20) | 260352 | 577.7 | fast tier |
| FO48_u4_c260608 (03:21) | 260608 | 205.9 | **collapsed** |
| FO49_c260480/544/576 (03:31) | 260608 | 206.4-208.1 | **collapsed** (3x, rounds to 260608) |
| FO50_c260608 (04:12) | 260608 | **808.7** | NOT collapsed, same config as above |
| FO50_c266240 (04:13) | 266240 | 738.4 | not collapsed |
| FO50_c274432 (04:15) | 274432 | 238.2 | collapsed |
| FO51_B_cold_260608 (04:20) | 260608 | 807.8 | not collapsed (cold) |
| FO51_A_warm_260352 (04:21) | 260352 | 807.0 | not collapsed (warm) |
| FO51_B_warm_260608 (04:22) | 260608 | 809.3 | not collapsed (warm) |
| FO51_A_hot_260352 (04:22) | 260352 | 806.9 | not collapsed (4th run) |
| FO52_u4_c260608 | 260608 | 801.6 | not collapsed |
| FO52_u4_c262144 | 262144 | 804.5 | not collapsed (was 208 at 03:11!) |
| FO52_u4_c266240 | 266240 | 735.4 | not collapsed |
| FO52_u4_c274432 | 274432 | 236.8 | collapsed |
| FO53_u4_c262144 | 262144 | 744.5 | not collapsed |
| FO53_u4_c265216 | 265216 | 682.6 | not collapsed (fast-ish) |
| FO53_u4_c268288 | 268288 | 240.2 | collapsed |
| FO53_u4_c274432 | 274432 | 220.3 | collapsed |
| FO53_u4_c274432_base | 274432 | **903.8** | base unaffected even here |
| FO54_u4_c265216 | 265216 | 238.9 | **collapsed** (was 682 at 04:4x!) |
| FO54_u4_c266240 | 266240 | 237.9 | collapsed |
| FO54_u4_c267264 | 267264 | 228.2 | collapsed |
| FO54_u4_c268288 | 268288 | 521.8 | *not* collapsed (bigger ctx, faster) |
| FO54_u4_c265216b | 265216 | 238.4 | collapsed, reproducible |
| FO54_u4_c268288b | 268288 | 526.6 | *not* collapsed, reproducible |
| FO54_u4_c268288_base | 268288 | **907.2** | base unaffected |

Verdict, with the boundary now observed at three different cells
(260608 on 09-30 03:2x, 268288 on 04:5x, 265216 on 04:6x) and FO54 showing a
*higher* ctx (268288) running 2.2x *faster* than a lower one (265216) in the
same sequence:

- **There is no fixed code threshold.** A boundary cannot move by thousands of
  tokens across sessions and still be a boundary. The collapses in FO46-49 were
  real measurements but of an *environmental* condition (GPU-PV power/thermal
  state - `amd-smi`/`rocm-smi`/hwmon are all unavailable on this GPU-PV path,
  so the state itself cannot be read), not a deterministic limit.
- **UD-specific within a bad session:** FO46's same-boot ladder had UD=189 but
  STRIX=630, IQ=637 at the same 262144. A global GPU degradation would scale all
  models; it did not. UD (Q4_K_S main + Q4_0 draft) is the sensitive one.
- **Base never collapses** (794-907 at every ctx tested, including 268288/274432).
- **Decode is unaffected in every collapsed run** (tg 49-63 across the whole
  table), so this is purely a prefill-path phenomenon.
- **Per-slot split is not a fix:** FO47's 2x131072 np2 was fast in the same boot
  that collapsed np1-262144 - but that is the *same* environmental condition
  avoiding the sensitive point, consistent with a per-slot resource threshold
  that the environment moves.

**Practical consequence:** UD MTP prefill at ctx >= ~260k is NOT benchmarkable
on this card as a single number - the same cells measure 206 to 808 pp t/s
depending on boot/GPU state. 13.2's matrix rows for UD at 262144 (189 pp t/s)
and the plot caption asserting "a cliff at exactly 2^18" must be read as
measurements taken in a degraded session, not as a property of the software.
Any shipping decision that depends on UD MTP throughput at >=260k should be
re-measured fresh in one sitting before trusting it.


### 13.3 Re-runs on the table's own terms (2026-09-30 / 2026-10-01)

Numbered N1-N5 so the row numbers above stay stable. **Conditions match the table rows they are compared with:**
the same prompts (`agentic(113182)` at ctx 131072, `network(258801)` = `fill255k.txt` at ctx 262144), **128 generated tokens**
(`probe.py`), greedy, q4_0 KV, default draft KV, `-b 2048 -ub 512 --no-kv-unified -np 1`, `LLAMA_ATTN_ROT_DISABLE=1`.

**None of N1-N5 uses the graph-key optimization** (`8a4f661e9`, "one CUDA graph per small batch size"): every run set
`GGML_CUDA_GRAPH_KEY_BY_BATCH=0`. That fix only changes speculative verify passes whose batch size varies (DFlash with a
draft `p-min`); it measured no effect on MTP (STRIX 131072: 51.87 off vs 52.08 on; Unsloth Q4 131072: 38.70 vs 38.73,
512-token runs) and none on base decode. Its measured gains (Q3_K_XL + DFlash2, 512 tokens, **not** table conditions:
+12% at 6k fill, +17% at 27.5k fill) are in `bench/decode_investigation/README.md` part 2 and are deliberately not in this
table, because the comparable DFlash config at 131072 / 113k fill hangs (N5).

Engine = `merged+exp`: llama.cpp tree 4 (`~/llama-rocmfp4-merge`), branch `exp/decode-bw` @ `8a4f661e9`, run with
`GGML_CUDA_GRAPH_KEY_BY_BATCH=0` so the behaviour equals `a3d2c2b52` (the diagnostics are env-gated). It is a different build
from the one that produced rows 57-63 (same source family, newer speculative-decoding code), so differences are not only
noise; acceptance differs between the rows.

| # | engine | model | spec | ctx | fill(tok) | KV | pp | tg | acc/len | VRAM(GB) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N1 | merged+exp [graph-key fix OFF] | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 547.5 | **44.33** | .685/4.38 | | compare #62: 523.4 / 27.13 / .52/3.60 |
| N2 | merged+exp [graph-key fix OFF] | STRIX fp4 | MTP n5 | 262144 | network(258801) | q4_0 | 323.1 | **3.32** | .99/5.77 | | same collapse as #60/#63 (240 / 2.77-2.86 / .94/4.75) |
| N3 | merged+exp [graph-key fix OFF] | UD-Q4_K_S dyn | MTP n5 | 131072 | agentic(113182) | q4_0 | 608.7 | **32.38** | .49/3.43 | | compare #28 (main e6ab7c1): 407 / 31.3 / .53/3.6 |
| N4 | merged+exp [graph-key fix OFF] | UD-Q4_K_S dyn | MTP n5 | 262144 | network(258801) | q4_0 | - | - | - | | **not obtained**: 9 launches in one session were all in the slow prefill state (184-203 t/s cumulative at 4096 tokens); one run was stopped at 86k/258,801 tokens at 131 t/s average and falling. See Part 4 of `bench/decode_investigation/README.md` |
| N5 | merged+exp [graph-key fix OFF; also hangs with it ON] | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic(113182) | q4_0 | - | - | - | | **hangs** after prefill (100% at ~187 s, 602 t/s), no output for > 6 min, needs `kill -9`; reproduced on two binaries; the first occurrence coincided with an AMD driver crash. Same config works at 6k and 27.5k fill  **Update: worked around with pre-warm requests, rows N28-N31 (README Part 15).** |

| N6 | merged+exp+footprint flags (`opt/vram-footprint` @ `a35a77184`, graph-key fix OFF) | UD-Q4_K_S dyn | MTP n3 | 262144 | network(258801) | q4_0 (draft KV q4_0) | 389.8 | **32.88** | 1.00/3.97 | ~22.0 (adapter 22.8 with ~0.7 GiB used by other apps) | flags: `--ctx-reserve 32768 --spec-draft-ctx 32768 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 3`. Whole 258,801-token prefill took 664 s with no cliff (instantaneous 881 t/s at 20k, 462 at 102k, 259 at 250k). Needs the other apps' GPU use low: with ~1.1 GiB used the same config hit the cliff at ~140k tokens and finished at 124 t/s / 2.14 t/s decode (n_max 5); with ~1.6 GiB it collapsed at ~37k. The generated continuation is repetitive (acceptance 1.00), so decode is optimistic for ordinary text. Compare #60/#63 (merged STRIX MTP, default flags): 2.77-2.86 |
| N7 | merged+exp+footprint flags (`opt/vram-footprint` @ `a35a77184`, graph-key fix OFF) | STRIX fp4 | MTP n3 | 262144 | network(258801) | q4_0 (draft KV q4_0) | 365.6 | **33.77** | 1.00/3.97 | ~22.0 (other apps ~0.8 GiB) | same flags as N6 (`--ctx-reserve 32768 --spec-draft-ctx 32768 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 3`). No cliff. Compare N2 (default flags): 323.1 / 3.32 - the footprint flags turn the collapse into 33.8 t/s. Repetitive continuation (acc 1.00), decode optimistic. |
| N8 | merged+exp+footprint tree (`a35a77184`, graph-key fix ON) | UD-Q4_K_S dyn | MTP n2 | 131072 | agentic(113182) | q4_0 | 610.1 | **37.40** | .86/2.70 | | sweep, see decode_investigation README Part 12 |
| N9 | same | UD-Q4_K_S dyn | MTP n3 | 131072 | agentic(113182) | q4_0 | 610.1 | **42.11** | .795/3.34 | | best n_max; compare N3 (n5): 608.7 / 32.38 |
| N10 | same | UD-Q4_K_S dyn | MTP n4 | 131072 | agentic(113182) | q4_0 | 609.4 | **34.25** | .61/3.43 | | |
| N11 | same | UD-Q4_K_S dyn | MTP n5 | 131072 | agentic(113182) | q4_0 | 608.0 | **31.62** | .49/3.43 | | reproduces N3 with the fix ON |
| N12 | same | UD-Q4_K_S dyn | MTP n5 p-min 0.5 | 131072 | agentic(113182) | q4_0 | 609.4 | **29.28** | .60/3.40 | | a draft-probability floor does not replace a small n_max |
| N13 | merged+exp tree `exp/fattn-tile-q4` @ `ac08925ea`, flag OFF | UD-Q4_K_S dyn | none | 131072 | agentic(113182) | q4_0 | 648.6 | **23.13** | - | | serial reference for N14 |
| N14 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | UD-Q4_K_S dyn | none | 131072 | agentic(113182) | q4_0 | 646.8 | **26.97** | - | | fused q4_0 tile attention, +16.6% over N13 |
| N15 | same, flag OFF, graph-key fix ON | UD-Q4_K_S dyn | MTP n3 | 131072 | agentic(113182) | q4_0 | 611-614 | **42.89** | .795/3.34 | | mean of 4 runs 42.1-42.95 |
| N16 | same, `GGML_CUDA_FATTN_TILE_Q4=1` (and =2: 45.95) | UD-Q4_K_S dyn | MTP n3 | 131072 | agentic(113182) | q4_0 | 617.3 | **46.05** | .779/3.32 | | +7%; continuation differs from N15 after some tokens |
| N17 | same, flag OFF, graph-key fix ON | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 558.7 | **42.98** | .685/4.38 | | compare N1 (fix OFF, older build): 547.5 / 44.33 |
| N18 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | STRIX fp4 | MTP n5 | 131072 | agentic(113182) | q4_0 | 550.5 | **47.43** | .685/4.38 | | +10.4%; identical output hash to N17 |
| N19 | tree 4 `exp/fattn-tile-q4` @ `ac08925ea`, flag OFF, footprint flags | STRIX fp4 | MTP n3 | 262144 | network(258801) | q4_0 (draft KV q4_0) | 364.3 | **34.42** | 1.00/3.97 | ~22.0 (other apps ~0.1 GiB) | reproduces N7 (365.6 / 33.77) |
| N20 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | STRIX fp4 | MTP n3 | 262144 | network(258801) | q4_0 (draft KV q4_0) | 362.6 | **41.07** | 1.00/3.97 | ~22.0 | **+19.3%** decode over N19; identical output hash (`f361034857a2`); prefill unchanged. Repetitive continuation (acc 1.00) |
| N21 | tree 4 `exp/fattn-tile-q4` @ `ac08925ea`, flag OFF | STRIX fp4 | MTP n3 | 131072 | agentic(113182) | q4_0 | 551.6 | **47.45** | .826/3.43 | | n_max sweep for STRIX (N17 = n5: 42.98) |
| N22 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | STRIX fp4 | MTP n2 | 131072 | agentic(113182) | q4_0 | 548.9 | **45.16** | .870/2.74 | | |
| N23 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | STRIX fp4 | MTP n3 | 131072 | agentic(113182) | q4_0 | 553.4 | **52.08** | .826/3.43 | | **best STRIX config at 113k: +21% over N17**; identical output hash in N17-N24 |
| N24 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | STRIX fp4 | MTP n4 | 131072 | agentic(113182) | q4_0 | 552.4 | **48.51** | .793/4.10 | | |
| N25 | tree 4 `exp/fattn-tile-q4` @ `ac08925ea`, flag OFF, graph-key fix ON | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic prefix(74549) | q4_0 | 689.4 | **35.92** | .537/3.23 | | 74.5k fill only: the full 113k prompt hangs (README Parts 11, 14) |
| N26 | same, `GGML_CUDA_FATTN_TILE_Q4=2` | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic prefix(74549) | q4_0 | 679.6 | **36.81** | .531/3.21 | | +2.5% (attention is a smaller share of this round); different continuation |
| N27 | replicates of N21/N23 (STRIX MTP n3, 113k fill, tree 4 `exp/fattn-tile-q4` @ `ac08925ea`) | STRIX fp4 | MTP n3 | 131072 | agentic(113182) | q4_0 | 549-559 | flag OFF: **45.88, 47.02, 47.45** (mean 46.8); flag ON (`=2`): **51.29, 52.01, 52.08** (mean 51.8) | .826/3.43 | | +10.7% from the flag at equal n_max; +20.5% over the n5 default (N17, 42.98); same output hash in all 6 runs |
| N28 | tree 4 `exp/fattn-tile-q4` @ `ac08925ea`, graph-key fix OFF, **pre-warm requests** | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic(113182) | q4_0 | 607.5 | **30.93** | .473/3.17 | | first completed DFlash row at the table prompt (N5 hung without the pre-warm, README Part 15) |
| N29 | same, graph-key fix ON (default), pre-warm | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic(113182) | q4_0 | 613.8 | **33.33** | .473/3.17 | | +7.8% over N28; identical output hash |
| N30 | same, fix ON + `GGML_CUDA_FATTN_TILE_Q4=2`, pre-warm | UD-Q3_K_XL | DFlash2 n5 p0.4 | 131072 | agentic(113182) | q4_0 | 602.6 | **35.12** | .470/3.17 | | +5.4% over N29, +13.5% over N28 |
| N31 | tree 4 `exp/fattn-tile-q4` @ `ac08925ea`, flag OFF, graph-key fix ON, pre-warm | UD-Q3_K_XL | DFlash2 n5 p0.4 | 262144 | network(258801) | q4_0 (draft KV q4_0) | 385.8 | **37.3** | 1.00/6.00 | ~22.0 (other apps ~0.1 GiB) | flags: `--ctx-reserve 32768 -ctkd q4_0 -ctvd q4_0` (`--spec-draft-ctx` is MTP-only). No hang and no cliff with the pre-warm. Repetitive continuation (acc 1.00, mean length 6), so decode is optimistic; output hash equals the STRIX runs (`f361034857a2`) |
| N32 | as N30 (fix ON, `GGML_CUDA_FATTN_TILE_Q4=2`, pre-warm) | UD-Q3_K_XL | DFlash2 n2 p0.4 | 131072 | agentic(113182) | q4_0 | 603.2 | **42.17** | .786/2.57 | | n_max sweep for DFlash (N30 = n5: 35.12) |
| N33 | same | UD-Q3_K_XL | DFlash2 n3 p0.4 | 131072 | agentic(113182) | q4_0 | 605.8 | **41.15** | .659/2.89 | | |
| N34 | same | UD-Q3_K_XL | DFlash2 n4 p0.4 | 131072 | agentic(113182) | q4_0 | 603.5 | **35.16** | .512/2.95 | | |
| N35 | fix ON, flag OFF, pre-warm | UD-Q3_K_XL | DFlash2 n3 p0.4 | 131072 | agentic(113182) | q4_0 | 603.4 | **39.49** | .677/2.95 | | pairs with N33: fused attention +4.2%; n3 vs n5 (N29 33.33) +18%; identical output hash in N28-N35 |

Not comparable to the table (different generation length or fill), kept in `bench/decode_investigation/models_ctx_ab_results.txt`:
512 generated tokens raise acceptance on this prompt (STRIX 131072: 52.1 t/s at .82; Unsloth Q4 131072: 38.7 t/s), and the
113k-fill-at-262144 runs (STRIX 9.3 t/s default draft KV, 29.0 with `-ctkd q4_0 -ctvd q4_0`; Unsloth Q4 8.0 t/s with q4_0 draft KV).

Reading: at 131072 the two MTP configs behave as before (Unsloth Q4 31-32 t/s; STRIX higher than #62, with higher acceptance
that this table cannot explain, so treat N1 as "new build" and not as a like-for-like improvement). At 262144 with the full
258,801-token fill, STRIX MTP is still in the 3 t/s collapse. The Unsloth Q4 MTP prefill slowdown at 262144 is session-persistent
and unexplained (Part 4 of the investigation README).

Update 2026-10-02 (rows N7-N26, all with the graph-key fix on unless stated; tree 4 `opt/vram-footprint` @ `a35a77184`, rows N13+ on `exp/fattn-tile-q4` @ `ac08925ea`):
- The cliff at 262144 is gone with the footprint flags and the other apps' GPU use low: STRIX MTP 258.8k fill 33.8-34.4 t/s (N7, N19) instead of 3.3 (N2).
- `--spec-draft-n-max 3` is the best setting for both MTP models at 113k fill: UD-Q4_K_S 42.1 t/s vs 31.6 at n5 (N9 vs N11); STRIX 47.5 vs 43.0 (N21 vs N17); a draft-probability floor does not replace it (N12).
- Opt-in `GGML_CUDA_FATTN_TILE_Q4` (fused q4_0 dequantization in the tile attention kernel) adds +7..+19% decode: no-spec 23.1 -> 27.0 (N13/N14), UD-Q4_K_S MTP n3 42.9 -> 46.1 (N15/N16),
  STRIX MTP n5 43.0 -> 47.4 and **n3 46.8 -> 51.8 (mean of 3, N27)**; n3 + flag is +20.5% over the n5 default (N17/N18, N23), 262144 / 258.8k fill 34.4 -> 41.1 (N19/N20), DFlash2 at 74.5k +2.5% (N25/N26). Details, exactness and negative results: `notes/tensorfold_analysis_2026-10-02.md`, `bench/decode_investigation/README.md` Parts 12-14.
- UD-Q3_K_XL + DFlash2 at the 113k fill (N5) hung after the prefill; cause located (a first-use load of a small-batch kernel variant at depth, README Parts 14-15) and **worked around by sending a few tiny requests right after the server starts** ("pre-warm"): DFlash2 at 113k = 30.9 t/s (graph-key fix off), 33.3 (fix on), 35.1 (fix on + fused attention): rows N28-N30.

### 13.4 Master matrix with the new settings (night of 2026-10-02 / 03): 7 models x 2 contexts

Decode speed (tok/s, **tg**) with and without the new settings on every model of the matrix. Conditions as in 13.3: agentic(113182) at `-c 131072` or `network(258801)` at `-c 262144`, 128 generated
tokens, greedy, q4_0 KV, `-np 1`, `-b 2048 -ub 512`, `LLAMA_ATTN_ROT_DISABLE=1`, graph-key fix on, one model copy in WSL at a time. "Fused" = `GGML_CUDA_FATTN_TILE_Q4=2` (tree 4 branch `exp/fattn-tile-q4`);
"spec n3" = `--spec-draft-n-max 3`; "spec n5 default" = the table's original draft length (measured at 131072 only: at 262144 the default flags collapse, see N2). The 262144 cells use
`--ctx-reserve 32768` (+ `-ctkd q4_0 -ctvd q4_0`, and `--spec-draft-ctx 32768` for the MTP models); every server got tiny pre-warm requests after start (sizes that hang on a model are dropped for it: 24/30/32 on the IQ3_XXS models).

| model | spec | ctx / fill | no spec, default | no spec, fused | gain | spec n5 default | spec n3 default | spec n3 fused | gain (n3) | n3 fused vs n5 default |
|---|---|---|---|---|---|---|---|---|---|---|
| STRIX fp4 | MTP | 131072 / 113,182 | 24.11 | 28.03 | +16.3% | 42.98 | 46.80 | 51.80 | +10.7% | +20.5% |
| STRIX fp4 | MTP | 262144 / 258,801 | 15.64 | 19.78 | +26.5% | - | 34.55 | 40.71 | +17.8% | - |
| UD-Q4_K_S dyn | MTP | 131072 / 113,182 | 23.13 | 26.97 | +16.6% | 31.62 | 42.89 | 45.95 | +7.1% | +45.3% |
| UD-Q4_K_S dyn | MTP | 262144 / 258,801 | 15.34 | 19.24 | +25.4% | - | 32.92 | 39.27 | +19.3% | - |
| UD-Q3_K_XL | DFlash2 | 131072 / 113,182 | 24.23 | 28.32 | +16.9% | 33.33 | 39.49 | 41.15 | +4.2% | +23.5% |
| UD-Q3_K_XL | DFlash2 | 262144 / 258,801 | 15.66 | 19.88 | +26.9% | - | 33.27 | 39.58 | +19.0% | - |
| Qwopus Q4_K_M | MTP | 131072 / 113,182 | 22.42 | 25.77 | +14.9% | 36.65 | 38.36 | 41.61 | +8.5% | +13.5% |
| Qwopus Q4_K_M | MTP | 262144 / 258,801 | 14.90 | 18.69 | +25.4% | - | 31.10 | 36.06 | +15.9% | - |
| UD-IQ3_XXS | DFlash2 | 131072 / 113,182 | 25.13 | 29.44 | +17.2% | 33.50 | 38.76 | 44.52 | +14.9% | +32.9% |
| UD-IQ3_XXS | DFlash2 | 262144 / 258,801 | 16.05 | 20.38 | +27.0% | - | 33.81 | 39.62 | +17.2% | - |
| GSQ-RCO IQ3_XXS | MTP (Unsloth Q4_0 head) | 131072 / 113,182 | 25.30 | 29.63 | +17.1% | 40.23 | 41.42 | 49.44 | +19.4% | +22.9% |
| GSQ-RCO IQ3_XXS | MTP (Unsloth Q4_0 head) | 262144 / 258,801 | 16.07 | 20.51 | +27.6% | - | 32.83 | 38.81 | +18.2% | - |
| Q4_K_S (plain) | DFlash2 | 131072 / 113,182 | 23.02 | 26.65 | +15.8% | 27.89 | 31.79 | 34.06 | +7.1% | +22.1% |
| Q4_K_S (plain) | DFlash2 | 262144 / 258,801 | 15.22 | 19.13 | +25.7% | - | 31.47 | 36.58 | +16.2% | - |

Notes and caveats:
- Output hashes: the speculative and no-speculation runs of one model give the same text on the 262144 prompt (all `f361034857a2`); at 131072 the fused path is hash-identical for STRIX and in some pairs of the others, and differs after some tokens in the rest (README Part 13).
- At 262144 the speculative acceptance is 1.00 (the continuation of that prompt repeats), so those decode figures are optimistic; the fused/default ratio is the informative part. The gain from the fused attention grows with context:
  +15-17% at 113k, +25-28% at 262144 without speculation.
- Cells marked from earlier rows (STRIX 131072 n5/n3, UD-Q4_K_S 131072 n5/n3 and no-spec, UD-Q3_K_XL 131072 spec n3/n5) come from rows N11-N35 and N13-N18 of 13.3; the others from `bench/decode_investigation/matrix_2026-10-02_raw.txt`
  (a cell measured twice keeps its latest run).
- **Driver state matters**: the 131072 cells of UD-IQ3_XXS, plain Q4_K_S and GSQ-RCO were first measured between 11:23 and 12:47 on 2026-10-02, while the AMD driver was resetting (slow prefill, 260-330 tok/s); they were re-run
  on 2026-10-03 (stage H) and the re-runs replace them. The 262144 cells ran with the GPU core and memory clocks capped at 2850 MHz (needed to stop the resets, `notes/amd_driver_crashes_2026-10-02.md`); the 131072 cells of
  STRIX, UD-Q4_K_S, UD-Q3_K_XL and Qwopus ran before the resets began.
- Every server held all layers on the GPU (65/65, draft 66/66 or 6/6); load reports with the memory occupancy per model: `bench/decode_investigation/load_reports_2026-10-02.txt` (made with `bench/sweep/load_report.sh`).

### 13.5 Which draft per quant (n_max 3, fused attention)

The same table for every draft we have, per quant. "Unsloth Q4_0 head" = the 1.37 GB `mtp-Qwen3.8-27B-Q4_0_dynamicV3.0.gguf` passed with `-md` next to the target file (any quant of Qwen3.8-27B).

| quant | draft | 131072 default | 131072 fused | acceptance (fused) | prefill 131072 | 262144 default | 262144 fused |
|---|---|---|---|---|---|---|---|
| STRIX fp4 | own MTP head (STRIX mtp file) | 46.80 | 51.80 | - | - | 34.55 | 40.71 |
| STRIX fp4 | Unsloth Q4_0 head | 46.66 | 52.25 | 0.83 | 554 | 34.35 | 41.52 |
| STRIX fp4 | DFlash2 | 39.12 | 43.62 | 0.70 | 544 | 34.35 | 40.87 |
| UD-Q4_K_S dyn | Unsloth Q4_0 head (its own) | 42.89 | 45.95 | - | - | 32.92 | 39.27 |
| UD-Q4_K_S dyn | DFlash2 | 34.11 | 37.91 | 0.64 | 604 | 32.93 | 38.23 |
| UD-Q3_K_XL | DFlash2 | 39.49 | 41.15 | - | - | 33.27 | 39.58 |
| UD-Q3_K_XL | Unsloth Q4_0 head | 47.37 | 50.62 | 0.83 | 615 | 33.72 | 40.45 |
| Qwopus Q4_K_M | embedded MTP | 38.36 | 41.61 | 0.77 | 608 | 31.10 | 36.06 |
| Qwopus Q4_K_M | Unsloth Q4_0 head | 38.89 | 42.33 | 0.77 | 608 | - | - |
| Qwopus Q4_K_M | DFlash2 | 33.31 | 35.38 | 0.66 | 601 | - | - |
| UD-IQ3_XXS | DFlash2 | 38.76 | 44.52 | 0.67 | 598 | 33.81 | 39.62 |
| UD-IQ3_XXS | Unsloth Q4_0 head | 42.44 | 46.70 | 0.74 | 609 | 33.82 | 40.41 |
| GSQ-RCO IQ3_XXS | own -mtp file alone | 37.50 | 45.90 | 0.76 | 601 | 32.66 | 38.77 |
| GSQ-RCO IQ3_XXS | Unsloth Q4_0 head | 41.42 | 49.44 | 0.84 | 600 | 32.83 | 38.81 |
| GSQ-RCO IQ3_XXS | DFlash2 | 33.49 | 35.75 | 0.59 | 591 | 33.18 | 39.16 |
| Q4_K_S (plain) | DFlash2 | 31.79 | 34.06 | 0.62 | 592 | 31.47 | 36.58 |
| Q4_K_S (plain) | Unsloth Q4_0 head | 43.80 | 47.11 | 0.88 | 603 | 31.32 | 37.16 |

| pairing (GSQ-RCO IQ3_XXS) | files loaded | 131072 decode | acceptance 131072 | 262144 decode | VRAM peak 262144 |
|---|---|---|---|---|---|
| plain + its own `-mtp` file as draft (`-md`) | 10.1 + 10.4 GB: two full models | 37.48 -> 46.13 | 0.64 / 0.76 | does not fit | does not fit (23.2 GiB + 2.4 GiB shared) |
| `-mtp` file alone (embedded MTP, no `-md`) | 10.4 GB | 37.50 -> 45.90 | 0.64 / 0.76 | 32.66 -> 38.77 | - |
| plain + Unsloth Q4_0 MTP head (`-md`) | 10.1 + 1.4 GB | 41.42 -> 49.44 | 0.77 / 0.84 | 32.83 -> 38.81 | 17.5 GiB |
| plain + DFlash2 (`-md`, p-min 0.4) | 10.1 + 1.1 GB | 33.49 -> 35.75 | 0.60 / 0.59 | 33.18 -> 39.16 | - |

Reading:
- **An MTP head beats DFlash2 on every quant tested**, at 131072 (Unsloth head vs DFlash2, fused): STRIX +20%, Qwopus +20%, UD-Q3_K_XL +23%, UD-IQ3_XXS +5%, plain Q4_K_S +38%, GSQ-RCO +38%; at 262144 the two are within 3% of each other (acceptance is 1.00 there).
  Prefill is the same for all drafts once the driver is healthy (about 600 tok/s at 131072, 390 at 262144; see the prefill column).
- The Unsloth head is interchangeable with a model's own head on this sample: with STRIX and with Qwopus it reproduces their own head's acceptance and output hash exactly (STRIX n3 fused 52.25 vs 51.8 tok/s with its own head); on GSQ-RCO
  its acceptance (0.84 fused) is *higher* than GSQ's own quantized MTP layer (0.76).
- `n_max` 3 stays the optimum with the Unsloth head (fused, 131072, rows `MX_q3u_113_n{2,4}_f2`, `MX_q4ku_113_n{2,4}_f2`): UD-Q3_K_XL n2 / n3 / n4 = 45.08 / 50.62 / 45.14 tok/s, plain Q4_K_S 42.79 / 47.11 / 41.51.
- **A "-" in the 262144 columns of Qwopus** means the combination does not fit in VRAM there (Qwopus Q4_K_M already peaks at 22.7 GiB with its embedded head): Qwopus + DFlash2 gave prefill 143 tok/s and no result in 30 min (twice), Qwopus + the Unsloth head 175 tok/s prefill and 1.18 tok/s decode (rows `MX_qwopusd_262`, `MX_qwopusu_262`).
- **DFlash2 Q8_0 drafter vs Q4_K_M** (stage J, 131072, n_max 3, default / fused): UD-Q3_K_XL 39.22 / 43.16 tok/s (acceptance .64) against 39.49 / 41.15 with Q4_K_M; UD-Q4_K_S dyn 35.01 / 35.72 (acceptance .60 / .59) against 34.11 / 37.91 (.61). No acceptance gain from Q8_0, so no sign that the Q4 drafter is the weak part. At 262144 UD-Q4_K_S + Q8_0 spills out of VRAM (decode 1.18 tok/s, row `MX_u4d8_262_n3_f0`).
- **Qwopus's own MTP layer works as the head for other quants** (stage M; the Unsloth head container with blk.64 replaced, `bench/sweep/patches/make_qwopus_head.py`; 131072, fused, n3): UD-Q3_K_XL 50.85 tok/s (acceptance .833, same as the Unsloth head's 50.62 / .833), UD-Q4_K_S dyn 45.55 (.76; Unsloth head 45.95), plain Q4_K_S 45.84 (.843; Unsloth head 47.11 / .876). Outputs hash-identical to the Unsloth head's.
- **A head without its own embedding and output matrices** (stage N; tree 4 branch `exp/mtp-share-io`, the head borrows the target's, 0.26 GiB instead of 1.27 GiB, `-fit off`): UD-Q3_K_XL 51.26 tok/s (acceptance .867, hash identical), Qwopus 131072 41.65 tok/s (Unsloth head 42.33). Qwopus at 262144 still does not fit with it (22.45 GiB dedicated + 1.3 GiB shared, prefill ~150 tok/s, no result in 30 min): the separate draft model brings its own KV and compute buffers.
- `GSQ-RCO-IQ3_XXS-mtp.gguf` is a complete model with the MTP layer appended (866 vs 851 tensors), not a head: use it alone, never as `-md` beside the plain file (two full models = 23.2 GiB + 2.4 GiB spilled at 262144).

### 13.6 Fused attention with q8_0 KV (2026-10-03, tree 4 branch `exp/fattn-tile-q8`)

The fused tile attention now also covers q8_0 KV (same flag `GGML_CUDA_FATTN_TILE_Q4`, head size 256). `test-backend-ops` FLASH_ATTN_EXT with q8_0 KV: same pass/fail pattern with and without the flag (the same two known tolerance
failures at kv=262144); 4-column attention at kv 113408: 2053 -> 1433 us per layer; 1 column unchanged (719 vs 725 us, the default vector kernel is already good for q8_0).

| model | spec | ctx / fill | KV | flag | pp | tg | acc/len | note |
|---|---|---|---|---|---|---|---|---|
| STRIX fp4 | MTP n3 | 131072 / 113,182 | q8_0 | off | 552.0 | **43.92** | .788/3.34 | `Q8KV_strix_f0` |
| STRIX fp4 | MTP n3 | 131072 / 113,182 | q8_0 | 1 (3-16 columns) | 553.2 | **50.29** | .788/3.34 | +14.5%, identical output hash |
| STRIX fp4 | MTP n3 | 131072 / 113,182 | q8_0 | 2 (also 1-2 columns) | 550.6 | **49.71** | .788/3.34 | identical hash |
| UD-Q4_K_S dyn | MTP n3 | 131072 / 113,182 | q8_0 | off | 615.3 | **41.79** | .817/3.41 | `Q8KV_u4_f0` |
| UD-Q4_K_S dyn | MTP n3 | 131072 / 113,182 | q8_0 | 1 | 616.4 | **46.74** | .817/3.41 | +11.8%, identical hash |
| Qwopus Q4_K_M | MTP n3 | 131072 / 113,182 | q8_0 | off | 607.2 | **35.61** | .731/3.17 | `Q8KV_qwopus_f0` |
| Qwopus Q4_K_M | MTP n3 | 131072 / 113,182 | q8_0 | 1 | 608.0 | **39.66** | .731/3.17 | +11.4%, identical hash |

Quality with q8_0 KV (STRIX, `-c 4096`, `-b 8 -ub 8`, Q8_0 reference logits, `llama-perplexity --kl-divergence`), default vs flag 1: **identical to every printed digit** - coding_q PPL 3.081347 +- 0.059548, mean KLD 0.042743 +- 0.001339, same top-1 88.149 +- 0.320 %;
wiki_q PPL 5.581524 +- 0.148503, KLD 0.092523 +- 0.007529, top-1 89.289 +- 0.342 % (the speed runs above show the path is engaged and give an identical output hash, so for q8_0 the fused loader reproduces the f16 conversion exactly, as far as these runs can tell).

q8_0 KV with the fused attention (50.3 t/s) is within 3% of q4_0 KV with it (51.8, N27) and much closer to f16 in quality (quality candidates `*_k8`, RESULTS 14); at 262144 q8_0 KV (9.2 GiB) does not fit next to the weights.

### 13.7 Decode speed vs context size (stage K of the master matrix, 2026-10-03 / 04)

Fused attention (`GGML_CUDA_FATTN_TILE_Q4=2`), `--spec-draft-n-max 3`, 128 generated tokens, greedy, q4_0 KV, GPU clocks capped at 2850 MHz, one run per cell, `-np 1`. The context is the server's `-c`; the fill is 0.864 x ctx
(14.1k / 28.3k / 56.5k / 169.9k tokens, prefixes of the 258,801-token fill text; the 131072 cell is the 113,182-token agentic prompt and the 262144 cell the full 258,801-token fill with the footprint flags, as in 13.4).
196608 also uses the footprint flags (`--ctx-reserve 32768`, `--spec-draft-ctx 32768`, `-ctkd q4_0 -ctvd q4_0`). The default-attention counterparts exist only at 131072 and 262144 (13.4). Decode tok/s, "-" = not measured or does not fit:

| model | draft | 16k | 32k | 64k | 128k | 192k | 256k |
|---|---|---|---|---|---|---|---|
| STRIX fp4 | no speculation | 37.9 | 35.7 | 32.5 | 28.0 | 24.4 | 19.8 |
| STRIX fp4 | own / embedded MTP head | 67.4 | 60.5 | 63.0 | 51.8 | 37.6 | 40.7 |
| STRIX fp4 | Unsloth Q4_0 head | 69.5 | 58.4 | 62.4 | 52.2 | 36.0 | 41.5 |
| STRIX fp4 | DFlash2 | 64.8 | 54.9 | 60.8 | 43.6 | 36.0 | 40.9 |
| UD-Q4_K_S dyn | no speculation | 36.4 | 33.5 | 30.9 | 27.0 | 23.6 | 19.2 |
| UD-Q4_K_S dyn | Unsloth Q4_0 head | 63.5 | 61.0 | 46.8 | 46.0 | 35.8 | 39.3 |
| UD-Q4_K_S dyn | DFlash2 | 61.3 | 54.6 | 45.7 | 37.9 | 33.9 | 38.2 |
| UD-Q3_K_XL | no speculation | 38.1 | 36.2 | 32.8 | 28.3 | 24.5 | 19.9 |
| UD-Q3_K_XL | Unsloth Q4_0 head | 63.2 | 62.3 | 63.2 | 50.6 | 50.7 | 40.5 |
| UD-Q3_K_XL | DFlash2 | 59.0 | 54.1 | 61.0 | 41.1 | 50.1 | 39.6 |
| Qwopus Q4_K_M | no speculation | 33.8 | 31.9 | 29.9 | 25.8 | 22.4 | 18.7 |
| Qwopus Q4_K_M | own / embedded MTP head | 65.2 | 52.1 | 57.0 | 41.6 | 43.5 | 36.1 |
| Qwopus Q4_K_M | Unsloth Q4_0 head | 68.3 | 52.4 | 58.2 | 42.3 | 44.8 | - |
| Qwopus Q4_K_M | DFlash2 | 61.7 | 58.4 | 57.3 | 35.4 | 44.0 | - |
| UD-IQ3_XXS | no speculation | 40.9 | 38.3 | 34.8 | 29.4 | 25.3 | 20.4 |
| UD-IQ3_XXS | Unsloth Q4_0 head | 70.9 | 59.5 | 55.6 | 46.7 | 50.7 | 40.4 |
| UD-IQ3_XXS | DFlash2 | 67.6 | 53.5 | 54.5 | 44.5 | 49.5 | 39.6 |
| GSQ-RCO IQ3_XXS | no speculation | 40.7 | 38.4 | 34.8 | 29.6 | 25.5 | 20.5 |
| GSQ-RCO IQ3_XXS | own / embedded MTP head | 60.7 | 53.9 | 48.3 | 45.9 | 49.0 | 38.8 |
| GSQ-RCO IQ3_XXS | Unsloth Q4_0 head | 65.4 | 53.3 | 47.3 | 49.4 | 47.6 | 38.8 |
| GSQ-RCO IQ3_XXS | DFlash2 | 61.1 | 50.2 | 47.6 | 35.8 | 48.6 | 39.2 |
| Q4_K_S (plain) | no speculation | 35.2 | 33.4 | 30.5 | 26.6 | 23.2 | 19.1 |
| Q4_K_S (plain) | Unsloth Q4_0 head | 54.0 | 52.3 | 52.6 | 47.1 | 35.6 | 37.2 |
| Q4_K_S (plain) | DFlash2 | 52.9 | 48.6 | 47.9 | 34.1 | 32.7 | 36.6 |

Reading: decode speed falls with the populated context for every variant (no speculation 34-41 tok/s at 16k -> 18.7-20.5 at 262k); with speculation the curves are noisy because the acceptance depends on the text each model generates (greedy output differs between quants): .6-.9 at 16k-196k, mostly 1.00 at 262144 where the prompt
repeats, so the 262144 speeds are optimistic. Compare variants of one model within a column, not across the noisy steps. Raw rows: `bench/decode_investigation/matrix_2026-10-02_raw.txt`,
figures `plots/quality_*_vs_speed.svg` and `plots/speed_output_vs_ctx_filled.svg`.

### 13.8 The same setups on a real pi session (stage P, 2026-10-04): new values next to the older ones

**Nothing above was replaced.** 13.4, 13.5 and 13.7 keep the older matrix values; those speculative speeds came from a synthetic agentic prompt (131072: an opencode-like system prompt, llama.cpp source files and an audit task) and from prefixes of a text in which one
JSON block repeats 1074 times (all other contexts, where the model copies the block and acceptance is inflated). Stage P re-measures every speculative setup on prompts cut from a **real pi session** (`~/.pi/agent/sessions/--mnt-f-programming-llm-pi-tg-gondolin--/2026-08-20T19-02-46-107Z_*.jsonl`,
the pi-tg-gondolin work: tool calls, tool results, thinking), rendered with the model's own chat template and cut at a turn boundary where the assistant answers next (`bench/sweep/patches/make_pi_prompts.py`; the tool list and system prompt are generic, the session does not store them).
Greedy, 512 generated tokens (the older rows: 128), q4_0 KV, `--spec-draft-n-max 3`, fused attention (default attention also at 131072 and 262144), tags `MP_`. No-speculation speeds do not depend on the text and are not repeated. Fills: 16k: 11.4k tokens, 32k: 27.7k tokens, 64k: 54.4k tokens, 128k: 111.2k tokens, 192k: 169.8k tokens, 256k: 244.6k tokens.

Decode tok/s, pi session (older prompt) and the difference; acceptance of the pi run in the last block:

| model | draft | 16k | 32k | 64k | 128k | 192k | 256k |
|---|---|---|---|---|---|---|---|
| STRIX fp4 | own / embedded head | 85.6 (67.4, +27%) | 63.4 (60.5, +5%) | 71.3 (63.0, +13%) | 49.1 (51.8, -5%) | 44.9 (37.6, +19%) | 37.8 (40.7, -7%) |
| STRIX fp4 | Unsloth head | 86.7 (69.5, +25%) | 64.6 (58.4, +11%) | 70.0 (62.4, +12%) | 48.2 (52.2, -8%) | 44.0 (36.0, +22%) | 36.0 (41.5, -13%) |
| STRIX fp4 | DFlash2 | 80.3 (64.8, +24%) | 58.3 (54.9, +6%) | 69.3 (60.8, +14%) | 44.5 (43.6, +2%) | 43.7 (36.0, +21%) | 32.2 (40.9, -21%) |
| UD-Q4_K_S dyn | Unsloth head | 78.1 (63.5, +23%) | 55.8 (61.0, -9%) | 66.4 (46.8, +42%) | 44.3 (46.0, -4%) | 46.2 (35.8, +29%) | 31.6 (39.3, -19%) |
| UD-Q4_K_S dyn | DFlash2 | 72.0 (61.3, +17%) | 57.9 (54.6, +6%) | 65.3 (45.7, +43%) | 41.0 (37.9, +8%) | 44.7 (33.9, +32%) | 30.2 (38.2, -21%) |
| UD-Q3_K_XL | Unsloth head | 82.5 (63.2, +31%) | 61.5 (62.3, -1%) | 70.6 (63.2, +12%) | 43.0 (50.6, -15%) | 47.0 (50.7, -7%) | 35.4 (40.5, -12%) |
| UD-Q3_K_XL | DFlash2 | 76.6 (59.0, +30%) | 53.9 (54.1, -0%) | 70.9 (61.0, +16%) | 38.9 (41.1, -6%) | 44.8 (50.1, -11%) | 31.4 (39.6, -21%) |
| Qwopus Q4_K_M | own / embedded head | 65.7 (65.2, +1%) | 50.0 (52.1, -4%) | 58.7 (57.0, +3%) | 36.3 (41.6, -13%) | 41.1 (43.5, -6%) | 33.4 (36.1, -7%) |
| Qwopus Q4_K_M | Unsloth head | 67.1 (68.3, -2%) | 52.2 (52.4, -0%) | 59.4 (58.2, +2%) | 36.4 (42.3, -14%) | 40.9 (44.8, -9%) | - |
| Qwopus Q4_K_M | DFlash2 | 64.1 (61.7, +4%) | 47.1 (58.4, -19%) | 60.0 (57.3, +5%) | 35.5 (35.4, +0%) | 39.3 (44.0, -11%) | - |
| UD-IQ3_XXS | Unsloth head | 82.4 (70.9, +16%) | 61.4 (59.5, +3%) | 71.8 (55.6, +29%) | 49.5 (46.7, +6%) | 44.7 (50.7, -12%) | 32.2 (40.4, -20%) |
| UD-IQ3_XXS | DFlash2 | 76.1 (67.6, +13%) | 52.4 (53.5, -2%) | 70.7 (54.5, +30%) | 42.3 (44.5, -5%) | 44.4 (49.5, -10%) | 30.8 (39.6, -22%) |
| GSQ-RCO IQ3_XXS | own / embedded head | 77.2 (60.7, +27%) | 56.9 (53.9, +6%) | 67.1 (48.3, +39%) | 44.5 (45.9, -3%) | 42.9 (49.0, -12%) | 31.9 (38.8, -18%) |
| GSQ-RCO IQ3_XXS | Unsloth head | 79.9 (65.4, +22%) | 56.5 (53.3, +6%) | 68.0 (47.3, +44%) | 43.9 (49.4, -11%) | 47.2 (47.6, -1%) | 32.3 (38.8, -17%) |
| GSQ-RCO IQ3_XXS | DFlash2 | 73.7 (61.1, +21%) | 52.7 (50.2, +5%) | 69.4 (47.6, +46%) | 41.5 (35.8, +16%) | 44.9 (48.6, -8%) | 30.4 (39.2, -22%) |
| Q4_K_S (plain) | Unsloth head | 70.5 (54.0, +31%) | 52.3 (52.3, -0%) | 50.3 (52.6, -4%) | 42.1 (47.1, -11%) | 40.0 (35.6, +12%) | - (37.2) |
| Q4_K_S (plain) | DFlash2 | 65.1 (52.9, +23%) | 48.4 (48.6, -0%) | 47.1 (47.9, -2%) | 37.4 (34.1, +10%) | 38.2 (32.7, +17%) | - (36.6) |

Acceptance (pi run) / mean accepted length:

| model | draft | 16k | 32k | 64k | 128k | 192k | 256k |
|---|---|---|---|---|---|---|---|
| STRIX fp4 | own / embedded head | 0.89 / 3.65 | 0.64 / 2.90 | 0.91 / 3.70 | 0.72 / 3.15 | 0.77 / 3.32 | 0.79 / 3.36 |
| STRIX fp4 | Unsloth head | 0.89 / 3.65 | 0.64 / 2.90 | 0.91 / 3.70 | 0.73 / 3.19 | 0.76 / 3.28 | 0.74 / 3.21 |
| STRIX fp4 | DFlash2 | 0.92 / 3.64 | 0.70 / 2.84 | 0.92 / 3.67 | 0.70 / 2.92 | 0.78 / 3.27 | 0.71 / 2.97 |
| UD-Q4_K_S dyn | Unsloth head | 0.89 / 3.68 | 0.61 / 2.82 | 0.93 / 3.78 | 0.69 / 3.07 | 0.88 / 3.65 | 0.66 / 2.97 |
| UD-Q4_K_S dyn | DFlash2 | 0.90 / 3.63 | 0.77 / 3.17 | 0.93 / 3.70 | 0.70 / 2.88 | 0.86 / 3.53 | 0.70 / 2.92 |
| UD-Q3_K_XL | Unsloth head | 0.89 / 3.67 | 0.64 / 2.92 | 0.94 / 3.81 | 0.62 / 2.84 | 0.86 / 3.57 | 0.74 / 3.21 |
| UD-Q3_K_XL | DFlash2 | 0.92 / 3.64 | 0.73 / 2.89 | 0.94 / 3.78 | 0.64 / 2.72 | 0.87 / 3.48 | 0.71 / 2.94 |
| Qwopus Q4_K_M | own / embedded head | 0.88 / 3.62 | 0.67 / 3.02 | 0.95 / 3.84 | 0.63 / 2.87 | 0.87 / 3.60 | 0.82 / 3.47 |
| Qwopus Q4_K_M | Unsloth head | 0.89 / 3.65 | 0.66 / 2.97 | 0.95 / 3.84 | 0.62 / 2.84 | 0.87 / 3.60 | - |
| Qwopus Q4_K_M | DFlash2 | 0.89 / 3.57 | 0.72 / 2.90 | 0.94 / 3.80 | 0.65 / 2.76 | 0.83 / 3.42 | - |
| UD-IQ3_XXS | Unsloth head | 0.89 / 3.68 | 0.64 / 2.93 | 0.98 / 3.93 | 0.77 / 3.30 | 0.82 / 3.45 | 0.65 / 2.94 |
| UD-IQ3_XXS | DFlash2 | 0.89 / 3.57 | 0.64 / 2.67 | 0.94 / 3.79 | 0.70 / 2.89 | 0.84 / 3.46 | 0.74 / 2.95 |
| GSQ-RCO IQ3_XXS | own / embedded head | 0.91 / 3.70 | 0.64 / 2.92 | 0.96 / 3.87 | 0.70 / 3.08 | 0.81 / 3.43 | 0.67 / 3.02 |
| GSQ-RCO IQ3_XXS | Unsloth head | 0.93 / 3.79 | 0.62 / 2.85 | 0.97 / 3.90 | 0.68 / 3.04 | 0.92 / 3.76 | 0.69 / 3.05 |
| GSQ-RCO IQ3_XXS | DFlash2 | 0.91 / 3.63 | 0.69 / 2.79 | 0.95 / 3.84 | 0.70 / 2.88 | 0.84 / 3.50 | 0.73 / 2.95 |
| Q4_K_S (plain) | Unsloth head | 0.93 / 3.78 | 0.65 / 2.94 | 0.73 / 3.17 | 0.72 / 3.15 | 0.81 / 3.43 | - |
| Q4_K_S (plain) | DFlash2 | 0.90 / 3.60 | 0.69 / 2.89 | 0.72 / 2.99 | 0.66 / 2.83 | 0.81 / 3.29 | - |

Reading: over the 98 cells measured on both prompts the speed changes by +5% on average (median +2%, range -22% to +46%); mean change per context: 16k +20%, 32k +1%, 64k +20%, 128k -3%, 192k +4%, 256k -17%. Acceptance on the pi session is 0.61-0.98; it still varies between prompt prefixes and between quants, so compare setups within a column.

### 13.9 Quant-format experiments, prefill ceiling and tool-calling bench (2026-10-05)

Full tables and method: `notes/quant_format_experiments_2026-10-05.md`, `notes/mmvq_by_quant_type_2026-10-05.md`. Conditions: tree 4 build-exp, pi-session prompts, fused attention, MTP n3 with the Unsloth head, 512 tokens (`MP_` rows).

| model | aggregate quality (14) | MTP n3 11k | MTP n3 111k | file |
|---|---|---|---|---|
| UD-Q4_K_S | 85.2 | 78.1 | 44.3 | 15.4 GB |
| plain Q4_K_S | 78.5 | 70.5 | 42.1 | 16.1 GB |
| **UDflat** (UD-Q4_K_S allocation, K-quants -> iq4_xs / iq4_nl / q8_0, imatrix) | 77.9 | 85.6 | 50.9 | 15.8 GB |
| IQ4_XS requant + imatrix | 71.5 | 85.6 | 48.9 | 15.3 GB |
| IQ4_XS requant, no imatrix | 66.4 | 86.5 | 50.7 | 15.4 GB |
| STRIX fp4 | 67.5 | 86.7 | 48.2 | |

- A 4-column verify costs 1.0-1.3x a single column for rocmfp4 / iq4_xs / iq4_nl / q8_0 / q4_0 and 1.5-1.9x for K-quants and IQ3, so the format decides the speculative speed. IQ3 sign patch (`exp/iq3-signs`): +1-2% end to end.
- Prefill weight-matmul ceiling ~65 TFLOPS (iq4_xs, iq4_nl, q8_0, q4_0, f16), IQ3 ~62, q4_K ~61, q6_K / q3_K ~53, rocmfp4 ~52 (16-wide scale groups). `-ub` 512 -> 2048: +2-3%. Deep prefill is attention-bound (35-39 TFLOPS). hipBLASLt solution 1277 reaches 76-79 TFLOPS on the FFN shape (default hipblasGemmEx 12): at most +4-5% prefill, no call path in ggml-cuda.
- tool-eval-bench (69 + 23 hard scenarios, thinking on, seed 42), score /100: greedy UDflat 91 (2091 s), STRIX 90; thinking-mode sampling (T 1.0, top_p .95, top_k 20) UDflat 86. Further runs: `~/tool_eval/summary.txt` in WSL.

### 13.10 Thinking-mode sampling on the pi session (2026-10-06): decode tok/s (draft acceptance), MTP n3, fused

Plan: `PLAN_thinking_matrix_2026-10-05.md` (ladder variant: one server per model, nested pi prompts pi16..pi262 in ascending order, so the prefix cache keeps prefill incremental). Per cell: 1 warm-up call + 3 calls at 512 tokens with `ignore_eos`, seeds 101-103, temperature 1.0, top_p 0.95, top_k 20, min_p 0; mean of the 3 calls, acceptance = accepted/drafted over them. Unsloth head (Qwopus not run), tree 4 commit c849dfd04, fused attention; STRIX at `-c 262144`, the others at `-c 229376` (pi262 not run for them). Raw: `bench/decode_investigation/thinking_matrix_raw_2026-10-06.txt`.

| model | 11k | 28k | 55k | 112k | 196k | 229k | 262k | greedy control at the last context vs sampled |
|---|---|---|---|---|---|---|---|---|
| STRIX | 80.2 (.79) | 56.0 (.50) | 47.2 (.47) | 38.8 (.47) | 41.4 (.69) | 39.7 (.77) | 29.8 (.54) | 262k: 34.1 vs 29.8 |
| UDflat | 77.5 (.80) | 50.6 (.46) | 56.4 (.69) | 38.3 (.50) | 40.8 (.73) | 33.9 (.64) | - | 229k: 43.6 vs 33.9 |
| UD-Q4_K_S | 74.2 (.83) | 48.3 (.47) | 56.8 (.73) | 37.2 (.49) | 40.9 (.74) | 42.6 (.90) | - | 229k: 43.4 vs 42.6 |
| plain Q4_K_S | 65.2 (.81) | 44.8 (.50) | 48.0 (.65) | 37.0 (.57) | 36.8 (.71) | 35.1 (.75) | - | 229k: 39.1 vs 35.1 |
| UD-IQ3_XXS | 79.7 (.83) | 50.1 (.44) | 66.9 (.85) | 36.6 (.44) | 40.6 (.66) | 38.3 (.72) | - | 229k: 45.7 vs 38.3 |
| UD-Q3_K_XL | 75.4 (.77) | 50.2 (.45) | 53.3 (.61) | 35.7 (.42) | 41.1 (.70) | 37.7 (.71) | - | 229k: 44.9 vs 37.7 |
| GSQ-RCO IQ3_XXS | 66.3 (.68) | 49.0 (.47) | 46.9 (.54) | 37.5 (.50) | 41.2 (.74) | 34.2 (.65) | - | 229k: 44.9 vs 34.2 |

Reading:
- **The acceptance is set by the text more than by the model**: all seven models show the same zigzag over contexts (low at 28k and 112k, high at 55k, 196k, 229k), because the 512 forced tokens continue a different part of the session at each depth (code listings accept well, prose and reasoning poorly). So compare models within a column, not along a row, and read the curves as "what this session does", not as depth decay.
- Within a column the models are close: at 112k all seven sit at 35.7-38.8 tok/s under sampling (greedy matrix: 42-51), at 11k the ranking is the greedy one (STRIX / IQ3 / UDflat fastest, plain Q4_K_S and GSQ slowest) with spreads up to 20% between models.
- Sampling costs 2-25% against greedy on the same server and prompt (control cells; the 229k control of UD-Q4_K_S is only -2% because its sampled acceptance there was .90).
- Seed-to-seed noise (3 calls) reaches 10-15% within a cell; a difference below ~10% between two models in one cell is not a result.
- No driver reset, retry or failure occurred during the 7 ladders (the WSL log has one `dxgkio_escape -75` after STRIX's last cell, at shutdown); the Strata and tool-eval-bench slots were separate.

Second part (same protocol, `-c 131072`, contexts to 112k; greedy control = the same server, pi113): decode tok/s (acceptance) under thinking sampling.
| setup | 11k | 28k | 55k | 112k | greedy at 112k |
|---|---|---|---|---|---|
| STRIX, no speculation | 38.1 | 36.2 | 33.3 | 28.3 | 28.2 |
| UDflat, no speculation | 36.8 | 35.0 | 32.4 | 27.5 | 27.4 |
| UD-Q4_K_S, no speculation | 36.7 | 34.9 | 32.1 | 27.3 | 27.3 |
| plain Q4_K_S, no speculation | 36.1 | 34.3 | 31.5 | 26.9 | 26.9 |
| Qwopus Q4_K_M, own MTP head | 61.8 (.81) | 43.0 (.50) | 49.3 (.72) | 32.5 (.50) | 37.7 |
| STRIX, DFlash2 | 71.5 (.77) | 47.1 (.50) | 47.9 (.56) | 36.9 (.52) | 44.0 |
| UDflat, DFlash2 | 70.9 (.83) | 47.6 (.54) | 58.7 (.73) | 38.1 (.55) | 41.8 |
| UD-Q4_K_S, DFlash2 | 65.7 (.80) | 45.4 (.53) | 50.9 (.68) | 33.7 (.52) | 41.4 |
| plain Q4_K_S, DFlash2 | 59.1 (.82) | 41.8 (.55) | 42.6 (.61) | 33.5 (.56) | 37.5 |
- Without speculation sampling changes nothing (28.3 vs 28.2 tok/s), as expected; the speculative speed-up is what sampling reduces. At 112k the sampled MTP numbers above (35.7-38.8) are 31-42% over these no-speculation values (26.9-28.3), against 51-76% for greedy.
- DFlash2's acceptance under sampling equals the MTP head's (.50-.56 at 28k/112k), so sampling does not change the earlier ranking: MTP n3 is ahead of DFlash2 by 5-11% at 11k, and the two are within noise at 112k (UDflat 38.3 vs 38.1, STRIX 38.8 vs 36.9, UD-Q4_K_S 37.2 vs 33.7, plain Q4_K_S 37.0 vs 33.5).
- Qwopus is the slowest at every depth (32.5 at 112k).

### 13.11 tool-eval-bench (69 scenarios + 23 Hard Mode, thinking on, seed 42, one run per cell), score /100

Benchmark of Ivan Fioravanti's post (SeraphimSerapis/tool-eval-bench); server = `run_model.sh` (MTP n3, fused, `-c 131072`), `bench/decode_investigation/teb_models.sh` / `teb_models2.sh`. Greedy = temperature 0; thinking = temperature 1.0, top_p 0.95, top_k 20, min_p 0 (Qwen's thinking-mode settings). Raw summary lines: `bench/decode_investigation/tool_eval_summary_2026-10-06.txt`; traces in WSL `~/tool_eval/runs`.
| model | greedy | thinking sampling | note |
|---|---|---|---|
| UD-IQ3_XXS | 95 | 91 | |
| plain Q4_K_S | 93 | 90 | |
| UDflat | 91 (4 timeouts excluded) | 86 | |
| STRIX | 90 | 89 (safety-capped) | |
| UD-Q3_K_XL | 90 | 90 | |
| UD-Q4_K_S | 89 | 90 | |
| Qwopus Q4_K_M | 89 | 86 | |
| GSQ-RCO IQ3_XXS | 88 (safety-capped) | 92 | |
All eight sit within 86-95 and flip order between the two sampling modes: one run per cell (and a safety cap that moves a model a full star level) cannot separate them, and the benchmark does not reproduce the aggregate quality ranking of section 14 (UD-Q4_K_S 85.2 ... IQ3 variants 44-46). Either the tool-calling scenarios are too easy to resolve the quantization differences, or the KLD metrics overstate them for this kind of task; repeats (3+ seeds) and harder, longer tasks (plan Part 2) are needed before drawing a conclusion.

Repeats with other seeds (thinking-mode sampling, `teb_models2.sh think43:...`), score /100:
| model | seed 42 | seed 43 | seed 44 | range |
|---|---|---|---|---|
| UDflat | 86 | 95 | 90 | 9 |
| UD-Q4_K_S | 90 | 92 | 95 | 5 |
| STRIX | 89 | 91 | - | 2 |
| plain Q4_K_S | 90 | 87 | - | 3 |
The same model moves by up to 9 points between seeds, as much as the whole spread between models, so no ranking can be read from a single run. **Strata 0.1.40** (Qwen3.8-Flash-Next Coder IQ1_M, 125B MoE, `STRATA_PREFILL_STREAM_MIN=65536`, `-c 131072`, seed 42, one run each): greedy 89 (2179 s), thinking sampling 88 (2404 s); `bench/strata_community/README.md` for the setup.

### 13.12 Vulkan on Windows (the fork's Vulkan build) and the q4_0 perplexity with the fused attention (2026-10-31)

**Vulkan build.** Our fork (branch `merge-upstream-2026-10-30`, commit `e3f26c928`) built with MSVC 2022 and the Vulkan SDK 1.4.363.0 (`bench/decode_investigation/build_win_vk.bat`; the build needs the two-line linkage patch `notes/win_vk_msvc_rocmfp4_linkage.patch`), run on Windows with the RX 7900 XTX (Vulkan0, 24560 MiB, KHR_coopmat, AMD proprietary driver; Adrenalin clocks, not the capped ones of the WSL runs). It contains the fork's MTP and DFlash2 code but **not** the fused attention (that kernel is CUDA/HIP only and `GGML_CUDA_FATTN_TILE_Q4` is ignored). The ROCm side is WSL `~/llama-up/build-up` (printed build `5b0c0af3b`, 12075). The two builds differ in time and OS, so the comparison is indicative, not a controlled A-B.

`llama-bench`, UDflat, `-ngl 99 -fa 1 -p 512 -n 128 -r 3` (`bench/decode_investigation/vk_vs_rocm_udflat_20261031.txt`):

| KV (K/V) | Vulkan pp512 | ROCm pp512 | Vulkan tg128 | ROCm tg128 |
|---|---|---|---|---|
| f16 / f16 | 906.7 | 1122.1 | 45.3 | 39.3 |
| q4_0 / q4_0 | 896.1 | 1134.9 | 44.8 | 38.1 |

Server, UDflat, q4_0 KV, `-c 131072`, `-b 2048 -ub 512`, `--spec-draft-n-max 3`, greedy, 256 tokens, pi16 / pi32 / pi113 prompts (11,376 / 27,688 / 111,217 tokens), pass 2 (`ab_vk_win.ps1`, results in `bench/decode_investigation/ab_vk_win_20261031/`); the ROCm columns are the A-B of `ab_dflash_q8_20261031_summary.txt` (build-up, fused attention on):

| prompt | Vulkan MTP decode / prefill | ROCm MTP | Vulkan DFlash2 (Q4_K_M) | ROCm DFlash2 (Q4_K_M) |
|---|---|---|---|---|
| 11k | **93.1** / 757 | 87.4 / 962 | 81.7 / 751 | 80.6 / 897 |
| 28k | 68.3 / 668 | 68.7 / 914 | 68.0 / 669 | 70.5 / 861 |
| 111k | 48.1 / 404 | **51.7** / 659 | 44.2 / 414 | 49.7 / 642 |

Draft acceptance on the last request: Vulkan MTP 0.732 (mean length 3.19), DFlash2 0.624 (2.87). Prefill on Vulkan is 21 % (11k), 27 % (28k) and 39 % (111k) below ROCm. **MTP plus DFlash2 at the same time does not work**: `--spec-type draft-mtp,draft-dflash` has one draft model (`-md`) and one draft context; with the MTP head file it stops at `GGML_ASSERT(target_layer_ids_n > 0 ...)`, with the DFlash2 file at `context type MTP requested but model doesn't contain MTP layers`. The first run of the script (kept out of the archive) was invalid: a PowerShell variable `$r` overwrote `$R`, so only the first probe of each server ran.

**q4_0 KV perplexity with the fused attention** (`bench/decode_investigation/ppl_fused_q4.sh`, results in `ppl_fused_q4_20261031/`). `llama-perplexity --kl-divergence` against the stored Q8_0 reference logits, `-c 4096 -b 8 -ub 8` (every batch 8 query columns, the range of the fused kernel; the default `-ub 512` would not reach it), `-fa on -ngl 99`, `-ctk q4_0 -ctv q4_0`, UDflat, `LLAMA_ATTN_ROT_DISABLE=1`, build-up `5b0c0af3b`. Flag off against `GGML_CUDA_FATTN_TILE_Q4=1`; "+-" is the standard error:

| corpus | flag | mean KLD | same top-1 | RMS dp | PPL(Q) | PPL(Q)/PPL(base) | seconds |
|---|---|---|---|---|---|---|---|
| wiki_q | off | 0.0511 +- 0.0066 | 93.61 +- 0.27 % | 5.42 % | 5.7803 | 1.00415 +- 0.0051 | 206 |
| wiki_q | 1 | 0.0528 +- 0.0071 | 93.69 +- 0.27 % | 5.40 % | 5.8127 | 1.00977 +- 0.0051 | 109 |
| coding_q | off | 0.02051 +- 0.00086 | 93.52 +- 0.24 % | 4.68 % | 3.0270 | 1.00751 +- 0.0023 | 180 |
| coding_q | 1 | 0.02033 +- 0.00086 | 93.52 +- 0.24 % | 4.61 % | 3.0235 | 1.00637 +- 0.0023 | 126 |

The flag changes the numbers slightly (not bit-identical) but every difference is far inside the standard error (wiki KLD +0.0017 against 0.007; coding KLD -0.0002 against 0.0009; top-1 within 0.07 points), and the kernel was engaged (the runs took 47 % (wiki_q) and 30 % (coding_q) less time). One model, context 4096, two corpora: it says nothing about the 111k-token case, where the greedy continuation does diverge from the default path (RESULTS 13.4, `ab_stock_20261030_summary.txt`).

### 13.13 Low-level gfx1100 experiments (2026-11-01; all off by default, branch `exp/gfx1100-lowlevel` of the fork, commits 5f022887a and 365ce901d on `e3f26c928`, build dir `build-ll`)

Raw data: `bench/decode_investigation/lowlevel_20261101/`; driver `lowlevel_tests.sh`; design in `notes/gfx1100_lowlevel_2026-10-31.md`. Reference check: with every flag off, build-ll matches build-up within 1 % (`test-backend-ops perf`, q4_0, GQA 6, kv 113408, 1 / 4 columns: 671 / 1327 us against 667 / 1324 us).

| item | flag | result |
|---|---|---|
| 1a more KV splits in the tile attention | `GGML_CUDA_FATTN_PBLOCKS=N` | 4 columns at 113k: 1317 us -> 1168 (N=48), 1162 (96), 1168 (192), i.e. -11 %; at 262k 3009 -> 2650 (96), -12 %; 1 column: 664 -> 603 at N=192 only (noise-level otherwise). End to end (`llama-bench -d 113408`, q4_0, flag 2, r=3): pp4 59.2 +- 7.8 -> 62.5 +- 9.1 (N=96) t/s, tg64 27.80 -> 28.02 (96), 28.90 (192) t/s, i.e. +1..4 %. Small, within the noise of pp4; not adopted, no quality check done (summation order of the split combine changes). |
| 1b two-block 32-bit loads in the fused loaders | `GGML_CUDA_FATTN_TILE_VLOAD=1` | **slower**: 674 -> 858 us (1 column), 1327 -> 1452 (4 columns) at 113k; 262k 1522 -> 1949 / 3030 -> 3315. Correctness the same as with the flag off (`test-backend-ops` FLASH_ATTN_EXT: 4108 OK / 24 failed off, 4107 / 25 on, where the 24 are existing f16 kv=65536 cases above the 5e-4 NMSE limit, the 25th is the summary line). The loaders are latency/occupancy bound, not instruction bound. Dropped. |
| 2 small-op fusions | measurement only | eager decode 19.2 t/s vs 38.3 with HIP graphs: the graph halves the step, so an eager per-op profile (57 ms / graph, `opprof_decode.txt`) cannot tell launch gaps from kernel time; the matmuls (iq4_xs 5120x17408 9.1 ms, 17408x5120 3.4, ...) dominate. No fusion was written. |
| 3 MMQ tile width | `GGML_CUDA_MMQ_JMAX`, `_JFORCE` | the default choice is the best at every point (pp512 / pp2048 at -ub 512: default 1062 / 1120 t/s; JMAX=64 1050 / 1090; JMAX=96 997 / 1027; JFORCE=48 973 / 997; JFORCE=64 1060 / 1087; JFORCE=96 1007 / 1029; -ub 1024 the same order). Dropped. |
| 4 Vulkan without the f16 scratch of a quantized K/V | `GGML_VK_FA_NO_DEQUANT_KV` | no effect: pp512 904.95 -> 901.80 t/s, @d16384 666.3 -> 667.7, @d65536 384.6 -> 385.7 (`llama-bench`, q4_0 KV, UDflat). Not worth a perplexity run. |

Conclusion: nothing here is a lever for the 262k decode gap; the only positive is a few percent from more KV splits. The fused tile kernel is not limited by instruction count or tile width; a software-pipelined (prefetching) main loop remains the untried idea.

## 14. QUALITY METRICS: what they mean, what we measured, what the vendors declare

### 14.1 Metric glossary

All quality numbers are per-token comparisons of a **candidate** model against a
**reference** model on the same corpus, computed by
`llama-perplexity --kl-divergence-base` via `scripts/rx7900xtx/quality-kld.sh`.

| metric | what it actually measures | direction | why it is here |
|---|---|---|---|
| `mean_kld` | Mean per-token KL divergence between the candidate's and the reference's next-token distributions. Captures *how differently* the two models spread probability, not just which token wins. | lower = closer to reference | the primary "did quantization break the distribution" number |
| `kld_99.0` | The KLD at the 99th percentile of tokens. Ignores the 99% easy tokens. | lower = better | robust tail measure; the gate metric of choice (see 14.4) |
| `kld_99.9` | KLD at the 99.9th percentile. | lower = better | **unstable** - swings 0.6 -> 13.8 by corpus, driven by rare tokens where both models put ~0 probability. Do not gate on it |
| `same_top_p` | % of tokens where candidate and reference pick the **same top-1 token**. Ignores the rest of the distribution. | higher = better | the shipping gate. Directly interpretable as "how often would it say the same thing" |
| `dp_rms` | RMS difference in per-token probability, symmetric. Mean of `|p_cand - p_ref|`-style noise across tokens. | lower = better | a per-token *noise* check that is sensitive where same-top is blind (a token can keep top-1 while its probability mass moves) |
| `mean_ppl_q` | The candidate's own perplexity on the corpus. | lower = better | standalone score, no reference needed |
| `mean_ppl_base` | The reference model's perplexity on the same corpus. | - | the denominator of the ratio |
| `ppl_ratio` | `mean_ppl_q / mean_ppl_base`. | 1.0 = identical; <1 = candidate scores *better* than the reference | the intuitive "is it worse" number |

Two things these metrics do **not** cover, which matters for reading 14.2:

- **Reference is `Q8_0`, not BF16.** The vendors compare against BF16/fp16. KLD
  magnitudes are therefore *not* numerically comparable between the two tables.
- **`llama-perplexity` is a plain forward pass.** Speculative decoding (MTP,
  DFlash2) is never exercised, so nothing here validates that a 30% decode speedup
  preserves the output distribution.

### 14.2 Our measured quality matrix

Method: reference `Qwen3.8-27B-Q8_0.gguf`, `-c 4096`, corpora `wiki_q` (natural
language) and `coding_q` (own code text), commit `5de4e6983` (source-identical to
the throughput commit `a3d2c2b52` over `common/ src/ ggml/ tools/`).
Machine-readable: `~/models/quality/matrix_q8/summary_dprms.csv`.

| candidate | file measured | corpus | mean_kld | kld_99.0 | same_top_p | dp_rms | ppl_ratio |
|---|---|---|---:|---:|---:|---:|---:|
| Q4_K_S (plain) | `Q4_K_S.gguf` | wiki | 0.0468 | 0.358 | **92.71** | 5.70 | 0.990 |
| Q4_K_S (plain) | | coding | 0.0165 | 0.176 | **92.34** | 4.18 | 1.002 |
| **UD-Q4_K_S dynV3.0** | `UD-Q4_K_S_dynamicV3.0.gguf` | wiki | **0.0438** | **0.217** | **94.06** | 4.96 | 1.005 |
| **UD-Q4_K_S dynV3.0** | | coding | **0.0169** | **0.161** | **92.47** | 4.17 | 1.006 |
| UD-Q3_K_XL | `UD-Q3_K_XL.gguf` | wiki | 0.0555 | 0.552 | **92.76** | 5.94 | 1.035 |
| UD-Q3_K_XL | | coding | 0.0279 | 0.311 | **92.48** | 5.14 | 1.012 |
| IQ3_XXS (UD) | `UD-IQ3_XXS.gguf` | wiki | 0.0948 | 0.889 | 90.08 | 7.70 | 1.032 |
| IQ3_XXS (UD) | | coding | 0.0560 | 0.560 | 89.46 | 7.40 | 1.030 |
| STRIX FP4 | `ROCMFP4-STRIX.gguf` | wiki | 0.0911 | 0.804 | 89.57 | 7.58 | 0.972 |
| STRIX FP4 | | coding | 0.0425 | 0.400 | 88.19 | 6.42 | 1.025 |
| Qwopus Q4_K_M | `Qwopus...-Q4_K_M.gguf` | wiki | 0.1116 | 1.178 | 89.00 | 8.18 | 0.929 |
| Qwopus Q4_K_M | | coding | 0.0501 | 0.451 | 89.63 | 6.80 | 1.014 |
| GSQ-RCO IQ3_XXS | `GSQ-RCO-IQ3_XXS.gguf` | wiki | 0.1482 | 1.528 | 87.19 | 10.20 | 0.979 |
| GSQ-RCO IQ3_XXS | | coding | 0.0630 | 0.675 | 89.61 | 7.69 | 1.030 |

**UD-Q4_K_S dynamicV3.0 is the best local quant on every wiki metric, and it beats
the plain Q4_K_S it is often confused with.** It is the file every `UD` throughput
row in this document actually uses, so from here on "UD-Q4_K_S" means this file
and nothing else:

| wiki metric | plain Q4_K_S | **UD-Q4_K_S dynV3.0** | delta |
|---|---:|---:|---:|
| mean_kld | 0.0468 | **0.0438** | -6.4% |
| kld_99.0 | 0.358 | **0.217** | -39% |
| same_top_p | 92.71 | **94.06** | **+1.35 pt** |
| dp_rms | 5.70 | **4.96** | -13% |

It is also 0.71 GiB smaller (14.30 vs 15.01) and 0.7 t/s faster at 196608, so the
"dynamicV3.0 is a downgrade" assumption baked into earlier notes was wrong on both
axes. On coding the two are within noise (0.0165 vs 0.0169 KLD) and both clear
`same_top >= 90`. **This is the only candidate that clears Tier A on both corpora
while also being the fastest local decode**, which retires the tension in gap 14.4#8.

### 14.2.1 The shipping KV config costs almost nothing (UD-Q4_K_S dynV3.0)

Measured 2026-09-29 in `~/models/quality/matrix_dyn/`, reusing the
`matrix_q8/*.ref.kld` reference logits (symlinked, so the multi-hour reference run
was skipped) at the same `-c 4096` / `--gpu-layers 32`. `-ctk/-ctv` sets the KV
cache type, which is the `-ctk q4_0 -ctv q4_0` we actually launch with at 131072+.

| corpus | KV type | mean_kld | kld_99.0 | kld_99.9 | same_top_p | dp_rms | ppl_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| wiki | f16 | 0.0438 | 0.217 | 11.39 | 94.06 | 4.96 | 1.0046 |
| wiki | **q4_0** | 0.0461 | 0.257 | 10.54 | 93.60 | 5.02 | 1.0043 |
| wiki | q8_0 | 0.0425 | 0.230 | 9.36 | **94.35** | 4.89 | **1.0021** |
| coding | f16 | 0.0169 | 0.161 | 0.93 | **92.47** | 4.17 | 1.0062 |
| coding | **q4_0** | 0.0195 | 0.183 | 1.04 | 92.05 | 4.50 | 1.0068 |
| coding | q8_0 | 0.0170 | 0.157 | 1.07 | 92.74 | 4.26 | 1.0054 |

Reading it:

- **q4_0 KV costs 0.46 pt of same_top on wiki and 0.42 pt on coding.** That is the
  entire measured price of the KV quantisation we ship to fit 262144 ctx in 20.8
  GiB. `mean_kld` moves +5.2% (wiki) / +15% (coding). Small, and it is the only
  thing that buys the long-context budget.
- **q8_0 KV is free or better** (wiki `same_top` 94.35 > f16's 94.06) but does not
  fit at 262144, so it is a small-ctx option, not the shipping one. Treat the
  wiki "better than f16" as noise: it is within the replicate spread, and q8_0
  being better than full precision is not physically meaningful.
- `ppl_ratio` stays within 1.002-1.007 in all six cells. Nothing here degrades
  output quality in the way the brief cares about.

Raw: `~/models/quality/matrix_dyn/summary.csv` and `matrix_dyn.log`.

### 14.2.2 Shipping-KV quality for all four candidates (the figures' y-axis)

The `plots/` figures put a quality number on y and a long-context speed on x, so
they must use the KV type we actually ship, not f16. Measured 2026-09-29 in
`~/models/quality/matrix_kvq4/`, `-c 4096`, same reused `matrix_q8` reference
logits, `rc=0`. These six rows plus the two `U4DYN_k4` rows from 14.2.1 are the
complete quality input to every figure.

| candidate | corpus | mean_kld | kld_99.0 | kld_99.9 | same_top_p | dp_rms | ppl_ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| UD-Q4_K_S dynV3.0 | wiki | 0.0461 | 0.257 | 10.54 | **93.60** | 5.02 | 1.0043 |
| UD-Q4_K_S dynV3.0 | coding | 0.0195 | 0.183 | 1.04 | **92.05** | 4.50 | 1.0068 |
| IQ3_XXS (UD) | wiki | 0.0620 | 0.580 | 2.14 | 89.77 | 7.50 | 1.0335 |
| IQ3_XXS (UD) | coding | 0.0607 | 0.626 | 2.55 | 89.40 | 7.94 | 1.0335 |
| STRIX FP4 | wiki | 0.0461 | 0.409 | 1.36 | 89.19 | 6.85 | 1.0028 |
| STRIX FP4 | coding | 0.0452 | 0.441 | 1.63 | 88.02 | 6.56 | 1.0293 |
| Qwopus Q4_K_M | wiki | 0.0600 | 0.627 | 1.72 | 88.89 | 7.46 | 0.9280 |
| Qwopus Q4_K_M | coding | 0.0523 | 0.465 | 1.71 | 89.29 | 7.00 | 1.0162 |

Reading it:

- **Only UD-Q4_K_S dynV3.0 clears `same_top >= 90` on both corpora under the
  shipping KV.** The other three land at 88.0-89.8, i.e. 1-2 pt short of Tier A.
  That is the headline of the whole quality/speed exercise: the fastest decoder at
  262144 (UD MTP, 49.5 t/s) is also the only faithful one.
- IQ3_XXS has the worst KLD-to-overlap ratio of the four: `mean_kld` 0.061 is
  2nd worst but `same_top` 89.4 is 3rd. Quantisation error that does not move the
  argmax is not free, it just happens to be hidden by the top-1 metric.
- **`ppl_ratio` cannot rank these.** All eight rows sit in 0.928-1.034 and the
  ordering it implies (Qwopus "best" at 0.928, IQ3 "worst" at 1.034) is the
  opposite of the `same_top` ordering. Perplexity on a 4k context is dominated by
  bulk-text statistics, not by the tokens that matter. This is why 12.3 rejects it
  as the aggregate score and the figures give it a dedicated panel instead.
- STRIX's two corpora disagree most (89.19 wiki vs 88.02 coding, 1.17 pt). The
  other three span under 0.5 pt. STRIX is the one candidate where the single
  aggregate number in 12.3 is hiding something, so its two-corpus gap is called
  out in the aggregate caption.

Raw: `~/models/quality/matrix_kvq4/summary.csv`, `matrix_kvq4.log`.

KV-cache variants were also measured (this is the config we ship):
Q4_K_S_k4 `-ctk q4_0 -ctv q4_0` moves mean_kld 0.0468 -> 0.0518 and same_top
92.71 -> 92.59; `_k8` `-ctk q8_0 -ctv q8_0` -> 0.0430 / 93.10. STRIX_k4/k8 are
within 0.3 pt of plain STRIX. **So KV quantisation is nearly free at 4k ctx**,
and a self-comparison control (`Q4KS_same`) returns KLD 0.000 / same_top 100.0,
validating the harness.

At 32k ctx on the 113k agentic corpus (STRIX only, rotation on): same_top drops to
86.8-87.0 for all KV variants, dp_rms rises to 13.6-13.8%, and `ppl_ratio` drifts
1.221 (f16) -> 1.236 (q8_0) -> 1.314 (q4_0). Ordering is preserved, so the 4k KV
conclusion still holds, but mean log-likelihood degrades faster than token overlap
does once context accumulates.

### 14.3 Vendor-declared values (HuggingFace model cards + plot images)

Extracted 2026-09-25; full text and images in
`llama-rocmfp4-merge/results/benchmarks/EXTRACTED_BENCHMARK_DATA.md`.
**These are declared by the vendors against BF16 logits, not measured by us.**

Unsloth "Dynamic 3.0", mean KLD vs BF16 over 300 sampled documents
(`huggingface.co/unsloth/Qwen3.8-27B-GGUF`):

| quant | declared mean KLD | "base" (pre-rerank) | declared size |
|---|---:|---:|---:|
| UD-Q4_K_XL | 0.0237 | 0.0249 | 15.64 GB |
| UD-Q3_K_XL | 0.0806 | 0.0878 | 12.76 GB |
| UD-Q2_K_XL | 0.2209 | 0.2297 | 9.95 GB |
| UD-IQ2_M | 0.2582 | 0.2655 | 8.96 GB |
| UD-IQ2_XXS | 0.5210 | 0.5358 | 7.31 GB |
| UD-IQ1_M | 0.8000 | 0.8323 | 6.51 GB |
| UD-IQ1_S | 0.9729 | 1.0357 | 6.06 GB |

ISTA-DASLab GSQ-RCO (`huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`),
5-task zero-shot average, "recovery" = % of the BF16 score:

| quant | declared bpw | declared size | declared zs-avg | declared recovery | AIME25 | GPQA-D | LCBv6 | wiki PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 reference | 16.0 | 53.8 GB | 74.34 | 100.0% | 100.0 | 89.90 | 85.71 | 7.05 |
| GSQ-RCO IQ3_XXS | 3.00 | 10.1 GB | 74.81 | **100.6%** | 100.0 | 88.89 | 84.57 | 7.20 |
| GSQ-RCO IQ3_S | 3.50 | 11.8 GB | 74.47 | 100.2% | 100.0 | 89.39 | 85.71 | 7.07 |
| UD-IQ3_S (their comp row) | 3.52 | 12.0 GB | 75.49 | 101.5% | 96.67 | 89.90 | 84.00 | - |

Also declared: unsloth ">10% better top-1 vs next-best provider at the same size",
UD-Q2_K_XL "+8% top-1", UD-IQ1_S "~72-77% top-1 agreement vs fp16 (89% smaller)",
NVFP4 "92-97% accuracy recovery". tokenstead AMD scorecard: UD-Q4_K_XL 17.56 GB,
UD-Q2_K_XL 9.83 GB, UD-IQ1_S ~6 GB.

Joining declared to measured, per model we actually benchmark:

| model | declared bpw / size | our measured size (BPW) | our KLD vs Q8_0 | declared KLD vs BF16 | agree? |
|---|---|---|---|---|---|
| UD-Q4_K_S **dynamicV3.0** | ~4.5 bpw / 15.4 GB (proxy) | 14.29 GiB (4.49) | **not measured** | none published | - |
| Q4_K_S (plain, imatrix) | 4.57 / 16.1 GB | 15.01 GiB | 0.0468 | ~0.09 (UD-Q3_K_S, approx) | yes, in family |
| UD-Q3_K_XL | 3.85 / 13.1 GB (proxy) | 12.24 GiB | 0.0555 | 0.0806 | yes |
| STRIX FP4 | 4.3 / 13.8 GB | 13.74 GiB (4.39) | 0.0911 | NVFP4 92-97% recovery | yes |
| IQ3_XXS (UD) | - | 10.17 GiB (3.20) | 0.0948 | none published | - |
| GSQ-RCO IQ3_XXS | 3.00 / 10.1 GB | 9.40 GiB | 0.1482 | 100.6% task recovery | **disagree** |
| Qwopus Q4_K_M | - | 15.65 GiB (4.92) | 0.1116 | none published | - |

Where both exist, vendor KLD-vs-BF16 and our KLD-vs-Q8_0 order the files the same
way. The one outright contradiction is **GSQ-RCO IQ3_XXS**: its own card calls it
task-lossless (100.6% recovery, AIME25 identical to base) while our wiki gate
discriminates it hardest of any local quant (same_top 87.19, kld_99.0 1.528). The
model card states the files are standard GGUF needing no special runtime, so this
is a metric-vs-metric divergence (5-task recovery vs token-overlap/wiki tail), not
a runtime artifact. It is not resolved.

### 14.4 Coverage gaps in the quality data

1. **CLOSED 2026-09-29 - `UD-Q4_K_S dynamicV3.0` is now quality-measured.** It is
   the best local quant on every wiki metric (same_top 94.06, kld_99.0 0.217) and
   beats the plain `Q4_K_S` it was previously confused with; see 14.2 and 14.2.1.
   The KV variants (`-ctk/-ctv q4_0`, `q8_0`) are also measured, so the 51-55 t/s
   decode figures now have a matching KLD. Unsloth still publishes no KLD for
   `UD-Q4_K_S` (their chart has `UD-Q4_K_XL`, a different quant), so section 9's
   "~99.6% retention" remains an interpolated proxy, not a declared number.
2. **Naming hazard: "IQ3_XXS" means two different files in our own data.**
   `Qwen3.8-27B-UD-IQ3_XXS.gguf` (10.17 GiB) is what this file's throughput
   benchmarks use, KLD 0.0948 / same_top 90.08. `Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf`
   (9.40 GiB) is the GSQ-RCO build, KLD 0.1482 / same_top 87.19, and is the file
   the 100.6% "lossless" claim belongs to. They are not interchangeable.
3. **All quality is `-c 4096`** (plus one 32k STRIX run). There is no KLD, PPL or
   same_top at 131072 / 196608 / 262144 - the exact region where the Qwopus
   decode cliff lives. If that cliff has a quality component it is currently
   invisible. **The `plots/` figures are affected by this and say so in every
   caption**: their y-axis is a 4k-context number and their x-axis is a
   131k-262k speed. Closing this needs a long-context reference run against
   `Q8_0`, which is the multi-hour `llama-perplexity` job that `matrix_kvq4`
   sidestepped by reusing cached reference logits - that trick only works at
   4096.
4. **BLOCKED, not merely open - no quality under speculative decoding.** MTP/DFlash2
   are never run through `llama-perplexity`, and this build **cannot** run them
   through it. Verified 2026-09-30 on tree 4 (`F/optimization` @ `5de4e6983`,
   `build/bin/llama-perplexity`): two independent blockers, either of which
   alone is fatal.
   1. **The spec args are gated away from this example.** `common/arg.cpp`
      registers `--spec-draft-n-max`, `--spec-type`, `--spec-draft-model` etc.
      with `.set_examples({ LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER,
      LLAMA_EXAMPLE_CLI })` (lines 4144-4253). `LLAMA_EXAMPLE_PERPLEXITY` is
      not in any of those sets, so the flags are not even parsed:
      `llama-perplexity --spec-type draft-mtp ... ` fails with
      `error: invalid argument: --spec-type`. Only the KV-quantisation pairs
      `--spec-draft-type-k/-v` survive, and those only name cache types for a
      draft context the tool never creates.
   2. **The tool builds no speculative graph.** `perplexity.cpp:2050` calls
      `common_init_from_params`, which (`common/common.cpp:1436`) constructs only
      the *target* model+context. The draft context comes from
      `common_speculative_init_from_params` / `common_speculative_init`
      (`common/speculative.cpp:2600,2617`), and the only caller in the whole
      tree is `tools/server/server-context.cpp:1132,1261`. Beyond that,
      `perplexity.cpp`'s `kl_divergence()` (line 1695) is a batched teacher-
      forced pass over logits with no sampling step, so it has no accepted
      draft token whose distribution could be compared.
   Closing this needs a source change in the fork - expose the spec args to
   `LLAMA_EXAMPLE_PERPLEXITY`, then construct and drive a spec controller from
   inside `kl_divergence()` - plus a WSL rebuild, so it is filed as a follow-up
   rather than silently assumed. **Until then, treat the 14.2.2 `same_top`
   numbers as base-only; no number in this file measures quality with
   speculation actually engaged.** The section 14.2.2 rows are therefore a
   quality/speed Pareto built on an *unverified assumption* - that the numbers
   hold with speculation on, which is what actually ships. That assumption
   should be called out before any buyout decision that rests on it.
   Side observation (harmless to the shipping path, flagged for the record):
   `common/speculative.cpp:2560` logs
   `params.speculative.draft.mparams.path` as the draft model but then calls
   `llama_model_load_from_file(params.model.path.c_str(), ...)` - the target
   path. In practice `has_draft()` and the MTP-with-no-draft-file case load the
   target model anyway, so measured numbers are unaffected; the log line is
   misleading and a separate draft file would silently be ignored.
5. **Cross-run base PPL differs** (`matrix_q8` 5.7565 vs `q4ksm` 5.6886 on
   `wiki_q`), so `ppl_ratio` must not be compared across those two directories.
   **Root cause found 2026-09-30: the two directories do not share a reference.**
   `q4ks_matrix.sh:15` passes `--ref "$Q4KS"`, i.e. the `Q4_K_S` file is its own
   reference. So `q4ksm/` rows are quant-vs-**Q4_K_S**, while every other directory
   is quant-vs-**Q8_0**, and the 5.6886 figure is Q4_K_S's own wiki PPL rather
   than a second Q8_0 measurement. `q4ksm/` is therefore excluded from
   `quality_all.csv` entirely, and its `STRIX*` rows (which appear to duplicate
   `matrix_q8` at ~3% different values) are not replicate noise - they are the
   same quant measured against a different reference. Anyone who diffs the two
   STRIX sets expecting run-to-run variance is reading a reference change.
6. **`matrix_q8/summary.csv` is stale** - 2 rows (GSQRIQ3 only) and no `dp_rms`.
   Use `summary_dprms.csv` (18 rows).
7. **Provenance risk:** the reference `Q8_0`, plus STRIX and `UD-IQ3_XXS`, exist
   only in WSL `~/models`. None is on `F:`. The model that defines every KLD
   number above has no F: backup.
8. **Gate note:** the user-framed gate is Tier A `same_top >= 90` both corpora;
   Tier B 85.0-89.9 **only if** the fastest 113k config reaches >= 50 t/s; Tier C
   `< 85` reject. Qwopus lands Tier B on same_top (89.00 / 89.63) but its best
   113k figure is 33.46 t/s, which does **not** clear the >= 50 t/s buyout. Its
   recorded verdict was "Tier B class"; under the gate as literally written it
   would not ship. Flagging rather than re-deciding, since the gate is a user
   framing and this model may have been dispositioned separately.
9. **CLOSED 2026-09-29 - all four candidates now have shipping-KV quality**, so
   the figures have a y-axis. See 14.2.2. Under the shipped `-ctk q4_0 -ctv q4_0`,
   **UD-Q4_K_S dynV3.0 is the only candidate clearing Tier A on both corpora**
   (93.60 / 92.05); IQ3_XXS 89.77/89.40, STRIX 89.19/88.02 and Qwopus
   88.89/89.29 all land Tier B. Gap 4 remains unmet and is now known to be
   *unreachable with the current quality harness* rather than merely unrun: none
   of these were measured with speculation actually running, which is what ships,
   and `llama-perplexity` cannot engage speculation in this build (gap 4 above).


### 14.5 Quality-data hygiene: what the plots can and cannot claim

Added 2026-09-30 while auditing the `plots/` figures. None of this changes a
number already in 14.2; it changes what those numbers are allowed to be used for.

#### 14.5.1 The `ppl_ratio < 1.0` anomaly is a CROSS-MODEL comparison, not a bug

The figures flag `Qwopus Q4_K_M` at `ppl_ratio 0.9721` - nominally better than the
model it is compared against - which is why `ppl_ratio` was rejected as the
aggregate metric. The mechanism is now known, and it is not noise.

`~/models/quality/qwopus_q8/PROVENANCE.md:25-26` records:

    ref model  : /home/thegiallo/models/Qwen3.8-27B-Q8_0.gguf
    candidate  : Qwopus3.8-27B-Flash-MTP-Q4_K_M.gguf

**These are different models.** `STRIX`, `UD-Q4_K_S`, `UD-Q3_K_XL`, `UD-IQ3_XXS`,
`Q4_K_S` and `GSQ-RCO IQ3_XXS` are all quantisations of the *same* Qwen3.8-27B
base and are compared against its Q8_0 - a like-for-like measurement of
quantisation error. Qwopus is a different release, so its row measures
(fine-tune delta + quantisation error), not quantisation error alone.

That accounts for the anomaly directly: on `wiki_q` Qwopus scores PPL 5.3461
against the reference's 5.7565, a genuine 7.2% *lower* perplexity, because it is
simply a different model; on `coding_q` it is 1.4% worse. Both numbers are
correct measurements of a correct quantity - they just do not measure
quantisation error.

#### 14.5.2 Consequence: the Qwopus quality row is not rankable, and its Tier B verdict is unsafe

Because a different fine-tune will disagree with the stock model's argmax
*regardless of how well it is quantised*, `same_top 88.996 / 89.634` is biased
**downward** for Qwopus. So:

* The quality axes in the `plots/` figures mix one cross-model measurement with
  six same-base measurements. Qwopus must not be ranked against the others on
  KLD, `same_top`, `dp_rms` or `ppl_ratio` as if the rows were comparable.
* The recorded **Tier B class** verdict for Qwopus rests on those numbers. If the
  bias is real, Qwopus may clear Tier A once measured against its *own* base. It
  is currently dispositioned on a metric that penalises it for being a different
  model, which should be re-checked before any buyout decision.
* Closing it needs a `Qwopus3.8-27B` Q8_0 (or BF16) reference, which **does not
  exist locally**. Fixing this therefore needs a model download and explicit
  approval - it cannot be closed offline.

#### 14.5.3 Two different quality bases coexist, and are reconciled here

The figures average `wiki_q` and `coding_q`; several 14.3 numbers quote `wiki_q`
alone. This is not an inconsistency, it is the same data at two granularities,
and both reconcile exactly against `quality_all.csv`:

| model | wiki_q KLD | coding_q KLD | 2-corpus mean | 14.3 quotes |
|---|---:|---:|---:|---:|
| Q4_K_S plain | 0.0468 | 0.0165 | 0.0316 | 0.0468 (wiki) |
| UD-Q3_K_XL | 0.0555 | 0.0279 | 0.0417 | 0.0555 (wiki) |
| STRIX fp4 | 0.0911 | 0.0425 | 0.0668 | 0.0911 (wiki) |
| UD-IQ3_XXS | 0.0948 | 0.0560 | 0.0754 | 0.0948 (wiki) |
| Qwopus Q4_K_M | 0.1116 | 0.0501 | 0.0809 | 0.1116 (wiki) |
| GSQ-RCO IQ3_XXS | 0.1482 | 0.0630 | 0.1056 | 0.1482 (wiki) |

Rule for reading this file: a 14.3 KLD is `wiki_q` alone and is therefore
uniformly *worse* than the corresponding plotted value for the same file. Do not
compare a 14.3 number with a plotted number as if they were the same statistic.

#### 14.5.4 The harness is validated by its own self-check row

`q4ksm/summary.csv` contains `Q4KS_same`: the `Q4_K_S` file scored against
**itself**. It returns `mean_kld -0.000010`, `kld_99 0.000000`, `same_top_p
100.000`, `ppl_ratio 1.0014` - a file compared with itself is identical to within
0.14% on perplexity. This is why the KLD / `same_top_p` figures are trusted above
the `ppl_ratio` figures: the divergence metrics demonstrably return 0 for a known
-identical pair, whereas `ppl_ratio` carries ~0.1-0.3% noise even in the best case.
It is drawn on the figures labelled `SELF-CHECK`.

#### 14.5.5 Q8_0 is a quality reference only, never a benchmark subject

`Qwen3.8-27B-Q8_0.gguf` is `29,047,086,048` bytes = **27.05 GiB** against a 24 GiB
card. It cannot be placed in the throughput or VRAM figures at all. Every figure
that mentions it is measuring *other* files against it; it is the y-axis, never a
data point. Recorded so a later pass does not try to benchmark it and report the
OOM as a finding.

#### 14.5.6 The KV cache is allocated at FULL context, so `fill 6022` does not make the cache small

This corrects a claim the VRAM figure caption used to make. The caption said
that because every ladder run has `fill 6022`, "the KV cache only ever holds
6022 real tokens" and the figure therefore measured the cost of allocating a
large context rather than filling it. The first half is true and the
conclusion drawn from it was wrong.

`main_kv` is reported by the server at every allocation and is **exactly linear
in `n_ctx` and completely independent of how many tokens are actually in the
prompt**:

| n_ctx | main_kv | per token |
|--------|---------|-----------|
| 8192 | 144.00 MiB | 0.01758 MiB |
| 131072 | 2304.00 MiB | 0.01758 MiB |
| 196608 | 3456.00 MiB | 0.01758 MiB |
| 262144 | 4608.00 MiB | 0.01758 MiB |
| 274432 | 4824.00 MiB | 0.01758 MiB |

The coefficient is `16 layers x 2 (K and V) x 32 head-dim x 0.5 byte` (q4_0) =
`512 B/token/layer` x 16 / 1024 = 0.01758 MiB/token, which reproduces the
measured value exactly. This is the expected llama.cpp behaviour: the KV cache
is **preallocated to `n_ctx` at context creation and never grown**. So a 6k
prompt into a 262144-token context still reserves the full 4.5 GiB, and no
speed test in this repo was ever running with a small KV cache.

What `fill 6022` actually hides is therefore the *compute* cost of the tokens
present, not the memory cost. That data does exist - `results.txt` holds 15
distinct prompt lengths across 253 runs - but the long-prompt runs are tagged
`FO9_262k_f130`-style and carry **no VRAM scrape**, so they never entered the
ladder:

| fill | runs | prefill t/s | decode t/s |
|------|------|-------------|-------------|
| 6022 | 151 | 188-907 | 27.8-64.0 |
| 12368 | 54 | 37-846 | 12.0-58.7 |
| 20651-116841 | 10 | 489-784 | 2.2-39.6 |
| 129945 | 11 | 448-570 | 6.4-31.5 |
| 149949-199626 | 14 | 130-543 | 3.5-24.8 |

At the same 262144 context, going from `fill 6022` to `fill 129945` costs
**decode roughly 2x** (50-54 t/s for the MTP ladder at fill 6022 vs **23.6-31.5**
for the fill-129945 runs) while **prefill largely overlaps** (188-804 vs 448-570).
That is the expected shape: prefill is bandwidth-bound on the weights and
attention is a small share of it at 262k, whereas decode must read the KV cache,
which grows with the prompt. The traffic arithmetic supports it - at fill 6022
the KV in use is 106 MiB, **0.7%** of the 14.3 GiB of weights and therefore
essentially free, while at fill 129945 it is 2284 MiB, **15.6%** of weight
traffic, and at fill 199626 it is 3509 MiB, **24.0%**.

**One outlier must not be used as the fill penalty.** `FO9_262k_f130_on` returns
6.41 t/s and `FO9_262k_f150_on` 7.81, against 23.6-31.5 for the other ten
fill-129945 runs. Those two are the *only* runs at ctx 262144 in that family and
they are the earliest tags in it, so fill is confounded with both ctx and
session. An earlier draft of this section quoted 6.41 as the cost of raising
fill ("~8x"); that was wrong and is retracted. Whether those two runs are a real
ctx-262144 long-fill cliff or the same unexplained cluster as the prefill
bimodality in 14.5.7 is **not resolvable from this data** and needs the C1/C3
instrumented rerun.

##### Allocated context does NOT slow decode; only populated tokens do

This is the direct answer to the question the KV numbers raise, and the two
observations are consistent once you separate *allocated* from *populated*.
Decode at `fill 6022` is flat across every context measured, while the
allocated cache grows 32x:

| model | 8192 | 32768 | 65536 | 131072 | 196608 | 245760 | 262144 |
|---|---|---|---|---|---|---|---|
| u4 decode t/s | 53.3 | 52.2 | 53.4 | 51.3 | 51.9 | 61.2 | 54.2 |
| u4 main_kv MiB | 144 | 576 | 1152 | 2304 | 3456 | 4320 | 4608 |
| iq decode t/s | 33.0 | 33.6 | 32.6 | 32.9 | 32.7 | - | 33.9 |
| st decode t/s | 61.1 | 64.0 | 59.5 | 60.0 | 60.3 | - | 62.3 |
| qw decode t/s | 42.6 | 44.5 | 42.5 | 43.2 | 44.5 | - | 37.3 |

Mechanism: attention reads the **populated prefix**, not the buffer. A
262144-token context holding a 6022-token prompt reads the same 6022 entries as
an 8192-token context holding the same prompt, so a 32x larger reservation
costs address space and nothing else. The quantity that degrades decode is the
number of tokens *in* the cache, i.e. `fill`.

So the speed-vs-ctx ladder holds `fill` fixed at 6022 precisely because `ctx` is
not the variable that costs decode throughput - and the flatness in
`speed_output_vs_ctx.svg` is the expected result, not evidence that context size
is free. It is free **for decode**, at a real cost in **VRAM** (the 23/24 GiB
lines in `vram_vs_ctx.svg`), and it stops being free the moment the context is
actually filled.

The VRAM figure is therefore valid for what it measures - process-allocated
VRAM as a function of `n_ctx` - and its 23/24 GiB lines are the right place to
read off the headroom. It is the fill-6022 caveat that was mis-stated, and the
caption now says the cache is allocated at full ctx with the per-token
coefficient attached.

#### 14.5.7 The prefill "collapse" is bimodal, not a ctx threshold

While regenerating the figures, `replicates.csv` exposed that
`UD-Q4_K_S + MTP @ ctx 262144` was run six times with default batch settings and
returned prefill **188, 188, 189, 209, 745, 804** t/s - two clusters ~4x apart
with nothing between. Consequences:

* The single prefill point for that cell in `plot_data.csv` is chosen by tag
  order, i.e. arbitrarily. `pp_spread_pct` was added so this is visible per cell.
* The collapse is **not monotonic in ctx**: the FO50-FO54 ladder shows 260608
  fast (801), then 260864-261632 slow (~206), then 262144 fast again, then
  263168+ slow again. No threshold on ctx produces that.
* It is **not batch or ub tuning**: every FO47 variant - ub128, ub2048, b512,
  b4096+ub2048 - lands in the slow cluster (205.9-208.0).
* It **tracks session and moves decode with it**: FO47 is slow, later FO52/FO53
  are fast, and the slow runs also return lower `tg` (49-54 t/s vs 61-63 t/s). A
  KV-allocation bug would hit prefill only; a uniformly slower run points at
  clock/thermal or GPU state. `amd-smi` / `rocm-smi` / hwmon are all unavailable
  on this WSL GPU-PV path, so this remains a correlation.

This supersedes the framing in 13.2.1 that the collapse "reappeared at
265216-268288 depending on boot": it appears at *any* ctx in both states.
Figure `plots/speed_prefill_collapse_bimodal.svg` shows all 34 runs.

#### 14.5.8 Reproducibility

`bench/sweep/build_plot_data.py` now emits four files and states its own
integrity rules:

| file | rows | contents |
|---|---:|---|
| `plot_data.csv` | 54 | systematic ctx ladder, one row per cell, quality attached |
| `collapse_data.csv` | 23 | FO50-FO54 cliff bisection probes (ctx 245760-274432) |
| `replicates.csv` | 139 | every fill-6022 run, undeduplicated |
| `quality_all.csv` | 27 | every locally measured candidate; `q4ksm` excluded |

The ladder/collapse split exists because the bisection probed 21 extra contexts
for one model only; merging them would put a single model on x-values the other
three never visit. `quality_all.csv` excludes `q4ksm/` per gap 5 and carries a
duplicate-name guard that reports any candidate claimed by two summary files.

##### The aggregate quality metric uses all five metrics, not two

The `quality_AGGREGATE_vs_speed_*` figures previously plotted `same_top_p` alone,
which is one metric wearing the name "aggregate". They now plot a composite over
**all five**: `same_top_p`, `mean_kld`, `kld_99`, `dp_rms` and
`|ppl_ratio - 1.000|`. Each is first mapped to a common 0-100 "percent of the best
candidate" scale, then averaged with equal weight (`composite_scores()` in
`make_plots.py`).

Two details matter:

* **Anchored on the best candidate, not min-max rescaled.** A min-max composite
  scores the *winner* of a metric 0 and the *loser* 100, which inverts a
  higher-is-better axis - an earlier draft of this did exactly that and ranked
  `U4DYN` (best `same_top_p` in the set) at 0.0. Ratio-to-best is monotone.
* **`ppl_ratio` enters as `|ratio - 1.000|`.** A candidate *nominally better*
  than its reference - `Qwopus Q4_K_M` at 0.9721 - must not score well for that.

| candidate | all-5 composite | excluding `ppl_ratio` |
|---|---:|---:|
| U4DYN (UD-Q4_K_S dynV3.0) | **85.2** | **100.0** |
| Q4KS (Q4_K_S plain) | 78.5 | 89.6 |
| STRIX (fp4) | 67.5 | 59.3 |
| Q3KXL (UD-Q3_K_XL) | 60.8 | 74.6 |
| IQ3XXS (UD-IQ3_XXS) | 45.5 | 55.8 |
| GSQRIQ3 (non-UD IQ3) | 44.5 | 47.9 |
| QWOPUS (cross-model) | 44.5 | 54.4 |

**The `ppl_ratio` weight is load-bearing for exactly one pair.** STRIX and Q3KXL
swap order depending on whether it is included, because STRIX's `ppl_ratio` is
the closest to 1.000 of any candidate while Q3KXL's is the furthest (1.033). That
single metric decides their relative rank, and it is the metric §14.5.1-2 flagged
as least trustworthy (it straddles 1.000, and one row is a cross-model
measurement). **Treat the STRIX/Q3KXL order as unresolved, not as a finding.**
Everything else is insensitive to the weighting, because the four divergences
track each other closely.

##### Why the non-UD IQ3 was missing from the quality plots, and how it is fixed

`GSQ-RCO IQ3_XXS` is a *different release* from `UD-IQ3_XXS` - not another
quantisation of the same weights - and it has always been measured for quality.
It was invisible because every quality figure was driven by `plot_data.csv`,
which exists only for models that have a **throughput** run, and no speed run has
been done for it (GPU-gated, deferred). The same is true of `UD-Q3_K_XL` and
`Q4_K_S plain`.

`quality_all.csv` is the quality-side source and carries 14 candidates. The new
`quality_vs_size_*.svg` figures are built from it and need no speed at all, so
all 7 f16-KV candidates now appear in the quality plots, including the non-UD
IQ3. The quality-vs-speed figures still show only 4 models, and that limit is
real: it reflects missing throughput, not missing quality data.
