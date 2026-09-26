# Cloud review: Phase Q round 2 (gate tiers, UD-Q3_K_XL depth fit, 32k KV run)

Reviewed `bb42eb2`…`2827d4d`.

## 1. UD-Q3_K_XL beats STRIX on speed as well, at every context

| depth | UD-Q3_K_XL q4_0 KV | STRIX q4_0 KV (09-24, rot off) | STRIX q4_0 KV (09-25, rot on) |
|---|---|---|---|
| fixed part | **25.15 ms** | 28.06 ms | 27.07 ms |
| 0 | **39.0 t/s** | 35.1 t/s | — |
| 131072 | **22.7 t/s** | 20.8 t/s | — |

(UD-Q3_K_XL q8_0 KV: 24.62 ms fixed, 23.8 t/s @131072.)

UD-Q3_K_XL is smaller (13.1 vs 13.8 GB), closer to the reference (Tier A vs Tier B), and faster
at short and long context. On this data STRIX has no remaining advantage. The Tier B "burst
buyout" still formally lets STRIX through at ≤ 16k, but UD-Q3_K_XL + DFlash2 at the same burst
lengths hasn't been measured and is likely at least as fast. **Request:** one burst
measurement (e.g. 4.5k context, same prompt class as the STRIX 43.7 / 60.4 t/s rows) for
UD-Q3_K_XL + DFlash2 before any config keeps STRIX for bursts.

## 2. The 32k run is a broken measurement, not a long-context degradation - RESOLVED

Every discriminating test was run. Verdict: **the 32k KLD numbers are a harness/reference
artifact. They measure nothing about the candidates and cannot gate anything.**

Evidence (all in `results/e0/2026-09-25/longctx_kv_round2/` and the `.kld` decode):

1. **Q4_K_S and UD-Q3_K_XL show the same blow-up.** Round-2 runs at identical `PPL_CTX=32768`
   on the same corpus:
   - Q4KS_f16: mean KLD 0.637, same-top 90.299, PPL ratio 1.624.
   - Q3KXL_f16: mean KLD 0.898, same-top 87.164, PPL ratio 1.493.
   Both Tier A candidates (clean ~0.02-0.05 KLD at 4k) jump to ~0.6-0.9 exactly like STRIX
   (0.936). Discriminating test rule 1 fails candidate-specifically: it is the setup.
2. **The per-chunk breakdown is impossible for real code.** The chunk rows are *cumulative*
   (perplexity.cpp prints `mean_and_uncertainty` over the running KLD sums, not per-chunk
   alone). Decomposing ref and every candidate to per-chunk-alone:
   - chunk1 is absurdly bad: ref 4661 (house accounting, `.kld` decode 1109), STRIX 2007,
     Q4KS 4727.
   - chunks 2-3 are absurdly good: ~1.10 / ~1.42 PPL for ref, STRIX, AND Q4KS. A PPL of 1.1
     on real mmq.cuh/mma.cuh C++ text is a measurement error, not a model property.
3. **Not corpus duplication.** Best longest-common-prefix between chunk2's scored span and the
   full 49k-token prior context is 25 tokens (chunk3: 9 tokens); position identicalness is 409
   rows of 16383. The reference does *not* leak a repeated 16k block. Scored text is normal
   CUDA code.
4. **Reference sanity at 4k is fine.** Q8_0 on the same corpus gives PPL 9.8814 (first 3
   chunks, `-c 4096`). At 32k the same reference scores chunks 2-3 at PPL ~1.1 - it is broken
   only in the long-context run.
5. **Internal save/read inconsistency.** The `.kld` file (the saved log-probs) decodes to
   chunk1 PPL 1109, while the run's own accounting said 4661 for the same chunk. The save path
   and the reported path disagree, further proof the 32k bookkeeping is corrupt.

**What is still usable from 32k:** the *within-corpus relative* order is consistent
(Q4KS 90.30 > Q3KXL 87.16 ≈ STRIX 87.00) and matches the 4k ordering of the same candidates.
The KV-cache comparison (f16 ≈ q8_0 ≈ q4_0, all ~86.8-87.0) was a *relative* comparison between
KV variants of one model, so its conclusion (KV quantization is not the driver) survives.

**What is not usable:** mean KLD, 99 % KLD, PPL ratio, Δp RMS, and any absolute tier check at
32k. Q4KS "passing" at 90.30 and STRIX "failing" at 87.00 are both scores from a broken
reference scoring path and must not gate. **The tier gate stays on the 4k matrix numbers.**
Rule 4 recommendation: confirm.

## 3. Gate framing

The 90 / 85 tiers with a 50 t/s buyout are the user's decision and are applied consistently.
One consequence to keep in view: the gate is on **same-top vs Q8_0 at 4k**, so it describes short
context. The daily load is 70–113k. Once §2 is resolved, the same tiers should be checked at 32k+
for the shipping candidates.

## 4. Minor

- The UD-Q3_K_XL run loaded the model from `/mnt/f/...` (Windows drive via WSL's 9P bridge).
  That only affects load time, not tok/s, but copying frequently used GGUFs to ext4 speeds up
  every run.
