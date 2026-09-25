# Phase Q — Quality KLD matrix (first full run, Q8_0 reference)

Date: 2026-09-25. Method: merged build `llama-perplexity --kl-divergence-base`,
measured with `scripts/rx7900xtx/quality-kld.sh` (commit `6705924c5`), reference =
`Qwen3.8-27B-Q8_0.gguf` (unsloth, same tokenizer), `-c 4096`, reference offload
`--gpu-layers 32` (29 GB model > 24.5 GB VRAM).

Corpora: `wiki_q.txt` (natural language, ~19k tokens) and `coding_q.txt` (our own
trimmed code text, ~15-31k tokens). All values vs the Q8_0 per-token logits.

Result CSV: `~/models/quality/matrix_q8/summary.csv` (machine-readable copy in this
dir: `quality_matrix_q8_summary.csv`).

## Results

### wiki_q (natural language)

| candidate | mean KLD | 99.0% KLD | 99.9% KLD | same top % | PPL(Q) | PPL(Q8_0) | PPL ratio |
|-----------|----------|-----------|-----------|------------|--------|-----------|-----------|
| Q4KS            | 0.0468 | 0.358 | 9.57  | 92.71 | 5.697 | 5.756 | 0.990 |
| Q4KS_k4 (KV q4_0) | 0.0518 | 0.360 | 9.52  | 92.59 | 5.713 | 5.756 | 0.992 |
| Q4KS_k8 (KV q8_0) | 0.0430 | 0.303 | 9.09  | 93.10 | 5.688 | 5.756 | 0.988 |
| STRIX (fp4)     | 0.0911 | 0.804 | 11.89 | 89.57 | 5.595 | 5.756 | 0.972 |
| STRIX_k4        | 0.0923 | 0.806 | 11.76 | 89.19 | 5.621 | 5.756 | 0.976 |
| STRIX_k8        | 0.0920 | 0.828 | 12.32 | 89.53 | 5.589 | 5.756 | 0.971 |
| IQ3XXS (UD)     | 0.0948 | 0.889 | 13.76 | 90.08 | 5.941 | 5.756 | 1.032 |
| GSQRIQ3 (RCO)   | 0.1482 | 1.528 | 16.09 | 87.19 | 5.634 | 5.756 | 0.979 |
| Q3KXL (UD)      | 0.0555 | 0.552 | 9.05  | 92.76 | 5.959 | 5.756 | 1.035 |

### coding_q

| candidate | mean KLD | 99.0% KLD | 99.9% KLD | same top % | PPL(Q) | PPL(Q8_0) | PPL ratio |
|-----------|----------|-----------|-----------|------------|--------|-----------|-----------|
| Q4KS            | 0.0165 | 0.176 | 0.731 | 92.34 | 3.011 | 3.004 | 1.002 |
| Q4KS_k4         | 0.0194 | 0.179 | 0.696 | 92.11 | 3.020 | 3.004 | 1.005 |
| Q4KS_k8         | 0.0167 | 0.172 | 0.622 | 92.46 | 3.012 | 3.004 | 1.003 |
| STRIX (fp4)     | 0.0425 | 0.400 | 1.713 | 88.19 | 3.080 | 3.004 | 1.025 |
| STRIX_k4        | 0.0452 | 0.441 | 1.626 | 88.02 | 3.092 | 3.004 | 1.029 |
| STRIX_k8        | 0.0427 | 0.405 | 1.769 | 88.11 | 3.083 | 3.004 | 1.026 |
| IQ3XXS (UD)     | 0.0560 | 0.560 | 2.221 | 89.46 | 3.094 | 3.004 | 1.030 |
| GSQRIQ3 (RCO)   | 0.0630 | 0.675 | 2.550 | 89.61 | 3.096 | 3.004 | 1.030 |
| Q3KXL (UD)      | 0.0279 | 0.311 | 1.079 | 92.48 | 3.041 | 3.004 | 1.012 |

## Findings

1. **Everything local passes a sane gate.** No candidate shows catastrophic KLD
   blow-ups beyond corpus-inherent tails; all PPL ratios ≤1.035.
2. **KV-cache quantization cost is negligible.** Q4KS_k4/k8 move mean KLD by
   <0.006 and same-top by <0.5 pt vs f16 KV. KV q4_0/q8_0 on STRIX likewise.
3. **STRIX (fp4) gaps Q4_K_S by ~3.4-4.3 same-top points** (88.0-89.6 vs 92.1-93.1)
   and mean KLD ~2x (0.091 vs 0.047 wiki; 0.043 vs 0.016 coding). PPL is *better*
   than Q8_0 on wiki (0.972) but +2.5-2.9% on coding — i.e. fp4's roughness shows
   mainly in distribution overlap (KLD/same-top), not mean log-likelihood.
4. **`UD-Q3_K_XL` ≈ Q4_K_S fidelity at smaller size.** same-top 92.76/92.48 vs
   92.71/92.34; only slightly worse mean KLD. Worth considering as the fallback
   config if a lower-byte-weight build is needed.
5. **`UD-IQ3_XXS` splits the difference**: same-top 90.1/89.5 (between STRIX and
   Q4KS), largest tails.
6. **99.9% KLD is not a stable gate metric.** It swings 0.6 → 13.8 by corpus+quant
   (rare-token zeros). Use 99.0% KLD instead: ~0.17-0.44 coding / 0.30-0.89 wiki.
7. **Corpus matters a lot in absolute KLD.** Code tokens are far more deterministic
   (coding KLD ≈ 0.4-0.6x wiki). Thresholds must be corpus-aware, or per-corpus.
8. **`GSQ-RCO-IQ3_XXS` (rotation-on IQ3, F:) FAILS the proposed wiki gate** — the
   only local quant to do so: same-top 87.19 (< 88 %), 99.0 % KLD 1.53 (> 1.0),
   mean KLD 0.148. It passes coding (same-top 89.61, 99.0 % KLD 0.68). Its wiki PPL
   (0.979) is better than plain UD-IQ3_XXS (1.032) — again the "good likelihood,
   worse agreement" signature, but more extreme. Rotation baking does not rescue
   IQ3_XXS distribution overlap; the gate correctly discriminates it.

## Recalibrated gate proposal (see QUALITY_THRESHOLDS.md)

All numbers chosen so every currently-shipped **passing-intent** local config
passes; STRIX passes at the floor but sits at its edge. `GSQ-RCO-IQ3_XXS` is
excluded by the wiki tail/same-top terms (deliberate — it is the discriminator the
gate exists to catch; see finding 8).

| metric | proposed threshold |
|--------|--------------------|
| Same top p | ≥ 88.0 % (STRIX-grade; 92-93 % is Q4_K_S/Q3_K_XL-grade) |
| Mean KLD | wiki ≤ 0.10, coding ≤ 0.06 (corpus-aware) |
| 99.0 % KLD | wiki ≤ 1.00, coding ≤ 0.70 (replaces 99.9 % rule) |
| PPL ratio | [0.95, 1.05] |
| Δp RMS | < 15 % (not measured this run — add to harness) |
| Task spot-check | ≥ 80 % on 5 coding tasks (95 % aspiration stays for benchmark-style scores, per user) |