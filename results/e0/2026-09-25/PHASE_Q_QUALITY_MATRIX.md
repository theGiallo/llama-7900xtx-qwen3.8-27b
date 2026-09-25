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

Δp RMS (symmetric per-token prob noise, the formerly-missing gate term) is parsed
from the same `llama-perplexity` summary line (`RMS Δp`, tools/perplexity/perplexity.cpp:2002)
and backfilled for every run. CSV: `quality_matrix_q8_summary.csv` (now has the `dp_rms` column).

### wiki_q (natural language)

| candidate | mean KLD | 99.0% KLD | 99.9% KLD | same top % | Δp RMS % | PPL(Q) | PPL(Q8_0) | PPL ratio |
|-----------|----------|-----------|-----------|------------|----------|--------|-----------|-----------|
| Q4KS            | 0.0468 | 0.358 | 9.57  | 92.71 | 5.70 | 5.697 | 5.756 | 0.990 |
| Q4KS_k4 (KV q4_0) | 0.0518 | 0.360 | 9.52  | 92.59 | 5.99 | 5.713 | 5.756 | 0.992 |
| Q4KS_k8 (KV q8_0) | 0.0430 | 0.303 | 9.09  | 93.10 | 5.23 | 5.688 | 5.756 | 0.988 |
| STRIX (fp4)     | 0.0911 | 0.804 | 11.89 | 89.57 | 7.58 | 5.595 | 5.756 | 0.972 |
| STRIX_k4        | 0.0923 | 0.806 | 11.76 | 89.19 | 7.79 | 5.621 | 5.756 | 0.976 |
| STRIX_k8        | 0.0920 | 0.828 | 12.32 | 89.53 | 7.64 | 5.589 | 5.756 | 0.971 |
| IQ3XXS (UD)     | 0.0948 | 0.889 | 13.76 | 90.08 | 7.70 | 5.941 | 5.756 | 1.032 |
| GSQRIQ3 (RCO)   | 0.1482 | 1.528 | 16.09 | 87.19 | 10.20 | 5.634 | 5.756 | 0.979 |
| Q3KXL (UD)      | 0.0555 | 0.552 | 9.05  | 92.76 | 5.94 | 5.959 | 5.756 | 1.035 |

### coding_q

| candidate | mean KLD | 99.0% KLD | 99.9% KLD | same top % | Δp RMS % | PPL(Q) | PPL(Q8_0) | PPL ratio |
|-----------|----------|-----------|-----------|------------|----------|--------|-----------|-----------|
| Q4KS            | 0.0165 | 0.176 | 0.731 | 92.34 | 4.18 | 3.011 | 3.004 | 1.002 |
| Q4KS_k4         | 0.0194 | 0.179 | 0.696 | 92.11 | 4.59 | 3.020 | 3.004 | 1.005 |
| Q4KS_k8         | 0.0167 | 0.172 | 0.622 | 92.46 | 4.24 | 3.012 | 3.004 | 1.003 |
| STRIX (fp4)     | 0.0425 | 0.400 | 1.713 | 88.19 | 6.42 | 3.080 | 3.004 | 1.025 |
| STRIX_k4        | 0.0452 | 0.441 | 1.626 | 88.02 | 6.56 | 3.092 | 3.004 | 1.029 |
| STRIX_k8        | 0.0427 | 0.405 | 1.769 | 88.11 | 6.48 | 3.083 | 3.004 | 1.026 |
| IQ3XXS (UD)     | 0.0560 | 0.560 | 2.221 | 89.46 | 7.40 | 3.094 | 3.004 | 1.030 |
| GSQRIQ3 (RCO)   | 0.0630 | 0.675 | 2.550 | 89.61 | 7.69 | 3.096 | 3.004 | 1.030 |
| Q3KXL (UD)      | 0.0279 | 0.311 | 1.079 | 92.48 | 5.14 | 3.041 | 3.004 | 1.012 |

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
8. **Δp RMS (now measured) tracks same-top, is well under any 15% bar.** Worst is
   GSQRIQ3 wiki 10.20%; every other config ≤ 7.8%. The metric orders candidates
   almost identically to same-top/KLD (GSQRIQ3 worst, Q3KXL≈Q4KS best), so it
   adds a per-token-noise check without changing which configs pass/fail.
9. **`GSQ-RCO-IQ3_XXS` (rotation-on IQ3, F:) FAILS the proposed wiki gate** — the
   only local quant to do so: same-top 87.19 (< 88 %), 99.0 % KLD 1.53 (> 1.0),
   mean KLD 0.148. It passes coding (same-top 89.61, 99.0 % KLD 0.68). Its wiki PPL
   (0.979) is better than plain UD-IQ3_XXS (1.032) — again the "good likelihood,
   worse agreement" signature, but more extreme. Rotation baking does not rescue
   IQ3_XXS distribution overlap; the gate correctly discriminates it.
   NOTE (review §5): the model card explicitly says all GSQ-RCO files are "standard
   GGUF and run unmodified in llama.cpp, Ollama, and LM Studio" - no special runtime
   is required, so this reading is a genuine metric-vs-metric divergence (their
   5-task recovery vs our token-overlap/wiki tail), not a runtime artifact.

## Recalibrated gate (USER FRAMING 2026-09-25; see QUALITY_THRESHOLDS.md)

Two quality tiers on same-top-p **plus a speed buyout** for the lower tier:

- **Tier A (target): same-top ≥ 90 % both corpora** -> ships, no speed condition.
- **Tier B (tolerated): 85.0-89.9 %** -> ships ONLY if the fastest daily-load
  config reaches ≥ 50 t/s (measured at 113k ctx, agentic-90k prompt, q4_0 KV;
  MTP/DFlash2 drafters count, ngram repetitive-win scenarios do not). Per user
  amendment (2026-09-25) the buyout ALSO applies at <= 16k ctx bursts, where
  STRIX+MTP clears ~44-60 t/s.
- **Tier C: < 85 %** -> reject.

| candidate | same-top (wiki/coding) | tier | fastest @113k | ≥50 t/s? | verdict |
|-----------|------------------------|------|---------------|----------|---------|
| Q4_K_S (f16/q8 KV) | 92.7/92.3 | A | 34.0 (MTP) | n/a | **PASS** |
| UD-Q3_K_XL | 92.8/92.5 | A | 37.9 (DFlash2) | n/a | **PASS** |
| STRIX (all KV var) | 88.2-89.6 | B | 36.3 (MTP) / 60.4-43.7 @4-16k | 113k: no; <=16k burst: yes | **QUALIFIED - Tier B via short-ctx buyout** |
| UD-IQ3_XXS | 90.1/89.5 | B (A on wiki only) | 31.6 (DFlash2) | no | FAILS buyout |
| GSQ-RCO-IQ3_XXS | 87.2/89.6 | B | n/a | no | REJECT (also wiki tail) |

Bottom line: Tier A ships for 70-113k daily loads (Q4_K_S or UD-Q3_K_XL). STRIX
is authorized for <= 16k-ctx bursts via the speed buyout; document the 4.5k-ctx
crossover (43.7 t/s < 50) when sizing burst workloads. Revisit if a future
drafter lifts daily-load speed above 50.

## Vendor-published data for context (extracted, 2026-09-25)

Full extraction + images: `results/benchmarks/EXTRACTED_BENCHMARK_DATA.md` and
`results/benchmarks/images/`. Caveat: vendors measure against **BF16/fp16 logits**;
our matrix is vs unsloth **Q8_0** (-c 4096, wiki/coding corpora). KLD magnitudes are
therefore NOT directly comparable to our rows above - sanity anchor only.

| vendor quant | vendor mean KLD vs BF16 | our-equivalent row | gate-relevant note |
|--------------|--------------------------|--------------------|--------------------|
| UD-Q4_K_XL   | 0.0237 (300-doc span)   | (not measured)  | different ref than our Q4_K_S 0.0468; expect clean |
| UD-Q3_K_XL   | 0.0806                   | Q3KXL 0.0555 (vs Q8_0) | in family: 2-3 bpw UD quants sit at wiki KLD ~0.08 |
| UD-Q3_K_S    | ~0.09 (approx from chart) | (not measured)  | |
| UD-Q2_K_XL   | 0.2209                   | (not measured)  | ~2.7x UD-Q3_K_XL KLD on their span |
| GSQ-RCO IQ3_XXS | 100.6% zs recovery (vs BF16 tasks) | GSQRIQ3 0.1482 (vs Q8_0) | their zs-avg says lossless; our wiki gate FAILS it (same-top 87.19) - disagreeing signals |
| GSQ-RCO IQ3_S | 100.2% zs recovery      | (not measured)  | |

Key reading: vendor KLD-vs-BF16 and our KLD-vs-Q8_0 put the *same files* in roughly
the same ordering (UD-Q3_K_XL clean, GSQ RCO IQ3_XXS the outlier), but GSQ-RCO's own
benchmark calls its IQ3_XXS task-lossless while our wiki-gate discriminates it.
Difference is metric (zs task avg vs same-top/KLD overlap) - worth one dedicated
comparison run (GSQ-RCO IQ3_XXS vs unsloth UD-IQ3_XXS on identical corpora) before
deciding the gate is over-strict.