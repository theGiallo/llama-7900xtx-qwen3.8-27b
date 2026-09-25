# Qwen3.8-27B - vendor-published benchmark data (extracted, 2026-09-25)

Text/number extractions from the model web pages + full-resolution plot images.
Collected by the machine agent; the plots in `images/` still need a vision model
(Claude Opus 5.5) for point-by-point reconciliation with the numbers below.

Sources:
- https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (README + model card)
- https://tokenstead.ai/models/qwen3-8-27b (AMD scorecard)
- https://unsloth.ai/docs/basics/dynamic-3.0-ggufs (Dynamic 3.0 KLD chart)
- https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF (README)
- https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65 (AtomicChat 4x RTX5090 harness KLD chart)

## 1. Unsloth Dynamic 3.0 - mean KL-Divergence vs BF16 original

Mean KLD over 300 sampled documents, Qwen3.8-27B. "base" = source quantization's own
KLD before unsloth's reranking pass; lower=`closer to full model`. File sizes from
the unsloth HF model card.

| quant | mean KLD | base | size (GB) |
|-------|----------|------|-----------|
| UD-Q4_K_XL   | 0.0237 | 0.0249 | 15.64 |
| UD-Q3_K_XL   | 0.0806 | 0.0878 | 12.76 |
| UD-Q2_K_XL   | 0.2209 | 0.2297 | 9.95 |
| UD-IQ2_M     | 0.2582 | 0.2655 | 8.96 |
| UD-IQ2_XXS   | 0.5210 | 0.5358 | 7.31 |
| UD-IQ1_M     | 0.8000 | 0.8323 | 6.51 |
| UD-IQ1_S     | 0.9729 | 1.0357 | 6.06 |

Unsloth claims:
- >10% better top-1 token accuracy vs "next-best provider" at the same size.
- UD-Q2_K_XL: +8% top-1 vs next-best (tokenstead: 9.83 GB).
- UD-IQ1_S (6 GB): ~72-77% top-1 agreement vs fp16 (89% smaller than Q4).
- NVFP4 (fp4, ~half Q4 size): 92-97% accuracy recovery.
- "Not overfitting to Wikitext": KLD chart held across mixed corpora, not just wiki
  (image `unsloth_kld_notoverfit.png`).

## 2. ISTA-DASLab GSQ-RCO (rotation-on, "Gold-Backed" RCO) - 5-task ZS sweep

README `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`. `zs-avg` = mean over 5 zero-shot
evals (AIME25 / GPQA-D / LCBv6 / +2). "recovery" = % of BF16 score. Other columns:
AIME25, GPQA-Diamond, LCBv6; last col = wikitext-2 PPL where reported.

| quant | bpw | size (GB) | zs-avg | recovery | AIME25 | GPQA-D | LCBv6 | wiki PPL |
|-------|-----|-----------|--------|----------|--------|--------|-------|----------|
| BF16 (reference)     | 16.0 | 53.8 | 74.34 | 100.0% | 100.0 | 89.90 | 85.71 | - |
| GSQ-RCO IQ3_XXS      | 3.00 | 10.1 | 74.81 | 100.6% | 100.0 | 88.89 | 84.57 | 7.20 |
| GSQ-RCO IQ3_S        | 3.50 | 11.8 | 74.47 | 100.2% | 100.0 | 89.39 | 85.71 | 7.07 |
| UD-IQ3_S (unsloth, for comparison) | 3.52 | 12.0 | 75.49 | 101.5% | 96.67 | 89.90 | 84.00 | - |

GSQ-RCO claims:
- IQ3_XXS at 10.1 GB **matches BF16 on AIME25** (100.0) and zs-avg is *above* BF16
  (74.81 vs 74.34).
- IQ3_S is "task-lossless" (no task under BF16 by >0.5; ~1-pt better than stock
  single-quant IQ3_S).
- wiki PPL: 7.20 (IQ3_XXS) / 7.07 (IQ3_S) vs 7.05 for the (fp16/base) model.

## 3. tokenstead.ai AMD scorecard (numbers: see images for charts)

- UD-Q4_K_XL amd: 17.56 GB.
- UD-Q2_K_XL amd: 9.83 GB.
- UD-IQ1_S amd: ~6 GB.
- Claims repeated from unsloth (top-1 vs next-best, IQ1_S agreement band).

## 4. AtomicChat cross-publisher KLD (4x RTX5090 harness, Qwen discussion 65)

Chart `atomicchat_crosspublisher_kld_chart.png` from cdn-uploads.huggingface.co
(Qwen/Qwen3.8-27B/discussions/65). Compares provider quants by mean KLD; exact series
values require a vision model pass (clipped/rendered into the screenshot).

## 5. Open: still image-only, awaiting Claude Opus 5.5 decode

- `unsloth_divergence300_32.png` - 32-token divergence chart over 300 docs.
- `unsloth_kld_bench_1.png` / `unsloth_kld_bench_2.png` - KLD detail charts.
- `gsqrco_task_avg_vs_bitwidth.png` - zs-avg recovery vs bpw sweep.
- `gsqrco_aime25.png` / `gsqrco_gpqa.png` / `gsqrco_lcb.png` - per-task vs bitwidth.
- `gsqrco_mtp_specdec.png` - MTP/spec-decoding acceptance/speedup charts.
- `tokenstead_amd_scorecard.jpg` / `tokenstead_amd_screenshot.png` - AMD provider table.
- `unsloth_qwen38_gguf_provider_quality.png` - KLD/top-1 vs size scatter, all providers.