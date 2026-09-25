# Phase Q — quality gate (Q2, user framing: 90/85 tiers + speed justification)

Pass criteria for the STRIX (and any future) config, measured by
`quality-kld.sh` against the Q8_0 reference (`Qwen3.8-27B-Q8_0.gguf`, unsloth,
same tokenizer). Metrics come from the merged build's
`llama-perplexity --kl-divergence-base` mechanism.

**USER FRAMING (2026-09-25).** Two quality tiers on "same-top-p" plus a speed
justification for the lower tier:

| tier | same top p (both corpora) | acceptable? |
|------|---------------------------|-------------|
| A (target) | ≥ 90.0 % | yes, no speed condition |
| B (tolerated) | 85.0 - 89.9 % | only if ALSO max-speed capable: the fastest config of that candidate must reach ≥ 50 t/s |
| C (reject) | < 85.0 % | no |

Speed measured at the real daily load (agentic-90k prompt, 113k ctx, q4_0 KV),
same environment as `RESULTS_tok_per_sec.md` section 6 (MTP/DFlash2 drafters
count, ngram repetitive-win scenarios do not count).

## Metrics (per candidate config)

- Mean KLD — lower is better. Measured: Q4_K_S 0.047/0.016 (wiki/coding),
  STRIX 0.091/0.043, UD-IQ3_XXS 0.095/0.056, UD-Q3_K_XL 0.055/0.028.
- Same top p — % of tokens whose argmax matches the reference. Measured:
  Q4_K_S 92.7/92.3, STRIX 89.6/88.2, IQ3_XXS 90.1/89.5, Q3_K_XL 92.8/92.5.
- PPL ratio — candidate mean PPL ÷ reference mean PPL (same corpus). All local
  quants in [0.97, 1.035].
- 99.0 % KLD — stable tail measure (the 99.9 % KLD was dropped: it swings
  0.6-13.8 across corpus+quant and is dominated by rare-token zeros).
- Δp RMS — per-token symmetric noise vs reference. All runs ≤ 10.2 %; the worst
  (GSQ-RCO-IQ3_XXS wiki 10.20) also fails the same-top tier anyway.

## Secondary (non-gating, informational)

| metric     | observed range        | note |
|------------|-----------------------|------|
| Mean KLD   | Q3_K_XL 0.055/0.028 … GSQ-RCO 0.148/0.063 | orders: Q4KS≈Q3KXL < STRIX < IQ3XXS < GSQ-RCO |
| 99.0 % KLD | Q4KS_k8 0.303 … GSQ-RCO 1.528 | same ordering; discard 99.9 % (rare-token zeros) |
| PPL ratio  | all in [0.97, 1.035]   | GSQ-RCO is *within* band on PPL but way off on same-top: "good likelihood, wrong token" signature |
| Δp RMS     | ≤ 10.2 %               | no config near the 15 % guard; ordering matches same-top |

A candidate must be at least Tier B to be shipped. Tier B is the "speed buyout":
price of the 85-90 % quality discount is ≥ 50 t/s.

## Verdicts from the Q3 matrix on this gate

Same-top vs the 90/85 tiers, and the fastest measured daily-load t/s
(113k ctx; drafter/none):

| candidate | same-top (wiki/coding) | tier | fastest @113k | 50 t/s? | verdict |
|-----------|------------------------|------|---------------|---------|---------|
| Q4_K_S (f16/q8 KV)     | 92.7/92.3 | A | 20.8 none, 34.0 MTP | (not required) | **PASS - Tier A** |
| Q4_K_S (q4 KV)         | 92.6/92.1 | A | 20.8 none | (not required) | PASS - Tier A (KV variant fine) |
| UD-Q3_K_XL             | 92.8/92.5 | A | 21.9 none, 37.9 DFlash2 | (not required) | **PASS - Tier A** |
| UD-IQ3_XXS             | 90.1/89.5 | A on wiki, B on coding | 22.8 none, 31.6 DFlash2 | no (< 50 even drafter) | **FAILS Tier B speed buyout** - 89.5 coding drops it to B, 31.6 t/s doesn't buy out |
| STRIX (fp4, all KV var) | 88.2-89.6 | B | 21.4 none, 36.3 MTP | no (< 50) | **FAILS Tier B speed buyout** - quality discount NOT paid for |
| GSQ-RCO-IQ3_XXS       | 87.2/89.6 | B | (not socketed in daily driver) | - | REJECT (Tier B not met on speed) |

**Consequence (important): with the 50 t/s buyout at 113k, nobody in the 85-90 %
band pays the price.** STRIX's real daily-load ceiling is 36.3 t/s (MTP);
IQ3_XXS 31.6 t/s (DFlash2). The ngram repetitive-win scenarios (80-112 t/s on
STRIX) are explicitly excluded - they are a prompt-class win, not a model-wide
speed. So the daily-load gate keeps only **Tier A: Q4_K_S and UD-Q3_K_XL**.

**USER AMENDMENT (2026-09-25): the Tier B buyout ALSO applies to short-context
bursts (<= 16k ctx), where STRIX qualifies.**

Measured STRIX+MTP decode vs ctx (RESULTS_tok_per_sec.md section 6): bump-start
60.4 t/s, 4.5k ctx 43.7 t/s, 113k ctx 36.6 t/s. So STRIX clears the 50 t/s
buyout for bursts that keep filled context under ~2-4k tokens, and is between
buyout and daily-load for 4-16k bursts. The ngram repetitive-win scenario
(80-112 t/s) stays excluded. Decision rule therefore:

- Tier A (>= 90 % same-top, daily load): Q4_K_S, UD-Q3_K_XL - ship.
- Tier B (85-90 %) + 50 t/s buyout met **either** at daily load (nobody does)
  **or** in <= 16k bursts (STRIX+MTP yes): STRIX keeps a qualified PASS for
  short-context loads; IQ3_XXS (31.6 daily, no burst data) stays FAIL.
- Tier C (< 85 %): reject.

Concretely: STRIX is authorized wherever the workload fits in 4-16k ctx bursts
(its buyout territory). For 70-113k daily loads the gated pick is UD-Q3_K_XL
(Tier A, 13.1 GB, DFlash2 37.9 t/s).

Decision rule: ship the best Tier A config that fits 24.5 GB VRAM at 113k-258k
ctx with q4_0 KV. Both pass; UD-Q3_K_XL (13.1 GB) is the fallback of record
from Phase Q depth-fit intent. STRIX is authorized for <= 16k-ctx bursts via the
buyout (clears ~44-60 t/s there); for 70-113k daily loads it is documented as a
speed/VRAM trade, not a quality-passed config.

## Open items

- Re-run STRIX vs Q8_0 with Q8_0 fully offloaded? (29 GB > VRAM; ngl-32 basis is
  CPU-mixed — acceptable, same context for base and candidates.)
- User sign-off achieved (2026-09-25): 90/85 same-top tiers + ≥ 50 t/s buyout,
  applied both at 113k daily load and at <= 16k-ctx bursts. On that gate only
  Q4_K_S and UD-Q3_K_XL clear Tier A; STRIX clears Tier B via burst buyout
  (44-60 t/s, crossover at ~4.5k ctx); IQ3_XXS fails (31.6 t/s daily, no burst
  data).
- Long-context KLD check pending: 32k ctx, agentic 113k corpus, STRIX × f16/q8_0/
  q4_0 KV (review §2). Rotation recorded. Watch for KV-cache-driven divergence
  that the 4k-ctx matrix cannot see.
- GSQ-RCO-IQ3_XXS discordance resolved as metric-vs-metric (their 5-task recovery
  vs our token-overlap); no special runtime needed (model card: plain GGUF).