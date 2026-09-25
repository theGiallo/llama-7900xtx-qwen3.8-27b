# Phase Q — quality gate (Q2, recalibrated from first full Q8_0 run)

Pass criteria for the STRIX (and any future) config, measured by
`quality-kld.sh` against the Q8_0 reference (`Qwen3.8-27B-Q8_0.gguf`, unsloth,
same tokenizer). Metrics come from the merged build's
`llama-perplexity --kl-divergence-base` mechanism.

**PENDING USER SIGN-OFF.** Values below were recalculated from the first full
8-candidate × 2-corpus matrix (see
`results/e0/2026-09-25/PHASE_Q_QUALITY_MATRIX.md`). Every current local quant
passes at these levels; STRIX passes but sits at the floor (88.0-89.6 same-top).

## Metrics (per candidate config)

- Mean KLD — lower is better. Measured: Q4_K_S 0.047/0.016 (wiki/coding),
  STRIX 0.091/0.043, UD-IQ3_XXS 0.095/0.056, UD-Q3_K_XL 0.055/0.028.
- Same top p — % of tokens whose argmax matches the reference. Measured:
  Q4_K_S 92.7/92.3, STRIX 89.6/88.2, IQ3_XXS 90.1/89.5, Q3_K_XL 92.8/92.5.
- PPL ratio — candidate mean PPL ÷ reference mean PPL (same corpus). All local
  quants in [0.97, 1.035].
- 99.0 % KLD — stable tail measure (the 99.9 % KLD was dropped: it swings
  0.6-13.8 across corpus+quant and is dominated by rare-token zeros).

## Proposed PASS gate (all must hold per corpus)

| metric           | threshold (recalibrated *)                       | rationale |
|------------------|--------------------------------------------------|-----------|
| Same top p       | ≥ 88.0 % (both corpora)                          | STRIX floor (89.6/88.2); Q4_K_S/Q3_K_XL-grade is 92-93 %; the 95 % plan figure is *not* observed for any 4-bit quant vs Q8_0 — keep it as a benchmark-score aspiration instead |
| Mean KLD         | wiki ≤ 0.10, coding ≤ 0.06                      | corpus-aware; bounds the worst tested (IQ3XXS 0.095/0.056) |
| 99.0 % KLD       | wiki ≤ 1.00, coding ≤ 0.70                      | stable tail; worst tested IQ3XXS 0.889/0.560 |
| PPL ratio        | [0.95, 1.05]                                    | all tested in [0.97, 1.035] |
| Δp RMS           | < 15 %                                          | symmetric noise check; NOT yet measured — add to harness (q3) |
| Task spot-check  | ≥ 80 % pass on 5 real coding tasks              | plan Q2; run on our own prompts; the "95 %" goal belongs here |

## Configs gated (Q3 matrix run)

Weight quant × KV type × rotation vs Q8_0, on `wiki_q.txt` + `coding_q.txt`:

- STRIX (fp4, rotation baked in) × {q4_0, q8_0, f16} KV — **PASS** at floor
- Q4_K_S × {q4_0, q8_0, f16} KV — **PASS** (f16 & q8_0 strongest)
- UD-IQ3_XXS, UD-Q3_K_XL × f16 KV — **PASS**; Q3_K_XL ≈ Q4_K_S fidelity

Decision rule: keep STRIX if it passes (it does, at the edge); otherwise fall
back to the best PASSING quant × KV variety that fits 24.5 GB VRAM at
113k-258k context (Q3_K_XL the prime fallback given 92.5-92.8 same-top).

## Open items

- Add Δp RMS to `quality-kld.sh` (needs per-token Δp in the tool output).
- Re-run STRIX vs Q8_0 with Q8_0 fully offloaded? (29 GB > VRAM; ngl-32 basis is
  CPU-mixed — acceptable, same context for base and candidates.)
- User sign-off on the gate; then fold into CI-ish nightly check if wanted.