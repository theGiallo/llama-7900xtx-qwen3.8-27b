# Phase Q — proposed quality gate (Q2)

Proposed pass criteria for the STRIX (and any future) config, measured by
`quality-kld.sh` against a high-precision reference (default: `Qwen3.8-27B-Q8_0.gguf`,
unsloth, same tokenizer). Metrics come from the merged build's
`llama-perplexity --kl-divergence-base` mechanism.

**PENDING USER SIGN-OFF.** Proposed values marked with * are initial guesses to be
recalibrated after the first full Q8_0-referenced run.

## Metrics (per candidate config)

- Mean KLD — lower is better; ~0.0014 for q8_0, ~0.031 for q4_K_M (llama.cpp scoreboard, FP16 base).
- Same top p — % of tokens whose argmax matches the reference. q4_K_M ≈ 91.9 % vs FP16 base.
- PPL ratio — candidate mean PPL ÷ reference mean PPL, on the same corpus.
- 99.0 %/99.9 % KLD and Δp RMS — tail/robustness measures.

## Proposed PASS gate (all must hold)

| metric           | threshold (initial*)                       | rationale                                    |
|------------------|--------------------------------------------|----------------------------------------------|
| Same top p       | ≥ 90.0 %                                   | q4_K_M-grade lands ~92 % vs high-precision base; 95 % (plan suggestion) held as the *target*, not a hard gate |
| Mean KLD         | ≤ 0.050*                                   | ~1.6× worse than q4_K_S-grade; recalibrate once STRIX number lands |
| PPL ratio        | ≤ 1.03                                     | bounded headroom vs reference                |
| Tail KLD         | 99.9 % KLD ≤ 10 × mean KLD + 0.1           | rejects sporadic blow-ups (rotation/kv-type bugs) |
| Δp RMS           | < 15 %                                     | symmetric noise check vs asymmetric quality loss |
| Task spot-check  | ≥ 80 % pass on 5 real coding tasks         | plan Q2 requirement, run on our own prompts  |

## Configs to gate (plan Q3 matrix)

Weight quant × KV type × rotation, all vs the Q8_0 reference, on both
`wiki_q.txt` (standard) and `coding_q.txt` (our own text):

- STRIX (fp4, rotation baked in) × {q4_0, q8_0, f16} KV
- Q4_K_S × {q4_0, q8_0, f16} KV
- UD-IQ3_XXS, UD-Q3_K_XL × {f16, q4_0} KV (low-bit reference points)

Decision rule afterwards: keep STRIX if it passes; otherwise fall back to the best
PASSING quant × KV variety that also fits the 24.5 GB VRAM at 113k-258k context.

## Notes

- The reference logits file (`.kld`) is corpus-specific and ~4-10 GB; generated once.
- Base run uses the same context as candidates so context-length bias cancels.
- KLD is defined over log-probs of the union of both vocabularies? No — same tokenizer,
  full vocab distributions compared per token (the tool stores full uint16 log-probs).