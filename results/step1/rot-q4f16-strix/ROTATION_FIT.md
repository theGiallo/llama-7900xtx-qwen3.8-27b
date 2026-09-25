# Rotation-ON depth fit — q4_0 KV (2026-09-25)

Same `run-step1.sh` battery as the 2026-09-24 STRIX run but with **no
`LLAMA_ATTN_ROT_DISABLE`** (Hadamard rotation enabled), model `q4_0_rocmfp4`, KV q4_0 and f16.

## Result: the rotation is NOT the missing ~8 ms

| metric | rotation OFF (09-24) | rotation ON (this run) |
|---|---|---|
| q4_0 fixed part | 28.06 ms/token | **27.07 ms/token** |
| q4_0 per-ctx slope | 0.151 ms/1k (r2 .998) | **0.132 ms/1k** (r2 0.989) |
| q4_0 @113k (fixed+ctx) | ~45.1 ms (raw) | ~42.0 ms |
| f16 depth fit | garbage (thrash @131k) | garbage (thrash @131k, 877 ms/tok) |

Direct same-build, same-command comparison: rotation costs **~0 to +1 ms fixed, within
run-to-run noise (±5-8 %)**, and the per-context slope is marginally *lower* with rotation on.
The kernel selection is unchanged (TILE for f16 KV; q4_0 decode still VEC per fatten log).

## Consequence

- The earlier ~35 ms fixed-part estimate (result rows 57/59) is **not explained by rotation**;
  it was some other config/branch difference. On this branch the fixed part is ~27-28 ms
  regardless.
- **Run 4-bit KV with rotation ON.** It is free and it is the quality-safe path for q4_0 KV
  (ameliorates distribution shift). q8_0 KV remains the faster choice up to ~160k (12.9 vs
  23.0 ms attention @113k, no rotation either way).
- Claude's handoff item "size the rotation cost" is now closed: negligible.