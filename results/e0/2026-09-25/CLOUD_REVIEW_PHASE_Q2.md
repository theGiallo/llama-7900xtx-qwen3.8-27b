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

## 2. The 32k run shows a large long-context degradation that isn't KV-related

The KV comparison holds: f16 ≈ q8_0 ≈ q4_0 on overlap metrics. But the absolute numbers are a red
flag, not noise:

| STRIX | mean KLD | 99 % KLD | PPL ratio |
|---|---|---|---|
| 4k ctx (wiki / code) | 0.091 / 0.043 | 0.80 / 0.40 | 0.97 / 1.03 |
| **32k ctx, f16 KV** (agentic) | **0.936** | **19.4** | **1.221** |

- A 99th-percentile KLD of ~19 nats means ~1 % of tokens where one model gives the other's
  choice ~e⁻¹⁹ probability: the two models genuinely disagree there, rather than being slightly
  noisy.
- It is the same with **f16 KV**, so it is not KV quantization. It comes from the weights path,
  the long-context attention/DeltaNet path, or the measurement setup.
- The reference's own PPL (11.98 on the scored tokens, 19.35 full corpus) is high for
  agentic/code text that scored ~3.0 at 4k. With more context PPL should normally go **down**.
  That points at the setup or at the reference run too.

**Discriminating test (one run each, same `PPL_CTX=32768`, same corpus):**

1. **Q4_K_S vs Q8_0** (and ideally **UD-Q3_K_XL vs Q8_0**, the shipping candidates).
   - If they also show KLD ≈ 0.9 / PPL ratio ≈ 1.2 → the setup or reference is the problem
     (see 3–4).
   - If they show ~0.02–0.06 → **STRIX specifically degrades at long context**: a serious
     finding for the FP4 path and one more reason to prefer UD-Q3_K_XL.
2. Look at the **per-chunk** lines (`chunk  PPL  ln(PPL(Q)/PPL(base))  KLD  Δp RMS  same top`)
   in the logs. Please commit the text logs (not the 24 GB `.kld`). If one chunk carries the
   blow-up, check what text is there (e.g. a boundary in the concatenated agentic prompt).
3. Reference sanity: PPL of Q8_0 on the same corpus at `-c 4096` vs `-c 32768`. If 32k is worse,
   the long-context run of the reference itself is suspect (CPU/GPU split at `--gpu-layers 32`,
   or batch settings; check that `-b`/`-ub` were the same for base and candidates).
4. Same-top at 32k (86.8–87.0) is below Tier A's 90 %. Until 1–3 are done, **don't apply the
   tier gate to 32k numbers**. The Tier A picks (Q4_K_S, UD-Q3_K_XL) were only verified at 4k.

## 3. Gate framing

The 90 / 85 tiers with a 50 t/s buyout are the user's decision and are applied consistently.
One consequence to keep in view: the gate is on **same-top vs Q8_0 at 4k**, so it describes short
context. The daily load is 70–113k. Once §2 is resolved, the same tiers should be checked at 32k+
for the shipping candidates.

## 4. Minor

- The UD-Q3_K_XL run loaded the model from `/mnt/f/...` (Windows drive via WSL's 9P bridge).
  That only affects load time, not tok/s, but copying frequently used GGUFs to ext4 speeds up
  every run.
