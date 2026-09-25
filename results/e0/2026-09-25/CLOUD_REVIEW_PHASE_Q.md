# Cloud review: Phase Q (quality-kld.sh, thresholds, first matrix)

Reviewed `39db893`…`80cf577`: `scripts/rx7900xtx/quality-kld.sh`, `QUALITY_THRESHOLDS.md`,
`PHASE_Q_QUALITY_MATRIX.md`, `results/benchmarks/`.

## What's good

- The harness is the right tool (`--kl-divergence-base`, per-corpus summary CSV). Using our own
  coding text alongside wiki was the right call: code KLD is ~0.4–0.6× wiki, so gates must be
  per corpus.
- Replacing 99.9 % KLD with 99.0 % KLD is sound.
- The matrix already answers a strategic question (§3).

## 1. Quick fix: Δp RMS is already printed

`llama-perplexity` prints `RMS Δp    : x.xxx ± y.yyy %` in the KLD summary
(`tools/perplexity/perplexity.cpp:2002`). Add one parse line next to the `Same top p` one:

```bash
dp_rms="$(sed -nE 's/^RMS Δp[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p' "$LOG")"
```

## 2. The KV-cache conclusion only holds at short context

The matrix ran at `-c 4096` (the script defaults to 8192). With `--kl-divergence-base`, KLD is
scored on the second half of each chunk, so every scored token saw **2–4k tokens of context**.
KV quantization error grows with the number of keys each query attends over. "KV quantization
cost is negligible" is shown for 4k context, **not for the 113k–258k this setup is for**.

Needed before relying on q4_0 KV at long context:

- One long-context run: `PPL_CTX=32768` (or 65536) on a long corpus of our own (e.g. the 113k
  agentic prompt, giving ≥ 2 chunks at 32k) for STRIX × {f16 (if it fits at that ctx), q8_0,
  q4_0} KV, plus the Q8_0 reference with the same `-c`.
- Record whether rotation was on for each KV row (`LLAMA_ATTN_ROT_DISABLE` unset/set). The
  matrix doesn't say, and q4_0 KV quality depends on it.

## 3. Strategic finding: UD-Q3_K_XL beats STRIX on quality at a smaller size

| | size | same top (wiki / code) | mean KLD (wiki / code) | base decode @113k | best spec decode @113k |
|---|---|---|---|---|---|
| STRIX (fp4) | 13.8 GB | 89.6 / 88.2 | 0.091 / 0.043 | 21.4 (fork) | 36.3 MTP (fork) |
| UD-Q3_K_XL | 13.1 GB | **92.8 / 92.5** | **0.056 / 0.028** | 21.9 (stock) | **37.9 DFlash2 (stock)** |

(Speed rows from `RESULTS_tok_per_sec.md` §6–7, same agentic 113k prompt.)

At the context lengths used, UD-Q3_K_XL matches STRIX on speed and is clearly closer to the
reference, at 0.7 GB less. Q4_K_S-level agreement, in fact. FP4's advantage would have to show up
in the **fixed part** (short context), which hasn't been measured for Q3_K_XL on this branch.

**Request:** run `scripts/rx7900xtx/run-step1.sh -k q4_0,q8_0` on UD-Q3_K_XL to get its
fixed ms/token and slope next to STRIX's 27–28 ms / 0.13–0.15 ms per 1k. If it's close, the
simplest route to "fastest at ≥ 95 % quality" is UD-Q3_K_XL + DFlash2 + q8_0 KV, with the
kernel work (attention, MMVQ) applying to it just the same.

## 4. The proposed gate is calibrated to pass what we have

The thresholds were set "so every currently-shipped passing-intent local config passes", with
STRIX "at the floor". That makes the gate describe the current candidates rather than the goal.
The user's requirement is **≥ 95 % of the main model's quality**, and the file moves the 95 % to
a 5-task spot check at an 80 % bar. That changes the goal, so it's the user's decision. Suggested
framing:

- **Primary gate (the 95 %):** task score ≥ 95 % of the Q8_0 reference's score on a fixed set of
  *our own* coding/agentic tasks. Five tasks is too coarse (one task = 20 %); use ≥ 20–30 tasks
  with deterministic checks (tests pass / exact output), same prompts, temp 0.
- **Early-warning gate (KLD metrics), anchored on a known-good quant, not on STRIX:** e.g. "no
  worse than 1.5× Q4_K_S's mean KLD and ≤ 2 points below its same-top, per corpus".
  - With the measured numbers that is mean KLD ≤ 0.070 wiki / 0.025 code and same-top ≥ 90.7 /
    90.3. **STRIX fails every term** (0.091 / 0.043, 89.6 / 88.2). UD-Q3_K_XL passes same-top on
    both corpora and wiki KLD, and misses code KLD by a hair (0.028 vs 0.025). That is exactly the
    kind of difference the gate should surface, not hide. The multiplier (1.5×, 2×) is the
    user's call.
- Keep PPL ratio [0.95, 1.05] and the 99.0 % KLD tails as sanity checks.

## 5. GSQ-RCO-IQ3_XXS: check the file's runtime requirements before trusting either number

Vendor task scores say near-lossless; our wiki same-top is 87.2 (worst in the matrix). Before
treating that as a real disagreement, check the model card: RCO-style files sometimes need a
matching runtime (a specific llama.cpp fork or version, or an online rotation / special kernel).
If our build doesn't provide it, the file will measure worse here than on its intended runtime.
Low priority, since it isn't the frontrunner, but noteworthy because it was the smallest file.

## Suggested next steps

1. Parse RMS Δp (§1), trivial.
2. Long-context KLD run for the KV types, with rotation recorded (§2).
3. UD-Q3_K_XL depth fit (§3).
4. User decides the gate framing (§4).
