# Live MTP fattn trace — q4_0 KV everywhere (2026-09-25)

Config: `llama-server 5efe75c30` + `Qwen3.8-27B-ROCMFP4-STRIX.gguf`, draft
`mtp-Qwen3.8-27B-ROCMFP4-STRIX.gguf`, `--spec-type draft-mtp --spec-draft-n-max 5`,
`-c 262144 -ngl 99 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -np 1`,
`GGML_CUDA_FATTN_LOG=1 LLAMA_ATTN_ROT_DISABLE=1`. Raw file lines in
`fattn_lines.txt` (also `../mtp_fattn_srv.log`).

## Distinct FlashAttention configs observed

| kernel | n_q | n_kv | K/V | conv_f16 per layer | context |
|--------|-----|------|-----|--------------------|---------|
| VEC | 2 | 256 | q4_0 | no | main decode    |
| TILE | 4 | 51968 | q4_0 | **203 MiB** (f16_conv) | MTP verify batch |
| MMA_F16 | 512 | 512..32768 (65536) | q4_0 | 2..128 MiB (f16_conv) | prompt fill chunks |
| MMA_F16 | 236/512 | 51456 | q4_0 | 201 MiB | final fill chunks |

## What this confirms

1. **MTP verify with quantized KV uses n_q=4 → TILE + whole-cache f16 conversion.** At 52k
   context that is 203 MiB per layer, i.e. ~3.2 GiB across 16 layers, **for every 4-token
   verify batch**. This is the mechanism behind the MTP collapse at long context and the
   3.3x when the verify KV got faster (see `RESULTS_tok_per_sec.md` rows 57-62).
2. **Batch-1/2 main decode runs VEC on q4_0 KV** (the ~10 % of peak path from step 1).
3. **New: prompt fill with quantized KV re-converts the whole cache to f16 per chunk.**
   Fill ran at only ~104 t/s at 52k context in this config - the same whole-cache conversion
   (128-203 MiB/layer per 512-token chunk) dominates ingest too. So quantized KV + f16
   conversion is the bottleneck for fill *and* MTP verify, not just small-batch decode.

## Consequence for Phase C

The kernel must serve **every flash-attn n_q (1-8 verify/apply, up to 512 fill) with quantized
KV without ever converting the visible cache to f16**: GQA-grouped decode for batch 1-8
(all 6 query heads per KV head, one dequant per KV block) plus a batched/fill path
(n_q up to 512, flash or MMA) that reads/dequantizes quantized KV in-kernel. Until then,
quantized KV is a net loss at depth: fill ~100 t/s, MTP verify 3.2 GiB/layer-step,
decode-attn 9-12 % of peak.

## Notes

- Generation ended after 1 token in this run (EOS-looking; irrelevant to the configs logged).
- `q4_0_rocmfp4` fill at 52k: 103.3 t/s. Compare f16-KV fill (not measured here) - the
  f16-KV fill path converts nothing and runs the WMMA path, so it is much faster.