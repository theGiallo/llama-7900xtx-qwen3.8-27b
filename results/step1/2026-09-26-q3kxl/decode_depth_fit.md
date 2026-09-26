# Decode time vs context depth (peak 960 GB/s)

## Qwen3.8-27B-UD-Q3_K_XL.gguf  K=q4_0 V=q4_0

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 39.02 | 25.63 |
| 32768 | 34.01 | 29.41 |
| 65536 | 29.25 | 34.18 |
| 131072 | 22.65 | 44.14 |

- fixed part: **25.15 ms/token** -> model bytes 13.14 GB at **522 GB/s (54% of peak)**
- per-context part: **0.143 ms per 1k tokens** (fit r2 = 0.997) -> KV 18.0 KiB per token of context (16 attention layers) read at **129 GB/s (13% of peak)**; at peak it would be 0.019 ms per 1k

## Qwen3.8-27B-UD-Q3_K_XL.gguf  K=q8_0 V=q8_0

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 39.51 | 25.31 |
| 32768 | 35.85 | 27.90 |
| 65536 | 29.98 | 33.35 |
| 131072 | 23.82 | 41.99 |

- fixed part: **24.62 ms/token** -> model bytes 13.14 GB at **534 GB/s (56% of peak)**
- per-context part: **0.131 ms per 1k tokens** (fit r2 = 0.990) -> KV 34.0 KiB per token of context (16 attention layers) read at **266 GB/s (28% of peak)**; at peak it would be 0.036 ms per 1k

