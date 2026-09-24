# Decode time vs context depth (peak 960 GB/s)

## Qwen3.8-27B-ROCMFP4-STRIX.gguf  K=f16 V=f16

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 37.91 | 26.38 |
| 32768 | 35.95 | 27.81 |
| 65536 | 32.87 | 30.42 |
| 131072 | 0.95 | 1056.91 |

- fixed part: **-178.35 ms/token** -> model bytes 14.75 GB at **nan GB/s (nan% of peak)**
- per-context part: **8.087 ms per 1k tokens** (fit r2 = 0.774)

## Qwen3.8-27B-ROCMFP4-STRIX.gguf  K=q4_0 V=q4_0

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 35.08 | 28.50 |
| 32768 | 30.71 | 32.56 |
| 65536 | 26.54 | 37.68 |
| 131072 | 20.82 | 48.03 |

- fixed part: **28.06 ms/token** -> model bytes 14.75 GB at **526 GB/s (55% of peak)**
- per-context part: **0.151 ms per 1k tokens** (fit r2 = 0.998)

## Qwen3.8-27B-ROCMFP4-STRIX.gguf  K=q8_0 V=q8_0

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 35.33 | 28.31 |
| 32768 | 35.26 | 28.36 |
| 65536 | 31.15 | 32.10 |
| 131072 | 24.23 | 41.27 |

- fixed part: **26.50 ms/token** -> model bytes 14.75 GB at **557 GB/s (58% of peak)**
- per-context part: **0.105 ms per 1k tokens** (fit r2 = 0.925)

