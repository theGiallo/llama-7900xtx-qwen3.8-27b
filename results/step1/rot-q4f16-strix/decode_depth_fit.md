# Decode time vs context depth (peak 960 GB/s)

## Qwen3.8-27B-ROCMFP4-STRIX.gguf  K=f16 V=f16

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 38.25 | 26.15 |
| 32768 | 36.18 | 27.64 |
| 65536 | 33.16 | 30.15 |
| 131072 | 1.14 | 877.58 |

- fixed part: **-142.74 ms/token** -> model bytes 14.75 GB at **nan GB/s (nan% of peak)**
- per-context part: **6.681 ms per 1k tokens** (fit r2 = 0.775)

## Qwen3.8-27B-ROCMFP4-STRIX.gguf  K=q4_0 V=q4_0

| depth | tok/s | ms/token |
|---:|---:|---:|
| 0 | 35.87 | 27.88 |
| 32768 | 32.45 | 30.82 |
| 65536 | 28.59 | 34.98 |
| 131072 | 22.27 | 44.91 |

- fixed part: **27.07 ms/token** -> model bytes 14.75 GB at **545 GB/s (57% of peak)**
- per-context part: **0.132 ms per 1k tokens** (fit r2 = 0.989)

