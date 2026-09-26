# FLASH_ATTN_EXT bandwidth on ROCm0 (peak 960 GB/s)

Effective GB/s counts each K/V byte once (the minimum any kernel must read).
verify x = time(nb) / time(nb=1): cost of a speculative verify step relative to one token.

## K=f16 V=f16 n_kv=16384  (K+V = 67.1 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 85.0 | 790 | 82% | 1.00 | TILE |
| 2 | 145.1 | 463 | 48% | 1.71 | TILE |
| 3 | 192.1 | 350 | 36% | 2.26 | TILE |
| 4 | 192.5 | 349 | 36% | 2.27 | TILE |
| 6 | 322.1 | 209 | 22% | 3.79 | TILE |
| 8 | 323.2 | 208 | 22% | 3.80 | TILE |

## K=f16 V=f16 n_kv=65536  (K+V = 268.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 409.5 | 656 | 68% | 1.00 | TILE |
| 2 | 556.5 | 483 | 50% | 1.36 | TILE |
| 3 | 726.9 | 370 | 39% | 1.78 | TILE |
| 4 | 727.1 | 370 | 39% | 1.78 | TILE |
| 6 | 1221.0 | 221 | 23% | 2.98 | TILE |
| 8 | 1141.7 | 236 | 25% | 2.79 | TILE |

## K=f16 V=f16 n_kv=113408  (K+V = 464.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 575.0 | 808 | 84% | 1.00 | TILE |
| 2 | 868.1 | 536 | 56% | 1.51 | TILE |
| 3 | 1146.9 | 406 | 42% | 1.99 | TILE |
| 4 | 1140.5 | 408 | 43% | 1.98 | TILE |
| 6 | 1956.0 | 238 | 25% | 3.40 | TILE |
| 8 | 1948.5 | 239 | 25% | 3.39 | TILE |

## K=f16 V=f16 n_kv=262144  (K+V = 1074.3 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1315.0 | 817 | 85% | 1.00 | TILE |
| 2 | 1973.1 | 545 | 57% | 1.50 | TILE |
| 3 | 2625.3 | 410 | 43% | 2.00 | TILE |
| 4 | 2629.3 | 409 | 43% | 2.00 | TILE |
| 6 | 4447.3 | 242 | 25% | 3.38 | TILE |
| 8 | 4492.3 | 240 | 25% | 3.42 | TILE |

## K=q4_0 V=q4_0 n_kv=16384  (K+V = 18.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 143.0 | 132 | 14% | 1.00 | VEC |
| 2 | 241.1 | 79 | 8% | 1.69 | VEC |
| 3 | 343.2 | 55 | 6% | 2.40 | TILE+f16conv |
| 4 | 338.3 | 56 | 6% | 2.37 | TILE+f16conv |
| 6 | 450.8 | 42 | 4% | 3.15 | TILE+f16conv |
| 8 | 454.3 | 42 | 4% | 3.18 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=65536  (K+V = 75.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 584.9 | 129 | 13% | 1.00 | VEC |
| 2 | 901.3 | 84 | 9% | 1.54 | VEC |
| 3 | 1361.7 | 56 | 6% | 2.33 | TILE+f16conv |
| 4 | 1344.9 | 57 | 6% | 2.30 | TILE+f16conv |
| 6 | 1799.6 | 42 | 4% | 3.08 | TILE+f16conv |
| 8 | 1835.6 | 42 | 4% | 3.14 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=113408  (K+V = 130.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1241.2 | 105 | 11% | 1.00 | VEC |
| 2 | 1829.1 | 72 | 7% | 1.47 | VEC |
| 3 | 2108.3 | 62 | 6% | 1.70 | TILE+f16conv |
| 4 | 2085.4 | 63 | 7% | 1.68 | TILE+f16conv |
| 6 | 2881.8 | 46 | 5% | 2.32 | TILE+f16conv |
| 8 | 2878.9 | 46 | 5% | 2.32 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=262144  (K+V = 302.5 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 2801.5 | 108 | 11% | 1.00 | VEC |
| 2 | 4142.4 | 73 | 8% | 1.48 | VEC |
| 3 | 4630.4 | 66 | 7% | 1.65 | TILE+f16conv |
| 4 | 4570.1 | 67 | 7% | 1.63 | TILE+f16conv |
| 6 | 6371.4 | 48 | 5% | 2.27 | TILE+f16conv |
| 8 | 6365.6 | 48 | 5% | 2.27 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=16384  (K+V = 35.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 112.7 | 317 | 33% | 1.00 | VEC |
| 2 | 233.8 | 153 | 16% | 2.08 | VEC |
| 3 | 351.1 | 102 | 11% | 3.12 | TILE+f16conv |
| 4 | 348.0 | 103 | 11% | 3.09 | TILE+f16conv |
| 6 | 460.5 | 78 | 8% | 4.09 | TILE+f16conv |
| 8 | 458.9 | 78 | 8% | 4.07 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=65536  (K+V = 142.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 434.7 | 328 | 34% | 1.00 | VEC |
| 2 | 876.9 | 163 | 17% | 2.02 | VEC |
| 3 | 1379.3 | 104 | 11% | 3.17 | TILE+f16conv |
| 4 | 1350.2 | 106 | 11% | 3.11 | TILE+f16conv |
| 6 | 1809.3 | 79 | 8% | 4.16 | TILE+f16conv |
| 8 | 1833.4 | 78 | 8% | 4.22 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=113408  (K+V = 247.0 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 733.3 | 337 | 35% | 1.00 | VEC |
| 2 | 1490.8 | 166 | 17% | 2.03 | VEC |
| 3 | 2105.2 | 118 | 12% | 2.87 | TILE+f16conv |
| 4 | 2097.5 | 118 | 12% | 2.86 | TILE+f16conv |
| 6 | 2905.9 | 85 | 9% | 3.96 | TILE+f16conv |
| 8 | 2907.6 | 85 | 9% | 3.97 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=262144  (K+V = 570.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 2096.1 | 272 | 28% | 1.00 | VEC |
| 2 | 3366.2 | 170 | 18% | 1.61 | VEC |
| 3 | 4943.2 | 116 | 12% | 2.36 | TILE+f16conv |
| 4 | 4808.0 | 119 | 12% | 2.29 | TILE+f16conv |
| 6 | 6593.8 | 87 | 9% | 3.15 | TILE+f16conv |
| 8 | 6625.4 | 87 | 9% | 3.16 | TILE+f16conv |

## Decode attention cost per token (16 full-attention layers, nb=1)

| n_kv | f16/f16 ms | q4_0/q4_0 ms | q8_0/q8_0 ms | at peak (q4_0) ms |
|---:|---:|---:|---:|---:|
| 16384 | 1.36 | 2.29 | 1.80 | 0.32 |
| 65536 | 6.55 | 9.36 | 6.96 | 1.26 |
| 113408 | 9.20 | 19.86 | 11.73 | 2.18 |
| 262144 | 21.04 | 44.82 | 33.54 | 5.04 |

