# FLASH_ATTN_EXT bandwidth on ROCm0 (peak 960 GB/s)

Effective GB/s counts each K/V byte once (the minimum any kernel must read).
verify x = time(nb) / time(nb=1): cost of a speculative verify step relative to one token.

## K=f16 V=f16 n_kv=16384  (K+V = 67.1 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 85.6 | 784 | 82% | 1.00 | TILE |
| 2 | 140.1 | 479 | 50% | 1.64 | TILE |
| 3 | 184.9 | 363 | 38% | 2.16 | TILE |
| 4 | 186.2 | 361 | 38% | 2.18 | TILE |
| 6 | 306.1 | 220 | 23% | 3.57 | TILE |
| 8 | 319.0 | 211 | 22% | 3.73 | TILE |

## K=f16 V=f16 n_kv=65536  (K+V = 268.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 392.9 | 683 | 71% | 1.00 | TILE |
| 2 | 544.7 | 493 | 51% | 1.39 | TILE |
| 3 | 718.9 | 374 | 39% | 1.83 | TILE |
| 4 | 711.1 | 378 | 39% | 1.81 | TILE |
| 6 | 1225.9 | 220 | 23% | 3.12 | TILE |
| 8 | 1210.1 | 223 | 23% | 3.08 | TILE |

## K=f16 V=f16 n_kv=113408  (K+V = 464.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 645.5 | 720 | 75% | 1.00 | TILE |
| 2 | 929.0 | 500 | 52% | 1.44 | TILE |
| 3 | 1227.4 | 379 | 39% | 1.90 | TILE |
| 4 | 1221.5 | 381 | 40% | 1.89 | TILE |
| 6 | 2061.3 | 226 | 24% | 3.19 | TILE |
| 8 | 2122.9 | 220 | 23% | 3.29 | TILE |

## K=f16 V=f16 n_kv=262144  (K+V = 1074.3 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1509.2 | 712 | 74% | 1.00 | TILE |
| 2 | 2122.8 | 506 | 53% | 1.41 | TILE |
| 3 | 2822.8 | 381 | 40% | 1.87 | TILE |
| 4 | 2796.7 | 385 | 40% | 1.85 | TILE |
| 6 | 4641.8 | 232 | 24% | 3.08 | TILE |
| 8 | 4833.5 | 223 | 23% | 3.20 | TILE |

## K=q4_0 V=q4_0 n_kv=16384  (K+V = 18.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 158.9 | 119 | 12% | 1.00 | VEC |
| 2 | 267.5 | 71 | 7% | 1.68 | VEC |
| 3 | 387.1 | 49 | 5% | 2.44 | TILE+f16conv |
| 4 | 378.7 | 50 | 5% | 2.38 | TILE+f16conv |
| 6 | 493.4 | 39 | 4% | 3.11 | TILE+f16conv |
| 8 | 490.9 | 39 | 4% | 3.09 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=65536  (K+V = 75.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 661.9 | 114 | 12% | 1.00 | VEC |
| 2 | 1013.6 | 75 | 8% | 1.53 | VEC |
| 3 | 1471.4 | 52 | 5% | 2.22 | TILE+f16conv |
| 4 | 1456.7 | 52 | 5% | 2.20 | TILE+f16conv |
| 6 | 1996.4 | 38 | 4% | 3.02 | TILE+f16conv |
| 8 | 1954.2 | 39 | 4% | 2.95 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=113408  (K+V = 130.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1439.3 | 91 | 9% | 1.00 | VEC |
| 2 | 1982.2 | 66 | 7% | 1.38 | VEC |
| 3 | 2263.0 | 58 | 6% | 1.57 | TILE+f16conv |
| 4 | 2255.0 | 58 | 6% | 1.57 | TILE+f16conv |
| 6 | 3063.8 | 43 | 4% | 2.13 | TILE+f16conv |
| 8 | 3094.3 | 43 | 4% | 2.15 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=262144  (K+V = 302.5 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 3318.0 | 91 | 9% | 1.00 | VEC |
| 2 | 4698.7 | 64 | 7% | 1.42 | VEC |
| 3 | 5143.9 | 59 | 6% | 1.55 | TILE+f16conv |
| 4 | 4950.0 | 61 | 6% | 1.49 | TILE+f16conv |
| 6 | 6891.8 | 44 | 5% | 2.08 | TILE+f16conv |
| 8 | 6770.7 | 45 | 5% | 2.04 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=16384  (K+V = 35.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 126.7 | 282 | 29% | 1.00 | VEC |
| 2 | 260.4 | 137 | 14% | 2.06 | VEC |
| 3 | 397.8 | 90 | 9% | 3.14 | TILE+f16conv |
| 4 | 375.7 | 95 | 10% | 2.97 | TILE+f16conv |
| 6 | 512.5 | 70 | 7% | 4.05 | TILE+f16conv |
| 8 | 493.8 | 73 | 8% | 3.90 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=65536  (K+V = 142.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 468.7 | 305 | 32% | 1.00 | VEC |
| 2 | 978.3 | 146 | 15% | 2.09 | VEC |
| 3 | 1496.0 | 96 | 10% | 3.19 | TILE+f16conv |
| 4 | 1473.8 | 97 | 10% | 3.14 | TILE+f16conv |
| 6 | 1965.7 | 73 | 8% | 4.19 | TILE+f16conv |
| 8 | 1919.1 | 75 | 8% | 4.09 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=113408  (K+V = 247.0 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 807.8 | 306 | 32% | 1.00 | VEC |
| 2 | 1635.8 | 151 | 16% | 2.03 | VEC |
| 3 | 2328.1 | 106 | 11% | 2.88 | TILE+f16conv |
| 4 | 2359.0 | 105 | 11% | 2.92 | TILE+f16conv |
| 6 | 3079.0 | 81 | 8% | 3.81 | TILE+f16conv |
| 8 | 3234.8 | 77 | 8% | 4.00 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=262144  (K+V = 570.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 2352.6 | 243 | 25% | 1.00 | VEC |
| 2 | 3668.1 | 156 | 16% | 1.56 | VEC |
| 3 | 5429.4 | 105 | 11% | 2.31 | TILE+f16conv |
| 4 | 5364.3 | 107 | 11% | 2.28 | TILE+f16conv |
| 6 | 7362.6 | 78 | 8% | 3.13 | TILE+f16conv |
| 8 | 7152.9 | 80 | 8% | 3.04 | TILE+f16conv |

## Decode attention cost per token (16 full-attention layers, nb=1)

| n_kv | f16/f16 ms | q4_0/q4_0 ms | q8_0/q8_0 ms | at peak (q4_0) ms |
|---:|---:|---:|---:|---:|
| 16384 | 1.37 | 2.54 | 2.03 | 0.32 |
| 65536 | 6.29 | 10.59 | 7.50 | 1.26 |
| 113408 | 10.33 | 23.03 | 12.93 | 2.18 |
| 262144 | 24.15 | 53.09 | 37.64 | 5.04 |

