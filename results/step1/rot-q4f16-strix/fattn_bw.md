# FLASH_ATTN_EXT bandwidth on ROCm0 (peak 960 GB/s)

Effective GB/s counts each K/V byte once (the minimum any kernel must read).
verify x = time(nb) / time(nb=1): cost of a speculative verify step relative to one token.

## K=f16 V=f16 n_kv=16384  (K+V = 67.1 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 79.7 | 842 | 88% | 1.00 | TILE |
| 2 | 132.5 | 507 | 53% | 1.66 | TILE |
| 3 | 173.7 | 387 | 40% | 2.18 | TILE |
| 4 | 175.3 | 384 | 40% | 2.20 | TILE |
| 6 | 304.4 | 221 | 23% | 3.82 | TILE |
| 8 | 307.6 | 219 | 23% | 3.86 | TILE |

## K=f16 V=f16 n_kv=65536  (K+V = 268.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 353.3 | 760 | 79% | 1.00 | TILE |
| 2 | 510.6 | 526 | 55% | 1.45 | TILE |
| 3 | 678.1 | 396 | 41% | 1.92 | TILE |
| 4 | 677.9 | 397 | 41% | 1.92 | TILE |
| 6 | 1152.4 | 234 | 24% | 3.26 | TILE |
| 8 | 1157.0 | 233 | 24% | 3.27 | TILE |

## K=f16 V=f16 n_kv=113408  (K+V = 464.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 584.2 | 796 | 83% | 1.00 | TILE |
| 2 | 880.3 | 528 | 55% | 1.51 | TILE |
| 3 | 1158.7 | 401 | 42% | 1.98 | TILE |
| 4 | 1157.8 | 402 | 42% | 1.98 | TILE |
| 6 | 1967.2 | 237 | 25% | 3.37 | TILE |
| 8 | 1965.8 | 237 | 25% | 3.37 | TILE |

## K=f16 V=f16 n_kv=262144  (K+V = 1074.3 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1362.6 | 788 | 82% | 1.00 | TILE |
| 2 | 1998.0 | 538 | 56% | 1.47 | TILE |
| 3 | 2659.3 | 404 | 42% | 1.95 | TILE |
| 4 | 2681.6 | 401 | 42% | 1.97 | TILE |
| 6 | 4528.5 | 238 | 25% | 3.32 | TILE |
| 8 | 4526.7 | 238 | 25% | 3.32 | TILE |

## K=q4_0 V=q4_0 n_kv=16384  (K+V = 18.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 149.0 | 127 | 13% | 1.00 | VEC |
| 2 | 269.2 | 70 | 7% | 1.81 | VEC |
| 3 | 352.4 | 54 | 6% | 2.37 | TILE+f16conv |
| 4 | 346.5 | 55 | 6% | 2.33 | TILE+f16conv |
| 6 | 460.6 | 41 | 4% | 3.09 | TILE+f16conv |
| 8 | 458.0 | 42 | 4% | 3.07 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=65536  (K+V = 75.6 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 612.3 | 124 | 13% | 1.00 | VEC |
| 2 | 1066.6 | 71 | 7% | 1.74 | VEC |
| 3 | 1386.1 | 55 | 6% | 2.26 | TILE+f16conv |
| 4 | 1365.8 | 56 | 6% | 2.23 | TILE+f16conv |
| 6 | 1833.4 | 42 | 4% | 2.99 | TILE+f16conv |
| 8 | 1833.5 | 42 | 4% | 2.99 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=113408  (K+V = 130.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 1309.5 | 100 | 10% | 1.00 | VEC |
| 2 | 2452.3 | 53 | 6% | 1.87 | VEC |
| 3 | 2163.2 | 61 | 6% | 1.65 | TILE+f16conv |
| 4 | 2128.7 | 62 | 6% | 1.63 | TILE+f16conv |
| 6 | 2933.4 | 45 | 5% | 2.24 | TILE+f16conv |
| 8 | 2932.4 | 45 | 5% | 2.24 | TILE+f16conv |

## K=q4_0 V=q4_0 n_kv=262144  (K+V = 302.5 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 2943.2 | 103 | 11% | 1.00 | VEC |
| 2 | 4344.5 | 70 | 7% | 1.48 | VEC |
| 3 | 4770.3 | 64 | 7% | 1.62 | TILE+f16conv |
| 4 | 4602.7 | 66 | 7% | 1.56 | TILE+f16conv |
| 6 | 6455.0 | 47 | 5% | 2.19 | TILE+f16conv |
| 8 | 6411.3 | 48 | 5% | 2.18 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=16384  (K+V = 35.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 116.2 | 307 | 32% | 1.00 | VEC |
| 2 | 247.3 | 144 | 15% | 2.13 | VEC |
| 3 | 366.2 | 98 | 10% | 3.15 | TILE+f16conv |
| 4 | 356.0 | 101 | 10% | 3.06 | TILE+f16conv |
| 6 | 470.0 | 76 | 8% | 4.04 | TILE+f16conv |
| 8 | 467.8 | 77 | 8% | 4.02 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=65536  (K+V = 142.7 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 443.0 | 322 | 34% | 1.00 | VEC |
| 2 | 898.2 | 159 | 17% | 2.03 | VEC |
| 3 | 1402.5 | 102 | 11% | 3.17 | TILE+f16conv |
| 4 | 1372.9 | 104 | 11% | 3.10 | TILE+f16conv |
| 6 | 1827.1 | 78 | 8% | 4.12 | TILE+f16conv |
| 8 | 1830.3 | 78 | 8% | 4.13 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=113408  (K+V = 247.0 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 775.0 | 319 | 33% | 1.00 | VEC |
| 2 | 1490.5 | 166 | 17% | 1.92 | VEC |
| 3 | 2159.5 | 115 | 12% | 2.79 | TILE+f16conv |
| 4 | 2142.4 | 116 | 12% | 2.76 | TILE+f16conv |
| 6 | 2929.5 | 85 | 9% | 3.78 | TILE+f16conv |
| 8 | 2943.2 | 84 | 9% | 3.80 | TILE+f16conv |

## K=q8_0 V=q8_0 n_kv=262144  (K+V = 570.9 MB per layer)

| nb | us/call | GB/s | % peak | verify x | kernel |
|---:|---:|---:|---:|---:|---|
| 1 | 2235.4 | 255 | 27% | 1.00 | VEC |
| 2 | 3529.2 | 162 | 17% | 1.58 | VEC |
| 3 | 5133.9 | 111 | 12% | 2.30 | TILE+f16conv |
| 4 | 4934.9 | 116 | 12% | 2.21 | TILE+f16conv |
| 6 | 6758.0 | 85 | 9% | 3.02 | TILE+f16conv |
| 8 | 6749.6 | 85 | 9% | 3.02 | TILE+f16conv |

## Decode attention cost per token (16 full-attention layers, nb=1)

| n_kv | f16/f16 ms | q4_0/q4_0 ms | q8_0/q8_0 ms | at peak (q4_0) ms |
|---:|---:|---:|---:|---:|
| 16384 | 1.28 | 2.38 | 1.86 | 0.32 |
| 65536 | 5.65 | 9.80 | 7.09 | 1.26 |
| 113408 | 9.35 | 20.95 | 12.40 | 2.18 |
| 262144 | 21.80 | 47.09 | 35.77 | 5.04 |

