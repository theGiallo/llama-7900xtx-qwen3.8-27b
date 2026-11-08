// mmvq-rdna3-tune.h (2026-11-07/08): nwarps / rows per block of the quantized matrix-vector kernel on RDNA3 (RX 7900 XTX), per weight type and number of columns.
// Measured with bench/decode_kernel/mmvq_tune.sh (65536 x 5120 matrices, larger than the infinity cache): 1 column: Q4_0 -22 %, IQ4_NL -10 %, Q8_0 -7 % time;
// 2..8 columns: 2 rows per block for the K-quants and the IQ1 / IQ2 / IQ3 types (-5..-28 %), 2 warps x 2 rows for the 4-bit legacy types, 8 warps x 2 rows for Q8_0.
// MMVQ_R3_NW1 / NW2 / ROWS2 > 0 force one configuration for all types (used by the sweep).
#pragma once
#ifndef MMVQ_R3_NW1
#define MMVQ_R3_NW1 -1
#endif
#ifndef MMVQ_R3_NW2
#define MMVQ_R3_NW2 -1
#endif
#ifndef MMVQ_R3_ROWS2
#define MMVQ_R3_ROWS2 -1
#endif
static constexpr __host__ __device__ int mmvq_r3_nwarps(ggml_type type, int ncols_dst) {
    if (ncols_dst == 1) {
        if (MMVQ_R3_NW1 > 0) {
            return MMVQ_R3_NW1;
        }
        switch (type) {
            case GGML_TYPE_Q4_0: return 2;
            case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: return 8;
            case GGML_TYPE_Q8_0: return 4;
            case GGML_TYPE_Q6_K: return 2;
            case GGML_TYPE_IQ4_NL: return 2;
            default: return 1;
        }
    }
    if (MMVQ_R3_NW2 > 0) {
        return MMVQ_R3_NW2;
    }
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_IQ4_NL: case GGML_TYPE_MXFP4: case GGML_TYPE_Q6_K: return 2;
        case GGML_TYPE_Q8_0: return 8;
        default: return 1;
    }
}
static constexpr __host__ __device__ int mmvq_r3_rows(ggml_type type, int ncols_dst, bool small_k, int nwarps) {
    (void) small_k; (void) nwarps;
    if (ncols_dst >= 2 && ncols_dst <= 8) {
        if (MMVQ_R3_ROWS2 > 0) {
            return MMVQ_R3_ROWS2;
        }
        switch (type) {
            case GGML_TYPE_IQ4_XS: case GGML_TYPE_F16: case GGML_TYPE_BF16: return 1;
            default: return 2;
        }
    }
    return 1;
}
