#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
    // H57: when set, the input state of sequence s is row state_ids[s] (of state_row floats) of the cache `state` (device-side gather, no GET_ROWS copy)
    const int32_t * state_ids = nullptr;
    int64_t         state_row = 0;
};

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream);
