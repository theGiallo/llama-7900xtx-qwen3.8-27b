#include "common.cuh"
#include "fattn-common.cuh"


// Grouped-query vector FlashAttention kernel.
//
// Unlike flash_attn_ext_vec (one block per query head, KV streamed once per query head),
// this kernel processes all GQA query heads that share a KV head in a single block:
//   - each KV block is loaded and dequantized once, then dotted with every query head,
//   - the nibble-unpack / dequantize ALU work (the q4_0/q8_0 bottleneck on RDNA) is amortized GQA x,
//   - KV bytes are read once per block instead of once per query head.
//
// Grid layout (via launch_fattn<D, 1, GQA_RATIO>):
//   blockIdx.x = Q column (token), gridDim.x = n_q
//   blockIdx.y = KV cache chunk,  gridDim.y = parallel_blocks
//   blockIdx.z = sequence * K->ne[2] + kv_head, gridDim.z = K->ne[2] * Q->ne[3]
// Registers per thread hold the q8_1 Q data for all GQA heads (nthreads_KQ == WARP_SIZE
// keeps this small: D/4/WARP_SIZE ints + float2s per head). Warps process disjoint KV rows;
// per-warp softmax partials are merged in the epilogue exactly like flash_attn_ext_vec.

static int ggml_cuda_fattn_vec_gqa_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_gqa_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int GQA, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size, GQA == query heads per KV head
__launch_bounds__(ggml_cuda_fattn_vec_gqa_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec_gqa(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_gqa_get_nthreads_device();
    // nthreads_KQ == WARP_SIZE so that each thread holds only D/4/WARP_SIZE ints (+ float2s)
    // of q8_1 Q data per query head, which keeps the register footprint of the GQA head group small:
    constexpr int nthreads_KQ = WARP_SIZE;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x; // Index of the Q column (token) this block handles.

    const int sequence = blockIdx.z / ne12;
    const int kvhead   = blockIdx.z - sequence*ne12;

    const int gqa_ratio = ne02 / ne12;
    GGML_UNUSED(gqa_ratio); // Guaranteed == GQA by the host dispatch (template-gated grid layout).

    Q += nb03*sequence + nb02*(kvhead*GQA)          + nb01*ic0;
    K += nb13*sequence + nb12*kvhead;
    V += nb23*sequence + nb22*kvhead;

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    constexpr int nwarps = nthreads / WARP_SIZE;
    static_assert(nwarps <= WARP_SIZE, "VEC_GQA epilogue assumes nwarps <= WARP_SIZE");

    // Shared memory layout:
    //   [0 .. smem_q_tmp)          q8_1 Q conversion buffer (lives only during the Q setup phase),
    //   [0 .. GQA*nthreads)        per-(head, KV row) KQ probabilities (KQ pass, overlaps the Q buffer above),
    //   [GQA*nthreads ..)          per-head VKQ scratch, nwarps*V_cols_per_iter*D elements (epilogue).
    constexpr int smem_q_tmp  = GQA*(D/sizeof(int) + 2*(D/QK8_1)); // head-major q8_1 temp: D/4 ints + D/QK8_1 float2s per head, in float elements
    constexpr int ne_combine  = nwarps*V_cols_per_iter*D;
    __shared__ float KQ[smem_q_tmp > GQA*nthreads + ne_combine ? smem_q_tmp : GQA*nthreads + ne_combine];
    float * VKQ_scratch = KQ + GQA*nthreads;
    half  * VKQ_scratch_h = (half *) VKQ_scratch;

    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    static_assert(GQA <= 8, "VEC_GQA supports at most 8 grouped query heads");
    const float slope[8] = { get_alibi_slope(max_bias, kvhead*GQA + 0, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 1, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 2, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 3, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 4, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 5, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 6, n_head_log2, m0, m1),
                             get_alibi_slope(max_bias, kvhead*GQA + 7, n_head_log2, m0, m1) };

#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[GQA][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
#else
    float2           VKQ[GQA][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[GQA];
    float KQ_sum[GQA];
#pragma unroll
    for (int jG = 0; jG < GQA; ++jG) {
        KQ_max[jG] = -FLT_MAX/2.0f;
        KQ_sum[jG] = 0.0f;
    }

    // Convert Q to q8_1 (quantized K) and store in registers. One block handles one Q column
    // (ic0), so there is no per-column boundary handling here.
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[GQA][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[GQA][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[GQA][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    float2 Q_ds[GQA][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];

    ggml_cuda_pdl_sync();
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int jG0 = 0; jG0 < GQA; jG0 += nwarps) {
            const int jG = jG0 + threadIdx.y;

            if (jG0 + nwarps > GQA && jG >= GQA) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[jG*(D/sizeof(int) + 2*(D/QK8_1))];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));
            const float * Q_f  = (const float *) (Q + jG*nb02);

            constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                quantize_q8_1_to_shared<float2, nthreads_quantize>
                    (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
        }

        __syncthreads();

#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
            int    * tmp_q_i32 = (int    *) &KQ[jG*(D/sizeof(int) + 2*(D/QK8_1))];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[jG][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[jG][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
            const float2 * Q_j = (const float2 *) (Q + jG*nb02);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[jG][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[jG][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
            const float2 * Q_j = (const float2 *) (Q + jG*nb02);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[jG][i0/nthreads_KQ],            &Q_j[i]);
                ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[jG][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[jG][k].x *= scale;
                Q_reg[jG][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;

    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[GQA]; // KQ in registers.

        float KQ_max_new[GQA];
#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
            KQ_max_new[jG] = KQ_max[jG];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

            // The KV row is loaded and dequantized once, then dotted with every query head of the group.
#pragma unroll
            for (int jG = 0; jG < GQA; ++jG) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[jG], Q_i32[jG], Q_ds[jG]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask) {
                    sum += slope[jG]*__half2float(maskh[i_KQ]);
                }

                KQ_max_new[jG] = fmaxf(KQ_max_new[jG], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[jG] = sum;
                }
            }
        }

#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[jG] = fmaxf(KQ_max_new[jG], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[jG], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[jG] - KQ_max_new[jG]);
            KQ_max[jG] = KQ_max_new[jG];

            KQ_reg[jG] = expf(KQ_reg[jG] - KQ_max[jG]);
            KQ_sum[jG] = KQ_sum[jG]*KQ_max_scale + KQ_reg[jG];
            KQ[jG*nthreads + tid] = KQ_reg[jG];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[jG][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[jG][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[jG][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

        ggml_cuda_syncwarp();

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[GQA];
#pragma unroll
            for (int jG = 0; jG < GQA; ++jG) {
                KQ_k[jG] = __half2half2(KQ[jG*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int jG = 0; jG < GQA; ++jG) {
                        VKQ[jG][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[jG];
                    }
                }
            }
#else
            float KQ_k[GQA];
#pragma unroll
            for (int jG = 0; jG < GQA; ++jG) {
                KQ_k[jG] = KQ[jG*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int jG = 0; jG < GQA; ++jG) {
                        VKQ[jG][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[jG];
                        VKQ[jG][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[jG];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
#pragma unroll
        for (int jG0 = 0; jG0 < GQA; jG0 += nwarps) {
            const int jG = jG0 + threadIdx.y;

            if (jG0 + nwarps > GQA && jG >= GQA) {
                break;
            }

            const int   head = kvhead*GQA + jG;
            const float sink = ((const float *) sinks)[head];

            const float kqmax_new_j = fmaxf(sink, KQ_max[jG]);
            const float KQ_max_scale = expf(KQ_max[jG] - kqmax_new_j);
            KQ_max[jG] = kqmax_new_j;

            KQ_sum[jG] = KQ_sum[jG]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[jG]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[jG][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[jG][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[jG][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[GQA][WARP_SIZE];
    __shared__ float KQ_sum_shared[GQA][WARP_SIZE];
#pragma unroll
    for (int jG = 0; jG < GQA; ++jG) {
        if (threadIdx.y == 0) {
            KQ_max_shared[jG][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[jG][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int jG = 0; jG < GQA; ++jG) {
        if (threadIdx.x == 0) {
            KQ_max_shared[jG][threadIdx.y] = KQ_max[jG];
        }
    }
    __syncthreads();

#pragma unroll
    for (int jG = 0; jG < GQA; ++jG) {
        const int head = kvhead*GQA + jG;

        float kqmax_new = KQ_max_shared[jG][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[jG] - kqmax_new);
        KQ_max[jG] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) VKQ_scratch + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[jG][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[jG][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) VKQ_scratch + threadIdx.y*(V_cols_per_iter*D/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[jG][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[jG][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[jG][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[jG][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[jG] *= kqmax_scale;
        KQ_sum[jG] = warp_reduce_sum(KQ_sum[jG]);

        if (threadIdx.x == 0) {
            KQ_sum_shared[jG][threadIdx.y] = KQ_sum[jG];
        }

        __syncthreads();

        if (nthreads <= D || tid < D) {
            KQ_sum[jG] = KQ_sum_shared[jG][threadIdx.x];
            KQ_sum[jG] = warp_reduce_sum(KQ_sum[jG]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        // Read the slot through a half view so the w*V_cols_per_iter*D stride is
                        // measured in half elements (each slot is V_cols_per_iter*D/2 half2s ==
                        // V_cols_per_iter*D half elements), matching the stride used when writing.
                        float val = float(VKQ_scratch_h[w*V_cols_per_iter*D + v*D + i0 + tid]);
                        dst_val += val;
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[jG];
                }
                dst[(((sequence*int(ne01.z) + ic0)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid] = dst_val;
            }
        }

        if (jG < GQA-1) {
            __syncthreads();
        }
    }

    if (gridDim.y != 1 && tid == 0) {
#pragma unroll
        for (int jG = 0; jG < GQA; ++jG) {
            const int head = kvhead*GQA + jG;
            dst_meta[((sequence*int(ne01.z) + ic0)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[jG], KQ_sum[jG]);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int GQA, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_gqa_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_gqa_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec_gqa<D, GQA, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D, 1, GQA>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false, false);
}

template <int D, int GQA, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_gqa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_gqa_case_impl<D, GQA, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_gqa_case_impl<D, GQA, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}
// Whether the GQA-grouped vector kernel is instantiated for this (head size, GQA ratio, K/V types) combination.
bool ggml_cuda_fattn_vec_gqa_supported(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V);

// Run the GQA-grouped vector kernel. Only valid if ggml_cuda_fattn_vec_gqa_supported() == true.
void ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
