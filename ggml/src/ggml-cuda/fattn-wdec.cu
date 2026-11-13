// fattn-wdec.cu: WMMA decode attention for quantized K/V (q4_0) on RDNA3, for verify batches of 2..5 query columns (written 2026-11-08).
//
// Replaces the per-column dot-product kernel (fattn-dec-q.cu) when several query columns share the same K/V: the quantized K and V are dequantized once per tile and
// consumed by 16x16x16 fp16 WMMA for all columns and all GQA query heads at once.
//   Block: 8 warps = one KV head (and sequence) x a slice of the KV rows, tiles of T = 32 rows, software pipelined (the K rows and raw V rows of the next tile are requested as soon as
//          the registers are free, and the barriers only wait for LDS so the loads stay in flight).
//   Q:     R = G*NB query rows (padded to a multiple of 16), kept as fp16 (pre-scaled) in LDS.
//   S:     NJOB = 2*NQT jobs (row tile rt of 16 KV rows x query tile qt of 16 query rows); the quantization blocks of the K rows are split over NKS = 8/NJOB warps per job. The K rows
//          are dequantized straight from the quantized data into the WMMA A operand (one KV row per lane, both lane halves load the same row); K never goes through LDS. The partial
//          S tiles are summed through LDS.
//   softmax: per query row over the 32 rows of the tile (the two row-tile jobs exchange their maxima through LDS), P as fp16 into LDS.
//   V:     the raw rows are staged in LDS and dequantized into a transposed fp16 tile Vt[d][t] (shared by all columns).
//   O:     warp w owns the output dims [32w, 32w+32): O += P * V with WMMA, rescaled by the softmax correction each tile.
// The partial results use the layout flash_attn_combine_results expects (same as fattn-dec-q.cu).
#include "common.cuh"
#include "fattn-common.cuh"

#if defined(GGML_USE_HIP) && defined(AMD_WMMA_AVAILABLE) && defined(FLASH_ATTN_AVAILABLE)
#define WDEC_AVAILABLE
#endif

// LDS-only workgroup barrier: waits for this wave's LDS accesses, not for outstanding global loads
#if defined(GGML_USE_HIP) && defined(AMD_WMMA_AVAILABLE)
#define WDEC_SYNC() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#else
#define WDEC_SYNC() __syncthreads()
#endif

#ifdef WDEC_AVAILABLE
typedef _Float16 wdec_half16 __attribute__((ext_vector_type(16)));
typedef float    wdec_float8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ wdec_half16 wdec_lds16(const half * p) { // 16 halfs from 16-byte aligned LDS
    union { uint4 u[2]; wdec_half16 h; } x;
    x.u[0] = *(const uint4 *) p;
    x.u[1] = *(const uint4 *) (p + 8);
    return x.h;
}

static __device__ __forceinline__ half2 wdec_u32_to_half2(const uint32_t v) {
    union { uint32_t u; half2 h; } x; x.u = v; return x.h;
}
#endif

template<int NB>
__launch_bounds__(256, 1)
static __global__ void flash_attn_wdec_q4(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const int  * KV_rows_ptr,
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
#ifdef WDEC_AVAILABLE
    constexpr int D    = 256;
    constexpr int G    = 6;
    constexpr int R    = G*NB;                 // query rows
    constexpr int NQT  = (R + 15)/16;          // query tiles (1 or 2)
    constexpr int RP   = NQT*16;               // padded query rows
    constexpr int T    = 32;                   // KV rows per tile
    constexpr int NT   = 256;
    constexpr int NJOB = 2*NQT;                // (row tile, query tile) S jobs
    constexpr int NKS  = 8/NJOB;               // warps per job: the quantization blocks of the K row are split over them
    constexpr int NBP  = 8/NKS;                // quantization blocks per split (even)
    constexpr int ND   = 9*(NBP/2);            // dwords of the K row per split
    constexpr int NV   = (T*18 + NT - 1)/NT;   // raw V units (8 bytes) per thread
    constexpr int QS   = D + 8;                // LDS strides (halfs)
    constexpr int PS   = T + 8;
    constexpr int VS   = T + 8;
    constexpr int RAWS = 148;                  // bytes per staged raw row (37 dwords, bank friendly)
    static_assert(NQT <= 2 && NBP % 2 == 0, "wdec: unsupported configuration");

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int tid  = w*32 + lane;
    const int job  = w % NJOB;
    const int part = w / NJOB;                 // k-split of the S computation
    const int rt   = job & 1;                  // KV row tile of the S computation
    const int qts  = job >> 1;                 // query tile of the S computation
    const int l16  = lane & 15;
    const int lhi  = lane >> 4;

    const int sequence = blockIdx.z / ne12;
    const int kvhead   = blockIdx.z - sequence*ne12;
    const char * Qb = Q_ptr + nb03*sequence + int64_t(nb02)*(kvhead*G);
    const char * Kb = K_ptr + nb13*sequence + int64_t(nb12)*kvhead;
    const char * Vb = V_ptr + nb23*sequence + int64_t(nb22)*kvhead;
    const char * maskb = mask_ptr ? mask_ptr + nb33*(sequence % ne33) : nullptr;

    __shared__ __align__(16) half  Qs[RP*QS];
    __shared__ __align__(16) half  Ps[RP*PS];
    __shared__ __align__(16) half  Vt[D*VS];
    __shared__ __align__(16) char  raw[T*RAWS];
    __shared__ float Sx[(NKS - 1)*NJOB*8*32];
    __shared__ float Mw[2][RP];
    __shared__ float As[RP];
    __shared__ float Ls[4][RP];
    __shared__ float Ms[RP];

    // K row part (one row per lane, both lane halves) and raw V rows of the current tile live in registers; the next tile is requested as soon as they are consumed
    uint32_t rw[ND];
    uint2 vv[NV];
    auto load_k = [&](const int row0n) {
        if (row0n < ne11) {
            const uint32_t * kp = (const uint32_t *) (Kb + int64_t(row0n + 16*rt + l16)*nb11 + 18*NBP*part);
#pragma unroll
            for (int i = 0; i < ND; ++i) {
                rw[i] = kp[i];
            }
        }
    };
    auto load_v = [&](const int row0n) {
#pragma unroll
        for (int r = 0; r < NV; ++r) {
            const int idx = tid + r*NT;
            vv[r] = make_uint2(0, 0);
            if (idx < T*18 && row0n < ne11) {
                const int i = idx / 18, u = idx - i*18;
                vv[r] = *(const uint2 *) (Vb + int64_t(row0n + i)*nb21 + u*8);
            }
        }
    };
    if (int(blockIdx.y)*T < ne11) {
        load_k(blockIdx.y*T);
        load_v(blockIdx.y*T);
    }

    // Q: scaled, fp16, zero padded
    for (int idx = tid; idx < RP*D; idx += NT) {
        const int q = idx / D, d = idx - q*D;
        float v = 0.0f;
        if (q < R) {
            const int c = q / G, g = q - c*G;
            v = ((const float *) (Qb + int64_t(nb01)*c + int64_t(nb02)*g))[d]*scale;
        }
        Qs[q*QS + d] = __float2half(v);
    }

    float m_run = -FLT_MAX/2.0f;
    float l_loc = 0.0f;
    wdec_float8 O[NQT][2];
#pragma unroll
    for (int qt = 0; qt < NQT; ++qt) {
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                O[qt][dt][i] = 0.0f;
            }
        }
    }
    WDEC_SYNC();

    for (int tile = blockIdx.y; tile*T < ne11; tile += gridDim.y) {
        const int row0 = tile*T;

        // ---- partial S = K * Q^T over the quantization blocks of this k-split
        wdec_float8 S;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            S[i] = 0.0f;
        }
#pragma unroll
        for (int bl = 0; bl < NBP; ++bl) {
            const int b = part*NBP + bl;
            uint32_t qw[4];
            uint32_t sc;
            if ((bl & 1) == 0) {                // block starts on a dword boundary, the quants are at +2 bytes
                const int o = 9*(bl >> 1);
                sc = rw[o] & 0xFFFF;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    qw[i] = (rw[o + i] >> 16) | (rw[o + i + 1] << 16);
                }
            } else {                            // the scale is the high half of a dword, the quants are dword aligned
                const int o = 9*(bl >> 1) + 4;
                sc = rw[o] >> 16;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    qw[i] = rw[o + 1 + i];
                }
            }
            const half2 d2 = __half2half2(__ushort_as_half((unsigned short) sc));
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {    // k-step = 16 dims: low nibbles (dims 0..15 of the block) then high nibbles
                union { uint32_t u[8]; wdec_half16 h; } A;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t x = hh == 0 ? (qw[i] & 0x0F0F0F0Fu) : ((qw[i] >> 4) & 0x0F0F0F0Fu);
                    const uint32_t a01 = ((x & 0x000Fu) | ((x & 0x0F00u) << 8)) | 0x64006400u;
                    const uint32_t y = x >> 16;
                    const uint32_t a23 = ((y & 0x000Fu) | ((y & 0x0F00u) << 8)) | 0x64006400u;
                    const half2 c = make_half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408)); // 1024 + n - 1032 = n - 8
                    const half2 h01 = (wdec_u32_to_half2(a01) - c)*d2;
                    const half2 h23 = (wdec_u32_to_half2(a23) - c)*d2;
                    A.u[2*i]     = *(const uint32_t *) &h01;
                    A.u[2*i + 1] = *(const uint32_t *) &h23;
                }
                const wdec_half16 Bq = wdec_lds16(&Qs[(16*qts + l16)*QS + 32*b + 16*hh]);
                S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A.h, Bq, S);
            }
        }
        load_k((tile + gridDim.y)*T); // next tile (the registers are dead now); out-of-range rows are not read
        if (part > 0) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                Sx[(((part - 1)*NJOB + job)*8 + i)*32 + lane] = S[i];
            }
        }
        WDEC_SYNC(); // X: partial S tiles

        if (part == 0) {
#pragma unroll
            for (int p = 1; p < NKS; ++p) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    S[i] += Sx[(((p - 1)*NJOB + job)*8 + i)*32 + lane];
                }
            }
            if (maskb) {
                const int q = 16*qts + l16;
                const int c = q < R ? q / G : 0;
                const half * mrow = (const half *) (maskb + int64_t(nb31)*c);
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    S[l] += __half2float(mrow[row0 + 16*rt + 2*l + lhi]);
                }
            }
            float mx = S[0];
#pragma unroll
            for (int l = 1; l < 8; ++l) {
                mx = fmaxf(mx, S[l]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 16, 32));
            if (lane < 16) {
                Mw[rt][16*qts + lane] = mx;
            }
        }
        WDEC_SYNC(); // A: tile maxima

        // ---- softmax, P to LDS
        if (part == 0) {
            const int q = 16*qts + l16;
            const float mt = fmaxf(Mw[0][q], Mw[1][q]);
            const float m_new = fmaxf(m_run, mt + FATTN_KQ_MAX_OFFSET);
            const float alpha = expf(m_run - m_new);
            m_run = m_new;
            float psum = 0.0f;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const half ph = __float2half(expf(S[l] - m_new));
                psum += __half2float(ph);
                Ps[q*PS + 16*rt + 2*l + lhi] = ph;
            }
            l_loc = l_loc*alpha + psum;
            if (rt == 0 && lane < 16) {
                As[q] = alpha;
            }
        }

        // ---- V: stage the raw rows, then request the next tile's rows
#pragma unroll
        for (int r = 0; r < NV; ++r) {
            const int idx = tid + r*NT;
            if (idx < T*18) {
                const int i = idx / 18, u = idx - i*18;
                uint32_t * dstw = (uint32_t *) (raw + i*RAWS + u*8);
                dstw[0] = vv[r].x; dstw[1] = vv[r].y;
            }
        }
        load_v((tile + gridDim.y)*T);
        WDEC_SYNC(); // B: P, As and the raw V rows are in LDS

        // ---- V: dequantize into the transposed tile Vt[d][t]
        {
            // each thread dequantizes two adjacent KV rows (t0, t0+1) of 4 groups of 8 dims and writes the pair as one half2 per dim
            // (half of the LDS store instructions of the scalar version); lanes 16..31 take the odd dim pairs of the group so that the
            // two lane halves do not hit the same banks
            const int t0   = 2*(tid & 15);
            const int gs   = tid >> 4;               // 0..15
            const int pset = gs & 1;                 // dim pairs p = pset, pset + 2 of each group
#pragma unroll
            for (int gi = 0; gi < 4; ++gi) {
                const int grp = (gs >> 1) + 8*gi;    // 8 dims per group
                const int e0  = grp*8;
                const int ub  = e0 >> 5;
                const int o   = e0 & 31;
                const char * blk0 = raw + t0*RAWS + ub*18;
                const char * blk1 = blk0 + RAWS;
                const half2 dv0 = __half2half2(*(const half *) blk0);
                const half2 dv1 = __half2half2(*(const half *) blk1);
                const uint16_t * qp0 = (const uint16_t *) (blk0 + 2 + (o < 16 ? o : o - 16));
                const uint16_t * qp1 = (const uint16_t *) (blk1 + 2 + (o < 16 ? o : o - 16));
#pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    const int p = pset + 2*pp;
                    uint32_t w0 = qp0[p];
                    uint32_t w1 = qp1[p];
                    if (o >= 16) {
                        w0 >>= 4;
                        w1 >>= 4;
                    }
                    const uint32_t x0 = ((w0 & 0x000Fu) | ((w0 & 0x0F00u) << 8)) | 0x64006400u;
                    const uint32_t x1 = ((w1 & 0x000Fu) | ((w1 & 0x0F00u) << 8)) | 0x64006400u;
                    const half2 c = make_half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));
                    const half2 h0 = (wdec_u32_to_half2(x0) - c)*dv0;   // row t0:   dims e0+2p, e0+2p+1
                    const half2 h1 = (wdec_u32_to_half2(x1) - c)*dv1;   // row t0+1
                    *(half2 *) &Vt[(e0 + 2*p)*VS + t0]     = make_half2(__low2half(h0),  __low2half(h1));
                    *(half2 *) &Vt[(e0 + 2*p + 1)*VS + t0] = make_half2(__high2half(h0), __high2half(h1));
                }
            }
        }
        WDEC_SYNC(); // C: Vt complete

        // ---- O = alpha*O + P * V for the output dims [32w, 32w+32)
#pragma unroll
        for (int qt = 0; qt < NQT; ++qt) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float a = As[16*qt + 2*l + lhi];
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) {
                    O[qt][dt][l] *= a;
                }
            }
        }
#pragma unroll
        for (int ks = 0; ks < T/16; ++ks) {
#pragma unroll
            for (int qt = 0; qt < NQT; ++qt) {
                const wdec_half16 Ap = wdec_lds16(&Ps[(16*qt + l16)*PS + 16*ks]);
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) {
                    const wdec_half16 Bv = wdec_lds16(&Vt[(32*w + 16*dt + l16)*VS + 16*ks]);
                    O[qt][dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Ap, Bv, O[qt][dt]);
                }
            }
        }
        WDEC_SYNC(); // D: Ps / Vt / raw / Sx may be overwritten by the next tile
    }

    // ---- epilogue: row sums and maxima
    if (part == 0) {
        Ls[rt*2 + lhi][16*qts + l16] = l_loc;
        if (rt == 0 && lane < 16) {
            Ms[16*qts + lane] = m_run;
        }
    }
    __syncthreads();
#pragma unroll
    for (int qt = 0; qt < NQT; ++qt) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int q = 16*qt + 2*l + lhi;
            if (q < R) {
                const float Lb = Ls[0][q] + Ls[1][q] + Ls[2][q] + Ls[3][q];
                const int c = q / G, g = q - c*G;
                const int head = kvhead*G + g;
                const int j_dst = ((sequence*int(ne01.z) + c)*ne02 + head)*gridDim.y + blockIdx.y;
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) {
                    const int d = 32*w + 16*dt + l16;
                    dst_ptr[int64_t(j_dst)*D + d] = gridDim.y == 1 ? O[qt][dt][l]/Lb : O[qt][dt][l];
                }
                if (w == 0 && l16 == 0 && gridDim.y > 1) {
                    dst_meta_ptr[j_dst] = make_float2(Ms[q], Lb);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // WDEC_AVAILABLE
}

template<int NB>
static void launch_wdec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_kernel_t kernel = flash_attn_wdec_q4<NB>;
    static const int occ = [] { const char * e = getenv("GGML_CUDA_FATTN_WDEC_OCC"); return e ? atoi(e) : 2; }();
    launch_fattn<256, NB, 6>(ctx, dst, kernel, 8, 0, 32, false, false, false, false, 32, false, occ);
}

bool ggml_cuda_flash_attn_ext_wdec_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (!mask || max_bias != 0.0f || logit_softcap != 0.0f || dst->src[4] != nullptr || Q->ne[3] != 1 || Q->ne[1] < 2 || Q->ne[1] > 5 || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || K->nb[1] % 8 != 0 || V->nb[1] % 8 != 0 || K->nb[2] % 8 != 0 || V->nb[2] % 8 != 0 || V->ne[0] != Q->ne[0]) {
        return false;
    }
    return Q->ne[0] == 256 && Q->ne[2] == 6*K->ne[2];
}

void ggml_cuda_flash_attn_ext_wdec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[1]) {
        case 2: launch_wdec<2>(ctx, dst); break;
        case 3: launch_wdec<3>(ctx, dst); break;
        case 4: launch_wdec<4>(ctx, dst); break;
        case 5: launch_wdec<5>(ctx, dst); break;
        default: GGML_ABORT("wdec: unsupported number of query columns");
    }
}
