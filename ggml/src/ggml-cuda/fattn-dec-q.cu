// fattn-dec-q.cu: decode attention for quantized K/V (q4_0 / q8_0) on RDNA (written 2026-11-07).
//
// One block = one KV head (and sequence) x interleaved 128-row KV tiles, with NB groups of 4 warps, one group per query column (NB <= 8: plain decode or an
// MTP verify batch). All G query heads of the KV head are handled by every group, so a group works on R = G query rows and the groups read the same K/V tile
// at the same time (served by L1/L2, the cache is read from memory once).
//   scores: one KV row per lane. The row (q4_0 144 B / q8_0 272 B for head size 256) is loaded with 8-byte loads into registers and dotted with the
//           int8-quantized Q rows (LDS, broadcast reads) with integer dot products; no cross-lane reductions are needed.
//   softmax: per warp (own running max per query row, partial sums per lane reduced at the end).
//   V:      lane = a slice of 8 (4 for head size 128) dims; the probabilities of the warp's 32 rows go through LDS; V is dequantized to half2 and
//           accumulated with packed fp16 FMA for the group's R rows.
// The partial results are written in the layout flash_attn_combine_results expects (same as the tile kernel).
#include "common.cuh"
#include "fattn-common.cuh"

#define DECQ_THREADS 128 // per group
#ifndef DECQ_MAXNC
#define DECQ_MAXNC 2 // query columns per warp in the multi-column kernel
#endif
#define DECQ_NGR(NB) (((NB) + DECQ_MAXNC - 1)/DECQ_MAXNC)
#ifndef DECQ_SCHED_BARRIER
#define DECQ_SCHED_BARRIER() __builtin_amdgcn_sched_barrier(0)
#endif

template<int D, int G, int NB, int kvq>
__launch_bounds__(DECQ_THREADS*NB, 1)
static __global__ void flash_attn_dec_q(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP)
    constexpr int R      = G;                          // query rows per group
    constexpr int RT     = G*NB;                       // query rows in total
    constexpr int NT     = DECQ_THREADS*NB;            // threads per block
    constexpr int NBLK   = D/32;                       // quantization blocks per row
    constexpr int BS     = kvq == 1 ? 18 : 34;         // bytes per block
    constexpr int ROWB   = NBLK*BS;                    // bytes per row
    constexpr int ROWU   = ROWB/8;                     // 8-byte units per row
    constexpr int DPL    = D/32;                       // V dims per lane
    constexpr int HPL    = DPL/2;                      // half2 per lane
    static_assert(ROWB % 8 == 0, "row size must be a multiple of 8 bytes");

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;                      // warp index in the block
    const int wg   = NB == 1 ? w : (w & 3);                            // warp index in its group
    const int cg   = NB == 1 ? 0 : (w >> 2);                           // query column of the group
    const int tid  = w*32 + lane;

    const int sequence = blockIdx.z / ne12;
    const int kvhead   = blockIdx.z - sequence*ne12;

    const char * Qb = Q_ptr + nb03*sequence + int64_t(nb02)*(kvhead*G);
    const char * Kb = K_ptr + nb13*sequence + int64_t(nb12)*kvhead;
    const char * Vb = V_ptr + nb23*sequence + int64_t(nb22)*kvhead;
    const char * maskb = mask_ptr ? mask_ptr + nb33*(sequence % ne33) + int64_t(nb31)*cg : nullptr;

    // shared memory: Q (int8) + Q scales + probabilities; reused by the epilogue
    constexpr int bytes_Qi = RT*NBLK*8*4;
    constexpr int bytes_Qd = RT*NBLK*8;
    constexpr int bytes_Ps = 4*NB*32*R*2;
    constexpr int bytes_ep = 4*D*4;                    // epilogue: one query row at a time
    constexpr int bytes_all = (bytes_Qi + bytes_Qd + bytes_Ps) > bytes_ep ? (bytes_Qi + bytes_Qd + bytes_Ps) : bytes_ep;
    __shared__ __align__(16) char smem[bytes_all];
    __shared__ float  Mw[4*NB][R];
    __shared__ float  Lw[4*NB][R];
    int    * Qi = (int    *) smem;
    float2 * Qd = (float2 *) (smem + bytes_Qi);
    half   * Ps = (half   *) (smem + bytes_Qi + bytes_Qd);

    // quantize Q (scaled by the softmax scale) to int8 per 32-element block
    for (int u = tid; u < RT*NBLK; u += NT) {
        const int r = u / NBLK, blk = u - r*NBLK;
        const int c = r / G, g = r - c*G;
        const float * qp = (const float *) (Qb + int64_t(nb01)*c + int64_t(nb02)*g) + blk*32;
        float v[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 t = ((const float4 *) qp)[k];
            v[4*k] = t.x*scale; v[4*k + 1] = t.y*scale; v[4*k + 2] = t.z*scale; v[4*k + 3] = t.w*scale;
        }
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            amax = fmaxf(amax, fabsf(v[k]));
        }
        const float d  = amax/127.0f;
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        int sum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int q = __float2int_rn(v[4*k + j]*id);
                sum += q;
                packed |= (q & 0xFF) << (8*j);
            }
            Qi[u*8 + k] = packed;
        }
        Qd[u] = make_float2(d, kvq == 1 ? 8.0f*d*float(sum) : 0.0f);
    }
    __syncthreads();

    float m_run[R];
    float l_run[R];
    half2 acc[R][HPL];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        m_run[r] = -FLT_MAX/2.0f;
        l_run[r] = 0.0f;
#pragma unroll
        for (int h = 0; h < HPL; ++h) {
            acc[r][h] = make_half2(0.0f, 0.0f);
        }
    }

    const int dbase = lane*DPL;
    const int vblk  = dbase/32;
    const int vo    = dbase - vblk*32;
    const int * Qi_g = Qi + cg*R*NBLK*8;
    const float2 * Qd_g = Qd + cg*R*NBLK;

    for (int tile = blockIdx.y; tile*128 < ne11; tile += gridDim.y) {
        const int row0 = tile*128 + wg*32;
        const int row  = row0 + lane;

        // ---- K row into registers
        uint32_t rw[ROWB/4];
        {
            const uint2 * kp = (const uint2 *) (Kb + int64_t(row)*nb11);
#pragma unroll
            for (int u = 0; u < ROWU; ++u) {
                const uint2 t = kp[u];
                rw[2*u] = t.x; rw[2*u + 1] = t.y;
            }
        }
        const float mk = maskb ? __half2float(((const half *) maskb)[row]) : 0.0f;

        // ---- scores
        float s[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[r] = 0.0f;
        }
        auto H16 = [&](const int idx) -> uint32_t { // 16-bit element idx of the row
            const uint32_t x = rw[idx >> 1];
            return (idx & 1) ? (x >> 16) : (x & 0xFFFF);
        };
#pragma unroll
        for (int blk = 0; blk < NBLK; ++blk) {
            int kwd[8];
            float dk;
            if constexpr (kvq == 1) {
                dk = __half2float(__ushort_as_half((unsigned short) H16(9*blk)));
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t word = H16(9*blk + 1 + 2*k) | (H16(9*blk + 2 + 2*k) << 16);
                    kwd[k]     = word & 0x0F0F0F0F;
                    kwd[4 + k] = (word >> 4) & 0x0F0F0F0F;
                }
            } else {
                dk = __half2float(__ushort_as_half((unsigned short) H16(17*blk)));
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    kwd[k] = H16(17*blk + 1 + 2*k) | (H16(17*blk + 2 + 2*k) << 16);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int4 qa = *(const int4 *) &Qi_g[(r*NBLK + blk)*8];
                const int4 qb = *(const int4 *) &Qi_g[(r*NBLK + blk)*8 + 4];
                int sumi = 0;
                sumi = ggml_cuda_dp4a(kwd[0], qa.x, sumi);
                sumi = ggml_cuda_dp4a(kwd[1], qa.y, sumi);
                sumi = ggml_cuda_dp4a(kwd[2], qa.z, sumi);
                sumi = ggml_cuda_dp4a(kwd[3], qa.w, sumi);
                sumi = ggml_cuda_dp4a(kwd[4], qb.x, sumi);
                sumi = ggml_cuda_dp4a(kwd[5], qb.y, sumi);
                sumi = ggml_cuda_dp4a(kwd[6], qb.z, sumi);
                sumi = ggml_cuda_dp4a(kwd[7], qb.w, sumi);
                const float2 qd = Qd_g[r*NBLK + blk];
                s[r] += dk*(qd.x*float(sumi) - qd.y);
                DECQ_SCHED_BARRIER();
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s[r] += mk;
        }

        // ---- online softmax per warp
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float mt = s[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mt = fmaxf(mt, __shfl_xor_sync(0xFFFFFFFF, mt, off, 32));
            }
            const float m_new = fmaxf(m_run[r], mt + FATTN_KQ_MAX_OFFSET);
            const float sc    = expf(m_run[r] - m_new);
            m_run[r] = m_new;
            const float p = expf(s[r] - m_new);
            const half ph = __float2half(p);
            l_run[r] = l_run[r]*sc + __half2float(ph);
            const half2 sc2 = __float2half2_rn(sc);
#pragma unroll
            for (int h = 0; h < HPL; ++h) {
                acc[r][h] *= sc2;
            }
            Ps[(w*32 + lane)*R + r] = ph;
        }
        ggml_cuda_syncwarp();

        // ---- V accumulation: lane = slice of DPL dims, the group's rows
#pragma unroll 4
        for (int kv = 0; kv < 32; ++kv) {
            const char * vrow = Vb + int64_t(row0 + kv)*nb21 + vblk*BS;
            const half dv = __ushort_as_half(*(const unsigned short *) vrow);
            const half2 dv2 = __half2half2(dv);
            half2 pv[HPL];
            if constexpr (kvq == 1) {
                const bool hi = vo >= 16;
                const unsigned short * qp = (const unsigned short *) (vrow + 2 + (hi ? vo - 16 : vo));
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    uint32_t wv = qp[h];
                    if (hi) {
                        wv >>= 4;
                    }
                    const uint32_t x = ((wv & 0x000F) | ((wv & 0x0F00) << 8)) | 0x64006400;
                    half2 t = *(const half2 *) &x;
                    t = t - make_half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408)); // 1024 + n - 1032
                    pv[h] = t*dv2;
                }
            } else {
                const unsigned short * qp = (const unsigned short *) (vrow + 2 + vo);
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    const uint32_t wv = qp[h];
                    const half2 t = make_half2(__int2half_rn((int) (int8_t) (wv & 0xFF)), __int2half_rn((int) (int8_t) (wv >> 8)));
                    pv[h] = t*dv2;
                }
            }
            const uint32_t * pr = (const uint32_t *) &Ps[(w*32 + kv)*R];
#pragma unroll
            for (int r2 = 0; r2 < R/2; ++r2) {
                const uint32_t pw = pr[r2];
                const half2 pp2 = *(const half2 *) &pw;
                const half2 p0 = __low2half2(pp2);
                const half2 p1 = __high2half2(pp2);
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    acc[2*r2][h]     += pv[h]*p0;
                    acc[2*r2 + 1][h] += pv[h]*p1;
                }
            }
        }
        ggml_cuda_syncwarp();
    }

    // ---- epilogue: combine the 4 warps of each group, write the partial result
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float l = l_run[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l += __shfl_xor_sync(0xFFFFFFFF, l, off, 32);
        }
        l_run[r] = l;
    }
    __syncthreads(); // all warps are done with Qi / Ps
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            Mw[w][r] = m_run[r];
            Lw[w][r] = l_run[r];
        }
    }
    __syncthreads();

    float * sm = (float *) smem;
    float fw[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float gmax = fmaxf(fmaxf(Mw[cg*4][r], Mw[cg*4 + 1][r]), fmaxf(Mw[cg*4 + 2][r], Mw[cg*4 + 3][r]));
        fw[r] = expf(m_run[r] - gmax);
    }

#pragma unroll
    for (int cc = 0; cc < NB; ++cc) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            if (cg == cc) {
                float * dstp = sm + wg*D + dbase;
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    const float2 f = __half22float2(acc[g][h]);
                    dstp[2*h]     = f.x*fw[g];
                    dstp[2*h + 1] = f.y*fw[g];
                }
            }
            __syncthreads();
            for (int d = tid; d < D; d += NT) {
                const float v = sm[0*D + d] + sm[1*D + d] + sm[2*D + d] + sm[3*D + d];
                const float gmax = fmaxf(fmaxf(Mw[cc*4][g], Mw[cc*4 + 1][g]), fmaxf(Mw[cc*4 + 2][g], Mw[cc*4 + 3][g]));
                float Lb = 0.0f;
#pragma unroll
                for (int ww = 0; ww < 4; ++ww) {
                    Lb += Lw[cc*4 + ww][g]*expf(Mw[cc*4 + ww][g] - gmax);
                }
                const int head = kvhead*G + g;
                const int j_dst = ((sequence*int(ne01.z) + cc)*ne02 + head)*gridDim.y + blockIdx.y;
                dst_ptr[int64_t(j_dst)*D + d] = gridDim.y == 1 ? v/Lb : v;
                if (d == 0 && gridDim.y > 1) {
                    dst_meta_ptr[j_dst] = make_float2(gmax, Lb);
                }
            }
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP)
}

// Multi-column variant (2..8 query columns, MTP / n-gram verify batches), 2026-11-08. One warp handles NC columns of its 32-row tile, so the K row load, the nibble
// unpack and the V dequantization are done once per row and shared by all columns (the one-group-per-column layout above repeated them per column).
// NB <= 4: NC = NB, one group of 4 warps. NB 5..8: two groups, NC = ceil(NB/2) columns per warp (the last group may be partial).
template<int D, int G, int NB, int kvq>
__launch_bounds__(DECQ_THREADS*DECQ_NGR(NB), 1)
static __global__ void flash_attn_dec_qm(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP)
    constexpr int NGR    = DECQ_NGR(NB);                // groups of 4 warps
    constexpr int NC     = (NB + NGR - 1)/NGR;         // columns per warp
    constexpr int R      = G;                          // query rows per column
    constexpr int RT     = G*NB;                       // query rows in total
    constexpr int NT     = DECQ_THREADS*NGR;           // threads per block
    constexpr int NBLK   = D/32;                       // quantization blocks per row
    constexpr int BS     = kvq == 1 ? 18 : 34;         // bytes per block
    constexpr int ROWB   = NBLK*BS;                    // bytes per row
    constexpr int ROWU   = ROWB/8;                     // 8-byte units per row
    constexpr int DPL    = D/32;                       // V dims per lane
    constexpr int HPL    = DPL/2;                      // half2 per lane
    static_assert(ROWB % 8 == 0, "row size must be a multiple of 8 bytes");

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;                      // warp index in the block
    const int wg   = w & 3;                            // warp index in its group
    const int cg   = w >> 2;                           // group index
    const int tid  = w*32 + lane;
    const int tidg = wg*32 + lane;                     // thread index in the group

    const int sequence = blockIdx.z / ne12;
    const int kvhead   = blockIdx.z - sequence*ne12;

    const char * Qb = Q_ptr + nb03*sequence + int64_t(nb02)*(kvhead*G);
    const char * Kb = K_ptr + nb13*sequence + int64_t(nb12)*kvhead;
    const char * Vb = V_ptr + nb23*sequence + int64_t(nb22)*kvhead;
    const char * maskb = mask_ptr ? mask_ptr + nb33*(sequence % ne33) : nullptr;

    constexpr int bytes_Qi = RT*NBLK*8*4;
    constexpr int bytes_Qd = RT*NBLK*8;
    constexpr int bytes_Ps = 4*NGR*32*NC*R*2;
    constexpr int bytes_ep = NGR*4*D*4;                // epilogue: one query row at a time per group
    constexpr int bytes_all = (bytes_Qi + bytes_Qd + bytes_Ps) > bytes_ep ? (bytes_Qi + bytes_Qd + bytes_Ps) : bytes_ep;
    __shared__ __align__(16) char smem[bytes_all];
    __shared__ float  Mw[4*NGR][NC][R];
    __shared__ float  Lw[4*NGR][NC][R];
    int    * Qi = (int    *) smem;
    float2 * Qd = (float2 *) (smem + bytes_Qi);
    half   * Ps = (half   *) (smem + bytes_Qi + bytes_Qd);

    // quantize Q (scaled by the softmax scale) to int8 per 32-element block
    for (int u = tid; u < RT*NBLK; u += NT) {
        const int r = u / NBLK, blk = u - r*NBLK;
        const int c = r / G, g = r - c*G;
        const float * qp = (const float *) (Qb + int64_t(nb01)*c + int64_t(nb02)*g) + blk*32;
        float v[32];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float4 t = ((const float4 *) qp)[k];
            v[4*k] = t.x*scale; v[4*k + 1] = t.y*scale; v[4*k + 2] = t.z*scale; v[4*k + 3] = t.w*scale;
        }
        float amax = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            amax = fmaxf(amax, fabsf(v[k]));
        }
        const float d  = amax/127.0f;
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        int sum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int q = __float2int_rn(v[4*k + j]*id);
                sum += q;
                packed |= (q & 0xFF) << (8*j);
            }
            Qi[u*8 + k] = packed;
        }
        Qd[u] = make_float2(d, kvq == 1 ? 8.0f*d*float(sum) : 0.0f);
    }
    __syncthreads();

    float m_run[NC][R];
    float l_run[NC][R];
    half2 acc[NC][R][HPL];
#pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            m_run[cc][r] = -FLT_MAX/2.0f;
            l_run[cc][r] = 0.0f;
#pragma unroll
            for (int h = 0; h < HPL; ++h) {
                acc[cc][r][h] = make_half2(0.0f, 0.0f);
            }
        }
    }

    const int dbase = lane*DPL;
    const int vblk  = dbase/32;
    const int vo    = dbase - vblk*32;

    for (int tile = blockIdx.y; tile*128 < ne11; tile += gridDim.y) {
        const int row0 = tile*128 + wg*32;
        const int row  = row0 + lane;

        // ---- K row into registers
        uint32_t rw[ROWB/4];
        {
            const uint2 * kp = (const uint2 *) (Kb + int64_t(row)*nb11);
#pragma unroll
            for (int u = 0; u < ROWU; ++u) {
                const uint2 t = kp[u];
                rw[2*u] = t.x; rw[2*u + 1] = t.y;
            }
        }
        float mk[NC];
#pragma unroll
        for (int cc = 0; cc < NC; ++cc) {
            const int ca = min(cg*NC + cc, NB - 1);
            mk[cc] = maskb ? __half2float(((const half *) (maskb + int64_t(nb31)*ca))[row]) : 0.0f;
        }

        // ---- scores
        float s[NC][R];
#pragma unroll
        for (int cc = 0; cc < NC; ++cc) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                s[cc][r] = 0.0f;
            }
        }
        auto H16 = [&](const int idx) -> uint32_t { // 16-bit element idx of the row
            const uint32_t x = rw[idx >> 1];
            return (idx & 1) ? (x >> 16) : (x & 0xFFFF);
        };
#pragma unroll
        for (int blk = 0; blk < NBLK; ++blk) {
            int kwd[8];
            float dk;
            if constexpr (kvq == 1) {
                dk = __half2float(__ushort_as_half((unsigned short) H16(9*blk)));
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t word = H16(9*blk + 1 + 2*k) | (H16(9*blk + 2 + 2*k) << 16);
                    kwd[k]     = word & 0x0F0F0F0F;
                    kwd[4 + k] = (word >> 4) & 0x0F0F0F0F;
                }
            } else {
                dk = __half2float(__ushort_as_half((unsigned short) H16(17*blk)));
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    kwd[k] = H16(17*blk + 1 + 2*k) | (H16(17*blk + 2 + 2*k) << 16);
                }
            }
#pragma unroll
            for (int cc = 0; cc < NC; ++cc) {
                const int ca = min(cg*NC + cc, NB - 1);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const int4 qa = *(const int4 *) &Qi[((ca*G + r)*NBLK + blk)*8];
                    const int4 qb = *(const int4 *) &Qi[((ca*G + r)*NBLK + blk)*8 + 4];
                    int sumi = 0;
                    sumi = ggml_cuda_dp4a(kwd[0], qa.x, sumi);
                    sumi = ggml_cuda_dp4a(kwd[1], qa.y, sumi);
                    sumi = ggml_cuda_dp4a(kwd[2], qa.z, sumi);
                    sumi = ggml_cuda_dp4a(kwd[3], qa.w, sumi);
                    sumi = ggml_cuda_dp4a(kwd[4], qb.x, sumi);
                    sumi = ggml_cuda_dp4a(kwd[5], qb.y, sumi);
                    sumi = ggml_cuda_dp4a(kwd[6], qb.z, sumi);
                    sumi = ggml_cuda_dp4a(kwd[7], qb.w, sumi);
                    const float2 qd = Qd[(ca*G + r)*NBLK + blk];
                    s[cc][r] += dk*(qd.x*float(sumi) - qd.y);
                    DECQ_SCHED_BARRIER();
                }
            }
        }

        // ---- online softmax per warp and column
#pragma unroll
        for (int cc = 0; cc < NC; ++cc) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                s[cc][r] += mk[cc];
                float mt = s[cc][r];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    mt = fmaxf(mt, __shfl_xor_sync(0xFFFFFFFF, mt, off, 32));
                }
                const float m_new = fmaxf(m_run[cc][r], mt + FATTN_KQ_MAX_OFFSET);
                const float sc    = expf(m_run[cc][r] - m_new);
                m_run[cc][r] = m_new;
                const float p = expf(s[cc][r] - m_new);
                const half ph = __float2half(p);
                l_run[cc][r] = l_run[cc][r]*sc + __half2float(ph);
                const half2 sc2 = __float2half2_rn(sc);
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    acc[cc][r][h] *= sc2;
                }
                Ps[((w*32 + lane)*NC + cc)*R + r] = ph;
            }
        }
        ggml_cuda_syncwarp();

        // ---- V accumulation: lane = slice of DPL dims; V is dequantized once per row and used for all NC columns
#pragma unroll 2
        for (int kv = 0; kv < 32; ++kv) {
            const char * vrow = Vb + int64_t(row0 + kv)*nb21 + vblk*BS;
            const half dv = __ushort_as_half(*(const unsigned short *) vrow);
            const half2 dv2 = __half2half2(dv);
            half2 pv[HPL];
            if constexpr (kvq == 1) {
                const bool hi = vo >= 16;
                const unsigned short * qp = (const unsigned short *) (vrow + 2 + (hi ? vo - 16 : vo));
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    uint32_t wv = qp[h];
                    if (hi) {
                        wv >>= 4;
                    }
                    const uint32_t x = ((wv & 0x000F) | ((wv & 0x0F00) << 8)) | 0x64006400;
                    half2 t = *(const half2 *) &x;
                    t = t - make_half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408)); // 1024 + n - 1032
                    pv[h] = t*dv2;
                }
            } else {
                const unsigned short * qp = (const unsigned short *) (vrow + 2 + vo);
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    const uint32_t wv = qp[h];
                    const half2 t = make_half2(__int2half_rn((int) (int8_t) (wv & 0xFF)), __int2half_rn((int) (int8_t) (wv >> 8)));
                    pv[h] = t*dv2;
                }
            }
#pragma unroll
            for (int cc = 0; cc < NC; ++cc) {
                const uint32_t * pr = (const uint32_t *) &Ps[((w*32 + kv)*NC + cc)*R];
#pragma unroll
                for (int r2 = 0; r2 < R/2; ++r2) {
                    const uint32_t pw = pr[r2];
                    const half2 pp2 = *(const half2 *) &pw;
                    const half2 p0 = __low2half2(pp2);
                    const half2 p1 = __high2half2(pp2);
#pragma unroll
                    for (int h = 0; h < HPL; ++h) {
                        acc[cc][2*r2][h]     += pv[h]*p0;
                        acc[cc][2*r2 + 1][h] += pv[h]*p1;
                    }
                }
            }
        }
        ggml_cuda_syncwarp();
    }

    // ---- epilogue: combine the 4 warps of each group, write the partial result
#pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float l = l_run[cc][r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                l += __shfl_xor_sync(0xFFFFFFFF, l, off, 32);
            }
            l_run[cc][r] = l;
        }
    }
    __syncthreads(); // all warps are done with Qi / Ps
    if (lane == 0) {
#pragma unroll
        for (int cc = 0; cc < NC; ++cc) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                Mw[w][cc][r] = m_run[cc][r];
                Lw[w][cc][r] = l_run[cc][r];
            }
        }
    }
    __syncthreads();

    float * sm = (float *) smem + cg*(4*D);
#pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        const int ca = cg*NC + cc;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float gmax = fmaxf(fmaxf(Mw[cg*4][cc][g], Mw[cg*4 + 1][cc][g]), fmaxf(Mw[cg*4 + 2][cc][g], Mw[cg*4 + 3][cc][g]));
            const float fw = expf(m_run[cc][g] - gmax);
            {
                float * dstp = sm + wg*D + dbase;
#pragma unroll
                for (int h = 0; h < HPL; ++h) {
                    const float2 f = __half22float2(acc[cc][g][h]);
                    dstp[2*h]     = f.x*fw;
                    dstp[2*h + 1] = f.y*fw;
                }
            }
            __syncthreads();
            if (ca < NB) {
                for (int d = tidg; d < D; d += DECQ_THREADS) {
                    const float v = sm[0*D + d] + sm[1*D + d] + sm[2*D + d] + sm[3*D + d];
                    float Lb = 0.0f;
#pragma unroll
                    for (int ww = 0; ww < 4; ++ww) {
                        Lb += Lw[cg*4 + ww][cc][g]*expf(Mw[cg*4 + ww][cc][g] - gmax);
                    }
                    const int head = kvhead*G + g;
                    const int j_dst = ((sequence*int(ne01.z) + ca)*ne02 + head)*gridDim.y + blockIdx.y;
                    dst_ptr[int64_t(j_dst)*D + d] = gridDim.y == 1 ? v/Lb : v;
                    if (d == 0 && gridDim.y > 1) {
                        dst_meta_ptr[j_dst] = make_float2(gmax, Lb);
                    }
                }
            }
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, KV_rows_ptr, dst_ptr, dst_meta_ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(GGML_USE_HIP)
}

template<int D, int G, int NB, int kvq>
static void launch_dec_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool use_multi = [] { const char * e = getenv("GGML_CUDA_FATTN_DECQ_MULTI"); return e == nullptr || atoi(e) != 0; }();
    if (NB >= 2 && use_multi) {
        fattn_kernel_t kernel = flash_attn_dec_qm<D, G, NB, kvq>;
        launch_fattn<D, NB, G>(ctx, dst, kernel, 4*DECQ_NGR(NB), 0, 128, false, false, false, false, 32);
        return;
    }
    fattn_kernel_t kernel = flash_attn_dec_q<D, G, NB, kvq>;
    launch_fattn<D, NB, G>(ctx, dst, kernel, 4*NB, 0, 128, false, false, false, false, 32);
}

template<int D, int G, int kvq>
static void switch_nb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[1]) {
        case 1: launch_dec_q<D, G, 1, kvq>(ctx, dst); break;
        case 2: launch_dec_q<D, G, 2, kvq>(ctx, dst); break;
        case 3: launch_dec_q<D, G, 3, kvq>(ctx, dst); break;
        case 4: launch_dec_q<D, G, 4, kvq>(ctx, dst); break;
        case 5: launch_dec_q<D, G, 5, kvq>(ctx, dst); break;
        case 6: launch_dec_q<D, G, 6, kvq>(ctx, dst); break;
        case 7: launch_dec_q<D, G, 7, kvq>(ctx, dst); break;
        case 8: launch_dec_q<D, G, 8, kvq>(ctx, dst); break;
        default: GGML_ABORT("dec_q: unsupported number of query columns");
    }
}

bool ggml_cuda_flash_attn_ext_dec_q_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (!mask || max_bias != 0.0f || logit_softcap != 0.0f || dst->src[4] != nullptr || Q->ne[3] != 1 || Q->ne[1] > 8 || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (K->type != V->type || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0) || K->nb[1] % 8 != 0 || V->nb[1] % 8 != 0 ||
            K->nb[2] % 8 != 0 || V->nb[2] % 8 != 0 || V->ne[0] != Q->ne[0]) {
        return false;
    }
    const int gqa = int(Q->ne[2] / K->ne[2]);
    return (Q->ne[0] == 256 && gqa == 6) || (Q->ne[0] == 128 && gqa == 4);
}

void ggml_cuda_flash_attn_ext_dec_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const bool q4 = K->type == GGML_TYPE_Q4_0;
    if (Q->ne[0] == 256) {
        if (q4) { switch_nb<256, 6, 1>(ctx, dst); } else { switch_nb<256, 6, 2>(ctx, dst); }
    } else {
        if (q4) { switch_nb<128, 4, 1>(ctx, dst); } else { switch_nb<128, 4, 2>(ctx, dst); }
    }
}
