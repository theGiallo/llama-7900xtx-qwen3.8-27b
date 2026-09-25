#include "fattn-vec-gqa.cuh"

#define DECL_FATTN_VEC_GQA_CASE(D, GQA, type_K, type_V) \
    template void ggml_cuda_flash_attn_ext_vec_gqa_case \
    <D, GQA, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst); \

#define DECL_ALL_GQA(D, type)     \
    DECL_FATTN_VEC_GQA_CASE(D,  2, type, type)   \
    DECL_FATTN_VEC_GQA_CASE(D,  4, type, type)   \
    DECL_FATTN_VEC_GQA_CASE(D,  6, type, type)   \
    DECL_FATTN_VEC_GQA_CASE(D,  8, type, type)   \

DECL_ALL_GQA(256, GGML_TYPE_Q4_0)
DECL_ALL_GQA(256, GGML_TYPE_Q8_0)

typedef void (* fattn_vec_gqa_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static bool ggml_cuda_get_fattn_vec_gqa_case(const int D, const int gqa, const ggml_type type_K, const ggml_type type_V, fattn_vec_gqa_case_t & out) {
    if (D != 256 || type_K != type_V) {
        return false;
    }
    switch (type_K) {
        case GGML_TYPE_Q4_0:
            switch (gqa) {
                case  2: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  2, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>; return true;
                case  4: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  4, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>; return true;
                case  6: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  6, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>; return true;
                case  8: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  8, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>; return true;
            }
            break;
        case GGML_TYPE_Q8_0:
            switch (gqa) {
                case  2: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  2, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>; return true;
                case  4: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  4, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>; return true;
                case  6: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  6, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>; return true;
                case  8: out = ggml_cuda_flash_attn_ext_vec_gqa_case<256,  8, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>; return true;
            }
            break;
        default:
            break;
    }
    return false;
}

bool ggml_cuda_fattn_vec_gqa_supported(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    fattn_vec_gqa_case_t unused = nullptr;
    return ggml_cuda_get_fattn_vec_gqa_case(int(Q->ne[0]), int(Q->ne[2]/K->ne[2]), K->type, V->type, unused);
}


void ggml_cuda_flash_attn_ext_vec_gqa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_gqa_case_t gqa_case = nullptr;
    const bool supported = ggml_cuda_get_fattn_vec_gqa_case(int(Q->ne[0]), int(Q->ne[2]/K->ne[2]), K->type, V->type, gqa_case);
    GGML_ASSERT(supported);
    gqa_case(ctx, dst);
}
