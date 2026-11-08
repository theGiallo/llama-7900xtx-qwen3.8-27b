#include "../fattn-mma-f16.cuh"
// fused-quant MMA decode kernel instances (kvq 1 = q4_0), RDNA 2026-11-07

template void ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 2, 1>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
template void ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 4, 4, 1>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
