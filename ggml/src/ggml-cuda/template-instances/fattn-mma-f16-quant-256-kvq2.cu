#include "../fattn-mma-f16.cuh"
// fused-quant MMA decode kernel instances (kvq 2 = q8_0), RDNA 2026-11-07

template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 2, 2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
template void ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 4, 4, 2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
