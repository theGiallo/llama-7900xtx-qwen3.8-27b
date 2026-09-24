# RX 7900 XTX measurement scripts

Tools for step 1 of [`CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md`](../../CLAUDE_OPUS55_SUGGESTIONS_AND_PLAN.md):
confirm whether decode at long context is limited by quantized-KV FlashAttention.

## One command

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target test-backend-ops llama-bench

scripts/rx7900xtx/run-step1.sh /path/to/Qwen3.8-27B-ROCMFP4-STRIX.gguf
# optional: -d 0,65536,131072,262144   -k q4_0,f16   -o out-dir   -B build-dir
```

`decode_depth_fit.py` reads the GGUF with the in-tree `gguf-py`, which needs `numpy`
(`pip install numpy`); without it the KV bandwidth line is skipped.

Results land in `step1-<timestamp>/` as `fattn_bw.md` and `decode_depth_fit.md`, plus the raw
SQL, JSONL and logs. Please share both `.md` files.

## Pieces

| file | what it does |
|---|---|
| `fattn_bw.py` | `test-backend-ops perf -o FLASH_ATTN_EXT` for D=256, 4 KV heads, gqa 6, n_kv 16k–256k, batch 1–8, f16/q8_0/q4_0 KV. Reports effective GB/s (K+V counted once), % of 960 GB/s, verify cost `t(nb)/t(1)`, and the kernel chosen. |
| `decode_depth_fit.py` | `llama-bench -p 0 -n 64 -d <depths>` per model and KV type; fits `ms/token = fixed + slope × depth`; converts the slope to KV GB/s using the attention tensor shapes read from the GGUF. |
| `run-step1.sh` | Runs both and stores everything in one directory. |

## Kernel choice logging

`GGML_CUDA_FATTN_LOG=1` (any CUDA/HIP binary, e.g. `llama-server`) prints one line per
distinct FlashAttention configuration, straight to stderr (not through the ggml log callback, so
`llama-bench` without `-v` and `llama-server` at default verbosity show it too):

```
fattn: kernel=VEC D=256/256 n_q=1 n_head=24 n_head_kv=4 gqa=6 K=q4_0 V=q4_0 n_kv=113408 K+V=... f16_conv_K=0 f16_conv_V=0 conv_f16=0.0 MiB
```

`f16_conv_K/V=1` means the whole visible K/V is converted to f16 before the kernel runs.
Running `llama-server` with speculative decoding and this variable set shows what the
verify batches (`n_q` = 2..8) use.

## What would confirm the diagnosis

- `fattn_bw.md`: q4_0 at `nb=1` well below peak (e.g. < 30 %), and f16 reaching a
  clearly higher share; q4_0 rows with `nb >= 3` marked `TILE+f16conv`.
- `decode_depth_fit.md`: KV bandwidth from the slope far below peak for q4_0, and
  f16 KV having a smaller ms-per-1k slope than q4_0 despite 3.5× the bytes.
