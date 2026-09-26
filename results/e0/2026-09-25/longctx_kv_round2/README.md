# long-context KLD round-2 (review Q2 #2): Tier A candidates at 32k
- git: 5de4e6983
- reuses ref .kld from round 1: (see longctx_kv/ round1)
- LLAMA_ATTN_ROT_DISABLE=<unset: rotation ON/baked>
- PPL_CTX=32768  corpus=/home/thegiallo/agentic90k_plain.txt (113182 tok)  ref=/home/thegiallo/models/Qwen3.8-27B-Q8_0.gguf
- purpose: if Q4K_S / UD-Q3_K_XL also show KLD~0.9 / PPL-ratio~1.2, the round-1 32k blow-up is a setup/corpus artifact, not STRIX-specific.

