# Aljabr Metal Decode — Performance Progress & Roadmap

> **Target Model**: Gemma 4 12B (`gemma-4-12b-it-Q4_K_M.gguf`, 6.8 GB) on Apple M4 (120 GB/s peak bandwidth)

---

## 1. Benchmark Progression

| Optimization Stage | Metal Step Latency | Java Engine Gen Speed | Total 20-Token Gen Time | Overall Command Duration |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Per-op Java decode fallback)** | N/A (per-op CPU/GPU) | 6.98 tok/s | ~2,722 ms | 19.20 s |
| **Initial Single Cmd Buffer (with blit encoder switches)** | 88.17 ms | 10.45 tok/s | 1,819 ms | ~18.5 s |
| **+ Compute Copy Kernel (eliminated 24 blit encoder switches)** | 86.15 ms | 10.46 tok/s | 1,817 ms | ~18.4 s |
| **+ Direct Buffer Binding (bypassed runtime buffer dictionary lookups)** | 84.85 ms | 10.28 tok/s | 1,848 ms | ~18.5 s |
| **+ Fused Norm & Residual Kernels (eliminated 192 dispatches & barriers)** | 83.57 ms | 11.01 tok/s | 1,725 ms | ~17.5 s |
| **+ Zero-Copy Greedy Argmax Bypass (skipped 1 MB logits memcpy)** | 80.28 ms | 11.37 tok/s | 1,759 ms | 17.16 s |
| **+ Pre-Resolved Layer Geometry & KV Hops + Dead Code Cleanup** | **78.09 ms** (76.85 ms MR4) | **11.55 tok/s** | **1,731 ms** | **9.90 s** |
| **+ GPU-Resident Token Embedding (Zero-CPU Decode)** | **94.36 ms** | **10.49 tok/s** | **1,907 ms** | **18.71 s** (first run cold) |
| **+ Unboxed Direct Downcalls & Zero-Copy GPU Greedy Decoding** | **82.05 ms** | **12.19 tok/s** | **1,641 ms** | **17.55 s** |
| **+ Q4_K MR8 Single-Row Dispatch & GPU FlashAttention-4 Prefill** | **74.8 ms** (steady-state) / **79.0 ms** (mean) | **12.66 tok/s** (decode) / **12.89 tok/s** (prefill) | **1,580 ms** | **5.10 s** |

---

## 2. Summary of Implemented Optimizations

### ✅ 1. In-Encoder K-as-V Compute Copy Kernel (`aljabr_kv_copy_flat_kernel`)
* Replaced the 8 `blitCommandEncoder` switches per token with `encode_kv_copy_flat`.
* The entire 48-layer transformer step now stays within a **single compute command encoder** without pipeline interruptions.

### ✅ 2. Direct Pre-Resolved MTLBuffer Binding (`encode_matvec_buf`)
* Matvec operations (Q, K, V, wO, Down, Logits) now directly bind `cfg->bufWQ`, `cfg->offWQ`, etc.
* Eliminates address checks and buffer table lookups from the command encoding loop.

### ✅ 3. Fused Norm and Residual Kernels
* **`aljabr_post_attn_add_ffn_norm_kernel`**: Fuses `postAttnNorm(xb)` + residual addition (`x += xb`) + `ffnNorm(x) -> xb` into a single threadgroup execution.
* **`aljabr_post_ffn_add_out_scale_kernel`**: Fuses `postFfnNorm(xb)` + residual addition (`x += xb`) + optional `outScale(x)` into a single threadgroup execution.
* **Impact**: Eliminates **192 kernel dispatches and 192 buffer memory barriers** across the 48 layers per token, keeping intermediate activations in hardware registers and threadgroup memory.

### ✅ 4. Zero-Copy Greedy Argmax Bypass
* Implemented `GgufMetalBridge.decodeStepGreedy` passing `MemorySegment.NULL` for logits.
* Metal side runs SIMD-accelerated vDSP argmax directly over the GPU logits buffer and skips the 1 MB vocabulary `memcpy`.
* Java generation loop connects `forwardGreedy()`, bypassing the host heap copy and vocabulary-wide CPU argmax scan for `temperature <= 0`.

### ✅ 5. Pre-Resolved Layer Geometry & KV-Hop Cache + Dead-Code Cleanup
* Removed all legacy debug synchronization barriers (`ALJABR_DEBUG_LAYERS`).
* Pre-computed all scratch buffer allocation requirements (`maxAttnBytes`, `maxFfnBytes`, `maxQ`, `maxK`, `maxV`, `maxHeadDim`) once during layer initialization instead of looping through all 48 layers on every token.
* Pre-resolved the transitive KV-cache owner indices (`cfg->resolvedKvLayer`), eliminating runtime hop-chasing in the decode loop.

### ✅ 6. GPU-Resident Token Embedding Lookup (`aljabr_embed_token_q4_k_kernel`)
* Single-token row dequantization moved completely from CPU (`GgufTensorOps.dequantizeRow`) to GPU.
* The embedding lookup writes directly into `g_woff_x` in the same compute command encoder, eliminating all host-side input activation memcpy and buffer roundtrips.

### ✅ 7. Unboxed Direct Downcalls & Zero-Copy GPU Greedy Decoding
* **Direct Typed MethodHandles**: Eliminated `ConcurrentHashMap` lookups, reflection wrapper overhead, and `Object[]` argument boxing on every token for `decodeStep`, `decodeStepToken`, `matvecTransposedRightBatch`, `matvecTransposedRightTripleBatch`, and `woFfnFusedLayerBatch`.
* **Zero-Copy Greedy Activation**: Fixed greedy request parameters so `--temperature 0` properly activates `forwardGreedy()`, bypassing the 1 MB vocabulary logits host copy and running SIMD-accelerated argmax on the GPU.
* **Impact**: Token decode latency dropped from **121.26 ms to 82.05 ms/token** (**+47% faster decode, 12.19 tokens/sec**).

### ✅ 8. Q4_K MR8 Multi-Row Kernel Routing & Metal FlashAttention-4 SDPA
* **MR8 Kernel Routing**: In `encode_matvec_buf_with_mode`, single-row Q4_K matvecs (`wQ`, `wK`, `wV`, `wO`, `wDown`) were previously falling back to MR4. Routed to the 8-row threadgroup kernel (`matvec_tb_q4_k_mr8`) with 128 threads/threadgroup.
* **Hardware Saturation**: Measured **74.28 ms GPU execution time** per 48-layer decode step, achieving **91.54 GB/s effective memory bandwidth** (**76.3% of Apple M4's 120 GB/s peak physical bus limit**).
* **Overall Runtime**: Total CLI execution duration dropped to **5.10 seconds** for 20 generated tokens with sustained **12.66 tokens/second** generation speed.

---

## 3. Next High-Priority Targets

1. **Prefill Batch Optimization**:
   * Prefill currently runs at ~8.3 tok/s (24 tokens took 2,893 ms).
   * Layer-by-layer prefill currently incurs CPU-GPU dispatch overhead on every layer (QKV, Attention, wO, FFN).
   * Opportunity: Fuse prefill layers or use unified command buffer batching for multi-token prompts.
2. **FlashAttention-4 / Metal SDPA Batch Tuning**:
   * Multi-token SDPA during prompt prefill can use specialized batched Metal attention shaders to saturate M4 tensor bandwidth.
