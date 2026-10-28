# GGUF Metal prefill and decode performance plan

This plan separates end-to-end throughput from phase throughput and defines the algorithmic work required to accelerate prompt prefill on Metal. It is a design and validation plan, not a claim that the target latency has already been achieved.

## 1. Measure the phases correctly

For a request with 24 prompt tokens, 19 generated tokens, 2,628 ms prefill, and 1,701 ms autoregressive decode:

| Metric | Formula | Result |
| --- | --- | ---: |
| Prefill throughput | prompt tokens / prefill seconds = 24 / 2.628 | 9.13 tokens/s |
| Decode throughput | decoded tokens / decode seconds = 19 / 1.701 | 11.17 tokens/s |
| End-to-end output rate | output tokens / full generate-call seconds = 19 / 4.327 | 4.39 tokens/s |
| Decode latency | decode milliseconds / decoded tokens = 1,701 / 19 | 89.53 ms/token |

The end-to-end output rate intentionally includes prompt prefill. It must not be labelled as pure autoregressive decode throughput. The open/model-load phase should be reported separately from the generate-call interval. If generation ends early or includes a stop token, use the runtime's explicit sampled/decoded/output token counters consistently and document which counter is used.

Useful CLI output should look like:

```text
Performance Metrics:
  open time      =    718.41 ms
  prompt prefill =   2628.00 ms (24 tokens @ 9.13 t/s)
  generation     =   1701.00 ms (19 decoded @ 11.17 t/s)
  decode latency =     89.53 ms/decoded token
  end-to-end rate=      4.39 t/s (19 output tokens / 4.327 s)
```

These are illustrative numbers supplied for the example; actual output must come from measured phase timers and counters.

## 2. Why prefill needs a different algorithm

Autoregressive decode usually processes one new token per layer. Its linear projections behave largely like matrix-vector products (GEMV), and attention reads the existing KV cache. Prompt prefill processes a sequence of tokens at once. Running the same per-token projection loop for every prompt token leaves available GPU parallelism and matrix-multiply throughput unused.

For a prompt length (T), hidden width (D), and projection output width (N), a linear projection can be written as:

`Y[T,N] = X[T,D] × W[D,N]`

Use a batched matrix-multiply (GEMM) path for the token dimension rather than dispatching a separate matrix-vector operation for each token. For a dense projection the arithmetic work scales approximately as (2TDN) floating-point operations, but elapsed time depends on weight format, memory traffic, occupancy, launch overhead, attention, and synchronization. This estimate alone does not predict latency.

## 3. Proposed layer algorithm

For each transformer layer during prefill:

1. **Load and stage weights once per execution plan.** Respect GGUF tensor type and quantization block layout. Do not silently dequantize the full model into host arrays just to use GEMM. Reuse packed/decoded weight representations only when their cache size and lifetime are controlled.
2. **Batch the token projections.** Compute Q, K, and V for all prompt tokens with tiled Metal GEMM kernels. Fuse compatible projection work only when tensor layouts, quantization formats, and numerical behavior allow it.
3. **Apply per-token normalization and position transforms.** Preserve architecture-specific Q/K normalization, RoPE scaling, partial rotary dimensions, embedding scaling, and Gemma-family post-attention/post-FFN norms.
4. **Compute attention over the prompt.** Parallelize by head and query-token tile; apply the correct causal mask for causal attention and the model-specific sliding-window behavior. Gemma 4 SWA prefill must preserve the implementation's required bidirectional attention within the sliding window; do not apply a generic causal mask indiscriminately.
5. **Project attention output.** Run the output projection as a batched GEMM and preserve residual connections and required normalization order.
6. **Batch the feed-forward network.** Run gate/up projections as GEMMs, apply the correct activation/gating function (including GeGLU where required), then run the down projection as a GEMM.
7. **Keep intermediate activations device-resident.** Avoid host round-trips between projections, attention, and FFN. Reduce command-buffer and encoder switches where correctness and resource lifetime permit.
8. **Produce prompt KV state and final prompt logits.** Commit KV entries in the same token/position order expected by the incremental decoder. The final prompt position supplies the logits for the first generated token.

Autoregressive decode remains a distinct path initially. Optimize it separately with KV-cache locality, efficient quantized GEMV/matvec, fewer command submissions, and correct persistent device state. Do not assume a faster prefill kernel automatically improves decode.

## 4. Metal implementation sequence

1. **Establish a trustworthy baseline.** Record prompt length, output/decoded token counts, model revision and quantization, warm/cold state, device, JDK, backend route, per-phase timers, and median/dispersion across repeated runs.
2. **Instrument per-layer timings.** Separate QKV projection, attention, output projection, FFN, normalization/activation, KV writes, synchronization/copies, and dispatch overhead. Use profiling builds only for diagnosis and compare unprofiled runs for final claims.
3. **Implement a small, testable GEMM primitive.** Start with representative supported GGUF types and shapes. Tile token rows and output channels, use a K-loop appropriate to the quantization block format, accumulate at a precision supported by the reference, and handle tail dimensions explicitly.
4. **Compare GEMM against the existing Java reference.** Validate intermediate projection outputs and logits for short and longer prompts before wiring it into every layer.
5. **Integrate batched QKV, then FFN and output projection.** Keep a per-operation fallback and a model/shape eligibility guard. Roll out one operation at a time to make regressions attributable.
6. **Optimize attention separately.** Verify masking and sliding-window semantics first; then tune query/key tiles, softmax stability, KV writes, and fusion.
7. **Reduce dispatch and memory traffic.** Reuse command buffers/pipelines where safe, avoid redundant buffer copies, and keep activations on the GPU. Measure whether each change reduces elapsed time rather than counting fewer launches as a proxy.
8. **Run end-to-end benchmark gates.** Compare median prefill, decode, generate-call, and process wall times with identical prompts and output limits. Report correctness and throughput together.

## 5. Target and arithmetic

A proposed prefill target below 300 ms for the example is an optimization hypothesis, not a guaranteed result. From 2,628 ms to 300 ms would require an 8.76x prefill speedup.

If decode remained 1,701 ms and prefill reached exactly 300 ms, excluding any unchanged overhead:

- Generate-call time would be approximately 2,001 ms.
- End-to-end output rate would be (19 / 2.001 approx 9.50) tokens/s.
- Decode throughput would remain 11.17 tokens/s; prefill acceleration does not change that phase's rate.

Measure actual time because launch, synchronization, logits, sampling, and runtime overhead may not fit neatly into these two buckets. Do not advertise the sub-300-ms target until a repeatable benchmark proves it on the specified hardware and model.

## 6. Correctness and acceptance gates

- Compare intermediate activations/logits with the existing Java reference using dtype- and quantization-appropriate tolerances.
- Compare greedy next-token IDs on fixed prompts and generated text over multiple prompt lengths.
- Test empty/minimal prompts, short prompts, longer prompts, batch/tile tails, and context-boundary cases.
- Test each supported architecture's normalization, RoPE, attention mask, sliding-window, per-layer embedding, and KV-sharing rules.
- Ensure a failed native/GPU step cannot silently continue with a partially updated KV cache. Fall back only if the host cache remains complete; otherwise restart/rebuild the request state.
- Test CPU fallback and unsupported shapes/dtypes explicitly.
- Report median and dispersion over repeated warm-up and measured runs; retain raw logs and commit SHAs.

## 7. Recommended metrics

Use distinct names and scopes so dashboards do not conflate them:

- `aqli.llm.model.open_time`: model open/load interval.
- `aqli.llm.prefill.duration` and `aqli.llm.prefill.speed`: prompt-only phase.
- `aqli.llm.decode.duration`, `aqli.llm.decode.speed`, and `aqli.llm.decode.token_latency`: autoregressive phase.
- `aqli.llm.generation.duration`: full generate-call interval.
- `aqli.llm.generation.end_to_end_speed`: output tokens divided by full generate-call seconds.
- Per-layer/operator histograms or benchmark logs: QKV, attention, output projection, FFN, KV, copies, and dispatch.

Keep unit conventions explicit. The CLI can display milliseconds and tokens/second; telemetry should document whether duration values are milliseconds or seconds and must use a consistent unit across versions.
