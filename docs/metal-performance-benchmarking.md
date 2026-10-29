# Metal inference benchmark protocol

Use this protocol before claiming a Metal decode or prefill optimization. A command's wall time includes process startup, model loading, prompt processing, generation, and shutdown; it is not a decode-step measurement.

## Repeatable end-to-end runs

From a checkout with the AQLI CLI available on `PATH`:

```bash
java scripts/BenchmarkAqli.java \
  --model hf:unsloth/gemma-4-12b-it-GGUF \
  --prompt "who are you" \
  --max-tokens 50 \
  --warmup 1 \
  --runs 5

# Optional JBang source-file launcher:
jbang scripts/BenchmarkAqli.java --model hf:unsloth/gemma-4-12b-it-GGUF --prompt "who are you" --max-tokens 50 --warmup 1 --runs 5
```

The JDK 25 source-file runner uses only the Java standard library. It writes `build/metal-benchmark/runs.csv` and one combined output log per run. JBang is an optional launcher, not a runtime dependency. Keep model revision/quantization, prompt, token limit, runtime flags, power mode, and machine state unchanged between revisions. Compare medians and dispersion, not a single run. The optional `--disable-metal` flag sets `-Daljabr.metal.disable=true` for a reference comparison; it is not guaranteed to select an equivalent or faster fallback for every model.

## Java decode-profile summary

The Java source-file tool replaces the Python profile summarizer and supports table, CSV, JSON, multiple log files, or standard input:

```bash
java scripts/MetalDecodeProfileSummary.java metal-fused.log
java scripts/MetalDecodeProfileSummary.java metal-fallback.log
java scripts/MetalDecodeProfileSummary.java --warmup 2 --format csv metal-fused.log
java scripts/MetalDecodeProfileSummary.java --format json metal-fused.log

# Multiple files are grouped by the reported execution path:
java scripts/MetalDecodeProfileSummary.java metal-fused.log metal-fallback.log
```

The profile summary reports median, nearest-rank p95, min, and max for `wall_ms`, `encode_ms`, `submit_wait_ms`, and `gpu_ms`, grouped by path. Warm-up samples are discarded independently for each path. These profile values are diagnostic and should be compared with unprofiled benchmark runs; do not treat instrumented timings as final performance numbers.

## What this runner does and does not measure

- Records end-to-end wall time, exit status, command, model, prompt, machine/OS, and raw logs.
- Retains a reported tokens/second value only when it can recognize one in the CLI output. This is diagnostic metadata; the CLI's timing scope may differ between versions.
- Does **not** isolate model load, prefill, individual decode steps, GPU command-buffer time, or CPU fallbacks.
- Does **not** claim that a run used the full-GPU decoder. Verify the runtime's decode-eligibility diagnostics and active backend separately.

## Before/after report checklist

Include the exact commit SHA, JDK and AQLI/Aljabr versions, model file/revision and quantization, prompt and generated-token count, warm-up policy, number of measured runs, median and min/max, and the separate prefill/decode metrics if the runtime exposes them. If only end-to-end time is available, label it as such.

For GPU token embedding or prefill changes, correctness is a hard gate: compare embedding outputs or next-token logits against the Java reference, verify greedy token IDs on fixed prompts, and test short and longer prompts. Keep eligibility guards for unsupported per-layer embeddings, partial RoPE, or KV layouts until their implementation and parity are proven.


## Aggregate prefill profiling (opt-in)

Enable the aggregate per-prefill summary with either:

```bash
ALJABR_METAL_PREFILL_PROFILE=1 aqli run --model hf:unsloth/gemma-4-12b-it-GGUF --no-banner --prompt --java "who are you" --max-tokens 50
```

or add `-Daljabr.metal.prefill.profile=true` to the Java process. The summary reports the prompt-token count, layer-loop time, accumulated QKV projection time, attention time, and how many layers used Metal FA4 attention versus Java/fallback attention. It also reports output-projection and FFN timings for non-fused layers, plus combined WO+FFN time for fused layers. These are inclusive phase totals across layers, not additive wall-clock buckets: host-side normalization, RoPE, KV writes, copies, and dispatch overhead can overlap the measured regions or remain outside them. Do not sum the fields to estimate total prefill time.

Keep this profiler disabled for final performance comparisons. Use it to identify the dominant phase, then compare unprofiled repeated runs with the same model, prompt, token limit, and backend eligibility. The current output is deliberately aggregate-level; add finer per-kernel GPU timestamps only when the aggregate profile identifies a phase that needs further decomposition.

## Q4_K batched matvec dispatch and correctness

The Metal Q4_K batched matvec dispatch uses the following policy:

- **Batch size 1:** prefer MR8 when its pipeline is available and MR8 is enabled; otherwise use MR4 when available, with the single-row base kernel as the final fallback.
- **Batch sizes 2–15:** use the B4/MR4 kernel when its pipeline is available. This range is the current correctness-validated B4 range.
- **Batch sizes greater than 15:** require the batch-aware MR4 kernel. If that pipeline is unavailable, dispatch fails rather than using the single-row base kernel, which cannot correctly write every row of a multi-row batch.

`MetalTripleBatchCorrectnessTest` compares batched mixed-quantization results against repeated single-row calls. The current local test run passed both correctness test cases, including the representative-weight variant, with zero failures or errors. The wider Metal backend test run also reported zero failures and errors; one benchmark test was skipped.

The batch-size boundary is a correctness-based dispatch guard, not a claim that MR4 is faster than B4 for every batch size. Revisit it only after correctness parity and repeatable performance measurements support a change.
