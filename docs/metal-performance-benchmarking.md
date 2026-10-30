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
- **Batch sizes greater than 15:** AUTO selects MR4. It does not widen the B4 range or fall back to a per-row dispatch loop.

`MetalTripleBatchCorrectnessTest` compares batched mixed-quantization results against repeated single-row calls. The current local test run passed both correctness test cases, including the representative-weight variant, with zero failures or errors. The wider Metal backend test run also reported zero failures and errors; one benchmark test was skipped.

The batch-size boundary is a correctness-based dispatch guard, not a claim that MR4 is faster than B4 for every batch size. Revisit it only after correctness parity and repeatable performance measurements support a change.


## Q4_K B4 versus MR4 measured status (Apple Silicon, 2026-10-30)

The opt-in isolated dispatch benchmark was run on the developer's Apple Silicon Mac with the following fixed shape and protocol:

- Matrix shape: K=3840, N=4096; Q4_K weights.
- Batch sizes: 2, 4, 8, 15, 16, 17, 20, 24, 27, 28, 31 and 32.
- Five warm-up calls and 20 measured calls per path/batch.
- Forced B4 and MR4 outputs were compared before timing; all tested batches passed the benchmark's B4-versus-MR4 parity assertion.
- Timings below are host-observed call latency in milliseconds, including the native dispatch call and command completion wait. They are not full model decode tokens/s.

| Batch | B4 mean ms | MR4 mean ms | MR4 speedup |
|---:|---:|---:|---:|
| 2 | 0.700 | 0.336 | 2.08x |
| 4 | 1.210 | 0.479 | 2.53x |
| 8 | 2.083 | 0.784 | 2.66x |
| 15 | 3.702 | 1.320 | 2.80x |
| 16 | 3.786 | 1.457 | 2.60x |
| 17 | 4.222 | 1.561 | 2.70x |
| 20 | 4.631 | 1.721 | 2.69x |
| 24 | 5.580 | 2.066 | 2.70x |
| 27 | 6.478 | 2.241 | 2.89x |
| 28 | 6.425 | 2.412 | 2.66x |
| 31 | 7.340 | 2.660 | 2.76x |
| 32 | 7.442 | 2.644 | 2.81x |

At this shape, MR4 is consistently faster; at batch 32 it measures 2.644 ms versus 7.442 ms for B4 (2.81x). The result does not justify widening the production B4 range. Keep the automatic dispatch policy conservative and MR4 as the measured performance baseline for this shape.

### Kernel implementation notes and next investigation

The source-level comparison shows that the B4 kernel processes two input tokens per threadgroup tile while retaining four independent accumulators (two output rows for each of two tokens). It constructs separate input-vector fragments and dot-product accumulations for both tokens, while decoding the same Q4_K weight tile. The MR4 path processes one token and reuses decoded weight fragments across two output rows. This makes B4's extra live values and duplicated per-token arithmetic a plausible explanation for its poorer result, but register pressure or occupancy has not yet been confirmed by a Metal GPU capture or compiler resource report.

Do not label register pressure as a measured root cause. The next optimization step is to collect shape-diverse measurements and, where available, Metal pipeline resource/occupancy evidence before changing the shader. Preserve the production B4 boundary at 15; batches above it use MR4. Revisit that boundary only after broader correctness and shape-diverse performance evidence.

### Current capability status

- **Implemented:** Q4_K single-row matvec, MR4 batched matvec, MR8 path where enabled, B4 two-token experimental path, opt-in forced-dispatch benchmark hook, and benchmark-side output parity checks.
- **Validated in the reported local run:** B4/MR4 parity for the deterministic benchmark fixture at the tested batch sizes; measured latency comparison for K=3840/N=4096.
- **Experimental:** B4 dispatch beyond batch 15 and any claim that B4 is a generally faster multi-token strategy.
- **Not yet established:** performance across a representative matrix-shape/model suite, real GGUF Q4_K weight corpus parity for this benchmark path, and the precise hardware bottleneck responsible for the B4 slowdown.


### Production AUTO-to-MR4 fallback

The production Q4_K batch entry point now routes batch sizes above 15 to MR4 while retaining B4 for batches 2–15. A new `MetalQ4KProductionDispatchCorrectnessTest` exercises the public batch binding at K/N shapes 1024/1024 and 3840/2048, comparing each batched row with a repeated calls to the dedicated production single-row Q4_K API across batches 2–32. This new test requires a fresh native Metal build and local Apple Silicon execution; it has not yet been run.
