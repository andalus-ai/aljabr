# Metal inference benchmark protocol

Use this protocol before claiming a Metal decode or prefill optimization. A command's wall time includes process startup, model loading, prompt processing, generation, and shutdown; it is not a decode-step measurement.

## Repeatable end-to-end runs

From a checkout with the AQLI CLI available on `PATH`:

```bash
python3 scripts/benchmark-aqli.py \
  --model hf:unsloth/gemma-4-12b-it-GGUF \
  --prompt "who are you" \
  --max-tokens 50 \
  --warmup 1 \
  --runs 5
```

The runner writes `build/metal-benchmark/runs.csv` and one combined output log per run. Keep model revision/quantization, prompt, token limit, runtime flags, power mode, and machine state unchanged between revisions. Compare medians and dispersion, not a single run. The optional `--disable-metal` flag sets `-Daljabr.metal.disable=true` for a reference comparison; it is not guaranteed to select an equivalent or faster fallback for every model.

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
