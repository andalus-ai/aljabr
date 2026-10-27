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
