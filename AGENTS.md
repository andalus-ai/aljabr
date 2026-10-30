# AGENTS.md — Aljabr contributor guidance

## Project boundaries

- Aljabr is the reusable machine-learning engine and backend framework: tensors, operators, kernels, model/weight formats, quantization, execution, and training-oriented primitives belong here.
- AQLI is the inference and serving runtime. Keep serving orchestration, CLI/runtime UX, request scheduling, and deployment concerns out of Aljabr unless they are reusable engine capabilities.
- Tafkir is a separate training project; do not introduce a dependency on Tafkir for functionality that belongs in Aljabr.
- Prefer the fully Java implementation and Java FFM for native interoperability. JNI and external runtimes are compatibility/fallback paths only where explicitly required; do not make them the default architecture.
- Preserve existing module boundaries and public APIs unless the task explicitly requires a migration. Avoid moving engine functionality into AQLI.

## Engineering principles

- Inspect the owning module, its callers, tests, and existing conventions before changing code.
- Prefer small, cohesive changes with explicit contracts, deterministic behavior, and useful failure messages.
- Preserve binary compatibility and validate native/FFM signatures together. Keep Java layouts, native declarations, exported symbols, and argument order synchronized.
- Do not silently turn unsupported shapes, formats, or backends into success. Validate dimensions, byte sizes, alignment, batch bounds, and native return codes.
- For Metal code, keep dispatch policy separate from kernel correctness. Never widen an automatic dispatch range based only on a benchmark or parity between two optimized kernels.
- Use independent references where possible: compare batched rows with repeated single-row production calls, and use representative format-valid fixtures rather than arbitrary byte patterns alone.
- Keep performance claims scoped to the exact device, shape, data format, warmup, and measurement method. Isolated kernel timings are not end-to-end model throughput.
- Do not claim local hardware validation unless the corresponding test was actually run on that hardware.

## Current Metal Q4_K dispatch policy

- Batch size 1 uses the single-row Q4_K path.
- Batch sizes 2–15 use B4 (the current production correctness-validated range).
- Batch sizes above 15 must use MR4; preserve the B4 threshold at 15 until broader correctness and performance evidence justifies changing it.
- Explicit forced B4/MR4 selection is for opt-in tests/benchmarks, not a production routing API.
- A benchmark comparing B4 against MR4 does not establish correctness against an independent reference.
- Any change to this policy must include production-entry-point tests comparing every batched output row against repeated single-row calls, including boundary sizes 15, 16, and 17 and larger batches.

## Tests and validation

- Run the narrowest relevant unit test first, then the owning module's test task, followed by broader checks when practical.
- Metal GPU tests require macOS with a working Metal device and a freshly built native library. CI on non-Apple hosts cannot substitute for those hardware checks.
- Before interpreting a Metal test result, verify which dylib was loaded; prefer the module-local build output over an installed copy to avoid stale-library false results.
- For Q4_K production-dispatch changes, exercise the public production batch binding, compare each row to a repeated batch-size-1 invocation, assert finite outputs, and use a documented tolerance.
- Keep synthetic deterministic fixtures for reproducibility, but add format-valid representative quantization data and multiple shapes before making broad claims.
- Benchmark only after correctness checks pass. Include warmups, repeated samples, median/percentile reporting, and CSV output where the project already follows that convention.
- Document validation status, known limitations, and exact hardware/test scope alongside performance changes.

## Repository workflow

- Work on the task branch requested by the maintainer. Do not force-push, reset, or overwrite divergent work.
- Preserve uncommitted work and backups. If local and remote histories diverge, inspect and reconcile explicitly rather than discarding either side.
- Keep commits focused and messages descriptive. Do not merge or change the default branch unless explicitly requested.
- Ask the maintainer to run local tests only when the result depends on their Apple Silicon/Metal hardware or other unavailable local environment; complete code inspection, implementation, documentation, and CI-oriented work autonomously first.
- Never report a test, build, benchmark, or hardware validation as passing unless its output was observed.
