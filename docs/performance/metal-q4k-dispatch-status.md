# Metal Q4_K dispatch: implementation and progress

Updated: 2026-10-30

## Scope

This status tracks the Aljabr Metal Q4_K batched matrix-vector implementation and the evidence supporting its dispatch policy. It is an engine/backend capability, not an AQLI serving feature.

## Implemented

- Q4_K single-row Metal matvec kernel.
- Q4_K MR4 batched matvec kernel (one token tile, multiple output rows).
- Q4_K MR8 kernel where the corresponding pipeline and runtime policy enable it.
- Experimental B4 kernel, which tiles two input tokens and reuses the decoded Q4_K weight tile.
- Native opt-in forced B4/MR4 benchmark entry point.
- JUnit benchmark that verifies B4/MR4 output parity before collecting timings and emits a CSV report under the backend build reports directory.
- Correctness guard in automatic production dispatch; benchmark-only forced dispatch does not silently widen production eligibility.

## Latest measured result

Latest developer-reported Apple Silicon run (2026-10-30), K=3840, N=4096, deterministic Q4_K fixture, five warmups and twenty measured calls:

| Batch | B4 mean | MR4 mean | MR4 faster |
|---:|---:|---:|---:|
| 2 | 0.700 ms | 0.336 ms | 2.08x |
| 4 | 1.210 ms | 0.479 ms | 2.53x |
| 8 | 2.083 ms | 0.784 ms | 2.66x |
| 15 | 3.702 ms | 1.320 ms | 2.80x |
| 16 | 3.786 ms | 1.457 ms | 2.60x |
| 17 | 4.222 ms | 1.561 ms | 2.70x |
| 20 | 4.631 ms | 1.721 ms | 2.69x |
| 24 | 5.580 ms | 2.066 ms | 2.70x |
| 27 | 6.478 ms | 2.241 ms | 2.89x |
| 28 | 6.425 ms | 2.412 ms | 2.66x |
| 31 | 7.340 ms | 2.660 ms | 2.76x |
| 32 | 7.442 ms | 2.644 ms | 2.81x |

Interpretation: MR4 wins at every measured batch size for this one shape. These are isolated native dispatch-call timings, not end-to-end inference throughput. Do not extrapolate them to all dimensions, models, or Apple GPUs.

## Dispatch policy

- Keep MR4 as the performance baseline for the measured K=3840/N=4096 case.
- Keep B4 available for controlled experiments; the current AUTO selector only selects it through batch 15.
- AUTO selects the single-row kernel for batch 1, B4 for batches 2–15, and MR4 for batches above 15. The B4 threshold remains 15; larger batches use MR4 rather than widening B4 or rejecting a supported production batch.
- A forced benchmark mode is not production eligibility and must remain opt-in.

## Known limitations / next work

1. Extend the benchmark across several K/N shapes and representative real Q4_K model weights.
2. Capture Metal pipeline resource usage or GPU traces to test whether B4's larger live accumulator set causes register pressure/occupancy loss. This is currently a hypothesis, not a confirmed cause.
3. Compare end-to-end prefill/decode behavior after isolated kernel improvements; isolated matvec latency alone is insufficient.
4. Revisit dispatch thresholds only after parity and repeatable shape-specific performance evidence.

## Validation record

The latest developer-provided local run passed the focused `MetalQ4KDispatchBenchmarkTest`; B4/MR4 parity assertions passed for its tested batches. The focused test requires a macOS Metal device and cannot be fully validated by generic CI runners. The benchmark fixture is deterministic synthetic Q4_K data; a separate validation against representative model-produced Q4_K blocks remains necessary.

## Dispatch selector cleanup (2026-10-30)

Removed an unreachable duplicate MR4 selection branch from the Q4_K AUTO selector. Explicit MR4 selection is handled before AUTO selection, so the duplicate branch could never execute. This is a control-flow cleanup only: AUTO behavior is intentionally unchanged, and batches above 15 remain rejected by AUTO pending production-path MR4 correctness validation.

Validation required: rebuild the native Metal library and run the focused Q4_K dispatch benchmark/correctness tests on macOS. The source edit is not claimed as locally compiled or GPU-tested yet.


## Production-path MR4 fallback validation (2026-10-30)

- Updated Q4_K AUTO dispatch to retain the validated B4 range (batch 2–15) and route batches above 15 through the MR4 pipeline.
- Added `MetalQ4KProductionDispatchCorrectnessTest`, which calls the public production `matvecTransposedRightBatch` binding and compares every batched row against repeated production batch-size-1 calls.
- The test covers K/N shapes 1024/1024 and 3840/2048 and batch sizes 2, 4, 8, 15, 16, 17, 24, 27, 28, 31, and 32. It checks finite outputs and a documented relative tolerance.
- The test has been added but still requires a fresh native Metal build and execution on macOS/Apple Silicon. Do not consider the production MR4 fallback validated until that hardware test passes.
- The 2026-10-30 isolated benchmark passed with B4/MR4 parity for its synthetic fixture. That result is performance evidence only and does not replace the new production-path comparison.
