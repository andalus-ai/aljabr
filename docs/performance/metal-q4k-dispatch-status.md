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

Developer-reported Apple Silicon run, K=3840, N=4096, Q4_K fixture, five warmups and twenty measured calls:

| Batch | B4 mean | MR4 mean | MR4 faster |
|---:|---:|---:|---:|
| 2 | 0.684 ms | 0.380 ms | 1.80x |
| 4 | 1.018 ms | 0.483 ms | 2.11x |
| 8 | 1.915 ms | 0.801 ms | 2.39x |
| 15 | 3.725 ms | 1.364 ms | 2.73x |
| 16 | 3.769 ms | 1.495 ms | 2.52x |
| 17 | 4.199 ms | 1.532 ms | 2.74x |
| 20 | 4.664 ms | 1.771 ms | 2.63x |
| 24 | 5.530 ms | 2.080 ms | 2.66x |
| 27 | 6.469 ms | 2.309 ms | 2.80x |
| 28 | 6.450 ms | 2.349 ms | 2.75x |
| 31 | 7.384 ms | 2.633 ms | 2.80x |
| 32 | 7.377 ms | 2.697 ms | 2.74x |

Interpretation: MR4 wins at every measured batch size for this one shape. These are isolated native dispatch-call timings, not end-to-end inference throughput. Do not extrapolate them to all dimensions, models, or Apple GPUs.

## Dispatch policy

- Keep MR4 as the performance baseline for the measured K=3840/N=4096 case.
- Keep B4 available for controlled experiments; the current AUTO selector only selects it through batch 15.
- AUTO currently rejects Q4_K batches above 15 rather than automatically switching to MR4; callers requiring larger batches must use the explicit MR4 mode. This behavior should be reviewed as a separate dispatch-policy task.
- A forced benchmark mode is not production eligibility and must remain opt-in.

## Known limitations / next work

1. Extend the benchmark across several K/N shapes and representative real Q4_K model weights.
2. Capture Metal pipeline resource usage or GPU traces to test whether B4's larger live accumulator set causes register pressure/occupancy loss. This is currently a hypothesis, not a confirmed cause.
3. Compare end-to-end prefill/decode behavior after isolated kernel improvements; isolated matvec latency alone is insufficient.
4. Revisit dispatch thresholds only after parity and repeatable shape-specific performance evidence.

## Validation record

The latest developer-provided local run passed the focused `MetalQ4KDispatchBenchmarkTest`; B4/MR4 parity assertions passed for its tested batches. The focused test requires a macOS Metal device and cannot be fully validated by generic CI runners. The benchmark fixture is deterministic synthetic Q4_K data; a separate validation against representative model-produced Q4_K blocks remains necessary.
