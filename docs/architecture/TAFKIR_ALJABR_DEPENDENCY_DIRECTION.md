# Tafkir–Aljabr Dependency Direction

## Status

Architecture decision and migration plan. This document records the intended dependency direction; it does not claim that the current source tree already conforms.

## Decision

The dependency direction is:

```text
Tafkir (training, adaptation, calibration, quantization workflows)
    └── depends on Aljabr (tensor, numerical, compute, and quantization contracts)

Aljabr
    └── MUST NOT depend on Tafkir
```

Aljabr is the lower-level computational foundation. Tafkir is a higher-level training and model-adaptation framework. The dependency must remain acyclic and point from Tafkir to Aljabr.

## Ownership boundaries

### Aljabr owns

- Tensor and numerical primitives, memory layouts, and compute backends.
- SafeTensors data structures, readers/writers, and format-level interoperability.
- Generic quantization metadata and low-level quantization contracts that are useful independently of a training framework.
- A stable quantizer SPI if inference/runtime consumers need pluggable quantization implementations.
- Generic, self-contained quantization algorithms that have no dependency on Tafkir training workflows.

### Tafkir owns

- Training-time and post-training adaptation workflows.
- Calibration datasets, optimization loops, and training-aware quantization.
- Algorithm implementations and workflow-specific adapters for GPTQ, AWQ, AutoRound, TurboQuant, and BitsAndBytes where they rely on Tafkir-owned code.
- Integration that connects Tafkir quantizers to Aljabr tensor/SafeTensors contracts.

A quantization algorithm is not automatically Tafkir-owned merely because it is used during training. The deciding factor is its dependency and responsibility: reusable low-level numerical kernels may live in Aljabr; training/calibration orchestration and Tafkir-specific implementations belong in Tafkir.

## Current coupling found during source inspection

The current Aljabr checkout has an optional `withTafkir` composite build in `settings.gradle.kts`. When enabled, it includes `../tafkir` and includes `core:aljabr-safetensor-quantization`.

That Aljabr module declares dependencies on these Tafkir artifacts:

- `tech.kayys.tafkir:tafkir-quantizer-gptq`
- `tech.kayys.tafkir:tafkir-quantizer-awq`
- `tech.kayys.tafkir:tafkir-quantizer-autoround`
- `tech.kayys.tafkir:tafkir-quantizer-turboquant`

Its source also directly imports Tafkir classes in `GPTQQuantizerAdapter`, `AWQQuantizerAdapter`, `AutoRoundQuantizerAdapter`, and `TurboQuantAdapter`. `BnBQuantizerAdapter` imports `BnBDequantizer` from the Tafkir TurboQuant package. The module's `QuantizationEngine` constructs and registers these adapters directly, and `QuantizerRegistry` directly constructs the GPTQ adapter.

This is a real compile-time dependency inversion, not just a Gradle configuration inconvenience.

The reverse direction already exists in Tafkir: its settings include a sibling Aljabr checkout as a composite build, and Tafkir compiler modules declare dependencies on Aljabr tensor/core artifacts. The Tafkir README also describes Aljabr as its foundation.

## Target integration design

1. Keep the Aljabr quantization contract independent of Tafkir. The contract should describe tensor input/output, quantization metadata, supported configurations, and capability discovery without importing Tafkir classes.
2. Move Tafkir-specific adapters and workflow implementations into a Tafkir-owned integration module (under Tafkir's existing Aljabr extension area or a dedicated quantization integration module).
3. Make that Tafkir integration module depend on Aljabr's quantization/SafeTensors contracts and on Tafkir's own quantizer implementations.
4. Register optional quantizers through an explicit extension mechanism (for example, Java `ServiceLoader` or an explicit registration API). Aljabr must not instantiate Tafkir classes or require Tafkir artifacts to compile or run.
5. Remove Aljabr's `withTafkir` composite-build switch once no Aljabr-owned project needs it. Tafkir may continue to include Aljabr as a composite build for local development.
6. Preserve standalone published-artifact consumption: a Tafkir integration consumer should resolve Aljabr artifacts from a Maven repository when no sibling checkout is present.
7. Add dependency-boundary checks that reject `tech.kayys.tafkir` references and Tafkir project dependencies from Aljabr production modules.

## Migration sequence

### Phase 1 — establish the boundary

- Keep this decision visible in Aljabr architecture documentation.
- Inventory every Tafkir import, dependency coordinate, composite-build hook, and source-level registration in Aljabr.
- Distinguish generic Aljabr quantization code from Tafkir-specific adapters before moving files.

### Phase 2 — move the integration

- Add the Tafkir-owned integration module.
- Move Tafkir-specific adapters there and make them consume Aljabr contracts.
- Replace Aljabr's direct adapter construction with built-in Aljabr implementations plus extension discovery.
- Preserve existing public API behavior where possible; document any intentionally unsupported strategy rather than silently selecting a different algorithm.

### Phase 3 — remove reverse dependency and verify

- Remove Tafkir artifact declarations and `withTafkir` from Aljabr.
- Build Aljabr without a Tafkir checkout.
- Build Tafkir with Aljabr supplied by composite build and with published Aljabr artifacts.
- Add an automated dependency-direction test to both repositories.

## Acceptance criteria

- No Aljabr production source imports `tech.kayys.tafkir.*`.
- No Aljabr Gradle project declares `tech.kayys.tafkir` dependencies.
- Aljabr settings do not include Tafkir as a composite build.
- Tafkir integration modules depend on Aljabr contracts and Tafkir implementations, never the reverse.
- Core Aljabr builds and tests pass without a Tafkir checkout.
- Tafkir builds/tests pass with the supported Aljabr integration mode.
- Optional quantizers are discoverable without hard-coded Tafkir classes in Aljabr.

## Scope and compatibility notes

This decision does not require moving generic quantization metadata, tensor operations, SafeTensors readers/writers, or independent INT8/FP8 implementations out of Aljabr. It also does not require every quantizer implementation to move to Tafkir. Ownership follows the dependency and responsibility boundaries above.

The migration must not remove an algorithm or change its output format as an incidental side effect of removing the dependency inversion. Any algorithm that cannot yet be moved behind the SPI should remain clearly marked as a migration blocker rather than introducing a new reverse dependency.
