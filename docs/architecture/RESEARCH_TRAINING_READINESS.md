# Research and training readiness audit

**Audit baseline:** 2026-10-29  
**Purpose:** distinguish existing source code from capabilities required for Aljabr to serve as a PyTorch-like Java research framework.

This is a source-level audit, not a claim that every item was compiled or executed. Findings below must be resolved through executable tests and CI before a training workflow is advertised as working.

## Target responsibility split

- **Aljabr:** tensor/autograd primitives, differentiable operators, neural-network modules and parameters, optimizer algorithms, reusable training primitives, numerical tests and checkpoint formats/APIs. These are library capabilities needed by researchers and application developers.
- **Tafkir:** higher-level training orchestration, dataset/training pipelines, fine-tuning/adaptation workflows, experiment orchestration and optional integrations built on Aljabr.
- **AQLI:** inference and serving. It consumes Aljabr model-execution APIs and must not depend on training-only modules for ordinary inference.

This refines the boundary between foundational ML capabilities and training workflow orchestration. It does not make Tafkir mandatory for Aljabr or AQLI.

## Findings confirmed in the source tree

### 1. Autograd is excluded from the default Gradle build

Root `settings.gradle.kts` sets `skipAutograd = true` and only includes `:core:autograd` when that flag is false. Therefore the source under `core/autograd` is not exercised by the ordinary project build unless a separate build path enables it.

**Risk:** source presence and standalone compile status can be mistaken for supported framework functionality.

### 2. The documented canonical trainer paths are not present in the current repository tree

`core/autograd/README.md` refers to `:trainer:aljabr-trainer-api` and `:trainer:aljabr-trainer` under `trainer/`, but the inspected `main` tree has no `trainer/` directory. The autograd Gradle file references the published `aljabr-trainer-api:0.1.0-SNAPSHOT` artifact.

**Risk:** the documented migration destination and the source tree/build graph may be out of sync. Confirm whether these modules live in another branch/repository or whether the README/dependencies are stale before moving code.

### 3. Optimizer implementations include explicit no-op paths

- `core/autograd/src/main/java/tech/kayys/aljabr/train/SGD.java`: `step(Collection<Tensor>)` contains placeholder logic and does not update parameter values.
- `core/autograd/src/main/java/tech/kayys/aljabr/train/AdamW.java`: `step(Collection<Tensor>)` is empty.
- Both have a named-map overload, but those methods replace entries with newly allocated tensors. Their update semantics, temporary tensor cleanup, state cleanup, parameter-map mutability, and numerical parity require tests.

**Risk:** the collection-based optimizer API looks functional while silently performing no update. A training example must not use this path until the contract is repaired.

### 4. Autograd graph construction has incomplete capability signaling

`AutogradEngine.requiresGrad(GOp)` currently returns `true` unconditionally. The backward builder skips operations whose gradient function is missing, without surfacing a typed unsupported-gradient result.

**Risk:** a graph can appear to build a backward pass even when gradient coverage is incomplete. Add explicit gradient capability checks and tests for unsupported operators before claiming general autograd support.

### 5. Neural-network module APIs need canonicalization

`core/aljabr-nn` defines `NNModule`, while legacy autograd sources also reference a separate `tech.kayys.aljabr.nn.Module` API. The module and parameter abstractions need an inventory of public consumers before choosing a single canonical model API. Avoid mass package moves until binary/source compatibility and resource ownership are understood.

## Required implementation sequence

1. **Build visibility:** decide how the autograd/training primitives are included in an explicit, reproducible Gradle task without making them mandatory for inference-only builds.
2. **Autograd correctness:** gradient checks for supported operations; accumulation, fan-out, missing-gradient errors, graph lifecycle, and finite-difference reference tests.
3. **Optimizer contracts:** remove silent no-op behavior; specify whether parameters are mutated or replaced; support named parameters and collection parameters consistently; validate hyperparameters and gradient shape/device/dtype; release temporary tensors and optimizer state safely.
4. **Canonical model API:** choose one module/parameter registration contract and document ownership of parameter tensors and gradients.
5. **Executable research example:** train a tiny deterministic linear-regression model in Java; assert loss decreases and learned parameters approach known values. Run it in CI against the CPU reference backend.
6. **Backend extension:** test numerical parity on Metal and other backends only after the CPU reference example is stable.
7. **Tafkir integration:** move orchestration and adaptation workflows to Tafkir incrementally while keeping core tensor/autograd/optimizer primitives reusable in Aljabr.

## Acceptance criteria

- No optimizer API silently succeeds without updating parameters.
- Supported gradient operations have numerical gradient tests; unsupported operations fail explicitly.
- A minimal Java training example is compiled and run by CI, with deterministic data and asserted convergence.
- Default inference-only AQLI dependency resolution does not pull in Aljabr training modules or Tafkir.
- The public module/model/optimizer contracts and their resource-ownership rules are documented.
- Source-tree, Gradle inclusion and README claims agree.

## Explicit non-goals for this audit

This document does not claim that all current optimizers, autograd operations or module abstractions are production-ready. It does not prescribe a full API rewrite before call sites and tests are inspected.
