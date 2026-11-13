# ADR-0004: Tensor Autograd and Optimization Module Boundaries

- **Status:** Proposed
- **Date:** 2026-11-13
- **Decision owners:** Aljabr maintainers
- **Scope:** Aljabr tensor differentiation, optimizer algorithms, and legacy ML utility placement

## Context

The current Gradle project is `plugins/ml/autograd`, but it is not a working tensor autograd implementation.

The current `AutogradEngine` traverses `GGraph` / `GValueId` objects from the external `tech.kayys.aljabr:aljabr-ir:0.1.0-SNAPSHOT` artifact. Its `GradFn` rules construct IR operations rather than computing gradients for operations executed on Aljabr `Tensor` instances. The project also depends on an unpublished `aljabr-trainer-api` snapshot and is excluded from the default Gradle build by `withAutograd`.

The current project mixes several unrelated responsibilities:

- IR graph gradient construction;
- tensor and neural-network dependencies;
- optimizers and optimizer state;
- training loops and trainer compatibility classes;
- datasets and data loaders;
- metrics and evaluation;
- data augmentation;
- model hub, persistence and export helpers.

Enabling this project wholesale would introduce unresolved external artifacts and couple unrelated capabilities. Renaming the directory without correcting its contract would preserve these problems.

The tensor API already exposes `requiresGrad()`, `setRequiresGrad(boolean)`, `grad()`, `setGrad(Tensor)` and `backward()`. However, `DefaultTensor.backward()` explicitly throws when gradients are required. This is an API promise without an implemented execution path.

## Decision

### 1. Rename the module artifact, not the established Java package

The target project name and path are:

```text
plugins/ml/aljabr-autograd
Gradle path: :plugins:ml:aljabr-autograd
Java package: tech.kayys.aljabr.autograd
```

The Java package name is already descriptive and should remain stable. The module rename must not trigger an unnecessary package-wide rename.

The existing `plugins/ml/autograd` project is legacy migration input. It must not be silently treated as the implementation of the new module. Its removal or archival follows a source/consumer audit and migration of any genuinely used behavior.

### 2. Autograd is a tensor-level numerical capability

The new `aljabr-autograd` module provides reverse-mode automatic differentiation for eager Aljabr tensors. It must not require:

- the external `aljabr-ir` artifact for ordinary eager tensor differentiation;
- `aljabr-trainer-api` or a trainer implementation;
- AQLI, Tafkir, Farabi, model serving or dataset orchestration;
- a specific CPU/GPU backend implementation.

The initial engine should use an eager dynamic computation graph. Each differentiable result records the operation's parent tensors and the backward rule/context needed to propagate its gradient. Backward traversal is topological and accumulates contributions when a tensor participates in multiple paths.

The graph is associated with tensor values and their operation history, not with a process-global graph singleton. Gradient recording mode may be scoped and thread-aware, but graph ownership and tensor lifetime must remain explicit.

### 3. Preserve backend neutrality

The first correct implementation may express backward formulas using the public Aljabr tensor API. It must not reach into `CpuBuffer`, `MemorySegment`, Metal buffers, CUDA pointers or backend implementation classes.

Backend-specific fused backward kernels can be introduced later behind the same semantics and conformance tests. A backend that does not implement a required forward or backward capability must fail explicitly; it must not return fabricated gradients.

The autograd module may depend on the tensor API. The tensor API must not depend on the autograd implementation. Integration should use a narrow, reviewed tensor extension point or a wrapper/decorator boundary rather than creating a Gradle dependency cycle.

### 4. Define explicit gradient semantics

The public contract must define and test:

- leaf tensors and non-leaf tensors;
- whether a tensor requires gradients;
- scalar implicit gradient seeding and explicit upstream gradients for non-scalar outputs;
- gradient accumulation across repeated uses and repeated backward calls;
- gradient reset and graph retention/release behavior;
- no-grad/inference scopes;
- unsupported/non-differentiable operations;
- shape, dtype and device compatibility of gradients;
- broadcasting reduction in backward rules;
- ownership and release of saved tensors and graph references;
- behavior after tensors or their backend resources have been released.

Unknown differentiable operations must not be silently skipped. A missing rule must produce a diagnostic that identifies the operation and relevant tensor metadata.

In-place mutation is not part of the initial contract. Operations that can invalidate saved values must either be rejected during graph recording or require a future explicit mutation/versioning design.

### 5. Validate gradient correctness numerically

Each supported rule requires:

1. a focused analytical test with known values;
2. a finite-difference gradient check for representative smooth inputs;
3. edge cases for shapes, broadcasting and repeated graph use where applicable;
4. tests for non-gradient inputs and disabled gradient recording;
5. CPU reference execution, with backend conformance added as providers implement the capability.

The initial supported-operation list must be deliberately small and documented. Candidate first wave:

- add, subtract, multiply and divide;
- scalar arithmetic;
- matrix multiplication for the supported rank/layout contract;
- sum and mean reductions;
- common elementwise activations such as ReLU, sigmoid, tanh, exp, log and power.

Softmax, normalization, convolution, attention, embedding, indexing and quantized operations should be added only with correct saved context and independent gradient tests. Do not infer support from a forward kernel existing.

### 6. Separate optimizer algorithms from autograd and training orchestration

The target optimizer artifact is a separate, independently consumable module:

```text
plugins/ml/aljabr-optim
Gradle path: :plugins:ml:aljabr-optim
```

It owns reusable algorithms and optimizer state (for example SGD, Adam and AdamW), consuming parameters and gradients through stable Aljabr APIs. It does not own graph construction or backward traversal.

The optimizer module may depend on tensor and the parameter API it needs. It must not depend on the full autograd implementation merely to consume a gradient tensor. This permits externally computed gradients and alternative differentiation systems.

The current tensor arithmetic is predominantly functional/immutable. Optimizers must update stable parameter handles correctly, define ownership of replaced tensors and optimizer state, and release superseded resources. A collection-based method that cannot update its inputs must not remain advertised as a functioning optimizer step.

### 7. Keep training orchestration outside Aljabr autograd

Tafkir owns training workflows, epochs, data pipelines, callbacks, experiment lifecycle, fine-tuning and adaptation orchestration.

Aljabr may provide reusable numerical building blocks such as autograd, optimizers, loss operations, gradient clipping and learning-rate schedules where their ownership is coherent and their behavior is tested. It must not require a trainer runtime to use autograd or optimizers.

### 8. Audit the remaining legacy classes individually

Do not move every source file from `plugins/ml/autograd` into one replacement module.

- Trainer and training-loop classes: remove from the autograd boundary; assess Tafkir ownership and actual consumers.
- Dataset, batch and data-loader classes: assess against an independently useful data API; do not couple them to autograd.
- Metrics: keep evaluation algorithms independent from differentiation; choose a metrics/statistics extension boundary based on actual consumers.
- Augmentation: assess as an independent data/vision capability, not an autograd responsibility.
- Model hub, persistence and export: assess against model-format/model-I/O ownership and existing modules.
- IR graph gradient builder: retain only if an actual in-repository consumer requires it; if so, place it beside the owning IR/compiler contract as a distinct optional integration, not in tensor autograd.
- Optimizer implementations and state: migrate to `aljabr-optim` only after parameter update and resource ownership semantics are tested.

No legacy class should be migrated solely to preserve its current path.

## Intended dependency direction

```text
aljabr-tensor
      ^
      |
aljabr-autograd       aljabr-optim
      |                    |
      +------ aljabr-nn ---+
                 ^
                 |
              Tafkir
       (workflow orchestration)
```

This is a conceptual dependency map, not a mandate that every artifact depend on every other artifact. In particular, optimizer algorithms should not need autograd to consume gradients, and tensor must not depend on autograd.

AQLI remains a consumer of Aljabr computation/model APIs for inference; it does not depend on autograd or optimizer modules for ordinary inference.

## Migration sequence

1. **Inventory and boundary:** identify all legacy source files, external dependencies, tests and real consumers. Confirm the new artifact name and package compatibility.
2. **Tensor integration design:** choose and document the narrowest acyclic mechanism for recording operations from the existing `Tensor` API. Specify graph and saved-tensor ownership before implementation.
3. **Minimal engine:** create `plugins/ml/aljabr-autograd` with no external IR/trainer snapshot dependencies. Implement graph nodes, topological traversal, gradient accumulation and a small set of rules.
4. **Tensor API integration:** implement `backward`, gradient access/reset, no-grad behavior and explicit upstream gradient semantics without coupling the tensor module to the autograd module.
5. **Numerical validation:** add finite-difference and edge-case tests; report only the exact operations and backend combinations validated.
6. **Optimizer extraction:** create `plugins/ml/aljabr-optim` and migrate optimizer algorithms after stable parameter replacement and resource lifecycle semantics exist.
7. **Legacy retirement:** migrate or remove each remaining legacy responsibility based on consumers; remove `withAutograd` only when the new module is independently buildable and the legacy exclusion is no longer needed.

## Non-goals

- Reimplementing the entire external IR/compiler autograd system in the tensor module.
- Adding a mandatory global tape or process-wide mutable graph.
- Making every tensor operation differentiable in the first release.
- Pulling training orchestration, data science, model hub or serving into autograd.
- Claiming GPU/Metal/CUDA/HAT differentiation before backend-specific conformance tests pass.
- Treating successful compilation as evidence of numerical correctness.

## Acceptance criteria

The new module is ready for initial implementation review when:

- it is named `aljabr-autograd` and has no dependency on unpublished IR/trainer snapshots;
- the tensor-to-autograd integration has no Gradle dependency cycle;
- the graph lifecycle and gradient semantics are documented;
- supported rules have analytical and finite-difference tests;
- a simple model can compute gradients for its trainable parameters on the CPU reference backend;
- disabling gradient recording does not build a graph;
- unsupported operations fail clearly rather than silently omitting gradients;
- no AQLI/Tafkir/serving workflow is required to use the module.

Until those criteria pass, autograd is **planned/in progress**, not implemented or validated.
