# Aljabr Computational Capability Inventory

**Snapshot:** 2026-11-01  
**Branch audited:** `main`  
**Method:** GitHub recursive source-tree inspection, Gradle settings/build-file inspection and targeted source review  
**Status:** Source-level inventory; no Gradle build or test task was executed for this review.

> This inventory records observable repository structure and representative source evidence. A class or module existing in the tree does not prove production readiness, broad numerical correctness, or support on every backend.

## 1. Executive findings

Aljabr already has meaningful computational foundations: a public tensor abstraction, tensor operations, a CPU SIMD backend, neural-network layers, tokenizer algorithms/contracts, model-family modules, and legacy autograd/training code.

However, the current repository structure is still strongly oriented around tensor/model execution and LLM/model-format integration. The inspected tree does **not** show dedicated first-class modules for general-purpose linear algebra, probability/distributions, statistical inference, classical ML algorithms, time-series analysis, or scientific computing.

The immediate recommendation is **not** to add a large set of new Gradle modules yet. First stabilize the tensor/numerical contracts, extract and validate the reusable statistics and ML capability surface, and separate model-runtime concerns from the computational foundation.

## 2. Repository shape observed

The recursive tree returned 6,408 entries (not truncated). At this snapshot it contains:

| Area | Observed structure | Interpretation |
|---|---:|---|
| Core Gradle build files | 24 under `core/*` | Tensor, NN, SPI, model-format, storage and legacy autograd areas |
| Backend Gradle build files | 11 | CPU and several accelerator/native integration projects |
| Model Gradle build files | 259 | Very broad model-family integration surface |
| Dedicated statistics/math/ML/NLP top-level directories | None found | No clearly separated domain-computation module family yet |

The 259 model build files are dynamically included by `settings.gradle.kts` when the model directory contains a Gradle build file. This is a large model integration catalogue; it should not be mistaken for 259 independently validated algorithms or fully supported model execution paths.

## 3. Existing capabilities and evidence

### 3.1 Tensor and numerical execution — present

Primary module: [core/aljabr-tensor](../../core/aljabr-tensor)

Evidence:
- `Tensor` exposes arithmetic, scalar operations, matrix multiplication, reshape, slicing, reductions, transpose, softmax/log-softmax, common activations, normalization, convolution, pooling, attention and embedding operations.
- `Shape` validates non-negative dimensions and detects element-count overflow.
- `TensorOps.einsum` provides an explicit-equation F32 CPU reference implementation for supported einsum cases. Its own documentation states that it enumerates the full index space and is not an optimized large-contraction kernel.
- Tensor factory, default tensor, lazy graph and memory/buffer abstractions exist.
- Tests cover tensor shape, dtype, tensor operations, lazy execution, memory pools and a tensor research quickstart.

Representative sources:
- [Tensor.java](../../core/aljabr-tensor/src/main/java/tech/kayys/aljabr/core/tensor/Tensor.java)
- [TensorOps.java](../../core/aljabr-tensor/src/main/java/tech/kayys/aljabr/core/tensor/TensorOps.java)
- [Shape.java](../../core/aljabr-tensor/src/main/java/tech/kayys/aljabr/core/tensor/Shape.java)
- [TensorResearchQuickstartTest.java](../../core/aljabr-tensor/src/test/java/tech/kayys/aljabr/core/tensor/TensorResearchQuickstartTest.java)

**Assessment:** good low-level starting point. A general linear-algebra API is not yet established by the existence of tensor `matmul` or einsum. The reviewed public tensor interface does not expose a complete solver/decomposition surface (for example LU/QR/SVD/eigenvalue/least-squares APIs).

### 3.2 CPU compute backend — present

Primary module: [backend/cpu/aljabr-backend-cpu](../../backend/cpu/aljabr-backend-cpu)

Evidence:
- CPU operations use the Java Vector API for pointwise operations and activations.
- Dedicated matrix multiplication, normalization and quantize/dequantize operation classes exist.
- The CPU backend has focused tests for selected operations.

Representative sources:
- [CpuOps.java](../../backend/cpu/aljabr-backend-cpu/src/main/java/tech/kayys/aljabr/backend/cpu/ops/CpuOps.java)
- [MatmulCpu.java](../../backend/cpu/aljabr-backend-cpu/src/main/java/tech/kayys/aljabr/backend/cpu/ops/MatmulCpu.java)

**Assessment:** a useful CPU execution path exists, but operator coverage, numerical tolerances, concurrency, edge cases and parity must be tracked per operation. Do not infer complete scientific-computing coverage from a backend implementation.

### 3.3 Neural-network building blocks — present

Primary module: [core/aljabr-nn](../../core/aljabr-nn)

The inspected source includes:
- Linear, convolution, pooling and embedding layers.
- Activations, normalization, dropout and transformer/attention components.
- Loss functions including regression, classification and divergence/ranking families.
- `NNModule` parameter, buffer, child-module and state-dictionary abstractions.

Representative sources:
- [NNModule.java](../../core/aljabr-nn/src/main/java/tech/kayys/aljabr/core/nn/NNModule.java)
- [Loss.java](../../core/aljabr-nn/src/main/java/tech/kayys/aljabr/core/nn/Loss.java)

**Assessment:** neural-network construction primitives are present. This is not equivalent to a validated end-to-end training framework; autograd, optimizer semantics, parameter ownership and backend gradient coverage remain separate gates.

### 3.4 Autograd, training utilities and ML metrics — present but opt-in / legacy

Primary source area: [core/autograd](../../core/autograd)

The source tree contains:
- Gradient-function implementations and an autograd engine.
- SGD and AdamW optimizer classes, optimizer state and gradient utilities.
- Dataset, batch and data-loader abstractions.
- Trainer/training-loop, checkpoint, precision and scheduler utilities.
- Regression and classification metrics, NLP metrics, cross-validation and randomized search utilities.

Important build facts:
- `settings.gradle.kts` excludes `:core:autograd` from the default build; it is included only with `-PwithLegacyAutograd=true`.
- `core/autograd/build.gradle.kts` references `aljabr-ir:0.1.0-SNAPSHOT` and `aljabr-trainer-api:0.1.0-SNAPSHOT`, in addition to local tensor/core projects.
- The reviewed SGD and AdamW collection-based `step(Collection<Tensor>)` APIs explicitly throw `UnsupportedOperationException`, because the tensor API does not provide mutable parameter assignment. The named-map overload replaces map entries with new tensor values.
- The source audit [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md) records additional graph/gradient and API canonicalization gaps.

Representative sources:
- [settings.gradle.kts](../../settings.gradle.kts)
- [SGD.java](../../core/autograd/src/main/java/tech/kayys/aljabr/train/SGD.java)
- [AdamW.java](../../core/autograd/src/main/java/tech/kayys/aljabr/train/AdamW.java)
- [RegressionMetrics.java](../../core/autograd/src/main/java/tech/kayys/aljabr/metrics/RegressionMetrics.java)
- [ClassificationMetrics.java](../../core/autograd/src/main/java/tech/kayys/aljabr/metrics/ClassificationMetrics.java)

**Assessment:** there is a useful but non-default legacy ML/training codebase. Do not advertise general training or classical ML support until the module is build-visible through a supported path and the documented acceptance tests pass. Metrics are not statistical inference.

### 3.5 NLP tokenizer foundations — present

Primary module: [core/aljabr-spi-tokenizer](../../core/aljabr-spi-tokenizer)

The source tree includes tokenizer interfaces and implementations/contracts for BPE, GPT-2 pre-tokenization, Hugging Face BPE loading, chat templates and streaming decode.

Representative source:
- [Tokenizer.java](../../core/aljabr-spi-tokenizer/src/main/java/tech/kayys/aljabr/spi/tokenizer/Tokenizer.java)

The model catalogue includes many language and multimodal model-family projects (259 Gradle model builds across the catalogue).

**Assessment:** tokenizer and model-specific NLP integration foundations exist. General-purpose NLP feature engineering (e.g. TF-IDF/vectorizers), classical text classification/clustering and broad NLP evaluation should be separately inventoried and validated. Model-family directories alone do not establish those capabilities.

### 3.6 Model formats, storage and inference-oriented integrations — present

The core tree includes GGUF, Safetensors, RocksDB and HelixDB-related modules, alongside model/inference/plugin SPIs. These are important framework capabilities but are not substitutes for statistical or mathematical algorithm families.

## 4. Capability gap matrix

Status labels below describe this source inspection, not release readiness.

| Capability family | Inventory status | Evidence / gap |
|---|---|---|
| Tensor abstraction and core arithmetic | Present | Public tensor API and tests; operation semantics/coverage need a capability matrix |
| CPU numerical kernels | Present | SIMD operations and matmul; operator-level numerical parity remains to be established |
| General linear algebra | Partial foundation | Tensor matmul/einsum; no reviewed public decomposition/solver API |
| Probability and distributions | Not located as a dedicated capability | No dedicated module or reviewed distribution/sampling API family |
| Descriptive statistics | Not located as a dedicated capability | Some model metrics exist in legacy autograd, but not a reusable statistics API |
| Statistical inference and tests | Not located | No dedicated inference/test/CI/power API family observed |
| Regression algorithms | Partial / legacy evidence | Regression evaluation metrics exist; metrics do not implement regression estimators |
| Classical ML algorithms | Partial / legacy evidence | Cross-validation/search and metrics exist in opt-in legacy tree; core estimator catalog not established |
| Deep-learning layers | Present | NN module and layers exist |
| Autograd and optimizers | Present but gated / not fully validated | Legacy module opt-in; optimizer collection API explicitly unsupported |
| NLP tokenization | Present | Tokenizer contracts and BPE implementations |
| NLP vectorization / classical NLP | Not located as a dedicated capability | No reviewed TF-IDF/vectorizer/feature pipeline module |
| Time-series / survival / Bayesian / causal methods | Not located | No dedicated module or reviewed implementation family |
| Scientific simulation / numerical solvers | Not located | No dedicated module or reviewed API family |
| Hardware backends | Present, unevenly gated | CPU/CUDA/Metal included in settings; HAT requires special JDK 26; some backend projects commented/optional |
| Reproducible experiment system | Partial / legacy evidence | Trainer/checkpoint/metrics sources exist in opt-in legacy tree; supported integrated workflow not verified |

## 5. Recommended module direction

Do not make one Gradle module per algorithm. Keep modules cohesive and dependencies shallow. Proposed future boundaries, subject to ADRs and source-level dependency analysis:

| Proposed module family | Responsibility | Dependency direction |
|---|---|---|
| `aljabr-numerics` (or equivalent) | Stable scalar functions, tolerances, precision, random generators and numerical helpers | Depends on minimal core; no NN/agent/runtime dependencies |
| `aljabr-linear-algebra` | Decompositions, solvers, sparse structures and matrix algorithms | Depends on tensor/numerics contracts |
| `aljabr-statistics` | Descriptive statistics, distributions, estimation, tests, intervals, resampling and diagnostics | Depends on numerical/linear-algebra APIs; no deep-learning dependency |
| `aljabr-ml-classical` | Estimators, preprocessing, feature pipelines, model selection and evaluation | Depends on statistics/linear algebra; no GPU requirement |
| `aljabr-autograd` / NN training surface | Canonical gradient and optimizer APIs, parameter ownership and differentiable operations | Depends on tensor and NN; replace legacy integration only after parity tests |
| `aljabr-nlp` | General NLP preprocessing/vectorization/embeddings and evaluation primitives | Depends on tokenizer and classical ML contracts where appropriate |
| Optional scientific extensions | Time-series, Bayesian, survival, simulation, signal/spatial methods | Separate optional modules only when each has a real consumer and acceptance suite |

These are candidate names only. Before creating them, map current packages and external dependencies, especially the legacy autograd package and any math functionality hidden in existing model/backend modules.

## 6. Priority implementation sequence

1. **Capability registry and inventory:** record algorithm IDs, module, source location, maturity, test evidence, backend support and known limitations.
2. **Numerical contract baseline:** define precision/tolerance conventions, deterministic random sources, shape/dtype semantics and reference-test utilities.
3. **Linear algebra:** inventory and implement only the needed decomposition/solver primitives with CPU reference tests.
4. **Statistical core:** distributions, descriptive summaries, estimation, confidence intervals, hypothesis tests, resampling and effect sizes.
5. **Classical ML:** preprocessing, regression/classification estimators, clustering, dimensionality reduction and leakage-aware evaluation.
6. **Training readiness:** reconcile autograd/NN APIs and prove deterministic tiny-model convergence in CI.
7. **NLP foundation:** vectorization and reusable text features beyond tokenizer/model loading.
8. **Advanced domains:** time-series, Bayesian, survival, causal and scientific simulation as independently accepted extensions.

## 7. Immediate blockers and cautions

- **Build visibility:** `:core:autograd` is excluded by default and depends on snapshot artifacts not defined in this repository's inspected tree.
- **API ownership:** `Tensor` is documented as immutable for mathematical properties and lacks a public assignment/parameter mutation contract; optimizer APIs must be explicit about replacement versus mutation.
- **Numerical semantics:** public APIs need documented dtype/device support, edge-case behavior and tolerance policies before broad algorithm additions.
- **Module sprawl:** the model catalogue is already very large. Avoid adding hundreds of algorithm/model modules without stable shared contracts and meaningful independent test suites.
- **Backend parity:** a class in CPU/CUDA/Metal source is not evidence that the same operator is supported and numerically equivalent across devices.
- **JDK baseline:** Aljabr documents JDK 25; downstream consumers must align or use a process/service boundary.

## 8. Recommended next engineering decision

Approve a small foundational milestone before expanding algorithm breadth:

- Define `Numerics` and `LinearAlgebra` public API boundaries.
- Decide whether to adopt an established Java math library behind Aljabr-owned adapters or implement a small CPU reference subset first.
- Establish reference fixtures and tolerance policy.
- Publish one clean, minimal consumer example that resolves only tensor + CPU + numerical foundation.
- Keep statistics independent from deep learning, GPU, model formats and agent frameworks.

This creates the substrate Farabi needs without making Farabi's runtime depend on all of Aljabr.

## 9. Related documents

- [Computational Framework Scope](COMPUTATIONAL_FRAMEWORK_SCOPE.md)
- [Framework Architecture](FRAMEWORK_ARCHITECTURE.md)
- [Research and Training Readiness](RESEARCH_TRAINING_READINESS.md)
- [Tensor Research Quickstart](../examples/TENSOR_RESEARCH_QUICKSTART.md)
- [Farabi Statistical Intelligence](https://github.com/andalus-ai/farabi/blob/main/docs/STATISTICAL_INTELLIGENCE.md)
