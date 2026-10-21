# Aljabr Architecture

## Vision: A General-Purpose ML Framework

Aljabr is the **foundational ML framework** for the Wayang platform — analogous to PyTorch in the Python ecosystem. It provides everything an inference/serving engine needs to be built on top of it, with zero engine-specific assumptions.

```
┌───────────────────────────────────────────────────────┐
│                    Wayang Platform                    │ ← Agent orchestration, RAG, MCP
├───────────────────────────────────────────────────────┤
│           Aqli / Any Future Serving Engine          │ ← Pure runtime: runners, repos, SDK
├───────────────────────────────────────────────────────┤
│                   Aljabr SPI                     │ ← Inference + serving contracts
│  (ModelRunner, InferenceRequest, PluginSPI, ...)      │
├───────────────────────────────────────────────────────┤
│                Aljabr Core                       │ ← Tensors, ops, data types, errors
├───────────────────────────────────────────────────────┤
│              Aljabr Backends                     │ ← CPU / CUDA / Metal / ROCm / HAT
└───────────────────────────────────────────────────────┘
```

The key design rule: **any compliant serving engine** (Aqli, or a future hypothetical engine) should be implementable using only Aljabr contracts, without touching engine-specific code.

---

## Module Groups

### `core/` — Primitives & Contracts

| Module | Purpose | Nature |
|---|---|---|
| `aljabr-tensor` | `Tensor`, `DefaultTensor`, `Shape`, `DType`, `DeviceType`, `TensorOps` | Foundation |
| `aljabr-core` | Root API surface, exposes tensor + spi-model | API umbrella |
| `aljabr-error-code` | `ErrorCode` enum, error taxonomy | Foundation |
| `aljabr-nn` | Neural network layer ops (linear, softmax, attention) | Foundation |
| `aljabr-spi-model` | **Model-level contracts**: `ModelConfig`, `ModelArchitecture`, `ModelRunner`, `MultimodalRequest/Response`, `ModelRegistry`, `ModelRepository`, `RunnerMetadata` | SPI |
| `aljabr-gguf-api` | GGUF format API/contracts | SPI |
| `aljabr-gguf-core` | GGUF parsing implementation | Runtime |
| `aljabr-gguf-converter` | GGUF format converter | Runtime |
| `aljabr-safetensor-api` | Safetensor format contracts | SPI |
| `aljabr-safetensor-spi` | Safetensor loader SPI | SPI |
| `aljabr-safetensor-core` | Safetensor loading implementation | Runtime |
| `aljabr-rocksdb` | RocksDB KV store binding | Runtime |
| `aljabr-helixdb` | HelixDB vector store binding | Runtime |
| `aljabr-3d` | 3D tensor/rendering ops | Optional |
| `autograd` | Autograd engine (training only — excluded from inference builds) | Training |

### `backend/` — Hardware Compute Backends

All backends implement `ComputeBackend` from `aljabr-tensor`.

| Module | Hardware | Nature |
|---|---|---|
| `backend/cpu/aljabr-backend-cpu` | CPU (Java Vector API) | Runtime |
| `backend/cuda/aljabr-backend-cuda` | NVIDIA CUDA | Runtime |
| `backend/cuda/aljabr-kernel-cuda` | CUDA kernel implementations | Runtime |
| `backend/metal/aljabr-backend-metal` | Apple Metal (macOS/iOS) | Runtime |
| `backend/hat/aljabr-backend-hat` | Oracle Project Babylon HAT | Experimental |
| `backend/rocm/aljabr-kernel-rocm` | AMD ROCm | Runtime |

**Backend boundary rule**: backends depend only on `core/aljabr-tensor` and `core/aljabr-error-code`. They must never depend on each other.

### `models/` — Model Family Definitions

Pre-built model graph descriptors auto-discovered at build time. Each `aljabr-model-<name>` provides `ModelFamilyDescriptor` + `ModelFamilyPlugin` registrations consumed by `aljabr-spi-model`.

---

## The SPI Elevation Plan

Aljabr currently hosts model-level contracts in `aljabr-spi-model`. The next step is to **host all general inference/serving SPIs** here, so any engine (Aqli or otherwise) depends only on Aljabr.

### Contracts to be elevated from Aqli → Aljabr

| Contract | Current location | Target Aljabr module |
|---|---|---|
| `InferenceRequest`, `InferenceResponse`, `StreamingResponse` | `aqli-spi-inference` | `aljabr-spi-inference` (new) |
| `InferenceEngine`, `InferencePipeline`, `InferencePhase` | `aqli-spi-inference` | `aljabr-spi-inference` (new) |
| `BatchScheduler`, `BatchConfig`, `BatchStrategy` | `aqli-spi-inference` | `aljabr-spi-inference` (new) |
| `EmbeddingRequest`, `EmbeddingResponse` | `aqli-spi-inference` | `aljabr-spi-inference` (new) |
| `ModelRunner` (interface) | `aqli-spi-runner` → `aljabr-spi-model` | Already partially there |
| `RunnerStats`, `RunnerCapabilities`, `RunnerConfiguration` | `aqli-spi-runner` | `aljabr-spi-model` extension |
| `AqliPlugin` → `AljabrPlugin` | `aqli-spi-plugin` | `aljabr-spi-plugin` (new) |
| `PluginHealth`, `PluginState`, `PluginRegistry` | `aqli-spi-plugin` | `aljabr-spi-plugin` (new) |
| `Tokenizer`, `TokenizerPool`, `PreTokenizer` SPI | `aqli-tokenizer-core/spi/` | `aljabr-spi-tokenizer` (new) |

### Target module layout after elevation

```
aljabr/
  core/
    aljabr-tensor/        ← tensors (unchanged)
    aljabr-spi-model/     ← model configs, ModelRunner, multimodal (existing, extended)
    aljabr-spi-inference/ ← NEW: InferenceRequest/Response, Engine, Pipeline, Batch, Embedding
    aljabr-spi-plugin/    ← NEW: Plugin lifecycle SPI
    aljabr-spi-tokenizer/ ← NEW: Tokenizer contracts
    aljabr-error-code/    ← errors (unchanged)
    ...
  backend/                     ← unchanged
  models/                      ← unchanged
```

---

## Module Boundary Rules

### Core → Backend direction only
```
core/aljabr-tensor   ←── backend/cpu
                          ←── backend/cuda
                          ←── backend/metal
```
Backends depend on core. Core never depends on backends.

### SPIs are pure contracts
SPI modules must contain only interfaces, abstract classes, records, enums, and annotations. No I/O, no JNI, no `new ConcreteImpl()`.

### No cross-backend dependencies
```
backend/cpu  ✗→  backend/cuda   (forbidden)
backend/cuda ✗→  backend/metal  (forbidden)
```

### Published coordinates
All `core/` and `backend/` modules must declare:
```kotlin
group = "tech.kayys.aljabr"
version = "0.1.0-SNAPSHOT"
```

---

## Enforcement Checklist

Run `scripts/check-boundaries.sh` before merging PRs.

```bash
# 1. No backend module depends on another backend
grep -rn 'project(":backend:' --include="*.kts" backend/
# Expect: no output

# 2. No core module declares a backend package
grep -rn '^package tech\.kayys\.aljabr\.backend' --include="*.java" core/
# Expect: no output

# 3. No stale references to deleted modules
grep -rn 'aljabr-spi-provider\|aljabr-model-runner\|"aljabr-engine"' --include="*.kts" .
# Expect: no output

# 4. SPI modules contain no concrete implementations
grep -rn '^public class [A-Z].*implements\|^public class [A-Z].*extends' \
  core/aljabr-spi-*/src --include="*.java" | grep -v 'Abstract\|Exception\|Error'
# Review any output: should only be abstract base classes or exceptions
```

---

## Change Guide

| What you want to add | Where it goes |
|---|---|
| New tensor primitive or op | `core/aljabr-tensor` |
| New model config field | `core/aljabr-spi-model` |
| New inference contract (request/response shape) | `core/aljabr-spi-inference` (new) |
| New hardware compute backend | `backend/<vendor>/aljabr-backend-<vendor>` |
| New model format (e.g. GGML) | `core/aljabr-<format>-api` (SPI) + `core/aljabr-<format>-core` (impl) |
| New model family (e.g. Falcon) | `models/aljabr-model-falcon` |
| New tokenizer SPI | `core/aljabr-spi-tokenizer` (new) |
| New tokenizer implementation | Lives in Aqli `runtime/` or a future `aljabr-tokenizer-impls` module |
