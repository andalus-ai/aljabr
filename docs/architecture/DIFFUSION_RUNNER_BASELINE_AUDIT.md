# Diffusion Model Architecture and Runner Roadmap

**Status:** architecture guidance and implementation roadmap  
**Scope:** Aljabr diffusion model ecosystem  
**Java baseline:** JDK 25  
**Last reviewed:** 2026-11-21

## 1. Executive direction

Diffusion is a first-class model domain in Aljabr. Its shared mathematics and execution contracts belong in a dedicated diffusion subsystem, while model-family architectures and their checkpoint conventions belong in independently installable family modules.

Do not keep the complete diffusion implementation inside `:models:aljabr-model-common`. That module is a shared model layer, not the permanent home for Stable Diffusion, FLUX, SD3, or future generative-image architectures.

**Core principle: model families are pluggable; computational primitives are shared.**

Aljabr remains the compute and model-execution framework. AQLI may consume diffusion APIs for serving, but diffusion modules must not depend on AQLI. Tafkir may reuse shared training primitives without owning inference model-family implementations.

## 2. Architectural boundaries

The diffusion subsystem is organized into five responsibility layers:

1. **Diffusion API:** stable public contracts for model families, pipeline requests/results, conditioning, latent state, scheduler behavior, progress, cancellation and model capabilities.
2. **Diffusion core:** family-neutral orchestration primitives, noise/latent utilities, prediction semantics, timestep/sigma representations and reusable execution helpers.
3. **Shared components:** scheduler implementations, VAE/autoencoder components, conditioning abstractions, image preprocessing/postprocessing and reusable attention/transformer blocks where their contracts are genuinely shared.
4. **Model families:** architecture-specific pipelines, configs, denoisers, conditioning composition, weight mapping and checkpoint conventions.
5. **Compute backends:** existing Aljabr backend SPI and tensor/operator implementations. Diffusion must not create its own CPU/Metal/CUDA backend hierarchy.

Keep the boundaries explicit. A shared class belongs in diffusion core only when it is independent of a particular family and has stable semantics. Similar-looking operations are not automatically interchangeable: for example, a U-Net epsilon-prediction model and a flow-prediction MMDiT model have different denoising/integration contracts.

## 3. Proposed Gradle module organization

The following is the target logical layout. Introduce modules incrementally when there is a concrete implementation and test suite; do not create empty projects for every future family.

```text
models/
  diffusion/
    aljabr-diffusion-api/
    aljabr-diffusion-core/

    schedulers/
      aljabr-diffusion-scheduler-api/
      aljabr-diffusion-scheduler-ddim/
      aljabr-diffusion-scheduler-ddpm/
      aljabr-diffusion-scheduler-euler/
      aljabr-diffusion-scheduler-heun/
      aljabr-diffusion-scheduler-dpm-solver/
      aljabr-diffusion-scheduler-flow-matching/

    components/
      aljabr-diffusion-vae/
      aljabr-diffusion-conditioning/
      aljabr-diffusion-image-io/

    families/
      aljabr-diffusion-stable/
      aljabr-diffusion-sdxl/
      aljabr-diffusion-sd3/
      aljabr-diffusion-flux/
      aljabr-diffusion-pixart/
      aljabr-diffusion-kandinsky/
```

Names are illustrative until aligned with the repository's published artifact naming conventions. Avoid both extremes: one monolithic diffusion module and a project for every small class. A module should represent a meaningful dependency, release, ownership or optional-installation boundary.

### 3.1 Module responsibilities

| Module | Owns | Must not own |
|---|---|---|
| `aljabr-diffusion-api` | Public family-neutral contracts and immutable request/config/result types | Concrete scheduler math, family architectures, backend implementations |
| `aljabr-diffusion-core` | Pipeline orchestration primitives, latent/noise utilities, common prediction and execution semantics | Stable/FLUX-specific assumptions |
| `aljabr-diffusion-scheduler-api` | Scheduler contracts, timestep/sigma schedule values and prediction/integration metadata | DDIM-specific equations |
| Scheduler implementation modules | Their named algorithm, configuration validation, deterministic math and reference tests | Model-family checkpoint loading or device backend selection |
| `aljabr-diffusion-vae` | Reusable spatial autoencoder interfaces and implementations such as AutoencoderKL | Stable/FLUX pipeline orchestration |
| `aljabr-diffusion-conditioning` | Conditioning composition and reusable text/image conditioning contracts | A specific tokenizer vocabulary or family-specific encoder topology |
| Family modules | Family pipeline, architecture/config, component graph, weight mapping, conditioning assembly and family-specific validation | Duplicated tensor math, private backend stacks, AQLI serving policy |
| Image I/O module | Image preprocessing, normalization and output conversion | Model-family denoising logic |

### 3.2 Family modules and optional dependencies

- **Stable Diffusion:** classic latent diffusion with conditional U-Net, CLIP text conditioning, spatial VAE and supported discrete schedulers. SD 1.x, SD 2.x and inpainting variants should be configuration/variant boundaries where topology is truly compatible; do not claim compatibility from a shared class name.
- **SDXL:** dual text encoders, pooled embeddings, size/crop conditioning and its own U-Net/configuration/weight mapping. It must not be treated as a flag on an SD 1.x pipeline unless parity proves that abstraction.
- **SD3:** MMDiT-style denoiser and its text-conditioning composition, including the family-specific encoder combination and flow-oriented sampling semantics.
- **FLUX:** MMDiT/transformer denoiser, joint text/image sequence handling, family-specific latent packing and guidance inputs, with flow-matching integration. FLUX is not Stable Diffusion with a different scheduler.
- **PixArt, Kandinsky and future families:** independently added when architecture, weights and tests justify a module.

A consumer that needs only Stable Diffusion should not transitively install FLUX, SD3 or every other family. Family implementations should be independently resolvable artifacts and register optional providers without making the API/core depend on every family.

## 4. Scheduler contract: discrete diffusion and flow matching

The scheduler abstraction must not be DDIM-centric. A single generic `step(modelOutput, timestep, sample)` signature is insufficient if it silently assumes every model predicts epsilon in an alpha/beta discrete process.

The scheduler API should represent, explicitly and without forcing one family of equations:

- Schedule construction and immutable inference schedule values.
- Discrete timesteps and continuous sigma/time values.
- Model prediction semantics, such as epsilon, velocity, sample, or flow.
- Scheduler/integrator family and required model-output interpretation.
- Per-step inputs and outputs, including any scheduler state required by multistep methods.
- Deterministic seed/noise behavior where stochastic sampling is used.
- Configuration validation and terminal-step behavior.

Keep algorithm-specific implementations independent and test each against reference values. DDIM/DDPM and Euler/Heun/DPM-Solver have distinct numerical behavior; flow-matching schedulers must not be shoehorned into alpha-cumprod-only contracts. Family compatibility should be explicit and validated before the first denoising step.

## 5. Pluggable model-family provider SPI

The API may expose a small provider contract so optional family modules can be discovered without static dependencies. A conceptual shape is:

```java
public interface DiffusionModelProvider {
    String modelFamily();

    boolean supports(ModelDescriptor descriptor);

    DiffusionModel create(
        DiffusionModelConfig config,
        DiffusionComponents components
    );
}
```

This is illustrative, not a commitment to this exact public signature. Finalize it only after the model descriptor, weight-loading and component-lifecycle contracts are audited.

Use Java `ServiceLoader` as an optional discovery mechanism when it fits the existing Aljabr provider conventions. Keep direct dependency injection/explicit registration available for deterministic tests and embedded use. Provider discovery must not become a hidden global mutable registry.

Provider rules:

- A family module depends inward on diffusion API/core and Aljabr tensor/model APIs.
- API/core never depends on a family implementation.
- A provider declares supported family/config/checkpoint traits and rejects unsupported descriptors with actionable diagnostics.
- Provider creation validates required components, scheduler compatibility, latent shape and conditioning requirements.
- Model weights and tokenizer artifacts are resolved through Aljabr's existing format/loading contracts; family modules own mapping, not duplicate parsers.
- Backend choice remains an Aljabr execution concern and is injected through existing compute/tensor contracts.

## 6. Dependency direction and optionality

The dependency graph must point inward:

```text
Application / AQLI / user code
          |
          v
Optional family module (Stable / SDXL / SD3 / FLUX / ...)
          |
          +------> diffusion core
          +------> diffusion API
          +------> shared components (VAE / conditioning / scheduler impl)
          |
          v
Aljabr model, tensor, NN and operator APIs
          |
          v
Aljabr backend SPI and backend implementations
```

Required rules:

- No diffusion API/core dependency on AQLI, Tafkir, server frameworks or a specific family.
- No mandatory dependency on every family module.
- No duplicated tensor, memory ownership, backend selection or native integration layer.
- No model-name string conditionals in generic orchestration where a typed capability/configuration can express the requirement.
- No implicit CPU round-trips in the denoising loop.
- Family configs remain independent of concrete hardware backends.
- Optional provider absence must produce an actionable unsupported-family diagnostic.
- Avoid circular dependencies between schedulers, components and family modules.

## 7. Migration from the current source layout

The repository currently has diffusion classes under `:models:aljabr-model-common`. The following migration keeps the public surface deliberate and avoids a risky all-at-once rewrite.

| Existing capability | Target ownership | Migration guidance |
|---|---|---|
| `DiffusionScheduler`, `DiffusionSchedule`, `SchedulerStepParameters` | Scheduler API / diffusion API | Audit and generalize semantics before moving; preserve a compatibility adapter if public |
| `DDIMScheduler` | DDIM scheduler implementation | Keep DDIM equations and tests isolated from family pipelines |
| `StableDiffusionPipeline` | Stable family module | Move only after API/core contracts and component ownership are established |
| `AutoencoderKL` | Shared VAE component module | Keep generic spatial AutoencoderKL separate from family-specific checkpoint configs/mapping |
| `CLIP` and tokenizer integration | Conditioning component + family integration | Separate tokenizer, encoder and prompt/conditioning assembly; no synthetic embeddings in a compatibility claim |
| `UNet2DConditional` and SD blocks | Stable family module | Preserve as experimental until configuration fidelity, checkpoint mapping and reference parity are demonstrated |
| Generic latent/noise helpers | Diffusion core | Move only if family-neutral and backend/tensor ownership is explicit |
| Existing vector `VAE` and generic `DDPM` | Generic model/training scope | Do not relabel them as image latent-diffusion implementations |

Do not make `:models:aljabr-model-common` depend on all diffusion family modules. During migration, it may temporarily retain deprecated adapters, but new diffusion features should be added to the dedicated modules.

## 8. Current implementation baseline and compatibility claims

The initial source audit identified these issues:

- The original `DDPM.sample` path had an invalid hard-coded sample dimension and insufficient timestep validation. Keep generic vector DDPM separate from image latent diffusion and retain regression tests for its configuration.
- The original Stable Diffusion pipeline used placeholder scheduler coefficients, fixed latent dimensions and a fixed subtraction update. DDIM scheduler work has since landed; do not treat that alone as proof of a complete Stable Diffusion pipeline.
- The original `CLIP.encode(String)` path generated synthetic hash-seeded embeddings rather than performing tokenizer-backed checkpoint inference. This is not valid text conditioning for a pretrained-model compatibility claim.
- The original `VAE` is a vector/MLP educational VAE, not a spatial latent-diffusion autoencoder.
- The initial `UNet2DConditional` implementation is a simplified scaffold, not evidence of SD checkpoint compatibility.
- A spatial `AutoencoderKL` foundation has been added, but that alone does not establish compatibility with SD/SDXL/SD3/FLUX checkpoints.

The following milestones are present in repository history at the time of this document update:

- DDIM scheduler implementation and pipeline integration changes have merged.
- Spatial AutoencoderKL foundation has merged.
- Further AutoencoderKL pipeline integration is tracked separately and must be checked for merge/CI status before relying on it.

Do not advertise Stable Diffusion, SDXL, SD3 or FLUX checkpoint compatibility until the relevant family has real checkpoint loading, exact architecture/configuration, tokenizer/conditioning, weight mapping, scheduler compatibility and reference numerical tests.

## 9. Sequenced implementation roadmap

1. **Architecture and module boundary (this document):** establish family-neutral API/core, shared components, scheduler and family ownership.
2. **Repository module setup:** align nested module discovery in `settings.gradle.kts`, Gradle conventions, publishing and dependency verification; do not include empty future modules.
3. **Scheduler API:** support discrete and flow-oriented schedules and explicit prediction semantics without breaking existing DDIM callers silently.
4. **Extract scheduler implementations:** move DDIM behind the scheduler API; add DDPM/Euler/Heun/DPM-Solver only with reference fixtures and validated contracts.
5. **Shared VAE/conditioning components:** isolate AutoencoderKL and conditioning abstractions, with deterministic lifecycle and tensor ownership tests.
6. **Stable family vertical slice:** establish a family module with config-driven UNet, real tokenizer/CLIP, VAE checkpoint mapping and end-to-end reference parity.
7. **SDXL family support:** add dual-encoder and size/crop conditioning with independent fixtures.
8. **SD3 and FLUX family foundations:** implement MMDiT and flow-matching semantics as distinct architecture paths, not Stable pipeline conditionals.
9. **Additional families:** add PixArt, Kandinsky or others only when there is concrete demand and validated implementation.
10. **Performance:** device-resident latent/conditioning tensors, reusable workspaces, fused guidance, graph reuse and backend kernels after numerical correctness.

## 10. Acceptance criteria

### Shared API and module architecture

- Core and API build without any family implementation on the classpath.
- A family module can be added/removed without changing diffusion core.
- Optional provider discovery is deterministic and testable.
- Scheduler contracts can express both discrete alpha/beta schedules and continuous sigma/flow schedules.
- Model prediction type and scheduler compatibility are validated explicitly.
- Resource ownership, tensor aliasing, cancellation and deterministic seed semantics are documented and tested.

### Family compatibility

- Loads a known checkpoint/config using Aljabr-native format/weight APIs without Python or LibTorch at inference time.
- Uses actual tokenizer and text/image conditioning weights; no synthetic embeddings.
- Uses actual denoiser and autoencoder weights with correct topology and latent scaling/packing.
- Matches trusted reference intermediate tensors within documented tolerances.
- Matches scheduler timesteps/sigmas and deterministic update outputs.
- Produces reproducible output for a fixed seed in deterministic backend mode.
- Has family-specific tests and a separate opt-in end-to-end test profile that does not require network access in normal CI.

### Performance and backend behavior

- No hidden CPU tensor conversion in the iterative denoising loop.
- Backend capability requirements are reported before execution.
- Family modules do not own or fork hardware backend implementations.
- Performance measurements separate model load, conditioning, denoising and decode phases.

## 11. Non-goals

- Do not implement every diffusion family in one PR.
- Do not create empty modules for speculative future architectures.
- Do not claim full Stable Diffusion compatibility from a scheduler or AutoencoderKL class alone.
- Do not force FLUX into a Stable Diffusion U-Net/CLIP/alpha-scheduler abstraction.
- Do not add a second tensor/backend/memory system for diffusion.
- Do not make AQLI or Tafkir a dependency of diffusion modules.
- Do not broaden the current work into video diffusion until image-family contracts and resource semantics are stable.
