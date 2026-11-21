# Diffusion Runner Baseline Audit

**Status:** source audit; no runtime behavior changed  
**Scope:** `:models:aljabr-model-common` on the default branch at audit time  
**Purpose:** establish a correctness-first baseline before replacing the current diffusion path.

## Executive summary

The repository contains diffusion-related building blocks, but the current Stable Diffusion path is a scaffold, not a usable pretrained-model inference pipeline. The most urgent issue is a deterministic runtime failure in the generic DDPM sampler. The Stable Diffusion pipeline also uses placeholder scheduler coefficients and dummy text embeddings, while the current VAE is a small fully connected educational VAE rather than a spatial latent-diffusion VAE.

Do not advertise this implementation as Stable Diffusion-compatible until model loading, tokenizer/text encoder, architecture fidelity, scheduler math, and end-to-end numerical tests are in place.

## Current source inventory

| Source | Current role | Audit finding |
|---|---|---|
| `models/.../DDPM.java` | Generic DDPM training and sampling | Sampling dimension is initialized to -1, making allocation/shape invalid; schedule constructor does not validate timestep count. |
| `models/.../StableDiffusionPipeline.java` | Text-to-image orchestration | Hard-coded 1000-step placeholder alpha cumulative products; fixed 4x64x64 latent; unseeded random initialization; fixed 0.02 subtraction masquerading as scheduler; no cancellation; no model/config-driven latent scaling. |
| `models/.../CLIP.java` | Text conditioning | `encode(String)` returns deterministic hash-seeded random values; it does not tokenize or execute the registered transformer. |
| `models/.../UNet2DConditional.java` | Conditional denoiser | Source explicitly describes a simplified scaffold, not a complete Stable Diffusion UNet. Its forward contract validates rank/backend/dimensions, but that does not establish checkpoint compatibility. |
| `models/.../VAE.java` | Generic educational VAE | Fully connected vector VAE with sigmoid output; not the convolutional spatial latent VAE required by Stable Diffusion. |
| `models/.../ResnetBlock2D.java`, `SpatialTransformer.java` | UNet building blocks | Require separate architecture/shape/numerical parity audit before checkpoint loading can be claimed. |

## Confirmed correctness and compatibility gaps

### P0 — DDPM sampler cannot produce a valid sample

In `DDPM.sample(int batchSize)`, the local dimension is assigned `int D = (int) 0 - 1`. This yields -1, then allocates an array with a negative length and later constructs invalid tensor dimensions. The constructor also accepts invalid timestep counts; the linear schedule divides by `T - 1`, so `T <= 1` needs explicit rejection.

**Remediation:** require a configured positive sample dimension, validate `batchSize` and `timesteps >= 2`, and add deterministic seeded sampling tests. Keep this generic vector DDPM separate from image latent-diffusion orchestration.

### P0 — Stable Diffusion scheduler is mathematically a placeholder

The pipeline initializes `alphasCumprod[i]` using `pow(0.999f, i)` rather than loading/constructing the checkpoint's configured beta schedule. The denoising update subtracts `noise * 0.02` regardless of timestep, prediction type, sigma/alpha parameterization, or scheduler configuration. The loop's integer timestep mapping is not a scheduler implementation.

**Remediation:** introduce a standalone scheduler contract and immutable schedule/config values; implement and test DDIM first against reference equations. Support epsilon prediction explicitly before adding v-prediction or sample prediction. Do not silently infer scheduler behavior from model family names.

### P0 — Text conditioning is not real CLIP inference

`CLIP.encode(String)` explicitly creates random values from the prompt hash. The registered transformer is not invoked by this method and no tokenizer is present in this call path.

**Remediation:** separate tokenizer from text encoder; make the pipeline accept validated conditioning tensors initially. Add a real tokenizer/encoder integration only when vocabulary, special tokens, attention mask, positional embeddings, and checkpoint weights are available.

### P0 — Current VAE is not a Stable Diffusion VAE

The existing `VAE` is an MLP with vector input/output and sigmoid reconstruction. Stable Diffusion requires a spatial convolutional encoder/decoder, latent distribution handling, scaling factor, and correct image normalization.

**Remediation:** do not adapt the MLP VAE by renaming it. Introduce a distinct spatial AutoencoderKL implementation with explicit encode/decode contracts and checkpoint configuration.

### P1 — UNet scaffold is not checkpoint-compatible

The UNet class documents itself as simplified. It has a fixed small number of blocks and attention stages. Existing tests exercise backend ownership and input validation, not full forward parity or loading a real SD checkpoint.

**Remediation:** preserve the current class as an experimental scaffold until a configuration-driven block graph, skip topology, attention layout, normalization, time embedding, and weight mapping are verified.

## Recommended module boundaries

Keep the first implementation inside the existing model-family ownership unless the repository already has a stable generative-model SPI suitable for extraction. Avoid creating a parallel runtime or introducing a dependency on AQLI.

Proposed responsibilities:

- **Diffusion API:** generation request/result, progress, cancellation, seed and validation.
- **Scheduler:** training/inference timestep setup and one mathematically defined update step.
- **Conditioning:** tokenization and text encoder contracts; positive/negative conditioning.
- **Denoiser:** architecture-specific UNet/DiT forward contract.
- **Latent autoencoder:** encode/decode and latent scaling.
- **Pipeline:** orchestrates these components and owns no model-family-specific scheduler math.
- **Backend:** all tensor execution remains on the selected Aljabr backend; avoid CPU round-trips in the denoising loop.

## Sequenced implementation plan

1. **This audit:** record actual baseline and prevent false compatibility claims.
2. **DDPM correctness:** repair generic vector DDPM dimension/configuration defects with deterministic tests.
3. **Scheduler SPI + DDIM:** introduce independently testable scheduler math and reference-value tests.
4. **Stable Diffusion pipeline contract:** inject scheduler, denoiser, conditioning tensors, latent initializer, and decoder; add cancellation and seed.
5. **Spatial AutoencoderKL:** implement and test encode/decode shape and scaling semantics.
6. **Checkpoint-backed CLIP/tokenizer and UNet:** only claim SD 1.x after real weight mapping and reference parity.
7. **Performance:** device-resident latents, reusable workspaces, graph reuse, fused guidance, and backend kernels after correctness.

## Acceptance criteria for a real SD 1.x claim

- Loads a known SD 1.x checkpoint/config without Python or LibTorch at inference time.
- Uses actual tokenizer and CLIP weights; no synthetic embeddings.
- Uses actual spatial VAE weights and latent scaling.
- Matches reference scheduler timestep/alpha values and deterministic update outputs within documented tolerances.
- Matches denoiser intermediate tensors against a trusted reference on a small fixture.
- Reproducible output for a fixed seed and deterministic backend mode.
- No hidden CPU tensor conversions in the denoising loop.
- End-to-end image generation test is separated from fast unit tests and does not require network access in normal CI.

## Non-goals for this audit PR

- No speculative new public API.
- No new backend or external runtime dependency.
- No broad rewrite of UNet/VAE before contracts and parity fixtures exist.
- No claim that SDXL, SD3, FLUX, video diffusion, or image-to-image is supported.
