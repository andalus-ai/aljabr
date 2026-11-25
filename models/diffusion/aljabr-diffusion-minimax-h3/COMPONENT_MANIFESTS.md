# MiniMax-H3 component manifest mapping

MiniMax-H3 uses the family-neutral component inventory in
`aljabr-diffusion-api`. The family module contributes role identifiers only;
it does not own file discovery, format parsing, storage access or tensor
allocation.

| Role enum | Manifest role | Requirement policy |
|---|---|---|
| `DIFFUSION_MODEL` | `diffusion-model` | Required for generation |
| `TEXT_ENCODER` | `text-encoder` | Required when prompt encoding is performed locally |
| `VIDEO_AUTOENCODER` | `video-autoencoder` | Required when video latent decoding is requested |
| `AUDIO_AUTOENCODER` | `audio-autoencoder` | Optional until an audio output path is selected |
| `ADAPTER` | `adapter` | Optional; zero or more artifacts may be represented |
| `MODEL_PATCH` | `model-patch` | Optional |
| `PROMPT_EMBEDDING` | `prompt-embedding` | Optional when precomputed conditioning is supplied |

A `DiffusionComponentManifest` is an immutable inventory for one model
revision. Each artifact has a stable ID, semantic role, opaque location,
format, optional variant, optional SHA-256 digest, required flag and
string-valued metadata. The resolver does not open the location.

A capability supplies `DiffusionComponentRequirement` values for the
requested operation. The generic resolver filters candidates by accepted
format, prefers an exact requested variant, and otherwise preserves manifest
declaration order. Missing required roles are returned as structured
resolution state; no file I/O occurs.

The current resolver selects at most one artifact per required role. Multi-file
roles such as adapter stacks are intentionally represented as a later
extension: they need explicit cardinality and ordering semantics rather than
silently overloading the single-component contract.
