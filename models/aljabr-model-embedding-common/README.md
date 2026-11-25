# Aljabr Embedding Common

Model-independent post-processing primitives shared by embedding model families.

## Included primitives

- `MaskedMeanPooler`: attention-mask-aware mean pooling over sequence-major token states.
- `L2Normalizer`: float32 vector normalization with double-precision norm accumulation. The prefix overload truncates before normalization for models that support Matryoshka dimensions.

These classes are stateless, do not depend on a tensor backend or checkpoint format, and do not own tokenization, batching, model execution, or model-specific output-dimension policy.

Model-family modules should validate their own supported dimensions and then delegate the mathematical operation here. This keeps model-specific policy out of shared math utilities.
