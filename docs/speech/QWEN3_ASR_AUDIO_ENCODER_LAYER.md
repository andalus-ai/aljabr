# Qwen3-ASR acoustic transformer layer

## Implementation scope

`Qwen3AsrAudioEncoderLayer` is a scalar Java reference implementation of one acoustic transformer layer. It is deliberately independent of any tensor backend so its mathematical semantics can be tested on small deterministic fixtures. It is not the production performance path.

The layer accepts packed hidden states with cumulative sequence boundaries. Every interval is a separate bidirectional attention domain. No query in one interval can read keys or values from another interval. This implements the encoder's inference-window isolation contract.

## Layer sequence

For each packed sequence:

1. LayerNorm over the 1024-wide hidden dimension.
2. Independent learned Q, K and V projections, each with bias.
3. 16-head scaled dot-product attention with scale `1 / sqrt(head_dim)`; attention is bidirectional and bounded by the supplied cumulative boundaries.
4. Learned output projection with bias and residual addition.
5. LayerNorm over the hidden dimension.
6. Learned 4096-wide feed-forward input projection with bias.
7. Exact GELU activation.
8. Learned feed-forward output projection with bias and residual addition.

The layer uses the checkpoint's LayerNorm epsilon supplied with its weights. Weight matrices use `[out_features][in_features]` layout.

## Ownership and performance

Checkpoint arrays are retained by reference to avoid duplicating large model weights. The loader must not mutate them while the layer is in use. Inputs are not modified; outputs are newly allocated.

This scalar reference is intended for numerical conformance and small tests. It is not appropriate for full 1024-wide, 24-layer production inference. The next backend integration should map the same sequence to Aljabr's tensor/NN APIs and use block/variable-length attention primitives. Keep this class as a correctness oracle rather than creating a second runtime or tensor abstraction.

## Conformance still required

- Compare LayerNorm, Q/K/V projections, attention probabilities, attention output, GELU, FFN and residual outputs against upstream tensors.
- Test the exact checkpoint layer weights and verify all checkpoint tensor names/shapes.
- Compare full encoder outputs after Conv2D, positional embedding, all 24 layers and post LayerNorm.
- Ensure packed-window boundaries are generated from the valid feature mask and never merely from padded frame count.

The upstream reference uses LayerNorm-before-attention and LayerNorm-before-FFN, with independent non-causal attention sequences supplied through cumulative lengths. See the [Transformers Qwen3-ASR implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_asr/modeling_qwen3_asr.py).
