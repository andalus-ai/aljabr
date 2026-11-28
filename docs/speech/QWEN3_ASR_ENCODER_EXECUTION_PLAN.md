# Qwen3-ASR acoustic encoder execution plan

## Scope of this change

This change adds a pure-Java geometry planner for the Qwen3-ASR acoustic encoder. It does not claim to execute convolution, attention, transformer layers, or checkpoint inference. Tensor operations remain the responsibility of Aljabr's compute/model execution APIs; this class computes only deterministic chunk and attention-window boundaries.

## Checkpoint-specific execution geometry

For the Qwen3-ASR 1.7B-hf configuration:

- 128 Mel bins enter the acoustic encoder.
- Each convolution chunk contains `2 * n_window = 100` Mel frames.
- Three Conv2D layers each use kernel 3, stride 2 and padding 1.
- The three layers are applied independently to each 100-frame chunk, with GELU after each layer.
- The convolution output is flattened across the downsampled frequency and channel axes and projected to the 1024-dimensional encoder width.
- Sinusoidal positional embeddings restart at position zero for each independently convolved chunk.
- Eight 100-frame chunks are grouped into one inference attention window because `n_window_infer / (2 * n_window) = 800 / 100 = 8`.
- Attention is bidirectional within a window and must not cross the window boundary.
- The encoder contains 24 transformer layers, followed by post layer normalization and a two-layer output projection to 2048 dimensions.

The 3-stage time-axis output length for one convolution chunk is computed by applying `ceil(length / 2)` three times. A full 100-frame chunk therefore produces 13 time positions. Eight full chunks produce 104 positions in one inference attention window.

## Planner contract

`Qwen3AsrEncoderPlan` accepts the number of valid Mel frames for one utterance and returns:

- ordered convolution chunks with input-frame offsets and lengths;
- the expected output-token count for each chunk;
- a zero positional offset for every convolution chunk;
- ordered attention windows referencing contiguous convolution chunks and their flattened token ranges;
- the total projected token count before the acoustic-to-text decoder handoff.

The planner is immutable and has no dependency on a tensor backend, codec provider, or Suling. It is intended to be consumed by a later encoder executor and tested independently of GPU/CPU kernels.

## Numerical and integration gates still required

Before declaring executable Qwen3-ASR inference:

1. Implement the three Conv2D stages and flatten/projection using Aljabr tensor APIs.
2. Implement the exact Qwen3-ASR audio attention and pre-norm encoder layer, including Q/K/V/output biases and GELU FFN.
3. Build block-diagonal attention masks or variable-length attention metadata so tokens cannot attend across inference windows.
4. Implement sinusoidal positions with per-convolution-chunk reset semantics.
5. Implement post LayerNorm and the 1024 -> 1024 -> GELU -> 2048 output projection.
6. Map every checkpoint tensor by its exact upstream name and shape; do not infer mappings from layer names alone.
7. Compare intermediate activations and final audio embeddings against the upstream implementation using fixed audio and checkpoint fixtures.
8. Only then connect the audio embeddings to the shared Qwen3 text backbone and validate transcription, timestamps, language prompts and termination.

No media codec or container functionality belongs here. Encoded audio remains the responsibility of Suling or another provider; this module accepts model-facing features/PCM through its existing frontend path.
