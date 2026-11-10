package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Static functional API for neural network operations.
 * Mirrors {@code torch.nn.functional} but is backend-agnostic:
 * all methods delegate to the {@link Tensor} interface, which is
 * dispatched at runtime to the active
 * {@link tech.kayys.aljabr.core.backend.ComputeBackend}.
 *
 * <p>
 * Use these helpers when you need a functional (stateless) style and do not
 * want to instantiate a {@link NNModule} subclass.
 * @author bhangun
 */
public final class Functional {

    private Functional() {
    }

    // ── Activation functions ──────────────────────────────────────────

    /**
     * Rectified Linear Unit: {@code max(0, x)}.
     *
     * @param input input tensor
     * @return output tensor with negative values clamped to zero
     */
    public static Tensor relu(Tensor input) {
        return input.relu();
    }

    /**
     * Gaussian Error Linear Unit.
     *
     * @param input input tensor
     * @return GELU-activated tensor
     */
    public static Tensor gelu(Tensor input) {
        return input.gelu();
    }

    /**
     * Sigmoid activation: {@code 1 / (1 + exp(-x))}.
     *
     * @param input input tensor
     * @return sigmoid-activated tensor in range (0, 1)
     */
    public static Tensor sigmoid(Tensor input) {
        return input.sigmoid();
    }

    /**
     * Hyperbolic tangent activation.
     *
     * @param input input tensor
     * @return tanh-activated tensor in range (-1, 1)
     */
    public static Tensor tanh(Tensor input) {
        return input.tanh();
    }

    /**
     * Sigmoid Linear Unit (SiLU / Swish): {@code x * sigmoid(x)}.
     *
     * @param input input tensor
     * @return SiLU-activated tensor
     */
    public static Tensor silu(Tensor input) {
        return input.silu();
    }

    /**
     * Softmax along a dimension.
     * Each slice along {@code dim} sums to 1.
     *
     * @param input input tensor
     * @param dim   dimension along which softmax is computed
     * @return probability tensor
     */
    public static Tensor softmax(Tensor input, int dim) {
        return input.softmax(dim);
    }

    /**
     * Log-softmax along a dimension.
     * Numerically more stable than {@code log(softmax(x))}.
     *
     * @param input input tensor
     * @param dim   dimension along which log-softmax is computed
     * @return log-probability tensor
     */
    public static Tensor logSoftmax(Tensor input, int dim) {
        return input.logSoftmax(dim);
    }

    // ── Regularization ────────────────────────────────────────────────

    /**
     * Dropout regularisation.
     * Randomly zeroes elements with probability {@code p} during training.
     *
     * @param input    input tensor
     * @param p        probability of an element being zeroed
     * @param training if {@code false} the operation is a no-op (inference mode)
     * @return tensor with dropout applied
     */
    public static Tensor dropout(Tensor input, float p, boolean training) {
        return input.dropout(p, training);
    }

    // ── Normalization ─────────────────────────────────────────────────

    /**
     * Layer normalization over the last N dimensions.
     *
     * @param input           input tensor
     * @param normalizedShape shape of the axes to normalize (last N dims)
     * @param weight          optional learnable scale (gamma); may be {@code null}
     * @param bias            optional learnable shift (beta); may be {@code null}
     * @param eps             epsilon added to denominator for numerical stability
     * @return normalized tensor
     */
    public static Tensor layerNorm(Tensor input, long[] normalizedShape,
            Tensor weight, Tensor bias, float eps) {
        return input.layerNorm(normalizedShape, weight, bias, eps);
    }

    /**
     * RMS normalization (no mean subtraction).
     *
     * @param input  input tensor
     * @param weight learnable scale (gamma)
     * @param eps    epsilon for numerical stability
     * @return RMS-normalized tensor
     */
    public static Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
        return input.rmsNorm(weight, eps);
    }

    /**
     * Batch normalization.
     *
     * @param input       input tensor [N, C, ...]
     * @param weight      optional learnable scale per channel; may be {@code null}
     * @param bias        optional learnable bias per channel; may be {@code null}
     * @param runningMean running mean (updated in-place during training); may be
     *                    {@code null}
     * @param runningVar  running variance (updated in-place during training); may
     *                    be {@code null}
     * @param training    whether in training mode
     * @param momentum    momentum for running stats update
     * @param eps         epsilon added to variance for numerical stability
     * @return batch-normalized tensor
     */
    public static Tensor batchNorm(Tensor input,
            Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps) {
        return input.batchNorm(weight, bias, runningMean, runningVar, training, momentum, eps);
    }

    // ── Convolution ───────────────────────────────────────────────────

    /**
     * 2-D convolution.
     *
     * @param input    input tensor [N, C_in, H, W]
     * @param weight   weight tensor [C_out, C_in/groups, kH, kW]
     * @param bias     optional bias tensor [C_out]; may be {@code null}
     * @param stride   convolution stride (square)
     * @param padding  zero-padding added on all sides
     * @param dilation dilation of the kernel
     * @param groups   number of blocked connections from input to output channels
     * @return convolved output tensor
     */
    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
            int stride, int padding, int dilation, int groups) {
        return input.conv2d(weight, bias, stride, padding, dilation, groups);
    }

    // ── Pooling ───────────────────────────────────────────────────────

    /**
     * 2-D max pooling.
     *
     * @param input      input tensor [N, C, H, W]
     * @param kernelSize size of the pooling window
     * @param stride     stride of the pooling window
     * @param padding    zero-padding added on all sides
     * @return max-pooled tensor
     */
    public static Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) {
        return input.maxPool2d(kernelSize, stride, padding);
    }

    /**
     * Adaptive average pooling 2-D.
     * The output size is fixed regardless of the input size.
     *
     * @param input   input tensor [N, C, H, W]
     * @param outputH target output height
     * @param outputW target output width
     * @return adaptively averaged tensor [N, C, outputH, outputW]
     */
    public static Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) {
        return input.adaptiveAvgPool2d(outputH, outputW);
    }

    // ── Shape ─────────────────────────────────────────────────────────

    /**
     * Flatten a tensor into a 1-D tensor.
     *
     * @param input input tensor
     * @return flattened tensor
     */
    public static Tensor flatten(Tensor input) {
        return input.flatten();
    }

    /**
     * Insert a dimension of size 1 at position {@code dim}.
     *
     * @param input input tensor
     * @param dim   dimension to insert
     * @return tensor with new dimension
     */
    public static Tensor unsqueeze(Tensor input, int dim) {
        return input.unsqueeze(dim);
    }

    /**
     * Remove dimensions of size 1.
     *
     * @param input input tensor
     * @return tensor with size-1 dimensions removed
     */
    public static Tensor squeeze(Tensor input) {
        return input.squeeze();
    }

    // ── Attention ─────────────────────────────────────────────────────

    /**
     * Scaled dot-product attention: {@code softmax(Q * K^T / sqrt(d_k)) * V}.
     *
     * @param query query tensor [batch, heads, seq, d_k]
     * @param key   key tensor [batch, heads, seq, d_k]
     * @param value value tensor [batch, heads, seq, d_v]
     * @return attention output [batch, heads, seq, d_v]
     */
    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value) {
        return query.attention(key, value);
    }

    // ── Embedding ─────────────────────────────────────────────────────

    /**
     * Look up embeddings for a tensor of indices.
     *
     * @param input      integer index tensor [*]
     * @param weight     embedding table [vocabSize, embeddingDim]
     * @param paddingIdx index whose embedding is all zeros; use {@code -1} for none
     * @return embedded tensor [*, embeddingDim]
     */
    public static Tensor embedding(Tensor input, Tensor weight, long paddingIdx) {
        return input.embedding(weight, paddingIdx);
    }
}
