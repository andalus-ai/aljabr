package tech.kayys.aljabr.core.backend;

import tech.kayys.aljabr.core.tensor.Tensor;
import java.util.List;

/**
 * Backend SPI for tensor compute operations.
 *
 * <p>
 * Implementations must provide hardware-specific kernels for all methods.
 * This interface mirrors the full op-set of {@code libtorch::nn::functional}
 * so that {@code aljabr-runner-libtorch} can swap its native LibTorch FFM calls
 * for pure aljabr backends without changing caller code.
 *
 * <p>
 * All methods receive fully-allocated tensors and return a new tensor
 * backed by memory appropriate for the backend (off-heap for CPU/GPU).
 * Operations are strictly mathematical; they do not modify the input
 * {@code Tensor}
 * instances.
 * 
 * <p>
 * Performance note: Backends are expected to leverage SIMD or GPU acceleration
 * where possible, keeping the heavy lifting completely off the Java heap to
 * minimize
 * GC pressure and maximize throughput.
 * @author bhangun
 */
public interface ComputeBackend {

    // ── Element-wise binary ───────────────────────────────────────────
    Tensor add(Tensor a, Tensor b);

    Tensor sub(Tensor a, Tensor b);

    Tensor mul(Tensor a, Tensor b);

    Tensor div(Tensor a, Tensor b);

    // ── Scalar operations ─────────────────────────────────────────────
    Tensor mul(Tensor a, float scalar);

    Tensor div(Tensor a, float scalar);

    Tensor addScalar(Tensor a, float scalar);

    // ── Linear algebra ────────────────────────────────────────────────
    Tensor matmul(Tensor a, Tensor b);

    Tensor embedding(Tensor weight, Tensor input, long paddingIdx);

    default Tensor applyRoPE(Tensor input, int posOffset, float freqBase, boolean isNeox) {
        return input;
    }

    // ── Shape manipulation ────────────────────────────────────────────
    Tensor reshape(Tensor a, long... newShape);

    Tensor slice(Tensor a, long[] offsets, long[] sizes);

    List<Tensor> split(Tensor a, int axis, int parts);

    Tensor flatten(Tensor a);

    Tensor unsqueeze(Tensor a, int dim);

    Tensor squeeze(Tensor a);

    Tensor transpose(Tensor a);

    Tensor transpose(Tensor a, int dim0, int dim1);

    // ── Reduction ─────────────────────────────────────────────────────
    Tensor softmax(Tensor a);

    Tensor softmax(Tensor a, int dim);

    Tensor logSoftmax(Tensor a, int dim);

    Tensor mean(Tensor a);

    Tensor mean(Tensor a, int dim, boolean keepDim);

    Tensor sum(Tensor a);

    Tensor sum(Tensor a, int dim, boolean keepDim);

    Tensor max(Tensor a);

    Tensor abs(Tensor a);

    // ── Activations ───────────────────────────────────────────────────
    Tensor relu(Tensor a);

    Tensor gelu(Tensor a);

    Tensor silu(Tensor a);

    Tensor sigmoid(Tensor a);

    Tensor tanh(Tensor a);

    Tensor sqrt(Tensor a);

    Tensor exp(Tensor a);

    Tensor log(Tensor a);

    Tensor pow(Tensor a, float exponent);

    // ── Normalization ─────────────────────────────────────────────────
    /**
     * Layer normalization over the last {@code normalizedShape.length} dimensions.
     *
     * @param input           input tensor of any shape
     * @param normalizedShape shape of the last N dims to normalize (e.g.
     *                        {@code [hiddenDim]})
     * @param weight          optional learnable scale (gamma), may be null
     * @param bias            optional learnable shift (beta), may be null
     * @param eps             small value for numerical stability (typically 1e-5)
     */
    Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps);

    /**
     * RMS normalization (used in Llama, Gemma, Mistral).
     *
     * @param input  input tensor [seq, hidden]
     * @param weight learnable scale (gamma)
     * @param eps    epsilon for numerical stability
     */
    Tensor rmsNorm(Tensor input, Tensor weight, float eps);

    /**
     * Batch normalization (used in CNNs).
     */
    Tensor batchNorm(Tensor input, Tensor weight, Tensor bias,
            Tensor runningMean, Tensor runningVar,
            boolean training, float momentum, float eps);

    // ── Convolution / Pooling ─────────────────────────────────────────
    /**
     * 2D convolution.
     *
     * @param input    [N, C_in, H, W]
     * @param weight   [C_out, C_in/groups, kH, kW]
     * @param bias     optional [C_out], may be null
     * @param stride   stride
     * @param padding  zero-padding
     * @param dilation dilation
     * @param groups   blocked connections
     */
    Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
            int stride, int padding, int dilation, int groups);

    /** 2D max pooling. */
    Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding);

    /** 2D adaptive average pooling. */
    Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW);

    // ── Regularization ────────────────────────────────────────────────
    /**
     * Applies dropout during training (sets random elements to zero).
     * During inference ({@code training=false}) this is a no-op returning
     * {@code input} as-is.
     */
    Tensor dropout(Tensor input, float p, boolean training);

    // ── Loss functions ────────────────────────────────────────────────
    Tensor crossEntropy(Tensor pred, Tensor target);

    Tensor binaryCrossEntropy(Tensor pred, Tensor target);

    // ── Memory / device ───────────────────────────────────────────────
    Tensor attention(Tensor Q, Tensor K, Tensor V);

    Tensor zerosLike(Tensor a);

    Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype);

    Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device);

    long numel(Tensor a);
}
