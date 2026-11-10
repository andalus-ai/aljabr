package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Rotary positional embedding module.
 *
 * <p>The rotation operation remains a placeholder, but the module now participates
 * in explicit backend ownership and rejects tensors from a different backend.
 */
public class RotaryEmbedding extends NNModule {

    private final int headDim;
    private final int maxSeqLen;

    /** @deprecated Use {@link #RotaryEmbedding(ComputeBackend, int, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public RotaryEmbedding(int headDim, int maxSeqLen) {
        this(null, headDim, maxSeqLen);
    }

    public RotaryEmbedding(ComputeBackend backend, int headDim, int maxSeqLen) {
        super(backend);
        if (headDim <= 0 || maxSeqLen <= 0) {
            throw new IllegalArgumentException("headDim and maxSeqLen must be positive");
        }
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
    }

    public int headDim() {
        return headDim;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input; // placeholder
    }

    public Tensor apply(Tensor q, Tensor k) {
        validateInputBackend(q);
        validateInputBackend(k);
        if (q.backend() != k.backend()) {
            throw new IllegalArgumentException("Q and K tensors must use the same compute backend");
        }
        return q; // placeholder
    }
}
