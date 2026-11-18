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
    private final RotaryEmbeddingConfig config;

    /** @deprecated Use {@link #RotaryEmbedding(ComputeBackend, int, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public RotaryEmbedding(int headDim, int maxSeqLen) {
        this(null, headDim, maxSeqLen);
    }

    public RotaryEmbedding(ComputeBackend backend, int headDim, int maxSeqLen) {
        this(backend, headDim, maxSeqLen, RotaryEmbeddingConfig.standard(headDim));
    }


    public RotaryEmbedding(ComputeBackend backend, int headDim, int maxSeqLen, RotaryEmbeddingConfig config) {
        super(backend);
        if (headDim <= 0 || maxSeqLen <= 0) {
            throw new IllegalArgumentException("headDim and maxSeqLen must be positive");
        }
        this.config = java.util.Objects.requireNonNull(config, "config");
        if (config.rotaryDimension() > headDim) {
            throw new IllegalArgumentException("rotaryDimension cannot exceed headDim");
        }
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
    }


    public RotaryEmbeddingConfig config() { return config; }

    public int headDim() {
        return headDim;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInput(input);
        int batch = Math.toIntExact(input.shape().dim(0));
        int sequence = Math.toIntExact(input.shape().dim(2));
        long[] positionIds = new long[Math.multiplyExact(batch, sequence)];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < sequence; s++) positionIds[b * sequence + s] = s;
        }
        return forward(input, positionIds);
    }

    /** Applies RoPE with batch-major explicit position IDs [batch, sequence]. */
    public Tensor forward(Tensor input, long[] positionIds) {
        validateInput(input);
        java.util.Objects.requireNonNull(positionIds, "positionIds");
        int batch = Math.toIntExact(input.shape().dim(0));
        int sequence = Math.toIntExact(input.shape().dim(2));
        if (positionIds.length != Math.multiplyExact(batch, sequence)) {
            throw new IllegalArgumentException("positionIds length must equal batch * sequence");
        }
        for (long position : positionIds) {
            if (position < 0 || position >= maxSeqLen) {
                throw new IllegalArgumentException("position ID must be within [0, maxSeqLen)");
            }
        }
        return input.backend().applyRoPE(input, config.frequencyBase(), config.rotaryDimension(),
                config.pairing() == RotaryEmbeddingConfig.Pairing.NEOX_HALF_SPLIT,
                config.scaling().type() == RotaryEmbeddingConfig.ScalingType.LINEAR
                        ? config.scaling().factor() : 1.0f, positionIds);
    }

    private void validateInput(Tensor input) {
        validateInputBackend(input);
        if (input.shape().rank() != 4) throw new IllegalArgumentException("RoPE expects [batch, heads, sequence, headDim]");
        if (input.shape().dim(3) != headDim) throw new IllegalArgumentException("input head dimension does not match configured headDim");
        if (input.shape().dim(2) > maxSeqLen) throw new IllegalArgumentException("input sequence length exceeds maxSeqLen");
    }

    /**
     * Applies the same rotary configuration and sequential positions to query
     * and key tensors.
     *
     * <p>Query and key may have different head counts, but must share batch,
     * sequence, and head dimensions.
     */
    public QueryKey apply(Tensor q, Tensor k) {
        validateInput(q);
        validateInput(k);
        if (q.backend() != k.backend()) {
            throw new IllegalArgumentException("Q and K tensors must use the same compute backend");
        }
        if (q.shape().dim(0) != k.shape().dim(0)
                || q.shape().dim(2) != k.shape().dim(2)
                || q.shape().dim(3) != k.shape().dim(3)) {
            throw new IllegalArgumentException(
                    "Q and K tensors must have matching batch, sequence, and head dimensions");
        }
        if (q.dtype() != k.dtype()) {
            throw new IllegalArgumentException("Q and K tensors must have the same dtype");
        }
        Tensor rotatedQuery = forward(q);
        try {
            Tensor rotatedKey = forward(k);
            return new QueryKey(rotatedQuery, rotatedKey);
        } catch (RuntimeException | Error failure) {
            rotatedQuery.release();
            throw failure;
        }
    }

    /** The independently rotated query and key tensors. */
    public record QueryKey(Tensor query, Tensor key) {
        public QueryKey {
            java.util.Objects.requireNonNull(query, "query");
            java.util.Objects.requireNonNull(key, "key");
        }
    }
}
