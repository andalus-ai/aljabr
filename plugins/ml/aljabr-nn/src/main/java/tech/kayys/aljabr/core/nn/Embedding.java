package tech.kayys.aljabr.core.nn;

import java.util.Objects;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Embedding layer — a lookup table for fixed-size vectors.
 */
public class Embedding extends NNModule {

    private final long numEmbeddings;
    private final long embeddingDim;
    private final long paddingIdx;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Embedding(long numEmbeddings, long embeddingDim, long paddingIdx) {
        this(null, numEmbeddings, embeddingDim, paddingIdx);
    }

    public Embedding(ComputeBackend backend, long numEmbeddings, long embeddingDim, long paddingIdx) {
        super(backend);
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIdx = paddingIdx;
        registerParameter("weight", randomNormal(numEmbeddings, embeddingDim));
    }

    public Embedding(Tensor weight) {
        super(Objects.requireNonNull(weight, "weight must not be null").backend());
        this.numEmbeddings = weight.shape().dim(0);
        this.embeddingDim = weight.shape().dim(1);
        this.paddingIdx = -1;
        registerParameter("weight", weight);
    }

    /** @deprecated Use {@link #Embedding(ComputeBackend, long, long)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Embedding(long numEmbeddings, long embeddingDim) {
        this(null, numEmbeddings, embeddingDim, -1);
    }

    public Embedding(ComputeBackend backend, long numEmbeddings, long embeddingDim) {
        this(backend, numEmbeddings, embeddingDim, -1);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.embedding(parameters.get("weight"), paddingIdx);
    }

    public long numEmbeddings() { return numEmbeddings; }
    public long embeddingDim() { return embeddingDim; }
    public long paddingIdx() { return paddingIdx; }

    @Override
    public String toString() {
        return String.format("Embedding(%d, %d%s)", numEmbeddings, embeddingDim,
                paddingIdx >= 0 ? ", padding_idx=" + paddingIdx : "");
    }
}
