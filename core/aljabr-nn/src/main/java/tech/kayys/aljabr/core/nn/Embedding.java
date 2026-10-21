package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Embedding layer — a lookup table for fixed-size vectors.
 * @author bhangun
 */
public class Embedding extends NNModule {

    private final long numEmbeddings;
    private final long embeddingDim;
    private final long paddingIdx;

    public Embedding(long numEmbeddings, long embeddingDim, long paddingIdx) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIdx = paddingIdx;

        Tensor weight = Tensor.randn(numEmbeddings, embeddingDim);
        registerParameter("weight", weight);
    }

    public Embedding(Tensor weight) {
        this.numEmbeddings = weight.shape().dim(0);
        this.embeddingDim = weight.shape().dim(1);
        this.paddingIdx = -1;
        registerParameter("weight", weight);
    }

    public Embedding(long numEmbeddings, long embeddingDim) {
        this(numEmbeddings, embeddingDim, -1);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = parameters.get("weight");
        return input.embedding(weight, paddingIdx);
    }

    public long numEmbeddings() {
        return numEmbeddings;
    }

    public long embeddingDim() {
        return embeddingDim;
    }

    public long paddingIdx() {
        return paddingIdx;
    }

    @Override
    public String toString() {
        return String.format("Embedding(%d, %d%s)", numEmbeddings, embeddingDim,
                paddingIdx >= 0 ? ", padding_idx=" + paddingIdx : "");
    }
}
