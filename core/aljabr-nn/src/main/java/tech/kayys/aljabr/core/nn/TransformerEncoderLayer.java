package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Transformer encoder layer API.
 */
public class TransformerEncoderLayer extends NNModule {
    private final int dModel;
    private final int nHead;
    private final int dimFeedforward;
    private final float dropout;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public TransformerEncoderLayer(int dModel, int nHead, int dimFeedforward, float dropout) {
        this(null, dModel, nHead, dimFeedforward, dropout);
    }

    public TransformerEncoderLayer(ComputeBackend backend, int dModel, int nHead,
            int dimFeedforward, float dropout) {
        super(backend);
        this.dModel = dModel;
        this.nHead = nHead;
        this.dimFeedforward = dimFeedforward;
        this.dropout = dropout;

        registerModule("self_attn", new MultiheadAttention(backend, dModel, nHead));
        registerModule("linear1", new Linear(backend, dModel, dimFeedforward));
        registerModule("linear2", new Linear(backend, dimFeedforward, dModel));
        registerModule("norm1", new LayerNorm(backend, dModel));
        registerModule("norm2", new LayerNorm(backend, dModel));
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        // Forward composition remains a placeholder pending the transformer op implementation.
        return input;
    }
}
