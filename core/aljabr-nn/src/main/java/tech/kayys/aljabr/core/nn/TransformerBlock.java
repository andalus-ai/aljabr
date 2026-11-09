package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Transformer block API placeholder.
 */
public class TransformerBlock extends NNModule {

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public TransformerBlock(int dModel, int nHeads, int dFF, float dropout) {
        this(null, dModel, nHeads, dFF, dropout);
    }

    public TransformerBlock(ComputeBackend backend, int dModel, int nHeads, int dFF, float dropout) {
        super(backend);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input;
    }
}
