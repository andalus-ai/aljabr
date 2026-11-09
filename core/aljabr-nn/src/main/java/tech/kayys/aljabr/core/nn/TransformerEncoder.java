package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Transformer encoder API placeholder.
 */
public class TransformerEncoder extends NNModule {

    /** @deprecated Use {@link #builder(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public static Builder builder() {
        return new Builder(null);
    }

    public static Builder builder(ComputeBackend backend) {
        return new Builder(backend);
    }

    /** @deprecated Use {@link #TransformerEncoder(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public TransformerEncoder() {
        super();
    }

    public TransformerEncoder(ComputeBackend backend) {
        super(backend);
    }

    public static class Builder {
        private final ComputeBackend backend;

        private Builder(ComputeBackend backend) {
            this.backend = backend;
        }

        public TransformerEncoder build() {
            return new TransformerEncoder(backend);
        }

        public Builder nLayers(int n) { return this; }
        public Builder dModel(int d) { return this; }
        public Builder numHeads(int h) { return this; }
        public Builder dFF(int d) { return this; }
        public Builder dropout(float d) { return this; }
        public Builder vocabSize(int d) { return this; }
        public Builder nHeads(int d) { return this; }
        public Builder maxSeqLen(int d) { return this; }
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input;
    }
}
