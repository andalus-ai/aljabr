package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/** Dropout layer for regularization during training. */
public class Dropout extends NNModule {

    private final float probability;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Dropout(float p) {
        this(null, p);
    }

    public Dropout(ComputeBackend backend, float p) {
        super(backend);
        if (p < 0.0f || p >= 1.0f) {
            throw new IllegalArgumentException("Dropout probability must be in [0, 1)");
        }
        this.probability = p;
    }

    /** @deprecated Use {@link #Dropout(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Dropout() {
        this(null, 0.5f);
    }

    public Dropout(ComputeBackend backend) {
        this(backend, 0.5f);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.dropout(probability, training);
    }

    public float probability() {
        return probability;
    }

    @Override
    public String toString() {
        return String.format("Dropout(p=%.2f)", probability);
    }
}
