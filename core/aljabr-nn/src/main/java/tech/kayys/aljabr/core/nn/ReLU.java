package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/** ReLU activation layer. */
public class ReLU extends NNModule {

    private final boolean inplace;

    /** @deprecated Use the constructor that accepts an explicit compute backend. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public ReLU(boolean inplace) {
        this(null, inplace);
    }

    public ReLU(ComputeBackend backend, boolean inplace) {
        super(backend);
        this.inplace = inplace;
    }

    /** @deprecated Use {@link #ReLU(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public ReLU() {
        this(null, false);
    }

    public ReLU(ComputeBackend backend) {
        this(backend, false);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.relu();
    }

    @Override
    public String toString() {
        return "ReLU(inplace=" + inplace + ")";
    }
}
