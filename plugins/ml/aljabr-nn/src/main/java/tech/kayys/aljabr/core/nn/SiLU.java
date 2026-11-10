package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/** SiLU (Swish) activation layer. */
public class SiLU extends NNModule {

    /** @deprecated Use {@link #SiLU(ComputeBackend)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public SiLU() {
        this(null);
    }

    /**
     * Creates a SiLU activation bound to an explicit compute backend.
     *
     * @param backend owning compute backend; may be null only for the
     *                deprecated compatibility constructor
     */
    public SiLU(ComputeBackend backend) {
        super(backend);
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input.silu();
    }

    @Override
    public String toString() {
        return "SiLU()";
    }
}
