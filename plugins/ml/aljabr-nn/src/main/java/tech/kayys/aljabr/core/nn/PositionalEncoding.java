package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Positional encoding module.
 *
 * <p>The current implementation is an identity placeholder; backend ownership is
 * established now so concrete positional encodings can add buffers without
 * changing their construction contract.
 */
public class PositionalEncoding extends NNModule {

    private final int dModel;
    private final int maxLen;

    /** @deprecated Use {@link #PositionalEncoding(ComputeBackend, int, int)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public PositionalEncoding(int dModel, int maxLen) {
        this(null, dModel, maxLen);
    }

    public PositionalEncoding(ComputeBackend backend, int dModel, int maxLen) {
        super(backend);
        if (dModel <= 0 || maxLen <= 0) {
            throw new IllegalArgumentException("dModel and maxLen must be positive");
        }
        this.dModel = dModel;
        this.maxLen = maxLen;
    }

    public int dModel() {
        return dModel;
    }

    public int maxLen() {
        return maxLen;
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input; // placeholder
    }
}
