package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D upsampling module.
 *
 * <p>The current forward operation is intentionally a placeholder and returns
 * its input unchanged. It exists as an architectural slot until a backend
 * implementation of spatial interpolation is provided.</p>
 */
public class Upsample2d extends NNModule {

    private final float scaleFactor;

    /** @deprecated Use {@link #Upsample2d(ComputeBackend, float)}. */
    @Deprecated(since = "0.2.0", forRemoval = true)
    public Upsample2d(float scaleFactor) {
        this(null, scaleFactor);
    }

    public Upsample2d(ComputeBackend backend, float scaleFactor) {
        super(backend);
        if (!Float.isFinite(scaleFactor) || scaleFactor <= 0.0f) {
            throw new IllegalArgumentException("scaleFactor must be finite and positive");
        }
        this.scaleFactor = scaleFactor;
    }

    @Override
    public Tensor forward(Tensor input) {
        validateInputBackend(input);
        return input;
    }

    public float scaleFactor() {
        return scaleFactor;
    }
}
