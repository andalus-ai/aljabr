package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * 2D upsampling module.
 *
 * <p>Uses the backend's nearest-neighbor resize operation for NCHW tensors.</p>
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
        if (input.shape().rank() != 4) {
            throw new IllegalArgumentException("Upsample2d expects NCHW rank-4 input");
        }
        long outputH = (long) Math.floor(input.shape().dim(2) * (double) scaleFactor);
        long outputW = (long) Math.floor(input.shape().dim(3) * (double) scaleFactor);
        if (outputH <= 0 || outputW <= 0 || outputH > Integer.MAX_VALUE || outputW > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("scaleFactor produces invalid output dimensions: "
                    + outputH + "x" + outputW);
        }
        return input.upsampleNearest2d((int) outputH, (int) outputW);
    }

    public float scaleFactor() {
        return scaleFactor;
    }
}
