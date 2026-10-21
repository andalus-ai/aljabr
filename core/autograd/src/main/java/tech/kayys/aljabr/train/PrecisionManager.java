package tech.kayys.aljabr.train;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.DType;

/**
 * Manages precision casting for mixed-precision training.
 * Forward pass can run in reduced precision (FP16/BF16) for speed,
 * while gradients are accumulated in FP32 for numerical stability.
 * @author bhangun
 */
public final class PrecisionManager {
    private final PrecisionMode mode;

    public PrecisionManager(PrecisionMode mode) {
        this.mode = mode;
    }

    /**
     * Cast tensor to the configured forward-pass precision.
     * Returns the tensor unchanged if already in FP32 mode.
     */
    public Tensor castForward(Tensor t) {
        if (mode == PrecisionMode.FP32)
            return t;
        return t.cast(toDType(mode));
    }

    /**
     * Cast tensor back to FP32 for gradient accumulation.
     */
    public Tensor castBackward(Tensor t) {
        return t.toFP32();
    }

    private static DType toDType(PrecisionMode m) {
        return switch (m) {
            case FP32 -> DType.F32;
            case FP16 -> DType.F16;
            case BF16 -> DType.BF16;
        };
    }
}