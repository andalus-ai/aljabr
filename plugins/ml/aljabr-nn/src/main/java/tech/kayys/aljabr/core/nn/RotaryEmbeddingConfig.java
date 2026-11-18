package tech.kayys.aljabr.core.nn;

/**
 * Immutable, model-neutral configuration for rotary positional embeddings.
 *
 * <p>This describes the rotary geometry and optional frequency scaling. Execution
 * remains owned by the selected compute backend.</p>
 */
public record RotaryEmbeddingConfig(
        int rotaryDimension,
        float frequencyBase,
        Pairing pairing,
        Scaling scaling) {

    public enum Pairing {
        INTERLEAVED,
        NEOX_HALF_SPLIT
    }

    public enum ScalingType {
        NONE,
        LINEAR
    }

    public record Scaling(ScalingType type, float factor) {
        public Scaling {
            if (type == null) {
                throw new IllegalArgumentException("scaling type is required");
            }
            if (!Float.isFinite(factor) || factor <= 0f) {
                throw new IllegalArgumentException("scaling factor must be finite and positive");
            }
            if (type == ScalingType.NONE && factor != 1f) {
                throw new IllegalArgumentException("NONE scaling requires factor=1");
            }
            if (type == ScalingType.LINEAR && factor < 1f) {
                throw new IllegalArgumentException("LINEAR scaling factor must be >= 1");
            }
        }

        public static Scaling none() {
            return new Scaling(ScalingType.NONE, 1f);
        }

        public static Scaling linear(float factor) {
            return new Scaling(ScalingType.LINEAR, factor);
        }

        public float scaledPosition(float position) {
            return type == ScalingType.LINEAR ? position / factor : position;
        }
    }

    public RotaryEmbeddingConfig {
        if (rotaryDimension <= 0 || (rotaryDimension & 1) != 0) {
            throw new IllegalArgumentException("rotaryDimension must be positive and even");
        }
        if (!Float.isFinite(frequencyBase) || frequencyBase <= 1f) {
            throw new IllegalArgumentException("frequencyBase must be finite and > 1");
        }
        if (pairing == null) {
            throw new IllegalArgumentException("pairing is required");
        }
        if (scaling == null) {
            scaling = Scaling.none();
        }
    }

    public static RotaryEmbeddingConfig standard(int rotaryDimension) {
        return new RotaryEmbeddingConfig(rotaryDimension, 10_000f, Pairing.INTERLEAVED, Scaling.none());
    }
}
