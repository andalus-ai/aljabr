package tech.kayys.aljabr.cache.quantization;

import java.util.Objects;

/**
 * Reference QJL residual quantizer used by TurboQuant's inner-product path.
 *
 * <p>For a residual {@code r} of dimension {@code d}, this primitive stores
 * {@code sign(S * r)} using one bit per coordinate and stores {@code ||r||2}
 * as an FP64 scalar. {@code S} is a deterministic, seed-derived {@code d x d}
 * matrix with independent standard-normal entries. The matrix is regenerated
 * during decoding and is never materialized, keeping auxiliary memory O(d).</p>
 *
 * <p>Reconstruction follows the paper's QJL inverse map:
 * {@code sqrt(pi / 2) / d * ||r||2 * S^T * sign(S * r)}. The result is an
 * unbiased inner-product estimator; it is not an exact inverse and may have
 * greater vector MSE than the original residual. This is a CPU/reference
 * primitive with O(d^2) arithmetic, not an accelerator kernel.</p>
 *
 * <p>The input residual must already be expressed in the same coordinate
 * system as the vector being quantized. In a TurboQuant pipeline this means
 * applying the same rotation/transform to the source and its MSE reconstruction
 * before computing the residual.</p>
 */
public final class TurboQuantQjlResidual {

    private final int dimension;
    private final long seed;

    public TurboQuantQjlResidual(int dimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        this.dimension = dimension;
        this.seed = seed;
    }

    public int dimension() {
        return dimension;
    }

    public long seed() {
        return seed;
    }

    /**
     * Encodes one residual as {@code d} sign bits plus its FP64 L2 norm.
     * The sign bit is 1 for a non-negative projection and 0 for a negative one.
     */
    public EncodedResidual encode(float[] residual, int offset) {
        Objects.requireNonNull(residual, "residual");
        checkRange(residual.length, offset, "residual");

        double normSquared = 0.0;
        for (int i = 0; i < dimension; i++) {
            float value = residual[offset + i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("residual values must be finite");
            }
            normSquared += (double) value * value;
        }
        double norm = StrictMath.sqrt(normSquared);
        if (!Double.isFinite(norm)) {
            throw new IllegalArgumentException("residual norm must be finite");
        }

        int[] signs = new int[dimension];
        if (norm != 0.0) {
            GaussianSequence gaussian = new GaussianSequence(seed);
            for (int row = 0; row < dimension; row++) {
                double projection = 0.0;
                for (int column = 0; column < dimension; column++) {
                    projection += gaussian.nextGaussian() * residual[offset + column];
                }
                signs[row] = projection >= 0.0 ? 1 : 0;
            }
        } else {
            // The sign is immaterial for a zero residual because its norm is zero.
            java.util.Arrays.fill(signs, 1);
        }

        return new EncodedResidual(dimension, seed, norm,
                TurboQuantIndexBitPacking.pack(signs, 1));
    }

    /**
     * Reconstructs the QJL residual contribution. This is an estimator, not
     * an exact residual decode.
     */
    public float[] decode(EncodedResidual encoded) {
        Objects.requireNonNull(encoded, "encoded");
        if (encoded.dimension != dimension || encoded.seed != seed) {
            throw new IllegalArgumentException(
                    "encoded residual dimension or projection seed does not match this quantizer");
        }
        int[] signs = TurboQuantIndexBitPacking.unpack(encoded.packedSigns, dimension, 1);
        float[] reconstructed = new float[dimension];
        if (encoded.norm == 0.0) {
            return reconstructed;
        }

        double[] accumulation = new double[dimension];
        GaussianSequence gaussian = new GaussianSequence(seed);
        for (int row = 0; row < dimension; row++) {
            double sign = signs[row] == 1 ? 1.0 : -1.0;
            for (int column = 0; column < dimension; column++) {
                accumulation[column] += gaussian.nextGaussian() * sign;
            }
        }

        double scale = (StrictMath.sqrt(Math.PI / 2.0) / dimension) * encoded.norm;
        for (int i = 0; i < dimension; i++) {
            double value = accumulation[i] * scale;
            float narrowed = (float) value;
            if (!Double.isFinite(value) || !Float.isFinite(narrowed)) {
                throw new IllegalArgumentException(
                        "QJL reconstruction is outside the finite float range");
            }
            reconstructed[i] = narrowed;
        }
        return reconstructed;
    }

    private void checkRange(int arrayLength, int offset, String name) {
        if (offset < 0 || offset > arrayLength - dimension) {
            throw new IndexOutOfBoundsException(name + " does not contain a complete vector");
        }
    }

    /**
     * Immutable QJL payload. The sign payload uses the canonical one-bit
     * little-endian index layout; norm is retained separately as FP64.
     */
    public static final class EncodedResidual {
        private final int dimension;
        private final long seed;
        private final double norm;
        private final byte[] packedSigns;

        private EncodedResidual(int dimension, long seed, double norm, byte[] packedSigns) {
            this.dimension = dimension;
            this.seed = seed;
            this.norm = norm;
            this.packedSigns = packedSigns.clone();
        }

        public int dimension() {
            return dimension;
        }

        public long seed() {
            return seed;
        }

        public double norm() {
            return norm;
        }

        public byte[] packedSigns() {
            return packedSigns.clone();
        }

        public int payloadSizeBytes() {
            return packedSigns.length;
        }

        public int metadataSizeBytes() {
            return Double.BYTES;
        }
    }

    /**
     * Small deterministic Box-Muller stream. StrictMath keeps the reference
     * projection reproducible across supported Java runtimes.
     */
    private static final class GaussianSequence {
        private static final long GOLDEN_GAMMA = 0x9E3779B97F4A7C15L;
        private static final double TWO_POW_MINUS_53 = 0x1.0p-53;

        private long state;
        private boolean hasSpare;
        private double spare;

        private GaussianSequence(long seed) {
            this.state = seed;
        }

        private double nextGaussian() {
            if (hasSpare) {
                hasSpare = false;
                return spare;
            }
            double first = nextUnitOpen();
            double second = nextUnitOpen();
            double radius = StrictMath.sqrt(-2.0 * StrictMath.log(first));
            double angle = 2.0 * Math.PI * second;
            spare = radius * StrictMath.sin(angle);
            hasSpare = true;
            return radius * StrictMath.cos(angle);
        }

        private double nextUnitOpen() {
            long bits = nextLong() >>> 11;
            return (bits + 0.5) * TWO_POW_MINUS_53;
        }

        private long nextLong() {
            long value = (state += GOLDEN_GAMMA);
            value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
            value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
            return value ^ (value >>> 31);
        }
    }
}
