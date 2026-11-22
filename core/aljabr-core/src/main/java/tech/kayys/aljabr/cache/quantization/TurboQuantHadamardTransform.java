package tech.kayys.aljabr.cache.quantization;

import java.util.Objects;

/**
 * Deterministic randomized Walsh-Hadamard transform for reference KV-cache
 * quantization algorithms.
 *
 * <p>The transform is {@code H D}, where {@code H} is the normalized
 * Walsh-Hadamard matrix and {@code D} is a reproducible diagonal Rademacher
 * matrix derived from the supplied seed. Its inverse is {@code D H}. This
 * provides an orthogonal, data-oblivious mixing transform without allocating
 * or storing a dense matrix.</p>
 *
 * <p>This class is a numerical primitive, not a complete TurboQuant codec.
 * Quantization, residual correction (QJL), storage, and attention integration
 * remain separate responsibilities.</p>
 *
 * <p>The dimension must be a positive power of two. Callers are responsible
 * for padding non-power-of-two vectors and recording their original length.</p>
 */
public final class TurboQuantHadamardTransform {

    private static final long GOLDEN_GAMMA = 0x9E3779B97F4A7C15L;

    private final int dimension;
    private final long seed;

    public TurboQuantHadamardTransform(int dimension, long seed) {
        if (dimension <= 0 || (dimension & (dimension - 1)) != 0) {
            throw new IllegalArgumentException("dimension must be a positive power of two");
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

    /** Applies {@code H D} to one vector. Input and output may be the same array. */
    public void forward(float[] source, int sourceOffset, float[] destination, int destinationOffset) {
        copyAndValidate(source, sourceOffset, destination, destinationOffset);
        applySigns(destination, destinationOffset);
        applyNormalizedHadamard(destination, destinationOffset);
    }

    /** Applies {@code D H}, the exact inverse of {@link #forward}. */
    public void inverse(float[] source, int sourceOffset, float[] destination, int destinationOffset) {
        copyAndValidate(source, sourceOffset, destination, destinationOffset);
        applyNormalizedHadamard(destination, destinationOffset);
        applySigns(destination, destinationOffset);
    }

    private void copyAndValidate(
            float[] source, int sourceOffset, float[] destination, int destinationOffset) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(destination, "destination");
        checkRange(source.length, sourceOffset, "sourceOffset");
        checkRange(destination.length, destinationOffset, "destinationOffset");
        System.arraycopy(source, sourceOffset, destination, destinationOffset, dimension);
        for (int i = 0; i < dimension; i++) {
            if (!Float.isFinite(destination[destinationOffset + i])) {
                throw new IllegalArgumentException("vector values must be finite");
            }
        }
    }

    private void checkRange(int arrayLength, int offset, String name) {
        if (offset < 0 || offset > arrayLength - dimension) {
            throw new IndexOutOfBoundsException(name + " does not address a complete vector");
        }
    }

    private void applySigns(float[] values, int offset) {
        for (int i = 0; i < dimension; i++) {
            if ((mix64(seed + GOLDEN_GAMMA * i) & Long.MIN_VALUE) != 0) {
                values[offset + i] = -values[offset + i];
            }
        }
    }

    private void applyNormalizedHadamard(float[] values, int offset) {
        for (int width = 1; width < dimension; width <<= 1) {
            int block = width << 1;
            for (int base = 0; base < dimension; base += block) {
                for (int j = 0; j < width; j++) {
                    int left = offset + base + j;
                    int right = left + width;
                    float a = values[left];
                    float b = values[right];
                    values[left] = a + b;
                    values[right] = a - b;
                }
            }
        }
        float normalization = (float) (1.0 / Math.sqrt(dimension));
        for (int i = 0; i < dimension; i++) {
            values[offset + i] *= normalization;
        }
    }

    private static long mix64(long value) {
        value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
        value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
        return value ^ (value >>> 31);
    }
}
