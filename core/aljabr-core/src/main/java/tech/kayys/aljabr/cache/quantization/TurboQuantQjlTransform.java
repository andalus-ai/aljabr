package tech.kayys.aljabr.cache.quantization;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Deterministic reference implementation of the one-bit Quantized
 * Johnson-Lindenstrauss (QJL) residual transform used by TurboQuant-prod.
 *
 * <p>For a residual {@code r}, encoding stores {@code sign(S * r)} as one bit
 * per coordinate and stores {@code ||r||2} separately. The matrix {@code S}
 * has independent standard-normal entries and is regenerated from the
 * persisted seed. Decoding reconstructs
 * {@code sqrt(pi / 2) / d * ||r||2 * S^T * signs}.</p>
 *
 * <p>This implementation intentionally favors a transparent reference
 * contract over accelerator performance: it materializes a dense
 * {@code d x d} matrix and uses O(d^2) time and memory.</p>
 */
public final class TurboQuantQjlTransform {

    private final int dimension;
    private final long seed;
    private final double[] projection;

    public TurboQuantQjlTransform(int dimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        int matrixElements;
        try {
            matrixElements = Math.multiplyExact(dimension, dimension);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("dimension is too large for a dense QJL reference matrix", overflow);
        }
        this.dimension = dimension;
        this.seed = seed;
        this.projection = gaussianMatrix(matrixElements, seed);
    }

    public int dimension() {
        return dimension;
    }

    public long seed() {
        return seed;
    }

    /**
     * Encodes the residual as packed signs, using zero for -1 and one for +1.
     */
    public byte[] quantize(float[] residual, int offset) {
        Objects.requireNonNull(residual, "residual");
        checkRange(residual.length, offset, dimension, "offset");
        int[] signs = new int[dimension];
        for (int row = 0; row < dimension; row++) {
            double dot = 0.0;
            int matrixOffset = row * dimension;
            for (int column = 0; column < dimension; column++) {
                float value = residual[offset + column];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("residual values must be finite");
                }
                dot += projection[matrixOffset + column] * value;
            }
            signs[row] = dot >= 0.0 ? 1 : 0;
        }
        return TurboQuantBitPacking.packIndices(signs, 1);
    }

    /** Computes the exact L2 norm that must accompany the packed signs. */
    public double residualNorm(float[] residual, int offset) {
        Objects.requireNonNull(residual, "residual");
        checkRange(residual.length, offset, dimension, "offset");
        double squaredNorm = 0.0;
        for (int i = 0; i < dimension; i++) {
            float value = residual[offset + i];
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("residual values must be finite");
            }
            squaredNorm += (double) value * value;
        }
        double norm = Math.sqrt(squaredNorm);
        if (!Double.isFinite(norm)) {
            throw new IllegalArgumentException("residual norm is not finite");
        }
        return norm;
    }

    /**
     * Reconstructs the QJL residual estimate into the destination.
     * The destination is unchanged if validation or reconstruction fails.
     */
    public void reconstruct(byte[] packedSigns, double residualNorm, float[] destination, int offset) {
        Objects.requireNonNull(packedSigns, "packedSigns");
        Objects.requireNonNull(destination, "destination");
        if (!Double.isFinite(residualNorm) || residualNorm < 0.0) {
            throw new IllegalArgumentException("residualNorm must be finite and non-negative");
        }
        checkRange(destination.length, offset, dimension, "offset");
        if (residualNorm == 0.0) {
            java.util.Arrays.fill(destination, offset, offset + dimension, 0.0f);
            return;
        }
        int[] signs = TurboQuantBitPacking.unpackIndices(packedSigns, dimension, 1);
        double scale = Math.sqrt(Math.PI / 2.0) * residualNorm / dimension;
        float[] reconstructed = new float[dimension];
        for (int column = 0; column < dimension; column++) {
            double sum = 0.0;
            for (int row = 0; row < dimension; row++) {
                int sign = signs[row] == 1 ? 1 : -1;
                sum += projection[row * dimension + column] * sign;
            }
            double value = scale * sum;
            if (!Double.isFinite(value) || Math.abs(value) > Float.MAX_VALUE) {
                throw new IllegalArgumentException("QJL reconstruction exceeds finite float range");
            }
            reconstructed[column] = (float) value;
        }
        System.arraycopy(reconstructed, 0, destination, offset, dimension);
    }

    private static double[] gaussianMatrix(int count, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double[] values = new double[count];
        int index = 0;
        while (index < count) {
            double u1 = 1.0 - random.nextDouble();
            double u2 = random.nextDouble();
            double radius = Math.sqrt(-2.0 * Math.log(u1));
            double angle = 2.0 * Math.PI * u2;
            values[index++] = radius * Math.cos(angle);
            if (index < count) {
                values[index++] = radius * Math.sin(angle);
            }
        }
        return values;
    }

    private static void checkRange(int length, int offset, int count, String name) {
        if (offset < 0 || count < 0 || offset > length - count) {
            throw new IndexOutOfBoundsException(name + " does not address the complete QJL vector");
        }
    }
}
