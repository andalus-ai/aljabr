package tech.kayys.aljabr.backend.cpu;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Double-precision weighted-value accumulation candidates for CPU attention.
 *
 * <p>The production rank-4 attention path uses a double accumulator. These
 * methods isolate that arithmetic for benchmarking; they do not perform
 * MemorySegment loads or float-to-double conversion.</p>
 */
final class CpuAttentionDoubleValueOps {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private CpuAttentionDoubleValueOps() {
    }

    static void scalar(double[] accumulator, double[] values, int valueOffset,
            double previousScale, double currentScale, int length) {
        for (int d = 0; d < length; d++) {
            accumulator[d] = accumulator[d] * previousScale
                    + currentScale * values[valueOffset + d];
        }
    }

    static void vectorized(double[] accumulator, double[] values, int valueOffset,
            double previousScale, double currentScale, int length) {
        int bound = SPECIES.loopBound(length);
        DoubleVector previous = DoubleVector.broadcast(SPECIES, previousScale);
        DoubleVector current = DoubleVector.broadcast(SPECIES, currentScale);
        int d = 0;
        for (; d < bound; d += SPECIES.length()) {
            DoubleVector accumulated = DoubleVector.fromArray(SPECIES, accumulator, d);
            DoubleVector value = DoubleVector.fromArray(SPECIES, values, valueOffset + d);
            accumulated = accumulated.mul(previous).add(value.mul(current));
            accumulated.intoArray(accumulator, d);
        }
        for (; d < length; d++) {
            accumulator[d] = accumulator[d] * previousScale
                    + currentScale * values[valueOffset + d];
        }
    }
}
