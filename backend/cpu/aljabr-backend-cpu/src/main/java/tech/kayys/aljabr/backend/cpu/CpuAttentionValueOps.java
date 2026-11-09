package tech.kayys.aljabr.backend.cpu;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Candidate F32 weighted-value accumulation kernels for CPU attention.
 *
 * <p>These kernels are benchmark candidates only. The production attention
 * implementation continues to use its double-precision accumulator until
 * end-to-end performance and numerical behavior justify a change.</p>
 */
final class CpuAttentionValueOps {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private CpuAttentionValueOps() {
    }

    static void scalar(float[] accumulator, float[] values, int valueOffset,
            float previousScale, float currentScale, int length) {
        for (int d = 0; d < length; d++) {
            accumulator[d] = accumulator[d] * previousScale
                    + currentScale * values[valueOffset + d];
        }
    }

    static void vectorized(float[] accumulator, float[] values, int valueOffset,
            float previousScale, float currentScale, int length) {
        int bound = SPECIES.loopBound(length);
        FloatVector previous = FloatVector.broadcast(SPECIES, previousScale);
        FloatVector current = FloatVector.broadcast(SPECIES, currentScale);
        int d = 0;
        for (; d < bound; d += SPECIES.length()) {
            FloatVector accumulated = FloatVector.fromArray(SPECIES, accumulator, d);
            FloatVector value = FloatVector.fromArray(SPECIES, values, valueOffset + d);
            accumulated = accumulated.mul(previous).add(value.mul(current));
            accumulated.intoArray(accumulator, d);
        }
        for (; d < length; d++) {
            accumulator[d] = accumulator[d] * previousScale
                    + currentScale * values[valueOffset + d];
        }
    }
}
