package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** SIMD dot product for contiguous CPU attention feature vectors. */
final class CpuAttentionOps {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    private CpuAttentionOps() {
    }

    static float dot(MemorySegment left, long leftByteOffset,
            MemorySegment right, long rightByteOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int dimension = 0;
        FloatVector accumulator = FloatVector.zero(SPECIES);
        for (; dimension < bound; dimension += SPECIES.length()) {
            FloatVector a = FloatVector.fromMemorySegment(
                    SPECIES, left, leftByteOffset + (long) dimension * Float.BYTES, NATIVE_ORDER);
            FloatVector b = FloatVector.fromMemorySegment(
                    SPECIES, right, rightByteOffset + (long) dimension * Float.BYTES, NATIVE_ORDER);
            accumulator = accumulator.add(a.mul(b));
        }

        float result = accumulator.reduceLanes(VectorOperators.ADD);
        for (; dimension < length; dimension++) {
            float a = left.get(ValueLayout.JAVA_FLOAT,
                    leftByteOffset + (long) dimension * Float.BYTES);
            float b = right.get(ValueLayout.JAVA_FLOAT,
                    rightByteOffset + (long) dimension * Float.BYTES);
            result += a * b;
        }
        return result;
    }
}
