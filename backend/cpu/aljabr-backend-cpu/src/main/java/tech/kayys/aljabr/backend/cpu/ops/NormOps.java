package tech.kayys.aljabr.backend.cpu.ops;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public final class NormOps {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    private NormOps() {
    }

    public static void rmsNorm(MemorySegment input, MemorySegment weight, MemorySegment output, long numElements,
            int hiddenDim, float eps) {
        long numRows = numElements / hiddenDim;

        for (long i = 0; i < numRows; i++) {
            long rowOffset = i * hiddenDim * 4L;

            // 1. Compute sum of squares
            float sumSq = 0.0f;
            int j = 0;
            int bound = SPECIES.loopBound(hiddenDim);
            FloatVector sumSqVec = FloatVector.zero(SPECIES);

            for (; j < bound; j += SPECIES.length()) {
                FloatVector v = FloatVector.fromMemorySegment(SPECIES, input, rowOffset + j * 4L, NATIVE_ORDER);
                sumSqVec = sumSqVec.add(v.mul(v));
            }
            sumSq += sumSqVec.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);

            for (; j < hiddenDim; j++) {
                float val = input.get(ValueLayout.JAVA_FLOAT, rowOffset + j * 4L);
                sumSq += val * val;
            }

            // 2. Compute RMS
            float rms = (float) Math.sqrt(sumSq / hiddenDim + eps);
            float invRms = 1.0f / rms;

            // 3. Normalize and scale
            j = 0;
            for (; j < bound; j += SPECIES.length()) {
                FloatVector v = FloatVector.fromMemorySegment(SPECIES, input, rowOffset + j * 4L, NATIVE_ORDER);
                FloatVector w = FloatVector.fromMemorySegment(SPECIES, weight, j * 4L, NATIVE_ORDER);
                v.mul(invRms).mul(w).intoMemorySegment(output, rowOffset + j * 4L, NATIVE_ORDER);
            }

            for (; j < hiddenDim; j++) {
                float val = input.get(ValueLayout.JAVA_FLOAT, rowOffset + j * 4L);
                float w = weight.get(ValueLayout.JAVA_FLOAT, j * 4L);
                output.set(ValueLayout.JAVA_FLOAT, rowOffset + j * 4L, (val * invRms) * w);
            }
        }
    }
}
