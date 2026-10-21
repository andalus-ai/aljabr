package tech.kayys.aljabr.backend.cpu;

import java.lang.foreign.*;
import java.nio.ByteOrder;

/**
 * 
 * 
 */
public final class FlashAttentionKernel {
    private static final jdk.incubator.vector.VectorSpecies<Float> SPECIES = jdk.incubator.vector.FloatVector.SPECIES_PREFERRED;

    public static void forward(
            MemorySegment Q,
            MemorySegment K,
            MemorySegment V,
            MemorySegment O,
            int seqLen,
            int headDim) {
        final float scale = (float) (1.0 / Math.sqrt(headDim));
        for (int i = 0; i < seqLen; i++) {
            // ---- initialize ----
            float m_i = Float.NEGATIVE_INFINITY;
            float l_i = 0f;
            float[] acc = new float[headDim]; // output accumulator
            // ---- iterate over keys ----
            for (int j = 0; j < seqLen; j++) {
                float score = dot(Q, K, i, j, headDim) * scale;
                float m_new = Math.max(m_i, score);
                float exp_old = (float) Math.exp(m_i - m_new);
                float exp_new = (float) Math.exp(score - m_new);
                float l_new = l_i * exp_old + exp_new;

                float alpha = (l_i == 0f) ? 0f : (l_i * exp_old / l_new);
                float beta = exp_new / l_new;
                // update accumulator
                for (int d = 0; d < headDim; d++) {
                    float v = V.get(ValueLayout.JAVA_FLOAT,
                            ((long) j * headDim + d) * 4);
                    acc[d] = acc[d] * alpha + v * beta;
                }
                m_i = m_new;
                l_i = l_new;
            }
            // ---- write output ----
            for (int d = 0; d < headDim; d++) {
                O.set(ValueLayout.JAVA_FLOAT,
                        ((long) i * headDim + d) * 4,
                        acc[d]);
            }
        }
    }

    private static float dot(
            MemorySegment Q,
            MemorySegment K,
            int qi,
            int kj,
            int dim) {
        int stride = SPECIES.length();
        int d = 0;
        long baseQ = (long) qi * dim;
        long baseK = (long) kj * dim;
        jdk.incubator.vector.FloatVector acc = jdk.incubator.vector.FloatVector.zero(SPECIES);
        for (; d <= dim - stride; d += stride) {
            var vq = jdk.incubator.vector.FloatVector.fromMemorySegment(
                    SPECIES, Q, (baseQ + d) * 4, ByteOrder.nativeOrder());
            var vk = jdk.incubator.vector.FloatVector.fromMemorySegment(
                    SPECIES, K, (baseK + d) * 4, ByteOrder.nativeOrder());
            acc = acc.add(vq.mul(vk));
        }
        float sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; d < dim; d++) {
            float q = Q.get(ValueLayout.JAVA_FLOAT, (baseQ + d) * 4);
            float k = K.get(ValueLayout.JAVA_FLOAT, (baseK + d) * 4);
            sum += q * k;
        }
        return sum;
    }
}