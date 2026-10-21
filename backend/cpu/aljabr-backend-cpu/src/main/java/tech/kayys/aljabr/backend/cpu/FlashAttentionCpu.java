package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.concurrent.*;
import jdk.incubator.vector.*;

public final class FlashAttentionCpu {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int BLOCK_Q = 64;
    private static final int BLOCK_K = 64;

    private FlashAttentionCpu() {
    }

    // =========================================================
    // PUBLIC ENTRY (single-head)
    // =========================================================
    public static Tensor forward(
            Tensor Q,
            Tensor K,
            Tensor V,
            int numThreads) {
        int seqLen = (int) Q.shape().dim(0);
        int dim = (int) Q.shape().dim(1);
        CpuBuffer out = new CpuBuffer((long) seqLen * dim * 4);
        try {
            compute(
                    ((DefaultTensor) Q).buffer().segment(),
                    ((DefaultTensor) K).buffer().segment(),
                    ((DefaultTensor) V).buffer().segment(),
                    out.segment(),
                    seqLen,
                    dim,
                    numThreads);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return new DefaultTensor(
                new Shape(seqLen, dim),
                Q.dtype(),
                Q.device(),
                out,
                null // backend injected outside if needed
        );
    }

    // =========================================================
    // CORE KERNEL
    // =========================================================
    private static void compute(
            MemorySegment Q,
            MemorySegment K,
            MemorySegment V,
            MemorySegment O,
            int seqLen,
            int dim,
            int numThreads) throws InterruptedException {
        ExecutorService pool = Executors.newFixedThreadPool(numThreads);
        int qBlocks = (seqLen + BLOCK_Q - 1) / BLOCK_Q;
        CountDownLatch latch = new CountDownLatch(qBlocks);
        for (int qb = 0; qb < qBlocks; qb++) {
            final int qStart = qb * BLOCK_Q;
            final int qEnd = Math.min(qStart + BLOCK_Q, seqLen);
            pool.submit(() -> {
                computeQBlock(Q, K, V, O, qStart, qEnd, seqLen, dim);
                latch.countDown();
            });
        }
        latch.await();
        pool.shutdown();
    }

    // =========================================================
    // Q BLOCK COMPUTATION
    // =========================================================
    private static void computeQBlock(
            MemorySegment Q,
            MemorySegment K,
            MemorySegment V,
            MemorySegment O,
            int qStart,
            int qEnd,
            int seqLen,
            int dim) {
        float scale = (float) (1.0 / Math.sqrt(dim));

        for (int i = qStart; i < qEnd; i++) {
            float m_i = Float.NEGATIVE_INFINITY;
            float l_i = 0f;
            float[] acc = new float[dim];
            for (int kb = 0; kb < seqLen; kb += BLOCK_K) {
                int kEnd = Math.min(kb + BLOCK_K, seqLen);
                for (int j = kb; j < kEnd; j++) {
                    float score = dot(Q, K, i, j, dim) * scale;
                    float m_new = Math.max(m_i, score);
                    float exp_old = (float) Math.exp(m_i - m_new);
                    float exp_new = (float) Math.exp(score - m_new);
                    float l_new = l_i * exp_old + exp_new;
                    float alpha = (l_i == 0f) ? 0f : (l_i * exp_old / l_new);
                    float beta = exp_new / l_new;
                    updateAccumulator(acc, V, j, dim, alpha, beta);
                    m_i = m_new;
                    l_i = l_new;
                }
            }
            // write output
            long base = (long) i * dim;
            for (int d = 0; d < dim; d++) {
                O.set(ValueLayout.JAVA_FLOAT, (base + d) * 4, acc[d]);
            }
        }
    }

    // =========================================================
    // VECTOR ACCUMULATOR UPDATE (V3 STYLE)
    // =========================================================
    private static void updateAccumulator(
            float[] acc,
            MemorySegment V,
            int j,
            int dim,
            float alpha,
            float beta) {
        int stride = SPECIES.length();
        int d = 0;
        long base = (long) j * dim;
        for (; d <= dim - stride; d += stride) {
            FloatVector vacc = FloatVector.fromArray(SPECIES, acc, d);
            FloatVector vv = FloatVector.fromMemorySegment(
                    SPECIES,
                    V,
                    (base + d) * 4,
                    ByteOrder.nativeOrder());
            vacc = vacc.mul(alpha).add(vv.mul(beta));
            vacc.intoArray(acc, d);
        }
        for (; d < dim; d++) {
            float v = V.get(ValueLayout.JAVA_FLOAT, (base + d) * 4);
            acc[d] = acc[d] * alpha + v * beta;
        }
    }

    // =========================================================
    // VECTOR DOT PRODUCT
    // =========================================================
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
        FloatVector acc = FloatVector.zero(SPECIES);
        for (; d <= dim - stride; d += stride) {
            FloatVector vq = FloatVector.fromMemorySegment(
                    SPECIES, Q,
                    (baseQ + d) * 4,
                    ByteOrder.nativeOrder());
            FloatVector vk = FloatVector.fromMemorySegment(
                    SPECIES,
                    K,
                    (baseK + d) * 4,
                    ByteOrder.nativeOrder());
            acc = acc.add(vq.mul(vk));
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; d < dim; d++) {
            float q = Q.get(ValueLayout.JAVA_FLOAT, (baseQ + d) * 4);
            float k = K.get(ValueLayout.JAVA_FLOAT, (baseK + d) * 4);
            sum += q * k;
        }
        return sum;
    }
}