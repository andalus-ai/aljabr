package tech.kayys.aljabr.backend.cpu.ops;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

/**
 * Cache-blocked SIMD matrix multiplication for the CPU backend.
 */
public final class MatmulCpu {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final ByteOrder NATIVE_ORDER = ByteOrder.nativeOrder();

    private MatmulCpu() {
    }

    /**
     * Performs a matrix multiplication C = A * B.
     * Dimensions: A is [M x K], B is [K x N], C is [M x N]
     * 
     * @param a Memory segment for matrix A
     * @param b Memory segment for matrix B
     * @param c Memory segment for output matrix C
     * @param m Number of rows in A and C
     * @param n Number of columns in B and C
     * @param k Number of columns in A and rows in B
     */
    public static void matmul(MemorySegment a, MemorySegment b, MemorySegment c, int m, int n, int k) {
        // Block sizes for cache efficiency
        int BLOCK_M = 32;
        int BLOCK_N = 128;
        int BLOCK_K = 32;

        int vecLen = SPECIES.length();

        for (int i0 = 0; i0 < m; i0 += BLOCK_M) {
            int imax = Math.min(i0 + BLOCK_M, m);
            for (int k0 = 0; k0 < k; k0 += BLOCK_K) {
                int kmax = Math.min(k0 + BLOCK_K, k);
                for (int j0 = 0; j0 < n; j0 += BLOCK_N) {
                    int jmax = Math.min(j0 + BLOCK_N, n);

                    // Micro-kernel
                    for (int i = i0; i < imax; i++) {
                        long aRowOffset = (long) i * k * 4L;
                        long cRowOffset = (long) i * n * 4L;

                        for (int p = k0; p < kmax; p++) {
                            float a_ip = a.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, aRowOffset + p * 4L);
                            FloatVector va = FloatVector.broadcast(SPECIES, a_ip);

                            long bRowOffset = (long) p * n * 4L;

                            int j = j0;
                            int bound = j0 + SPECIES.loopBound(jmax - j0);
                            for (; j < bound; j += vecLen) {
                                FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b, bRowOffset + j * 4L,
                                        NATIVE_ORDER);
                                FloatVector vc = FloatVector.fromMemorySegment(SPECIES, c, cRowOffset + j * 4L,
                                        NATIVE_ORDER);

                                // vc = vc + va * vb
                                vc.add(va.mul(vb)).intoMemorySegment(c, cRowOffset + j * 4L, NATIVE_ORDER);
                            }

                            // Scalar tail
                            for (; j < jmax; j++) {
                                float valB = b.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, bRowOffset + j * 4L);
                                float valC = c.get(java.lang.foreign.ValueLayout.JAVA_FLOAT, cRowOffset + j * 4L);
                                c.set(java.lang.foreign.ValueLayout.JAVA_FLOAT, cRowOffset + j * 4L,
                                        valC + a_ip * valB);
                            }
                        }
                    }
                }
            }
        }
    }
}
