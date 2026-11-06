package tech.kayys.aljabr.backend.hat;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import static hat.NDRange.of2D;

/**
 * Project Babylon HAT tensor kernel for row-major F32 matrix multiplication.
 *
 * <p>Computes {@code C[M,N] = A[M,K] * B[K,N]}. The x dimension addresses
 * output columns and the y dimension addresses output rows.</p>
 */
public final class HatMatmulKernel {

    private HatMatmulKernel() {
    }

    /** Executes one output element per work item using the injected kernel context. */
    @Reflect
    public static void mxmF32(
            KernelContext kc,
            F32Array matrixA,
            F32Array matrixB,
            F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        int column = kc.gix;
        int row = kc.giy;

        if (column >= columns || row >= rows) {
            return;
        }

        float acc = 0.0f;
        for (int k = 0; k < reductionSize; k++) {
            acc += matrixA.array((long) row * reductionSize + k)
                    * matrixB.array((long) k * columns + column);
        }
        matrixC.array((long) row * columns + column, acc);
    }

    /** Dispatches a padded {@code columns x rows} work grid. */
    @Reflect
    public static void mxmF32Dispatch(
            ComputeContext cc,
            F32Array matrixA,
            F32Array matrixB,
            F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        int globalColumns = ((columns + 15) / 16) * 16;
        int globalRows = ((rows + 15) / 16) * 16;
        cc.dispatchKernel(of2D(globalColumns, globalRows, 16, 16),
                kc -> mxmF32(kc, matrixA, matrixB, matrixC, rows, columns, reductionSize));
    }
}
