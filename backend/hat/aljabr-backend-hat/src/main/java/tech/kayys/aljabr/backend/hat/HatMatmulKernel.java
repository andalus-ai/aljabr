package tech.kayys.aljabr.backend.hat;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;

import static hat.NDRange.of1D;

/**
 * Project Babylon HAT tensor kernel for row-major F32 matrix multiplication.
 *
 * <p>Computes {@code C[M,N] = A[M,K] * B[K,N]}. The installed Java HAT
 * backends execute a one-dimensional global range, so each work item maps
 * its global index to one flattened output element.</p>
 */
public final class HatMatmulKernel {

    private HatMatmulKernel() {
    }

    /** Executes one output element using a flattened one-dimensional index. */
    @Reflect
    public static void mxmF32(
            KernelContext kc,
            F32Array matrixA,
            F32Array matrixB,
            F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        int outputIndex = kc.gix;
        int outputElements = rows * columns;
        if (outputIndex >= outputElements) {
            return;
        }

        int row = outputIndex / columns;
        int column = outputIndex % columns;

        float acc = 0.0f;
        for (int k = 0; k < reductionSize; k++) {
            acc += matrixA.array((long) row * reductionSize + k)
                    * matrixB.array((long) k * columns + column);
        }
        matrixC.array(outputIndex, acc);
    }

    /** Dispatches exactly one work item per output element. */
    @Reflect
    public static void mxmF32Dispatch(
            ComputeContext cc,
            F32Array matrixA,
            F32Array matrixB,
            F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        cc.dispatchKernel(of1D(rows * columns),
                kc -> mxmF32(kc, matrixA, matrixB, matrixC, rows, columns, reductionSize));
    }
}
