package tech.kayys.aljabr.backend.hat;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

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

    /** Executes one output element per work item. */
    @Reflect
    public static void mxmF32(@MappableIface.RO KernelContext kc,
            @MappableIface.RO F32Array matrixA,
            @MappableIface.RO F32Array matrixB,
            @MappableIface.WO F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        int row = kc.giy;
        int column = kc.gix;
        if (row < rows && column < columns) {
            float acc = 0.0f;
            for (int k = 0; k < reductionSize; k++) {
                acc += matrixA.array(row * reductionSize + k)
                        * matrixB.array(k * columns + column);
            }
            matrixC.array(row * columns + column, acc);
        }
    }

    /** Dispatches an {@code N x M} work grid for an {@code M x N} result. */
    @Reflect
    public static void mxmF32Dispatch(@MappableIface.RO ComputeContext cc,
            @MappableIface.RO F32Array matrixA,
            @MappableIface.RO F32Array matrixB,
            @MappableIface.WO F32Array matrixC,
            int rows,
            int columns,
            int reductionSize) {
        // Use complete 16x16 work-groups. The logical matrix dimensions are
        // passed separately and checked by the kernel to mask padded work-items.
        int globalColumns = ((columns + 15) / 16) * 16;
        int globalRows = ((rows + 15) / 16) * 16;
        cc.dispatchKernel(of2D(globalColumns, globalRows, 16, 16),
                kc -> mxmF32(kc, matrixA, matrixB, matrixC, rows, columns, reductionSize));
        // Ensure device-side writes are copied back to the host-visible buffer
        
        
    }
}
