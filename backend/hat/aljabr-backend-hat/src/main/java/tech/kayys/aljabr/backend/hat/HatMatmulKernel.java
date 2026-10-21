package tech.kayys.aljabr.backend.hat;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import static hat.NDRange.of2D;

/**
 * Project Babylon HAT Tensor Kernel for Matrix Multiplication.
 *
 * This kernel uses the @Reflect annotation to allow the HAT code model
 * to analyze it and lower it to GPU instructions (like CUDA or MSL).
 */
public class HatMatmulKernel {

    /**
     * The actual kernel executed on the device.
     */
    @Reflect
    public static void mxmF32(@MappableIface.RO KernelContext kc,
            @MappableIface.RO F32Array matrixA,
            @MappableIface.RO F32Array matrixB,
            @MappableIface.WO F32Array matrixC,
            int size) {
        if (kc.gix < kc.gsx && kc.giy < kc.gsy) {
            float acc = 0.0f;
            for (int k = 0; k < size; k++) {
                acc += (matrixA.array(k * size + kc.giy) * matrixB.array(kc.gix * size + k));
            }
            matrixC.array(kc.gix * size + kc.giy, acc);
        }
    }

    /**
     * Dispatches the kernel over a 2D NDRange.
     */
    @Reflect
    public static void mxmF32Dispatch(@MappableIface.RO ComputeContext cc,
            @MappableIface.RO F32Array matrixA,
            @MappableIface.RO F32Array matrixB,
            @MappableIface.WO F32Array matrixC,
            int globalSize) {
        // Dispatch over a 2D grid matching the size of the matrix.
        cc.dispatchKernel(of2D(globalSize, globalSize, 16, 16),
                kc -> mxmF32(kc, matrixA, matrixB, matrixC, globalSize));
    }
}
