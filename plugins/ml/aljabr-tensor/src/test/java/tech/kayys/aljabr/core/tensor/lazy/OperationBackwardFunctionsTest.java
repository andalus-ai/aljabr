package tech.kayys.aljabr.core.tensor.lazy;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;

import java.util.List;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class OperationBackwardFunctionsTest {

    private final CpuBackend backend = new CpuBackend();

    private Tensor tensor(float... values) {
        return TensorFactory.of(values, backend, values.length);
    }

    private void assertGradient(Tensor actual, float... expected) {
        assertArrayEquals(expected, actual.toFloatArray(), 1.0e-6f);
    }

    @Test
    void addReturnsIndependentCopiesOfIncomingGradient() {
        Tensor gradient = tensor(10f, 20f);
        Tensor[] result = null;
        try {
            result = OperationBackwardFunctions.forOperation(OpType.ADD)
                    .apply(null, gradient, List.of(), new Object[0]);

            assertGradient(result[0], 10f, 20f);
            assertGradient(result[1], 10f, 20f);
            assertNotSame(gradient, result[0]);
            assertNotSame(gradient, result[1]);
            assertNotSame(result[0], result[1]);
        } finally {
            if (result != null) {
                result[0].release();
                result[1].release();
            }
            gradient.release();
            backend.close();
        }
    }

    @Test
    void arithmeticRulesProduceExpectedVectorJacobianProducts() {
        Tensor left = tensor(2f, 4f);
        Tensor right = tensor(5f, 10f);
        Tensor gradient = tensor(10f, 20f);
        Tensor[] subtract = null;
        Tensor[] multiply = null;
        Tensor[] divide = null;
        try {
            subtract = OperationBackwardFunctions.forOperation(OpType.SUB)
                    .apply(null, gradient, List.of(left, right), new Object[0]);
            multiply = OperationBackwardFunctions.forOperation(OpType.MUL)
                    .apply(null, gradient, List.of(left, right), new Object[0]);
            divide = OperationBackwardFunctions.forOperation(OpType.DIV)
                    .apply(null, gradient, List.of(left, right), new Object[0]);

            assertGradient(subtract[0], 10f, 20f);
            assertGradient(subtract[1], -10f, -20f);
            assertGradient(multiply[0], 50f, 200f);
            assertGradient(multiply[1], 20f, 80f);
            assertGradient(divide[0], 2f, 2f);
            assertGradient(divide[1], -0.8f, -0.8f);
        } finally {
            release(subtract);
            release(multiply);
            release(divide);
            left.release();
            right.release();
            gradient.release();
            backend.close();
        }
    }

    @Test
    void scalarRulesProduceExpectedVectorJacobianProducts() {
        Tensor gradient = tensor(10f, 20f);
        Tensor[] add = null;
        Tensor[] multiply = null;
        Tensor[] divide = null;
        try {
            add = OperationBackwardFunctions.forOperation(OpType.ADD_SCALAR)
                    .apply(null, gradient, List.of(), new Object[] { 3.0f });
            multiply = OperationBackwardFunctions.forOperation(OpType.MUL_SCALAR)
                    .apply(null, gradient, List.of(), new Object[] { 3.0f });
            divide = OperationBackwardFunctions.forOperation(OpType.DIV_SCALAR)
                    .apply(null, gradient, List.of(), new Object[] { 2.0f });

            assertGradient(add[0], 10f, 20f);
            assertGradient(multiply[0], 30f, 60f);
            assertGradient(divide[0], 5f, 10f);
            assertNotSame(gradient, add[0]);
        } finally {
            release(add);
            release(multiply);
            release(divide);
            gradient.release();
            backend.close();
        }
    }

    private static void release(Tensor[] tensors) {
        if (tensors != null) {
            for (Tensor tensor : tensors) {
                if (tensor != null) {
                    tensor.release();
                }
            }
        }
    }
}
