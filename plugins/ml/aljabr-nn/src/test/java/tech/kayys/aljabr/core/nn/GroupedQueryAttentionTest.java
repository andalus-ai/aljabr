package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GroupedQueryAttentionTest {

    @Test
    void gqaAcceptsMoreQueryHeadsThanKeyValueHeads() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 4, 2, 2);
        Tensor query = TensorFactory.of(new float[1 * 4 * 3 * 2], backend, 1, 4, 3, 2);
        Tensor key = TensorFactory.of(new float[1 * 2 * 3 * 2], backend, 1, 2, 3, 2);
        Tensor value = TensorFactory.of(new float[1 * 2 * 3 * 2], backend, 1, 2, 3, 2);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null, false);
            assertEquals(4, output.shape().dim(1));
            assertEquals(3, output.shape().dim(2));
            assertEquals(2, output.shape().dim(3));
            assertEquals(backend, output.backend());
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            attention.close();
        }
    }

    @Test
    void gqaSupportsDifferentQueryAndKeySequenceLengths() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 4, 1, 2);
        Tensor query = TensorFactory.of(new float[1 * 4 * 2 * 2], backend, 1, 4, 2, 2);
        Tensor key = TensorFactory.of(new float[1 * 1 * 5 * 2], backend, 1, 1, 5, 2);
        Tensor value = TensorFactory.of(new float[1 * 1 * 5 * 2], backend, 1, 1, 5, 2);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null, false);
            assertEquals(2, output.shape().dim(2));
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            attention.close();
        }
    }

    @Test
    void rejectsNonDivisibleHeadCounts() {
        CpuBackend backend = new CpuBackend();
        assertThrows(IllegalArgumentException.class,
                () -> new GroupedQueryAttention(backend, 3, 2, 4));
        backend.close();
    }

    @Test
    void rejectsMismatchedTensorHeadCounts() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 4, 2, 2);
        Tensor query = TensorFactory.of(new float[1 * 4 * 2 * 2], backend, 1, 4, 2, 2);
        Tensor key = TensorFactory.of(new float[1 * 1 * 2 * 2], backend, 1, 1, 2, 2);
        Tensor value = TensorFactory.of(new float[1 * 2 * 2 * 2], backend, 1, 2, 2, 2);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> attention.forward(query, key, value, null, false));
        } finally {
            value.release();
            key.release();
            query.release();
            attention.close();
        }
    }
}
