package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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
    void gqaMapsEachQueryHeadToItsGroupedKeyValueHead() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 4, 2, 1);
        Tensor query = TensorFactory.of(new float[4], backend, 1, 4, 1, 1);
        Tensor key = TensorFactory.of(new float[2], backend, 1, 2, 1, 1);
        Tensor value = TensorFactory.of(new float[] {1f, 2f}, backend, 1, 2, 1, 1);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null, false);
            assertArrayEquals(new float[] {1f, 1f, 2f, 2f}, output.toFloatArray(), 1e-6f);
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
    void gqaCanApplyRotaryEmbeddingBeforeAttention() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 4, 2, 2);
        RotaryEmbedding rope = new RotaryEmbedding(backend, 2, 16);
        Tensor query = TensorFactory.of(new float[1 * 4 * 3 * 2], backend, 1, 4, 3, 2);
        Tensor key = TensorFactory.of(new float[1 * 2 * 3 * 2], backend, 1, 2, 3, 2);
        Tensor value = TensorFactory.of(new float[1 * 2 * 3 * 2], backend, 1, 2, 3, 2);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null, false, rope);
            assertEquals(4, output.shape().dim(1));
            assertEquals(3, output.shape().dim(2));
            assertEquals(backend, output.backend());
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            rope.close();
            attention.close();
        }
    }

    @Test
    void gqaCanApplyRotaryEmbeddingWithExplicitPositions() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 2, 1, 2);
        RotaryEmbedding rope = new RotaryEmbedding(backend, 2, 16);
        Tensor query = TensorFactory.of(new float[1 * 2 * 2 * 2], backend, 1, 2, 2, 2);
        Tensor key = TensorFactory.of(new float[1 * 1 * 2 * 2], backend, 1, 1, 2, 2);
        Tensor value = TensorFactory.of(new float[1 * 1 * 2 * 2], backend, 1, 1, 2, 2);
        Tensor output = null;
        try {
            output = attention.forward(query, key, value, null, false, rope,
                    new long[] {3, 7});
            assertEquals(2, output.shape().dim(1));
            assertEquals(2, output.shape().dim(2));
        } finally {
            if (output != null) output.release();
            value.release();
            key.release();
            query.release();
            rope.close();
            attention.close();
        }
    }

    @Test
    void rotaryEmbeddingChangesAttentionScoresUsingRotatedQueryAndKey() {
        CpuBackend backend = new CpuBackend();
        GroupedQueryAttention attention = new GroupedQueryAttention(backend, 1, 1, 2);
        RotaryEmbedding rope = new RotaryEmbedding(backend, 2, 16);
        Tensor query = TensorFactory.of(new float[] {
                1f, 0f,
                1f, 0f
        }, backend, 1, 1, 2, 2);
        Tensor key = TensorFactory.of(new float[] {
                1f, 0f,
                0f, 1f
        }, backend, 1, 1, 2, 2);
        Tensor value = TensorFactory.of(new float[] {
                0f, 0f,
                10f, 0f
        }, backend, 1, 1, 2, 2);
        Tensor baseline = null;
        Tensor rotated = null;
        try {
            baseline = attention.forward(query, key, value, null, false);
            rotated = attention.forward(query, key, value, null, false, rope,
                    new long[] {0, 1});

            // The second query position is rotated together with its matching key.
            // Its attention distribution must therefore differ from unrotated attention.
            assertNotEquals(baseline.toFloatArray()[2], rotated.toFloatArray()[2], 1e-4f);
        } finally {
            if (rotated != null) rotated.release();
            if (baseline != null) baseline.release();
            value.release();
            key.release();
            query.release();
            rope.close();
            attention.close();
            backend.close();
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
