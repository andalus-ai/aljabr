package tech.kayys.aljabr.core.nn;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RotaryEmbeddingQueryKeyTest {

    @Test
    void rotatesQueryAndKeyWithDifferentGroupedQueryHeadCounts() {
        CpuBackend backend = new CpuBackend();
        RotaryEmbedding rope = new RotaryEmbedding(backend, 4, 16);
        Tensor query = TensorFactory.of(new float[1 * 4 * 3 * 4], backend, 1, 4, 3, 4);
        Tensor key = TensorFactory.of(new float[1 * 2 * 3 * 4], backend, 1, 2, 3, 4);
        RotaryEmbedding.QueryKey result = null;
        try {
            result = rope.apply(query, key);

            assertEquals(backend, result.query().backend());
            assertEquals(backend, result.key().backend());
            assertEquals(4, result.query().shape().dim(1));
            assertEquals(2, result.key().shape().dim(1));
            assertEquals(3, result.query().shape().dim(2));
            assertEquals(3, result.key().shape().dim(2));
            assertNotSame(query, result.query());
            assertNotSame(key, result.key());
        } finally {
            if (result != null) {
                result.query().release();
                result.key().release();
            }
            query.release();
            key.release();
            rope.close();
            backend.close();
        }
    }

    @Test
    void rejectsQueryAndKeyWithDifferentSequenceLengths() {
        CpuBackend backend = new CpuBackend();
        RotaryEmbedding rope = new RotaryEmbedding(backend, 4, 16);
        Tensor query = TensorFactory.of(new float[1 * 4 * 3 * 4], backend, 1, 4, 3, 4);
        Tensor key = TensorFactory.of(new float[1 * 2 * 2 * 4], backend, 1, 2, 2, 4);
        try {
            assertThrows(IllegalArgumentException.class, () -> rope.apply(query, key));
        } finally {
            query.release();
            key.release();
            rope.close();
            backend.close();
        }
    }

    @Test
    void rejectsKeyOwnedByAnotherBackendBeforeRotatingQuery() {
        CpuBackend backend = new CpuBackend();
        CpuBackend otherBackend = new CpuBackend();
        RotaryEmbedding rope = new RotaryEmbedding(backend, 4, 16);
        Tensor query = TensorFactory.of(new float[1 * 4 * 3 * 4], backend, 1, 4, 3, 4);
        Tensor key = TensorFactory.of(new float[1 * 2 * 3 * 4], otherBackend, 1, 2, 3, 4);
        try {
            assertThrows(IllegalArgumentException.class, () -> rope.apply(query, key));
        } finally {
            query.release();
            key.release();
            rope.close();
            backend.close();
            otherBackend.close();
        }
    }
}
