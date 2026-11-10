package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class GgufCacheTest {
    @Test
    void cacheHitDoesNotReorderEvictionQueue() {
        GgufPreparedMatrixCache<String, Matrix> cache = new GgufPreparedMatrixCache<>();
        Matrix first = new Matrix(10);
        Matrix second = new Matrix(10);
        Matrix third = new Matrix(10);

        cache.put("first", first, 20);
        cache.put("second", second, 20);
        assertSame(first, cache.get("first"));

        cache.put("third", third, 20);

        assertNull(cache.get("first"));
        assertSame(second, cache.get("second"));
        assertSame(third, cache.get("third"));
        assertEquals(20L, cache.bytes());
    }

    private record Matrix(long estimatedBytes) implements PreparedMatrix {
    }
}
