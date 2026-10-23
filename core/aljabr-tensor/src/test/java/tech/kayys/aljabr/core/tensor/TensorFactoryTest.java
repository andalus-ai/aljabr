package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class TensorFactoryTest {
    @Test
    void rejectsShapesTooLargeForJavaArraysBeforeAllocation() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.zeros(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.ones(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.randn(Long.MAX_VALUE));
    }

    @Test
    void rejectsDataWhoseLengthDoesNotMatchShape() {
        assertThrows(IllegalArgumentException.class, () -> TensorFactory.of(new float[] {1f}, 2));
    }

    @Test
    void rejectsQuantizedStorageWithIncompleteBlockShape() {
        var segment = java.lang.foreign.Arena.ofAuto().allocate(16);
        assertThrows(IllegalArgumentException.class,
                () -> TensorFactory.ofQuantized(segment, DType.Q8_0, 31));
    }
}
