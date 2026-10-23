package tech.kayys.aljabr.core.tensor;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ShapeTest {
    @Test
    void rejectsNegativeDimensions() {
        assertThrows(IllegalArgumentException.class, () -> new Shape(2, -1));
    }

    @Test
    void rejectsElementCountOverflow() {
        assertThrows(IllegalArgumentException.class, () -> new Shape(Long.MAX_VALUE, 2));
    }

    @Test
    void protectsInternalDimensionsFromMutation() {
        long[] dims = {2, 3};
        Shape shape = new Shape(dims);
        dims[0] = 99;
        assertArrayEquals(new long[] {2, 3}, shape.dims());

        long[] returned = shape.dims();
        returned[1] = 77;
        assertArrayEquals(new long[] {2, 3}, shape.dims());
        assertEquals(6, shape.numel());
    }

    @Test
    void acceptsZeroSizedDimensionsAndScalarShape() {
        assertEquals(0, new Shape(2, 0, 3).numel());
        assertEquals(1, new Shape().numel());
    }
}
