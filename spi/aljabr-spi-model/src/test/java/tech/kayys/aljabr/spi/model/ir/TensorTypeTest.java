package tech.kayys.aljabr.spi.model.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

class TensorTypeTest {

    @Test
    void representsStaticShapesAndScalarElementCounts() {
        TensorShape shape = TensorShape.of(2, 3, 4);
        assertEquals(3, shape.rank());
        assertTrue(shape.isStatic());
        assertEquals(List.of(2L, 3L, 4L), shape.staticDimensions());
        assertEquals(OptionalLong.of(24), shape.elementCount());
        assertEquals(OptionalLong.of(1), TensorShape.scalar().elementCount());
    }

    @Test
    void representsSymbolicDimensionsAndBounds() {
        TensorShape shape = new TensorShape(List.of(
                Dimension.symbolic("batch", 1, 64), Dimension.fixed(128)));
        assertFalse(shape.isStatic());
        assertTrue(shape.elementCount().isEmpty());
        assertEquals("batch", shape.dimensions().getFirst().symbol());
        assertThrows(IllegalStateException.class, shape::staticDimensions);
    }

    @Test
    void detectsElementCountOverflowWithoutLosingShape() {
        TensorShape shape = TensorShape.of(Long.MAX_VALUE, 2);
        assertTrue(shape.elementCount().isEmpty());
        assertEquals(2, shape.rank());
    }

    @Test
    void supportsExtensibleDataTypesAndLayouts() {
        TensorDataType fp8 = TensorDataType.of("float8_e4m3fn", TensorDataType.Category.FLOATING_POINT, 8);
        TensorType type = TensorType.of(fp8, TensorShape.of(2, 4))
                .withLayout(TensorLayout.strided(List.of(4L, 1L)));
        assertEquals("float8_e4m3fn", type.dataType().id());
        assertEquals(List.of(4L, 1L), type.layout().strides());
        assertEquals(TensorDataType.Category.OPAQUE, TensorDataType.opaque("vendor:block-float").category());
    }

    @Test
    void validatesLayoutAndQuantizationAgainstShape() {
        assertThrows(IllegalArgumentException.class,
                () -> new TensorType(TensorDataType.F32, TensorShape.of(2, 3),
                        TensorLayout.strided(List.of(3L)), Optional.empty()));

        QuantizationDescriptor quantization = new QuantizationDescriptor(
                "per_channel_affine", TensorDataType.I8, OptionalInt.of(1),
                Optional.of(new BigDecimal("0.125")), OptionalLong.of(0), Map.of("source", "test"));
        assertThrows(IllegalArgumentException.class,
                () -> TensorType.of(TensorDataType.F32, TensorShape.of(2))
                        .withQuantization(quantization));
    }

    @Test
    void integratesTypedInformationWithoutBreakingLegacyDescriptorConstructors() {
        TensorDescriptor legacy = new TensorDescriptor(
                "weight", "f32", List.of(2L, 3L), "contiguous", null, null);
        assertTrue(legacy.tensorType().isEmpty());

        TensorType type = TensorType.of(TensorDataType.F32, TensorShape.symbolic("out", "in"))
                .withLayout(TensorLayout.CONTIGUOUS);
        TensorDescriptor typed = TensorDescriptor.typed("projection.weight", type, null, Map.of());
        assertTrue(typed.tensorType().isPresent());
        assertEquals(TensorShape.symbolic("out", "in"), typed.tensorType().orElseThrow().shape());
        assertTrue(typed.shape().isEmpty()); // Legacy shape cannot encode symbolic dimensions.
    }

    @Test
    void rejectsContradictoryLegacyAndTypedMetadata() {
        TensorType type = TensorType.of(TensorDataType.F32, TensorShape.of(2, 3));
        assertThrows(IllegalArgumentException.class, () -> new TensorDescriptor(
                "weight", "f32", List.of(3L, 2L), "unknown", null, null, Map.of(), Optional.of(type)));
    }

    @Test
    void validatesDimensionConstraints() {
        assertThrows(IllegalArgumentException.class, () -> Dimension.symbolic("batch", 8, 2));
        assertThrows(IllegalArgumentException.class, () -> Dimension.fixed(-1));
        assertThrows(IllegalArgumentException.class, () -> Dimension.symbolic("batch", -1, 8));
    }
}
