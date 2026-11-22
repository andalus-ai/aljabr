package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.Dimension;
import tech.kayys.aljabr.spi.model.ir.QuantizationDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorDataType;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorShape;
import tech.kayys.aljabr.spi.model.ir.TensorType;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.*;

class AmirTensorTypeCodecTest {

    @Test
    void roundTripsSymbolicShapeStridesAndQuantizationDetails() {
        var shape = new TensorShape(List.of(
                new Dimension(OptionalLong.of(128), null, OptionalLong.of(64), OptionalLong.of(256)),
                Dimension.symbolic("sequence", 1, 4096)));
        var quantization = new QuantizationDescriptor("vendor.per-axis-int8", TensorDataType.I8,
                OptionalInt.of(1), Optional.of(new BigDecimal("0.1250")),
                OptionalLong.of(-3), Map.of("vendor.rounding", "nearest-even"));
        var type = new TensorType(TensorDataType.F16, shape,
                new tech.kayys.aljabr.spi.model.ir.TensorLayout("vendor.blocked", List.of(4096L, 1L)),
                Optional.of(quantization));
        var source = TensorDescriptor.typed("weight/主", type, null, Map.of("source", "extension"));

        Map<String, String> encoded = AmirTensorTypeCodec.encode(List.of(source));
        List<TensorDescriptor> restored = AmirTensorTypeCodec.restore(encoded,
                List.of(new TensorDescriptor("weight/主", "F16", List.of(128L, 32L),
                        "unknown", null, null)));

        assertEquals(Optional.of(type), restored.get(0).tensorType());
        assertEquals("extension", restored.get(0).extensions().get("source"));
        assertTrue(AmirTensorTypeCodec.isEncodedKey(encoded.keySet().iterator().next()));
    }

    @Test
    void rejectsTypedMetadataForUnknownTensor() {
        String key = AmirTensorTypeCodec.encode(List.of(
                TensorDescriptor.typed("known", TensorType.of(TensorDataType.F32, TensorShape.scalar()),
                        null, Map.of()))).keySet().iterator().next();
        assertThrows(IllegalArgumentException.class, () -> AmirTensorTypeCodec.restore(
                Map.of(key, "{}"), List.of()));
    }
}
