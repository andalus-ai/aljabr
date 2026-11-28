package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.ComponentCollection;
import tech.kayys.aljabr.spi.model.ir.ComponentGraphBinding;
import tech.kayys.aljabr.spi.model.ir.ControlFlowRegion;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;
import tech.kayys.aljabr.spi.model.ir.GraphCollection;
import tech.kayys.aljabr.spi.model.ir.GraphDescriptor;
import tech.kayys.aljabr.spi.model.ir.GraphSignature;
import tech.kayys.aljabr.spi.model.ir.GraphValue;
import tech.kayys.aljabr.spi.model.ir.ModelComponent;
import tech.kayys.aljabr.spi.model.ir.ModelComponentKind;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.OperationDescriptor;
import tech.kayys.aljabr.spi.model.ir.TensorDataType;
import tech.kayys.aljabr.spi.model.ir.TensorShape;
import tech.kayys.aljabr.spi.model.ir.TensorType;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class AmirGraphCodecTest {
    @Test
    void roundTripsGraphsOperationsControlFlowAndComponentBindings() {
        TensorType tensor = TensorType.of(TensorDataType.F32, TensorShape.of(1));
        GraphDescriptor graph = new GraphDescriptor("encoder",
                List.of(new GraphValue("input", tensor), new GraphValue("output", tensor, "embedding")),
                List.of(new OperationDescriptor("project", "linear",
                        Map.of("x", List.of("input")), Map.of("y", "output"),
                        Map.of("bias", new ConfigurationValue.BooleanValue(false)), Map.of("vendor.op", "opaque"))),
                new GraphSignature(Map.of("pixels", "input"), Map.of("embedding", "output")),
                List.of(new ControlFlowRegion("body", ControlFlowRegion.Kind.LOOP, Optional.empty(),
                        List.of("project"), List.of("input"), List.of("output"), Map.of("vendor.region", "x"))),
                Map.of("vendor.graph", "opaque"));
        ComponentCollection components = new ComponentCollection(List.of(
                new ModelComponent("vision", ModelComponentKind.IMAGE_PROCESSOR, Map.of(),
                        List.of(), List.of(new ComponentGraphBinding("encoder",
                        Map.of("pixels", "pixels"), Map.of("embedding", "embedding"))), Map.of())));
        ModelDescriptor source = model().withGraphCollection(new GraphCollection(List.of(graph)))
                .withComponentCollection(components);

        Map<String, String> encoded = AmirModelLayerCodec.encode(source);
        assertTrue(encoded.containsKey(AmirModelLayerCodec.GRAPHS_KEY));
        assertTrue(encoded.containsKey(AmirModelLayerCodec.COMPONENTS_KEY));

        ModelDescriptor restored = AmirModelLayerCodec.restore(new java.util.LinkedHashMap<>(encoded), model());

        assertEquals(source.graphCollection(), restored.graphCollection());
        assertEquals(source.componentCollection(), restored.componentCollection());
    }

    @Test
    void rejectsUnsupportedGraphEnvelopeVersionAndUnknownFields() {
        assertThrows(IllegalArgumentException.class,
                () -> AmirGraphCodec.decode("{\"version\":2,\"graphs\":[]}"));
        assertThrows(IllegalArgumentException.class,
                () -> AmirGraphCodec.decode("{\"version\":1,\"graphs\":[],\"future\":true}"));
    }

    private static ModelDescriptor model() {
        return new ModelDescriptor(ModelIdentity.of("graph-envelope-test"),
                "unknown", "safetensors", Map.of(), Map.of(), List.of());
    }
}
