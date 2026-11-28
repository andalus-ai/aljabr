package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.AssetCollection;
import tech.kayys.aljabr.spi.model.ir.AssetReference;
import tech.kayys.aljabr.spi.model.ir.ComponentCollection;
import tech.kayys.aljabr.spi.model.ir.ComponentGraphBinding;
import tech.kayys.aljabr.spi.model.ir.ConfigurationValue;
import tech.kayys.aljabr.spi.model.ir.GraphCollection;
import tech.kayys.aljabr.spi.model.ir.GraphDescriptor;
import tech.kayys.aljabr.spi.model.ir.GraphSignature;
import tech.kayys.aljabr.spi.model.ir.GraphValue;
import tech.kayys.aljabr.spi.model.ir.TensorDataType;
import tech.kayys.aljabr.spi.model.ir.TensorShape;
import tech.kayys.aljabr.spi.model.ir.TensorType;
import tech.kayys.aljabr.spi.model.ir.ModelComponent;
import tech.kayys.aljabr.spi.model.ir.ModelComponentKind;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.*;

class AmirModelLayerCodecTest {

    @Test
    void roundTripsAssetsAndTypedComponentsWithoutResolvingPayloads() {
        AssetCollection assets = new AssetCollection(List.of(
                new AssetReference("tokenizer", "file:assets/tokenizer.json",
                        Optional.of("application/json"), OptionalLong.of(128),
                        Optional.of("abc123"), Optional.of("sha256"),
                        Map.of("vendor.asset", "opaque"))));
        ComponentCollection components = new ComponentCollection(List.of(
                new ModelComponent("text-tokenizer", ModelComponentKind.TOKENIZER,
                        Map.of(
                                "revision", new ConfigurationValue.StringValue("v2"),
                                "padding", new ConfigurationValue.BooleanValue(true),
                                "scale", new ConfigurationValue.DecimalValue(new BigDecimal("0.125")),
                                "special-ids", new ConfigurationValue.IntegerListValue(List.of(1L, 2L)),
                                "languages", new ConfigurationValue.StringListValue(List.of("en", "id"))),
                        List.of("tokenizer"), List.of(), Map.of("vendor.component", "opaque"))));

        ModelDescriptor source = model().withAssetCollection(assets).withComponentCollection(components);
        Map<String, String> encoded = AmirModelLayerCodec.encode(source);
        ModelDescriptor restored = AmirModelLayerCodec.restore(
                new java.util.LinkedHashMap<>(encoded), model());

        assertEquals(assets, restored.assetCollection().orElseThrow());
        assertEquals(components, restored.componentCollection().orElseThrow());
        assertEquals("file:assets/tokenizer.json",
                restored.assetCollection().orElseThrow().assets().getFirst().location());
    }

    @Test
    void graphBoundComponentsAreNotEncodedWithoutTheirGraphCollection() {
        ComponentCollection components = new ComponentCollection(List.of(
                new ModelComponent("vision-encoder", ModelComponentKind.IMAGE_PROCESSOR,
                        Map.of(), List.of(), List.of(new ComponentGraphBinding(
                                "vision-forward", Map.of("pixels", "input"), Map.of("embedding", "output"))),
                        Map.of())));
        ModelDescriptor source = model()
                .withGraphCollection(new GraphCollection(List.of(GraphDescriptor.of(
                        "vision-forward",
                        List.of(new GraphValue("input", TensorType.of(TensorDataType.F32, TensorShape.of(1)))),
                        List.of(),
                        new GraphSignature(Map.of("pixels", "input"), Map.of("embedding", "input"))))))
                .withComponentCollection(components);

        Map<String, String> encoded = AmirModelLayerCodec.encode(source);

        assertFalse(encoded.containsKey(AmirModelLayerCodec.COMPONENTS_KEY));
        assertTrue(AmirModelLayerCodec.hasNoGraphBindings(
                new ComponentCollection(List.of(ModelComponent.of("tokenizer", ModelComponentKind.TOKENIZER)))));
    }

    @Test
    void rejectsUnsupportedEnvelopeVersionsAndUnexpectedFields() {
        assertThrows(IllegalArgumentException.class, () ->
                AmirModelLayerCodec.restore(
                        new java.util.LinkedHashMap<>(Map.of(
                                AmirModelLayerCodec.ASSETS_KEY,
                                "{\\"version\\":2,\\"assets\\":[]}")),
                        model()));
        assertThrows(IllegalArgumentException.class, () ->
                AmirModelLayerCodec.restore(
                        new java.util.LinkedHashMap<>(Map.of(
                                AmirModelLayerCodec.COMPONENTS_KEY,
                                "{\\"version\\":1,\\"components\\":[],\\"future\\":true}")),
                        model()));
    }

    private static ModelDescriptor model() {
        return new ModelDescriptor(ModelIdentity.of("model-layer-test"),
                "unknown", "safetensors", Map.of(), Map.of(), List.of());
    }
}
