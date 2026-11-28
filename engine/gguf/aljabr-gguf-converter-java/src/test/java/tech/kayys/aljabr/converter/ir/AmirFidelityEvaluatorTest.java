package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AmirFidelityEvaluatorTest {

    private final AmirFidelityEvaluator evaluator = new AmirFidelityEvaluator();

    @Test
    void builtInProfilesDeclareTheEnvelopesTheyPreserve() {
        var safetensors = AmirFormatCapabilities.safetensors();
        var gguf = AmirFormatCapabilities.gguf();

        for (var feature : List.of(
                AmirFormatCapabilities.Feature.TYPED_TENSOR_METADATA,
                AmirFormatCapabilities.Feature.GRAPH_COLLECTION,
                AmirFormatCapabilities.Feature.ASSET_COLLECTION,
                AmirFormatCapabilities.Feature.COMPONENT_COLLECTION)) {
            assertTrue(safetensors.preserves(feature), "Safetensors must preserve " + feature);
            assertTrue(gguf.preserves(feature), "GGUF must preserve " + feature);
        }
        assertFalse(safetensors.preserves(AmirFormatCapabilities.Feature.OPAQUE_MODEL_EXTENSIONS));
        assertFalse(safetensors.preserves(AmirFormatCapabilities.Feature.OPAQUE_TENSOR_EXTENSIONS));
        assertTrue(gguf.preserves(AmirFormatCapabilities.Feature.OPAQUE_MODEL_EXTENSIONS));
        assertTrue(gguf.preserves(AmirFormatCapabilities.Feature.OPAQUE_TENSOR_EXTENSIONS));
    }

    @Test
    void emptyDescriptorIsFullFidelityForBothBuiltInTargets() {
        ModelDescriptor model = new ModelDescriptor(
                ModelIdentity.of("empty-fixture"), "fixture", "safetensors",
                Map.of(), Map.of(), List.of());

        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                evaluator.evaluate(model, AmirFormatCapabilities.safetensors()).fidelity());
        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                evaluator.evaluate(model, AmirFormatCapabilities.gguf()).fidelity());
    }

    @Test
    void unknownTargetProfileIsRejectedRatherThanAssumedLossless() {
        assertThrows(IllegalArgumentException.class,
                () -> AmirFormatCapabilities.forTarget("some-future-format"));
    }
}
