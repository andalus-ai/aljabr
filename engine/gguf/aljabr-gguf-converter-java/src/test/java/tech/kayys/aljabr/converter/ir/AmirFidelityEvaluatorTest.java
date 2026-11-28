package tech.kayys.aljabr.converter.ir;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
                AmirFormatCapabilities.Feature.TYPED_MODEL_METADATA,
                AmirFormatCapabilities.Feature.ARCHITECTURE_DESCRIPTOR,
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
    void reportsExactModelAndTensorExtensionPathsThatSafetensorsCannotPreserve() {
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L),
                "contiguous", null, null, Map.of("quant.scale", "0.125"));
        var model = new ModelDescriptor(ModelIdentity.of("lossy-fixture"), "llama", "gguf",
                Map.of(), Map.of("vendor.metadata", "opaque"), List.of(tensor));

        var report = evaluator.evaluate(model, AmirFormatCapabilities.safetensors());

        assertEquals(CompatibilityReport.ConversionFidelity.PARTIAL, report.fidelity());
        assertEquals(Set.of(
                        "/extensions/vendor.metadata",
                        "/tensors/weight/extensions/quant.scale"),
                report.diagnostics().stream()
                        .map(diagnostic -> diagnostic.path().orElseThrow())
                        .collect(java.util.stream.Collectors.toSet()));
        assertTrue(report.diagnostics().stream().allMatch(diagnostic ->
                diagnostic.code().equals("UNTRANSFERRED_OPAQUE_MODEL_EXTENSIONS")
                        || diagnostic.code().equals("UNTRANSFERRED_OPAQUE_TENSOR_EXTENSIONS")));
    }

    @Test
    void ggufPreservesTheSameOpaqueExtensionPaths() {
        var tensor = new TensorDescriptor("weight", "F32", List.of(2L),
                "contiguous", null, null, Map.of("quant.scale", "0.125"));
        var model = new ModelDescriptor(ModelIdentity.of("lossless-fixture"), "llama", "safetensors",
                Map.of(), Map.of("vendor.metadata", "opaque"), List.of(tensor));

        assertEquals(CompatibilityReport.ConversionFidelity.FULL,
                evaluator.evaluate(model, AmirFormatCapabilities.gguf()).fidelity());
    }


    @Test
    void escapesJsonPointerSegmentsForNamesAndMetadataKeys() {
        var tensor = new TensorDescriptor("weight/0~draft", "F32", List.of(2L),
                "contiguous", null, null, Map.of("vendor/scale~raw", "0.5"));
        var model = new ModelDescriptor(ModelIdentity.of("pointer-fixture"), "llama", "gguf",
                Map.of(), Map.of("vendor/metadata~raw", "opaque"), List.of(tensor));

        var report = evaluator.evaluate(model, AmirFormatCapabilities.safetensors());

        assertEquals(Set.of(
                        "/extensions/vendor~1metadata~0raw",
                        "/tensors/weight~10~0draft/extensions/vendor~1scale~0raw"),
                report.diagnostics().stream()
                        .map(diagnostic -> diagnostic.path().orElseThrow())
                        .collect(java.util.stream.Collectors.toSet()));
    }

    @Test
    void unknownTargetProfileIsRejectedRatherThanAssumedLossless() {
        assertThrows(IllegalArgumentException.class,
                () -> AmirFormatCapabilities.forTarget("some-future-format"));
    }
}
