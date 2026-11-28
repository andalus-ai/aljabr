package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Evaluates the fidelity of one AMIR descriptor against a target format profile.
 *
 * <p>Findings are emitted at the narrowest stable descriptor path available.
 * The evaluator is metadata-only and never opens tensor payload references.
 */
public final class AmirFidelityEvaluator {

    public CompatibilityReport evaluate(
            ModelDescriptor model, AmirFormatCapabilities capabilities) {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(capabilities, "capabilities must not be null");

        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>();
        model.tensors().forEach(tensor -> {
            if (tensor.tensorType().isPresent()) {
                requireAtPath(AmirFormatCapabilities.Feature.TYPED_TENSOR_METADATA,
                        capabilities, diagnostics, "tensors." + tensor.name() + ".type");
            }
            tensor.extensions().keySet().forEach(key ->
                    requireAtPath(AmirFormatCapabilities.Feature.OPAQUE_TENSOR_EXTENSIONS,
                            capabilities, diagnostics,
                            "tensors." + tensor.name() + ".extensions." + key));
        });

        if (model.architectureDescriptor().isPresent()) {
            requireAtPath(AmirFormatCapabilities.Feature.ARCHITECTURE_DESCRIPTOR,
                    capabilities, diagnostics, "architectureDescriptor");
        }
        model.typedMetadata().keySet().forEach(key ->
                requireAtPath(AmirFormatCapabilities.Feature.TYPED_MODEL_METADATA,
                        capabilities, diagnostics, "typedMetadata." + key));

        model.graphCollection().ifPresent(collection ->
                requireAtPath(AmirFormatCapabilities.Feature.GRAPH_COLLECTION,
                        capabilities, diagnostics, "graphs"));
        model.assetCollection().ifPresent(collection ->
                requireAtPath(AmirFormatCapabilities.Feature.ASSET_COLLECTION,
                        capabilities, diagnostics, "assets"));
        model.componentCollection().ifPresent(collection ->
                requireAtPath(AmirFormatCapabilities.Feature.COMPONENT_COLLECTION,
                        capabilities, diagnostics, "components"));

        model.extensions().keySet().forEach(key ->
                requireAtPath(AmirFormatCapabilities.Feature.OPAQUE_MODEL_EXTENSIONS,
                        capabilities, diagnostics, "extensions." + key));

        if (capabilities.requiresKnownArchitecture()
                && "unknown".equalsIgnoreCase(model.architecture())) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "UNKNOWN_ARCHITECTURE",
                    "Target format '" + capabilities.format()
                            + "' requires a recognized architecture for full-fidelity conversion",
                    "architecture"));
        }

        CompatibilityReport.ConversionFidelity fidelity = diagnostics.isEmpty()
                ? CompatibilityReport.ConversionFidelity.FULL
                : CompatibilityReport.ConversionFidelity.PARTIAL;
        return new CompatibilityReport(fidelity, List.copyOf(diagnostics));
    }

    private static void requireAtPath(
            AmirFormatCapabilities.Feature feature,
            AmirFormatCapabilities capabilities,
            List<CompatibilityReport.Diagnostic> diagnostics,
            String path) {
        if (!capabilities.preserves(feature)) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "UNTRANSFERRED_" + feature.name(),
                    "Target format '" + capabilities.format()
                            + "' cannot preserve AMIR feature " + feature.name(),
                    path));
        }
    }
}
