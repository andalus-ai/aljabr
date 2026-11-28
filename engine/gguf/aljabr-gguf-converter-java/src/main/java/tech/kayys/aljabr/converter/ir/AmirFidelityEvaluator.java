package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Evaluates the fidelity of one AMIR descriptor against a target format profile.
 *
 * <p>The evaluator only reports semantics that are actually present in the
 * descriptor. A target's inability to preserve an absent feature therefore
 * does not downgrade an otherwise lossless conversion.
 */
public final class AmirFidelityEvaluator {

    public CompatibilityReport evaluate(
            ModelDescriptor model, AmirFormatCapabilities capabilities) {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(capabilities, "capabilities must not be null");

        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>();
        requireWhenPresent(model.tensors().stream().anyMatch(tensor -> tensor.type().isPresent()),
                AmirFormatCapabilities.Feature.TYPED_TENSOR_METADATA,
                capabilities, diagnostics);
        requireWhenPresent(model.graphCollection().isPresent(),
                AmirFormatCapabilities.Feature.GRAPH_COLLECTION,
                capabilities, diagnostics);
        requireWhenPresent(model.assetCollection().isPresent(),
                AmirFormatCapabilities.Feature.ASSET_COLLECTION,
                capabilities, diagnostics);
        requireWhenPresent(model.componentCollection().isPresent(),
                AmirFormatCapabilities.Feature.COMPONENT_COLLECTION,
                capabilities, diagnostics);
        requireWhenPresent(!model.extensions().isEmpty(),
                AmirFormatCapabilities.Feature.OPAQUE_MODEL_EXTENSIONS,
                capabilities, diagnostics);
        requireWhenPresent(model.tensors().stream().anyMatch(tensor -> !tensor.extensions().isEmpty()),
                AmirFormatCapabilities.Feature.OPAQUE_TENSOR_EXTENSIONS,
                capabilities, diagnostics);

        if (capabilities.requiresKnownArchitecture()
                && "unknown".equalsIgnoreCase(model.architecture())) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "UNKNOWN_ARCHITECTURE",
                    "Target format '" + capabilities.format()
                            + "' requires a recognized architecture for full-fidelity conversion"));
        }

        CompatibilityReport.ConversionFidelity fidelity = diagnostics.isEmpty()
                ? CompatibilityReport.ConversionFidelity.FULL
                : CompatibilityReport.ConversionFidelity.PARTIAL;
        return new CompatibilityReport(fidelity, List.copyOf(diagnostics));
    }

    private static void requireWhenPresent(
            boolean present,
            AmirFormatCapabilities.Feature feature,
            AmirFormatCapabilities capabilities,
            List<CompatibilityReport.Diagnostic> diagnostics) {
        if (present && !capabilities.preserves(feature)) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "UNTRANSFERRED_" + feature.name(),
                    "Target format '" + capabilities.format()
                            + "' cannot preserve AMIR feature " + feature.name()));
        }
    }
}
