package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.TargetFormatCapability;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Metadata-only compatibility assessment for writing AMIR tensors as
 * Safetensors. Quantized GGML block encodings are not silently treated as
 * ordinary scalar dtypes.
 */
public final class SafetensorsTargetCapability implements TargetFormatCapability {

    private static final Set<String> SUPPORTED = Set.of(
            "BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2",
            "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "BOOL");

    @Override
    public String targetFormat() {
        return "safetensors";
    }

    @Override
    public CompatibilityReport assess(ModelDescriptor source) {
        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>();
        boolean partial = false;

        for (TensorDescriptor tensor : source.tensors()) {
            String type = tensor.dataType().toUpperCase(Locale.ROOT);
            if (!SUPPORTED.contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.ERROR,
                        "SAFETENSORS_DTYPE_UNSUPPORTED",
                        "Tensor '" + tensor.name() + "' uses '" + tensor.dataType()
                                + "'. Safetensors requires a supported scalar dtype; quantized block data "
                                + "must be dequantized or handled by a dedicated extension."));
            }
        }

        if ("unknown".equalsIgnoreCase(source.architecture())) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "ARCHITECTURE_UNSPECIFIED",
                    "Safetensors can store the tensor inventory, but architecture configuration is unavailable."));
            partial = true;
        }

        if (!source.extensions().isEmpty()) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.INFO,
                    "OPAQUE_METADATA_REQUIRES_MAPPING",
                    "Opaque source extensions are not part of the Safetensors tensor format and require sidecar preservation."));
            partial = true;
        }

        boolean unsupported = diagnostics.stream().anyMatch(d -> d.severity() == CompatibilityReport.Severity.ERROR);
        return new CompatibilityReport(
                unsupported ? CompatibilityReport.ConversionFidelity.UNSUPPORTED
                        : partial ? CompatibilityReport.ConversionFidelity.PARTIAL
                        : CompatibilityReport.ConversionFidelity.FULL,
                diagnostics);
    }
}
