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
            if (Set.of("IQ3_S", "IQ4_NL", "IQ4_XS", "Q2_K", "Q3_K", "Q4_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0").contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.WARNING,
                        "GGML_QUANTIZATION_DEQUANTIZED",
                        "Tensor '" + tensor.name() + "' will be dequantized to F32; quantized storage "
                                + "and exact quantized values are not preserved."));
                partial = true;
            } else if (!SUPPORTED.contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.ERROR,
                        "SAFETENSORS_DTYPE_UNSUPPORTED",
                        "Tensor '" + tensor.name() + "' uses '" + tensor.dataType()
                                + "'. No verified Safetensors conversion decoder is available."));
            }
        }

        boolean hasFlattenedLegacyMetadata = source.extensions().keySet().stream()
                .anyMatch(key -> key.startsWith("gguf.metadata."));
        if (hasFlattenedLegacyMetadata) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "GGUF_TYPED_METADATA_FLATTENED",
                    "Legacy AMIR descriptors contain rendered GGUF metadata strings without their original types."));
            partial = true;
        }

        if ("unknown".equalsIgnoreCase(source.architecture())) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "ARCHITECTURE_UNSPECIFIED",
                    "Safetensors can store the tensor inventory, but architecture configuration is unavailable."));
            partial = true;
        }

        if (source.graphCollection().isPresent()
                || source.componentCollection().filter(collection ->
                        !AmirModelLayerCodec.hasNoGraphBindings(collection)).isPresent()) {
            diagnostics.add(new CompatibilityReport.Diagnostic(
                    CompatibilityReport.Severity.WARNING,
                    "AMIR_TYPED_LAYER_NOT_REPRESENTABLE",
                    "Safetensors preserves assets and unbound components, but graph collections and "
                            + "components with graph bindings are not yet encoded."));
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
