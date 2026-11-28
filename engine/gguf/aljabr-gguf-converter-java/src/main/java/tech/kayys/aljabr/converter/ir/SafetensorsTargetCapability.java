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

    private static final Set<String> GGML_QUANTIZED = Set.of(
            "MXFP4", "NVFP4", "TQ1_0", "TQ2_0", "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS",
            "IQ4_NL", "IQ4_XS", "Q2_0", "Q2_K", "Q3_K", "Q4_K",
            "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0");

    @Override
    public String targetFormat() {
        return "safetensors";
    }

    @Override
    public CompatibilityReport assess(ModelDescriptor source) {
        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>(
                new AmirFidelityEvaluator().evaluate(source, AmirFormatCapabilities.safetensors()).diagnostics());

        for (TensorDescriptor tensor : source.tensors()) {
            String type = tensor.dataType().toUpperCase(Locale.ROOT);
            String path = "/tensors/" + pointerSegment(tensor.name()) + "/dataType";
            if (GGML_QUANTIZED.contains(type)) {
            if (Set.of("MXFP4", "NVFP4", "TQ1_0", "TQ2_0", "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "Q2_0", "Q2_K", "Q3_K", "Q4_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0").contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.WARNING,
                        "GGML_QUANTIZATION_DEQUANTIZED",
                        "Tensor '" + tensor.name() + "' will be dequantized to F32; quantized storage "
                                + "and exact quantized values are not preserved.",
                        path));
            } else if (!SUPPORTED.contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.ERROR,
                        "SAFETENSORS_DTYPE_UNSUPPORTED",
                        "Tensor '" + tensor.name() + "' uses '" + tensor.dataType()
                                + "'. No verified Safetensors conversion decoder is available.",
                        path));
            }
        }

        return report(diagnostics);
    }

    private static String pointerSegment(String value) {
        return value.replace("~", "~0").replace("/", "~1");
    }

    private static CompatibilityReport report(List<CompatibilityReport.Diagnostic> diagnostics) {
        boolean unsupported = diagnostics.stream().anyMatch(
                diagnostic -> diagnostic.severity() == CompatibilityReport.Severity.ERROR);
        boolean partial = diagnostics.stream().anyMatch(
                diagnostic -> diagnostic.severity() == CompatibilityReport.Severity.WARNING);
        return new CompatibilityReport(
                unsupported ? CompatibilityReport.ConversionFidelity.UNSUPPORTED
                        : partial ? CompatibilityReport.ConversionFidelity.PARTIAL
                        : CompatibilityReport.ConversionFidelity.FULL,
                diagnostics);
    }
}
