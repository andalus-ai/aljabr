package tech.kayys.aljabr.converter.ir;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.spi.model.ir.CompatibilityReport;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.TargetFormatCapability;
import tech.kayys.aljabr.spi.model.ir.TensorDescriptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Metadata-only preflight for GGUF tensor and AMIR envelope capabilities. */
public final class GgufTargetCapability implements TargetFormatCapability {

    private static final Set<String> SUPPORTED = java.util.Arrays.stream(GgmlType.values())
            .filter(type -> type.blockSize > 0 && type.typeSize > 0)
            .map(type -> type.label.toUpperCase(Locale.ROOT))
            .collect(java.util.stream.Collectors.toUnmodifiableSet());

    @Override
    public String targetFormat() {
        return "gguf";
    }

    @Override
    public CompatibilityReport assess(ModelDescriptor source) {
        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>(
                new AmirFidelityEvaluator().evaluate(source, AmirFormatCapabilities.gguf()).diagnostics());

        for (TensorDescriptor tensor : source.tensors()) {
            String type = tensor.dataType().toUpperCase(Locale.ROOT);
            String path = "/tensors/" + pointerSegment(tensor.name());
            if (!SUPPORTED.contains(type)) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.ERROR,
                        "GGUF_DTYPE_UNSUPPORTED",
                        "Tensor '" + tensor.name() + "' uses dtype '" + tensor.dataType()
                                + "', which has no supported GGML storage type.",
                        path + "/dataType"));
                continue;
            }

            try {
                GgmlType ggmlType = GgmlType.fromLabel(type);
                long elements = elementCount(tensor);
                if (elements % ggmlType.blockSize != 0) {
                    diagnostics.add(new CompatibilityReport.Diagnostic(
                            CompatibilityReport.Severity.ERROR,
                            "GGUF_QUANTIZATION_BLOCK_MISMATCH",
                            "Tensor '" + tensor.name() + "' has " + elements
                                    + " elements, not divisible by GGML block size "
                                    + ggmlType.blockSize + " for " + ggmlType.label + ".",
                            path + "/shape"));
                }
            } catch (IllegalArgumentException | ArithmeticException exception) {
                diagnostics.add(new CompatibilityReport.Diagnostic(
                        CompatibilityReport.Severity.ERROR,
                        "GGUF_SHAPE_INVALID",
                        "Tensor '" + tensor.name() + "' cannot be represented: " + exception.getMessage(),
                        path + ".shape"));
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

    private static long elementCount(TensorDescriptor tensor) {
        long elements = 1;
        for (long dimension : tensor.shape()) {
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }
}
