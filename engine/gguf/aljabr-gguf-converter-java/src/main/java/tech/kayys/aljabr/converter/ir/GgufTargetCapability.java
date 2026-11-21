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

/**
 * Metadata-only compatibility assessment for writing AMIR tensors as GGUF.
 *
 * <p>This provider checks representability, not whether a particular runtime
 * implements the model architecture or every kernel for its tensor types.
 */
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
        List<CompatibilityReport.Diagnostic> diagnostics = new ArrayList<>();
        boolean partial = false;

        if ("unknown".equalsIgnoreCase(source.architecture())) {
            diagnostics.add(warning("ARCHITECTURE_UNSPECIFIED",
                    "GGUF can store tensors, but model architecture metadata is unspecified."));
            partial = true;
        }

        for (TensorDescriptor tensor : source.tensors()) {
            String type = tensor.dataType().toUpperCase(Locale.ROOT);
            if (!SUPPORTED.contains(type)) {
                diagnostics.add(error("GGUF_DTYPE_UNSUPPORTED",
                        "Tensor '" + tensor.name() + "' uses dtype '" + tensor.dataType()
                                + "', which has no supported GGML storage type."));
                continue;
            }

            try {
                GgmlType ggmlType = GgmlType.fromLabel(type);
                long elements = elementCount(tensor);
                if (elements % ggmlType.blockSize != 0) {
                    diagnostics.add(error("GGUF_QUANTIZATION_BLOCK_MISMATCH",
                            "Tensor '" + tensor.name() + "' has " + elements
                                    + " elements, not divisible by GGML block size "
                                    + ggmlType.blockSize + " for " + ggmlType.label + "."));
                }
            } catch (IllegalArgumentException exception) {
                diagnostics.add(error("GGUF_SHAPE_INVALID",
                        "Tensor '" + tensor.name() + "' cannot be represented: " + exception.getMessage()));
            }
        }

        if (!source.extensions().isEmpty()) {
            diagnostics.add(info("OPAQUE_METADATA_REQUIRES_MAPPING",
                    "Source extensions must be explicitly mapped to GGUF metadata by the conversion executor."));
            partial = true;
        }

        boolean unsupported = diagnostics.stream().anyMatch(d -> d.severity() == CompatibilityReport.Severity.ERROR);
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

    private static CompatibilityReport.Diagnostic error(String code, String message) {
        return new CompatibilityReport.Diagnostic(CompatibilityReport.Severity.ERROR, code, message);
    }

    private static CompatibilityReport.Diagnostic warning(String code, String message) {
        return new CompatibilityReport.Diagnostic(CompatibilityReport.Severity.WARNING, code, message);
    }

    private static CompatibilityReport.Diagnostic info(String code, String message) {
        return new CompatibilityReport.Diagnostic(CompatibilityReport.Severity.INFO, code, message);
    }
}
