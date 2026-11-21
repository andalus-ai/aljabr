package tech.kayys.aljabr.spi.model.ir;

import java.net.URI;
import java.util.List;
import java.util.Objects;

/**
 * Format-neutral summary of a completed model conversion.
 *
 * <p>The result is returned only after the target executor has committed its
 * output. Metadata that could not be represented is listed explicitly.
 */
public record ConversionExecutionResult(
        String sourceFormat,
        String targetFormat,
        URI output,
        int tensorCount,
        long payloadBytes,
        CompatibilityReport compatibility,
        List<String> untransferredMetadataKeys) {

    public ConversionExecutionResult {
        sourceFormat = requireText(sourceFormat, "sourceFormat");
        targetFormat = requireText(targetFormat, "targetFormat");
        Objects.requireNonNull(output, "output must not be null");
        if (tensorCount < 0) {
            throw new IllegalArgumentException("tensorCount must be non-negative");
        }
        if (payloadBytes < 0L) {
            throw new IllegalArgumentException("payloadBytes must be non-negative");
        }
        Objects.requireNonNull(compatibility, "compatibility must not be null");
        untransferredMetadataKeys = List.copyOf(
                Objects.requireNonNull(untransferredMetadataKeys, "untransferredMetadataKeys must not be null"));
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
