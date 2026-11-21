package tech.kayys.aljabr.spi.model.ir;

import java.net.URI;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Format-neutral request to execute a preflighted model conversion.
 *
 * @param output destination URI owned by the target-format executor
 * @param acknowledgePartial explicit consent to execute a PARTIAL plan
 * @param options immutable target-specific options; keys must be namespaced
 */
public record ConversionRequest(
        URI output,
        boolean acknowledgePartial,
        Map<String, String> options) {

    public ConversionRequest {
        Objects.requireNonNull(output, "output must not be null");
        if (!output.isAbsolute()) {
            throw new IllegalArgumentException("output URI must be absolute");
        }
        options = Collections.unmodifiableMap(
                new LinkedHashMap<>(Objects.requireNonNull(options, "options must not be null")));
        options.forEach((key, value) -> {
            if (key == null || key.isBlank() || !key.contains(".") || value == null) {
                throw new IllegalArgumentException(
                        "options must contain namespaced keys and non-null values");
            }
        });
    }

    public static ConversionRequest to(URI output) {
        return new ConversionRequest(output, false, Map.of());
    }
}
