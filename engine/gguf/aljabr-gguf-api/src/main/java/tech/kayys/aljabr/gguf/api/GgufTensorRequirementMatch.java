package tech.kayys.aljabr.gguf.api;

import java.util.List;
import java.util.Objects;

/** Result of validating required tensor names, shapes and GGML types. */
public record GgufTensorRequirementMatch(List<String> diagnostics) {

    public GgufTensorRequirementMatch {
        diagnostics = List.copyOf(Objects.requireNonNull(diagnostics, "diagnostics"));
        diagnostics.forEach(value -> {
            Objects.requireNonNull(value, "diagnostic");
            if (value.isBlank()) {
                throw new IllegalArgumentException("diagnostics must not contain blank values");
            }
        });
    }

    public boolean compatible() {
        return diagnostics.isEmpty();
    }
}
