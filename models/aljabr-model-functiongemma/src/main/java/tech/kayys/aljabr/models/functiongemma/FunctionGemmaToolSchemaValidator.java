package tech.kayys.aljabr.models.functiongemma;

import tech.kayys.aljabr.core.tool.ToolCall;
import tech.kayys.aljabr.core.tool.ToolDeclaration;
import tech.kayys.aljabr.core.tool.ToolSchemaValidator;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Backward-compatible FunctionGemma facade over the model-independent
 * {@link ToolSchemaValidator}.
 *
 * <p>This class validates untrusted model output only. It does not authorize,
 * resolve, dispatch, or invoke tools. New integrations should use the core
 * validator directly.</p>
 */
public final class FunctionGemmaToolSchemaValidator {
    private final ToolSchemaValidator delegate;

    public FunctionGemmaToolSchemaValidator() {
        this(new ToolSchemaValidator());
    }

    public FunctionGemmaToolSchemaValidator(ToolSchemaValidator delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    /** Validates raw arguments against a raw parameter schema (legacy API). */
    public ValidationResult validate(Map<String, ?> arguments, Map<String, ?> toolSchema) {
        return adapt(delegate.validate(arguments, toolSchema));
    }

    /** Validates a legacy FunctionGemma call against a raw parameter schema. */
    public ValidationResult validate(FunctionGemmaToolCallParser.FunctionCall call,
            Map<String, ?> toolSchema) {
        Objects.requireNonNull(call, "call");
        return adapt(delegate.validate(FunctionGemmaToolCallParser.toToolCall(call).arguments(), toolSchema));
    }

    /** Validates a core tool call against a raw parameter schema. */
    public ValidationResult validate(ToolCall call, Map<String, ?> toolSchema) {
        Objects.requireNonNull(call, "call");
        return adapt(delegate.validate(call.arguments(), toolSchema));
    }

    /** Validates a core tool call against a named, model-independent declaration. */
    public ValidationResult validate(ToolCall call, ToolDeclaration declaration) {
        return adapt(delegate.validate(call, declaration));
    }

    private static ValidationResult adapt(ToolSchemaValidator.ValidationResult result) {
        return new ValidationResult(result.errors().stream()
                .map(error -> new ValidationError(error.path(), error.message()))
                .toList());
    }

    /** Immutable compatibility result preserving the original FunctionGemma API. */
    public record ValidationResult(List<ValidationError> errors) {
        public ValidationResult {
            errors = List.copyOf(Objects.requireNonNull(errors, "errors"));
        }
        public boolean valid() { return errors.isEmpty(); }
    }

    /** One path-addressed validation diagnostic. */
    public record ValidationError(String path, String message) {
        public ValidationError {
            Objects.requireNonNull(path, "path");
            Objects.requireNonNull(message, "message");
        }
        @Override public String toString() { return path + ": " + message; }
    }
}
