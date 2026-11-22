package tech.kayys.aljabr.core.tool;

import java.util.Objects;

/**
 * Model-independent identifier for the wire protocol used to encode tool calls.
 *
 * <p>This value describes a parsing protocol, not a model family. Applications
 * associate it with model configuration or response-format metadata and use
 * it to select a {@link ToolCallParser}. Core intentionally does not infer
 * protocol identifiers from model names.</p>
 *
 * @param id stable, application-recognized protocol identifier
 */
public record ToolCallProtocolDescriptor(String id) {

    public ToolCallProtocolDescriptor {
        Objects.requireNonNull(id, "id");
        if (id.isBlank()) {
            throw new IllegalArgumentException("id must not be blank");
        }
    }

    /** Creates a descriptor for an application-defined protocol identifier. */
    public static ToolCallProtocolDescriptor of(String id) {
        return new ToolCallProtocolDescriptor(id);
    }
}
