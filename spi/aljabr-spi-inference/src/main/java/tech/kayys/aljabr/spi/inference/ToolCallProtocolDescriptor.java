package tech.kayys.aljabr.spi.inference;

import java.util.Objects;

/**
 * Stable identifier for the wire protocol used to encode tool calls.
 *
 * <p>The descriptor belongs to the inference SPI so request and response
 * contracts can carry protocol selection without depending on runtime/core
 * implementation details. It describes a protocol, not a model family.
 * Applications explicitly associate it with model or response configuration;
 * implementations must not infer it from a model name.</p>
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
