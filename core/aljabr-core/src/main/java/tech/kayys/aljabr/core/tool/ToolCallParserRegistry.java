package tech.kayys.aljabr.core.tool;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable registry of tool-call parsers keyed by an application-defined
 * protocol identifier.
 *
 * <p>The identifier is deliberately opaque to core. Applications decide
 * which parser key applies to a model or response format; this registry does
 * not inspect model metadata, discover plugins, or execute parsed calls.</p>
 */
public final class ToolCallParserRegistry {

    private final Map<String, ToolCallParser> parsers;

    private ToolCallParserRegistry(Map<String, ToolCallParser> parsers) {
        this.parsers = Collections.unmodifiableMap(new LinkedHashMap<>(parsers));
    }

    /** Creates a builder for an explicitly configured registry. */
    public static Builder builder() {
        return new Builder();
    }

    /** Returns the parser registered for the supplied protocol identifier. */
    public Optional<ToolCallParser> find(String protocolId) {
        return Optional.ofNullable(parsers.get(requireProtocolId(protocolId)));
    }

    /**
     * Returns the parser registered for the supplied protocol identifier.
     *
     * @throws IllegalArgumentException if no parser is registered
     */
    public ToolCallParser require(String protocolId) {
        String id = requireProtocolId(protocolId);
        ToolCallParser parser = parsers.get(id);
        if (parser == null) {
            throw new IllegalArgumentException("No tool-call parser registered for protocol: " + id);
        }
        return parser;
    }

    /** Returns registered protocol identifiers in registration order. */
    public Set<String> protocolIds() {
        return parsers.keySet();
    }

    public int size() {
        return parsers.size();
    }

    public boolean isEmpty() {
        return parsers.isEmpty();
    }

    private static String requireProtocolId(String protocolId) {
        Objects.requireNonNull(protocolId, "protocolId");
        if (protocolId.isBlank()) {
            throw new IllegalArgumentException("protocolId must not be blank");
        }
        return protocolId;
    }

    /** Mutable construction helper; built registries are immutable snapshots. */
    public static final class Builder {
        private final Map<String, ToolCallParser> parsers = new LinkedHashMap<>();

        private Builder() {
        }

        /**
         * Registers a parser under an application-defined protocol identifier.
         *
         * @throws IllegalArgumentException if the identifier is blank or already registered
         */
        public Builder register(String protocolId, ToolCallParser parser) {
            String id = requireProtocolId(protocolId);
            Objects.requireNonNull(parser, "parser");
            if (parsers.putIfAbsent(id, parser) != null) {
                throw new IllegalArgumentException("A tool-call parser is already registered for protocol: " + id);
            }
            return this;
        }

        /** Builds an immutable snapshot of the current registrations. */
        public ToolCallParserRegistry build() {
            return new ToolCallParserRegistry(parsers);
        }
    }
}
