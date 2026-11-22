package tech.kayys.aljabr.core.tool;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Immutable registry of tool-call parsers keyed by an application-defined
 * protocol descriptor.
 *
 * <p>Applications decide which descriptor applies to a model or response
 * format. This registry does not inspect model metadata, discover plugins,
 * or execute parsed calls.</p>
 */
public final class ToolCallParserRegistry {

    private final Map<ToolCallProtocolDescriptor, ToolCallParser> parsers;

    private ToolCallParserRegistry(Map<ToolCallProtocolDescriptor, ToolCallParser> parsers) {
        this.parsers = Collections.unmodifiableMap(new LinkedHashMap<>(parsers));
    }

    /** Creates a builder for an explicitly configured registry. */
    public static Builder builder() {
        return new Builder();
    }

    /** Returns the parser registered for the supplied protocol descriptor. */
    public Optional<ToolCallParser> find(ToolCallProtocolDescriptor protocol) {
        return Optional.ofNullable(parsers.get(Objects.requireNonNull(protocol, "protocol")));
    }

    /** Returns the parser registered for a protocol identifier. */
    public Optional<ToolCallParser> find(String protocolId) {
        return find(ToolCallProtocolDescriptor.of(protocolId));
    }

    /**
     * Returns the parser registered for the supplied protocol descriptor.
     *
     * @throws IllegalArgumentException if no parser is registered
     */
    public ToolCallParser require(ToolCallProtocolDescriptor protocol) {
        Objects.requireNonNull(protocol, "protocol");
        ToolCallParser parser = parsers.get(protocol);
        if (parser == null) {
            throw new IllegalArgumentException("No tool-call parser registered for protocol: " + protocol.id());
        }
        return parser;
    }

    /** Returns the parser registered for a protocol identifier. */
    public ToolCallParser require(String protocolId) {
        return require(ToolCallProtocolDescriptor.of(protocolId));
    }

    /** Returns registered protocol descriptors in registration order. */
    public Set<ToolCallProtocolDescriptor> protocols() {
        return parsers.keySet();
    }

    /** Returns registered protocol identifiers in registration order. */
    public Set<String> protocolIds() {
        LinkedHashSet<String> ids = new LinkedHashSet<>();
        parsers.keySet().forEach(protocol -> ids.add(protocol.id()));
        return Collections.unmodifiableSet(ids);
    }

    public int size() {
        return parsers.size();
    }

    public boolean isEmpty() {
        return parsers.isEmpty();
    }

    /** Mutable construction helper; built registries are immutable snapshots. */
    public static final class Builder {
        private final Map<ToolCallProtocolDescriptor, ToolCallParser> parsers = new LinkedHashMap<>();

        private Builder() {
        }

        /** Registers a parser under an application-defined protocol descriptor. */
        public Builder register(ToolCallProtocolDescriptor protocol, ToolCallParser parser) {
            Objects.requireNonNull(protocol, "protocol");
            Objects.requireNonNull(parser, "parser");
            if (parsers.putIfAbsent(protocol, parser) != null) {
                throw new IllegalArgumentException(
                        "A tool-call parser is already registered for protocol: " + protocol.id());
            }
            return this;
        }

        /** Convenience overload for callers that already have a protocol identifier. */
        public Builder register(String protocolId, ToolCallParser parser) {
            return register(ToolCallProtocolDescriptor.of(protocolId), parser);
        }

        /** Builds an immutable snapshot of the current registrations. */
        public ToolCallParserRegistry build() {
            return new ToolCallParserRegistry(parsers);
        }
    }
}
