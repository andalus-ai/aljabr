package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallParserRegistryTest {

    private static final ToolCallParser NO_CALLS =
            output -> new ToolCallParseResult(List.of(), List.of());

    @Test
    void resolvesParserByApplicationDefinedProtocolId() {
        var registry = ToolCallParserRegistry.builder()
                .register("functiongemma", NO_CALLS)
                .build();

        assertSame(NO_CALLS, registry.find("functiongemma").orElseThrow());
        assertSame(NO_CALLS, registry.require("functiongemma"));
        assertEquals(1, registry.size());
        assertEquals(List.of("functiongemma"), List.copyOf(registry.protocolIds()));
    }

    @Test
    void builtRegistryIsSnapshotAndCannotBeMutatedThroughItsIds() {
        var builder = ToolCallParserRegistry.builder().register("protocol-a", NO_CALLS);
        var registry = builder.build();

        builder.register("protocol-b", output ->
                new ToolCallParseResult(List.of(new ToolCall("later", Map.of())), List.of()));

        assertEquals(1, registry.size());
        assertTrue(registry.find("protocol-b").isEmpty());
        assertThrows(UnsupportedOperationException.class,
                () -> registry.protocolIds().add("protocol-c"));
    }

    @Test
    void rejectsDuplicateAndInvalidRegistrations() {
        var builder = ToolCallParserRegistry.builder().register("protocol-a", NO_CALLS);

        assertThrows(IllegalArgumentException.class,
                () -> builder.register("protocol-a", NO_CALLS));
        assertThrows(IllegalArgumentException.class,
                () -> builder.register(" ", NO_CALLS));
        assertThrows(NullPointerException.class,
                () -> builder.register("protocol-b", null));
    }

    @Test
    void distinguishesUnknownProtocolFromEmptyRegistry() {
        var registry = ToolCallParserRegistry.builder().build();

        assertTrue(registry.isEmpty());
        assertTrue(registry.find("unknown").isEmpty());
        assertThrows(IllegalArgumentException.class, () -> registry.require("unknown"));
        assertThrows(NullPointerException.class, () -> registry.find(null));
    }
}
