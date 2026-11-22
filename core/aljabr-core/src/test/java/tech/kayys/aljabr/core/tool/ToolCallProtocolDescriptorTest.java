package tech.kayys.aljabr.core.tool;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallProtocolDescriptorTest {

    @Test
    void descriptorIsValueBasedAndCanBeUsedForRegistrySelection() {
        ToolCallParser parser = output -> new ToolCallParseResult(java.util.List.of(), java.util.List.of());
        var protocol = ToolCallProtocolDescriptor.of("vendor-neutral-call-format");
        var registry = ToolCallParserRegistry.builder()
                .register(protocol, parser)
                .build();

        assertEquals(new ToolCallProtocolDescriptor("vendor-neutral-call-format"), protocol);
        assertSame(parser, registry.find(ToolCallProtocolDescriptor.of("vendor-neutral-call-format")).orElseThrow());
        assertSame(parser, registry.require(protocol));
        assertEquals(java.util.List.of(protocol), java.util.List.copyOf(registry.protocols()));
    }

    @Test
    void rejectsNullAndBlankIdentifiers() {
        assertThrows(NullPointerException.class, () -> new ToolCallProtocolDescriptor(null));
        assertThrows(IllegalArgumentException.class, () -> ToolCallProtocolDescriptor.of(" "));
    }
}
