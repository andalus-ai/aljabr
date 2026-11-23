package tech.kayys.aljabr.spi.inference;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ToolCallProtocolDescriptorTest {

    @Test
    void descriptorIsValueBased() {
        var protocol = ToolCallProtocolDescriptor.of("vendor-neutral-call-format");

        assertEquals(new ToolCallProtocolDescriptor("vendor-neutral-call-format"), protocol);
        assertEquals("vendor-neutral-call-format", protocol.id());
    }

    @Test
    void rejectsNullAndBlankIdentifiers() {
        assertThrows(NullPointerException.class, () -> new ToolCallProtocolDescriptor(null));
        assertThrows(IllegalArgumentException.class, () -> ToolCallProtocolDescriptor.of(" "));
    }
}
