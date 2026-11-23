package tech.kayys.aljabr.spi.inference;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.Message;

import static org.junit.jupiter.api.Assertions.*;

class InferenceRequestToolCallProtocolTest {

    @Test
    void protocolIsOptionalByDefault() {
        var request = InferenceRequest.builder()
                .model("example-model")
                .message(Message.user("hello"))
                .build();

        assertTrue(request.getToolCallProtocol().isEmpty());
    }

    @Test
    void protocolSurvivesBuildAndToBuilderCopy() {
        var protocol = ToolCallProtocolDescriptor.of("functiongemma-native");
        var request = InferenceRequest.builder()
                .model("example-model")
                .message(Message.user("hello"))
                .toolCallProtocol(protocol)
                .build();

        assertEquals(protocol, request.getToolCallProtocol().orElseThrow());
        assertEquals(protocol, request.toBuilder().build().getToolCallProtocol().orElseThrow());
    }

    @Test
    void protocolParticipatesInRequestEquality() {
        var base = InferenceRequest.builder()
                .requestId("same-request")
                .model("example-model")
                .message(Message.user("hello"))
                .build();
        var configured = base.toBuilder()
                .toolCallProtocol(ToolCallProtocolDescriptor.of("functiongemma-native"))
                .build();

        assertNotEquals(base, configured);
    }
}
