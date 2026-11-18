package tech.kayys.aljabr.spi.embedding;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class EmbeddingRequestTest {

    @Test
    void preservesLegacyFourArgumentConstruction() {
        var request = new EmbeddingRequest("request-1", "model-1", List.of("hello"), Map.of());

        assertEquals(EmbeddingTask.UNSPECIFIED, request.task());
        assertNull(request.outputDimension());
    }

    @Test
    void buildsTypedRetrievalRequestAndOutputDimension() {
        var request = EmbeddingRequest.builder()
                .requestId("request-2")
                .model("embedding-model")
                .inputs(List.of("query", "document"))
                .task(EmbeddingTask.RETRIEVAL_QUERY)
                .outputDimension(256)
                .build();

        assertEquals(EmbeddingTask.RETRIEVAL_QUERY, request.task());
        assertEquals(256, request.outputDimension());
        assertEquals(List.of("query", "document"), request.inputs());
    }

    @Test
    void rejectsNonPositiveOutputDimension() {
        assertThrows(IllegalArgumentException.class, () -> new EmbeddingRequest(
                "request-3", "model-1", List.of("hello"), Map.of(),
                EmbeddingTask.SEMANTIC_SIMILARITY, 0));
    }

    @Test
    void defaultsNullTaskToUnspecified() {
        var request = new EmbeddingRequest(
                "request-4", "model-1", List.of("hello"), Map.of(), null, null);

        assertEquals(EmbeddingTask.UNSPECIFIED, request.task());
    }
}
