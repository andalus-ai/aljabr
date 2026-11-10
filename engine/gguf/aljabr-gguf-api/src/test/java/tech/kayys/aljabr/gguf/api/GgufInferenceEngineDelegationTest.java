package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.embedding.EmbeddingRequest;
import tech.kayys.aljabr.spi.embedding.EmbeddingResponse;
import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.inference.InferenceResponse;
import tech.kayys.aljabr.spi.inference.StreamingInferenceChunk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class GgufInferenceEngineDelegationTest {

    @Test
    void delegatesStreamingToBackend() throws Exception {
        Path model = Files.createTempFile("gguf-stream", ".gguf");
        AtomicBoolean streamCalled = new AtomicBoolean();
        GgufBackend backend = new TestBackend() {
            @Override
            public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
                streamCalled.set(true);
                return Multi.createFrom().items(
                        StreamingInferenceChunk.textDelta(request.getRequestId(), 0, "hello"),
                        StreamingInferenceChunk.finalChunk(request.getRequestId(), 1, " world"));
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend)) {
            List<StreamingInferenceChunk> chunks = engine.streamExecute("model", request())
                    .collect().asList().await().indefinitely();
            assertTrue(streamCalled.get());
            assertEquals(2, chunks.size());
            assertEquals("hello", chunks.get(0).delta());
            assertTrue(chunks.get(1).finished());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void delegatesEmbeddingToBackend() throws Exception {
        Path model = Files.createTempFile("gguf-embedding", ".gguf");
        AtomicBoolean embedCalled = new AtomicBoolean();
        EmbeddingResponse expected = new EmbeddingResponse(
                "request", "model", List.of(new float[]{1f, 2f}), 2, Map.of());
        GgufBackend backend = new TestBackend() {
            @Override
            public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
                embedCalled.set(true);
                return Uni.createFrom().item(expected);
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend)) {
            EmbeddingResponse actual = engine.executeEmbedding("model",
                    EmbeddingRequest.builder().requestId("request").model("model").input("text").build())
                    .await().indefinitely();
            assertTrue(embedCalled.get());
            assertSame(expected, actual);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void defaultBackendStreamingAdaptsSynchronousResponse() throws Exception {
        Path model = Files.createTempFile("gguf-default-stream", ".gguf");
        GgufBackend backend = new TestBackend() {
            @Override
            public InferenceResponse execute(InferenceRequest request) {
                return response("sync output");
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend)) {
            List<StreamingInferenceChunk> chunks = engine.streamExecute("model", request())
                    .collect().asList().await().indefinitely();
            assertEquals(1, chunks.size());
            assertEquals("sync output", chunks.get(0).delta());
            assertTrue(chunks.get(0).finished());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static GgufInferenceEngine engine(Path model, GgufBackend backend) throws Exception {
        GgufBackendProvider provider = new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("test", Set.of(), Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                return backend;
            }
        };
        GgufInferenceEngine engine = new GgufInferenceEngine(List.of(provider));
        engine.loadModel("model", model, Map.of("gguf.backend", "test"));
        return engine;
    }

    private static InferenceRequest request() {
        return InferenceRequest.builder().requestId("request").model("model").prompt("hello").build();
    }

    private static InferenceResponse response(String content) {
        return new InferenceResponse("request", content, "model", 0, 1, 1, 1, null, Map.of(), null,
                InferenceResponse.FinishReason.STOP, null);
    }

    private static class TestBackend implements GgufBackend {
        @Override public String name() { return "test"; }
        @Override public InferenceResponse execute(InferenceRequest request) { return response("sync"); }
        @Override public void close() {}
    }
}
