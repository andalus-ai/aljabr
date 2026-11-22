package tech.kayys.aljabr.gguf.api;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.subscription.Cancellable;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.cache.HeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.execution.ModelExecutionContext;
import tech.kayys.aljabr.diagnostics.ExecutionEvent;
import tech.kayys.aljabr.diagnostics.ExecutionObserver;
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
import java.util.concurrent.atomic.AtomicReference;

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
    void streamingEmitsStartedAndSucceededOnlyAfterSubscription() throws Exception {
        Path model = Files.createTempFile("gguf-stream-lifecycle", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean streamCreated = new AtomicBoolean();
        GgufBackend backend = new TestBackend() {
            @Override
            public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
                streamCreated.set(true);
                return Multi.createFrom().item(
                        StreamingInferenceChunk.finalChunk(request.getRequestId(), 0, "done"));
            }
        };

        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            Multi<StreamingInferenceChunk> stream = engine.streamExecute("model", request());
            assertTrue(events.isEmpty(), "lifecycle must not start while the Multi is only assembled");
            assertFalse(streamCreated.get());

            stream.collect().asList().await().indefinitely();

            assertTrue(streamCreated.get());
            assertEquals(2, events.size());
            assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
            assertEquals(ExecutionEvent.Type.SUCCEEDED, events.get(1).type());
            assertEquals("gguf.stream", events.get(1).context().operation());
            assertEquals("test", events.get(1).context().provider());
            assertTrue(events.get(1).elapsedNanos() >= 0);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void streamingFailureEmitsFailedAndDoesNotCaptureFailureMessage() throws Exception {
        Path model = Files.createTempFile("gguf-stream-failure", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackend backend = new TestBackend() {
            @Override
            public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
                return Multi.createFrom().failure(new IllegalStateException("private failure detail"));
            }
        };

        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            assertThrows(IllegalStateException.class,
                    () -> engine.streamExecute("model", request()).collect().asList().await().indefinitely());

            assertEquals(2, events.size());
            assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
            assertEquals(ExecutionEvent.Type.FAILED, events.get(1).type());
            assertEquals("java.lang.IllegalStateException", events.get(1).failureType());
            assertFalse(events.get(1).context().attributes().values().stream()
                    .anyMatch(value -> value.contains("private failure detail")));
            assertEquals(1, engine.getStats().failedInferences());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void streamCancellationEmitsAbandoned() throws Exception {
        Path model = Files.createTempFile("gguf-stream-cancel", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackend backend = new TestBackend() {
            @Override
            public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
                return Multi.createFrom().emitter(emitter -> { });
            }
        };

        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            Cancellable subscription = engine.streamExecute("model", request())
                    .subscribe().with(item -> { }, failure -> fail(failure));
            subscription.cancel();

            assertEquals(2, events.size());
            assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
            assertEquals(ExecutionEvent.Type.ABANDONED, events.get(1).type());
            assertTrue(events.get(1).elapsedNanos() >= 0);
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
    void forwardsModelExecutionContextToBackend() throws Exception {
        Path model = Files.createTempFile("gguf-execution-context", ".gguf");
        ModelExecutionContext context = ModelExecutionContext.withoutKVCache("model", "request");
        AtomicBoolean executeCalled = new AtomicBoolean();
        AtomicBoolean streamCalled = new AtomicBoolean();
        GgufBackend backend = new TestBackend() {
            @Override
            public InferenceResponse execute(
                    InferenceRequest request, ModelExecutionContext suppliedContext) {
                assertSame(context, suppliedContext);
                executeCalled.set(true);
                return response("context output");
            }

            @Override
            public Multi<StreamingInferenceChunk> stream(
                    InferenceRequest request, ModelExecutionContext suppliedContext) {
                assertSame(context, suppliedContext);
                streamCalled.set(true);
                return Multi.createFrom().item(
                        StreamingInferenceChunk.finalChunk(request.getRequestId(), 0, "context stream"));
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend)) {
            assertEquals("context output", engine.execute("model", request(), context).getContent());
            List<StreamingInferenceChunk> chunks = engine.streamExecute("model", request(), context)
                    .collect().asList().await().indefinitely();
            assertTrue(executeCalled.get());
            assertTrue(streamCalled.get());
            assertEquals("context stream", chunks.get(0).delta());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void defaultContextMethodsRejectUnconsumedExternalCacheStorage() {
        HeapKVCacheStorage storage = new HeapKVCacheStorage(KVCacheLayout.uniform(1, 8, 1, 4));
        ModelExecutionContext context = ModelExecutionContext.withKVCacheStorage("model", "request", storage);
        GgufBackend backend = new TestBackend();

        try {
            UnsupportedOperationException executeFailure = assertThrows(
                    UnsupportedOperationException.class,
                    () -> backend.execute(request(), context));
            assertTrue(executeFailure.getMessage().contains("must override"));

            assertThrows(
                    UnsupportedOperationException.class,
                    () -> backend.stream(request(), context));
        } finally {
            storage.close();
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
        return engine(model, backend, ExecutionObserver.NOOP);
    }

    private static GgufInferenceEngine engine(
            Path model, GgufBackend backend, ExecutionObserver observer) throws Exception {
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
        GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, observer);
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
