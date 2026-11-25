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

import static org.junit.jupiter.api.Assertions.*;

class GgufInferenceEngineDelegationTest {

    @Test
    void asyncInferenceUsesOneDeferredSynchronousLifecycleScope() throws Exception {
        Path model = Files.createTempFile("gguf-async-infer-lifecycle", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean executeCalled = new AtomicBoolean();
        GgufBackend backend = new TestBackend() {
            @Override
            public InferenceResponse execute(InferenceRequest request) {
                executeCalled.set(true);
                return response("async");
            }
        };

        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            Uni<InferenceResponse> inference = engine.executeAsync("model", request());

            assertTrue(events.isEmpty(), "async inference must remain lazy until subscription");
            assertFalse(executeCalled.get());

            InferenceResponse result = inference.await().indefinitely();

            assertTrue(executeCalled.get());
            assertEquals("async", result.getContent());
            assertEquals(2, events.size(), "the async adapter must not add a duplicate lifecycle scope");
            assertEquals(ExecutionEvent.Type.STARTED, events.get(0).type());
            assertEquals(ExecutionEvent.Type.SUCCEEDED, events.get(1).type());
            assertEquals("gguf.infer", events.get(1).context().operation());
            assertEquals(events.get(0).context().correlationId(), events.get(1).context().correlationId());
        } finally {
            Files.deleteIfExists(model);
        }
    }

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
    void embeddingEmitsStartedAndSucceededOnlyAfterSubscription() throws Exception {
        Path model = Files.createTempFile("gguf-embedding-lifecycle", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean embedCreated = new AtomicBoolean();
        EmbeddingResponse expected = new EmbeddingResponse(
                "request", "model", List.of(new float[]{1f, 2f}), 2, Map.of());
        GgufBackend backend = new TestBackend() {
            @Override
            public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
                embedCreated.set(true);
                return Uni.createFrom().item(expected);
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            Uni<EmbeddingResponse> embedding = engine.executeEmbedding("model", embeddingRequest());
            assertTrue(events.isEmpty(), "lifecycle must not start while the Uni is only assembled");
            assertFalse(embedCreated.get());

            assertSame(expected, embedding.await().indefinitely());

            assertTrue(embedCreated.get());
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.SUCCEEDED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals("gguf.embed", events.get(0).context().operation());
            assertEquals("test", events.get(0).context().provider());
            assertTrue(events.get(1).elapsedNanos() >= 0);
            assertEquals(0, engine.getStats().activeInferences());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void embeddingFailureEmitsFailedWithoutCapturingFailureMessage() throws Exception {
        Path model = Files.createTempFile("gguf-embedding-failure", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackend backend = new TestBackend() {
            @Override
            public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
                return Uni.createFrom().failure(new IllegalStateException("private embedding failure"));
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> engine.executeEmbedding("model", embeddingRequest()).await().indefinitely());
            assertEquals("private embedding failure", failure.getMessage());
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.FAILED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals(IllegalStateException.class.getName(), events.get(1).failureType());
            assertTrue(events.get(1).context().attributes().isEmpty());
            assertEquals(1, engine.getStats().failedInferences());
            assertEquals(0, engine.getStats().activeInferences());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void embeddingCancellationEmitsAbandonedAndReleasesActiveCount() throws Exception {
        Path model = Files.createTempFile("gguf-embedding-cancel", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackend backend = new TestBackend() {
            @Override
            public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
                return Uni.createFrom().emitter(emitter -> { });
            }
        };
        try (GgufInferenceEngine engine = engine(model, backend, events::add)) {
            Cancellable subscription = engine.executeEmbedding("model", embeddingRequest())
                    .subscribe().with(item -> { }, failure -> fail(failure));
            subscription.cancel();

            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.ABANDONED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals(0, engine.getStats().activeInferences());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void embeddingObserverFailuresDoNotChangeResult() throws Exception {
        Path model = Files.createTempFile("gguf-embedding-observer-failure", ".gguf");
        EmbeddingResponse expected = new EmbeddingResponse(
                "request", "model", List.of(new float[]{1f, 2f}), 2, Map.of());
        GgufBackend backend = new TestBackend() {
            @Override
            public Uni<EmbeddingResponse> embed(EmbeddingRequest request) {
                return Uni.createFrom().item(expected);
            }
        };
        ExecutionObserver observer = event -> {
            throw new IllegalStateException("observer failure");
        };
        try (GgufInferenceEngine engine = engine(model, backend, observer)) {
            assertSame(expected, engine.executeEmbedding("model", embeddingRequest()).await().indefinitely());
            assertTrue(engine.executionObserverFailureCount() > 0);
            assertEquals(0, engine.getStats().activeInferences());
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

    @Test
    void providerProbeEmitsIndependentLifecycleEvents() throws Exception {
        Path model = Files.createTempFile("gguf-provider-probe", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackend backend = new TestBackend();
        GgufBackendProvider provider = provider((path, config) -> backend);
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            assertSame(backend, engine.loadModel("model", model, Map.of()));
            assertEquals(List.of(
                            "gguf.provider.probe:STARTED",
                            "gguf.provider.probe:SUCCEEDED",
                            "gguf.model.load:STARTED",
                            "gguf.model.load:SUCCEEDED"),
                    events.stream().map(event -> event.context().operation() + ":" + event.type()).toList());
            assertEquals("test", events.get(0).context().provider());
            assertTrue(events.get(0).context().attributes().isEmpty());
            assertNotNull(events.get(0).context().correlationId());
            assertTrue(events.get(1).elapsedNanos() >= 0);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void providerProbeFailureEmitsFailedWithoutCapturingPathOrMessage() throws Exception {
        Path model = Files.createTempFile("gguf-provider-probe-failure", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackendProvider provider = new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) throws Exception {
                throw new IllegalStateException("private probe detail");
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                return new TestBackend();
            }
        };
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            assertThrows(tech.kayys.aljabr.spi.exception.InferenceException.class,
                    () -> engine.loadModel("model", model, Map.of()));
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.FAILED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals("gguf.provider.probe", events.get(1).context().operation());
            assertEquals(IllegalStateException.class.getName(), events.get(1).failureType());
            assertTrue(events.get(1).context().attributes().isEmpty());
            assertFalse(events.toString().contains(model.toString()));
            assertFalse(events.toString().contains("private probe detail"));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void explicitProviderSelectionBypassesReadinessProbeLifecycle() throws Exception {
        Path model = Files.createTempFile("gguf-provider-explicit", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean probed = new AtomicBoolean();
        GgufBackendProvider provider = new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                probed.set(true);
                throw new AssertionError("explicit selection must not probe");
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                return new TestBackend();
            }
        };
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            engine.loadModel("model", model, Map.of("gguf.backend", "test"));
            assertFalse(probed.get());
            assertEquals(List.of("gguf.model.load:STARTED", "gguf.model.load:SUCCEEDED"),
                    events.stream().map(event -> event.context().operation() + ":" + event.type()).toList());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void failedProviderProbeIsObservedBeforeAutoFallback() throws Exception {
        Path model = Files.createTempFile("gguf-provider-probe-fallback", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        GgufBackendProvider first = new GgufBackendProvider() {
            @Override public String id() { return "first"; }
            @Override public boolean isAvailable() { return true; }
            @Override public int priority() { return 10; }
            @Override public GgufBackendCapability probe(Path path) {
                throw new IllegalStateException("private first-provider probe failure");
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                throw new AssertionError("failed provider must not be created");
            }
        };
        GgufBackendProvider second = new GgufBackendProvider() {
            @Override public String id() { return "second"; }
            @Override public boolean isAvailable() { return true; }
            @Override public int priority() { return 0; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("second", Set.of(), Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                return new TestBackend();
            }
        };
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(first, second), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            engine.loadModel("model", model, Map.of());
            assertEquals(List.of(
                            "gguf.provider.probe:first:STARTED",
                            "gguf.provider.probe:first:FAILED",
                            "gguf.provider.probe:second:STARTED",
                            "gguf.provider.probe:second:SUCCEEDED",
                            "gguf.model.load:second:STARTED",
                            "gguf.model.load:second:SUCCEEDED"),
                    events.stream().map(event -> event.context().operation() + ":"
                            + event.context().provider() + ":" + event.type()).toList());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void duplicateModelIdIsRejectedWithoutReplacingOrClosingRegisteredBackend() throws Exception {
        Path model = Files.createTempFile("gguf-duplicate-model-id", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean closed = new AtomicBoolean();
        AtomicBoolean replacementCreated = new AtomicBoolean();
        GgufBackend original = new TestBackend() {
            @Override public InferenceResponse execute(InferenceRequest request) {
                return response("original");
            }
            @Override public void close() { closed.set(true); }
        };
        GgufBackendProvider provider = provider((path, config) -> {
            if (replacementCreated.getAndSet(true)) {
                throw new AssertionError("duplicate registration must be rejected before provider creation");
            }
            return original;
        });
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            assertSame(original, engine.loadModel("model", model, Map.of("gguf.backend", "test")));
            events.clear();

            assertThrows(IllegalStateException.class,
                    () -> engine.loadModel("model", model, Map.of("gguf.backend", "test")));

            assertFalse(closed.get(), "the registered backend must remain usable");
            assertTrue(events.isEmpty(), "rejected duplicate registration must not start a load lifecycle");
            assertEquals("original", engine.execute("model", request()).getContent());

            engine.unloadModel("model");
            assertTrue(closed.get(), "the original backend must close when explicitly unloaded");
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void modelLoadEmitsLifecycleEventsWithoutModelPathMetadata() throws Exception {
        Path model = Files.createTempFile("gguf-model-load", ".gguf");
        List<ExecutionEvent> events = new java.util.ArrayList<>();
        GgufBackend backend = new TestBackend();
        GgufBackendProvider provider = provider((path, config) -> backend);
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            assertSame(backend, engine.loadModel("model", model, Map.of("gguf.backend", "test")));
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.SUCCEEDED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals("gguf.model.load", events.get(0).context().operation());
            assertEquals("test", events.get(0).context().provider());
            assertNotNull(events.get(0).context().correlationId());
            assertTrue(events.get(0).context().attributes().isEmpty());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void modelLoadFailureEmitsFailedWithoutExceptionMessage() throws Exception {
        Path model = Files.createTempFile("gguf-model-load-failure", ".gguf");
        List<ExecutionEvent> events = new java.util.ArrayList<>();
        GgufBackendProvider provider = provider((path, config) -> {
            throw new IllegalStateException("private model initialization detail");
        });
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> engine.loadModel("model", model, Map.of("gguf.backend", "test")));
            assertEquals("private model initialization detail", failure.getMessage());
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.FAILED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals(IllegalStateException.class.getName(), events.get(1).failureType());
            assertTrue(events.get(1).context().attributes().isEmpty());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void modelUnloadEmitsLifecycleEventsAndClosesBackend() throws Exception {
        Path model = Files.createTempFile("gguf-model-unload", ".gguf");
        List<ExecutionEvent> events = new java.util.ArrayList<>();
        AtomicBoolean closed = new AtomicBoolean();
        GgufBackend backend = new TestBackend() {
            @Override public void close() { closed.set(true); }
        };
        GgufBackendProvider provider = provider((path, config) -> backend);
        try (GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add)) {
            engine.loadModel("model", model, Map.of("gguf.backend", "test"));
            events.clear();
            engine.unloadModel("model");
            assertTrue(closed.get());
            assertEquals(List.of(ExecutionEvent.Type.STARTED, ExecutionEvent.Type.SUCCEEDED),
                    events.stream().map(ExecutionEvent::type).toList());
            assertEquals("gguf.model.unload", events.get(0).context().operation());
            assertEquals("test", events.get(0).context().provider());
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    void closeAttemptsEveryBackendAndPreservesFailuresAndUnloadEvents() throws Exception {
        Path firstModel = Files.createTempFile("gguf-close-first", ".gguf");
        Path secondModel = Files.createTempFile("gguf-close-second", ".gguf");
        Path thirdModel = Files.createTempFile("gguf-close-third", ".gguf");
        List<ExecutionEvent> events = new java.util.concurrent.CopyOnWriteArrayList<>();
        AtomicBoolean firstClosed = new AtomicBoolean();
        AtomicBoolean secondClosed = new AtomicBoolean();
        AtomicBoolean thirdClosed = new AtomicBoolean();
        GgufBackendProvider provider = new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("test", Set.of(), Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                if (path.equals(firstModel)) {
                    return new TestBackend() {
                        @Override public String name() { return "first"; }
                        @Override public void close() {
                            firstClosed.set(true);
                            throw new IllegalStateException("first close failed");
                        }
                    };
                }
                if (path.equals(secondModel)) {
                    return new TestBackend() {
                        @Override public String name() { return "second"; }
                        @Override public void close() {
                            secondClosed.set(true);
                            throw new IllegalArgumentException("second close failed");
                        }
                    };
                }
                return new TestBackend() {
                    @Override public String name() { return "third"; }
                    @Override public void close() { thirdClosed.set(true); }
                };
            }
        };
        GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, events::add);
        try {
            engine.loadModel("first", firstModel, Map.of("gguf.backend", "test"));
            engine.loadModel("second", secondModel, Map.of("gguf.backend", "test"));
            engine.loadModel("third", thirdModel, Map.of("gguf.backend", "test"));
            events.clear();

            RuntimeException failure = assertThrows(RuntimeException.class, engine::close);

            assertTrue(failure instanceof IllegalStateException || failure instanceof IllegalArgumentException);
            assertEquals(1, failure.getSuppressed().length);
            assertTrue(failure.getSuppressed()[0] instanceof IllegalStateException
                    || failure.getSuppressed()[0] instanceof IllegalArgumentException);
            assertTrue(firstClosed.get());
            assertTrue(secondClosed.get(), "a second failing backend must still be attempted");
            assertTrue(thirdClosed.get(), "a failing backend must not prevent later backend cleanup");
            assertEquals(6, events.size(), "all three unload scopes must emit STARTED and a terminal event");
            assertEquals(3, events.stream()
                    .filter(event -> event.context().operation().equals("gguf.model.unload")
                            && event.type() == ExecutionEvent.Type.STARTED).count());
            assertEquals(2, events.stream()
                    .filter(event -> event.context().operation().equals("gguf.model.unload")
                            && event.type() == ExecutionEvent.Type.FAILED).count());
            assertEquals(1, events.stream()
                    .filter(event -> event.context().operation().equals("gguf.model.unload")
                            && event.type() == ExecutionEvent.Type.SUCCEEDED).count());
            assertEquals(0, engine.getStats().activeInferences());
            assertThrows(tech.kayys.aljabr.spi.exception.InferenceException.class,
                    () -> engine.execute("first", request()));
            assertDoesNotThrow(engine::close, "a second close must be harmless after cleanup was attempted");
        } finally {
            Files.deleteIfExists(firstModel);
            Files.deleteIfExists(secondModel);
            Files.deleteIfExists(thirdModel);
        }
    }

    @Test
    void closeWithoutObserverStillClosesRemainingBackendsAfterFailure() throws Exception {
        Path firstModel = Files.createTempFile("gguf-close-noop-first", ".gguf");
        Path secondModel = Files.createTempFile("gguf-close-noop-second", ".gguf");
        AtomicBoolean secondClosed = new AtomicBoolean();
        GgufBackendProvider provider = new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("test", Set.of(), Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) {
                if (path.equals(firstModel)) {
                    return new TestBackend() {
                        @Override public void close() { throw new IllegalArgumentException("close failure"); }
                    };
                }
                return new TestBackend() {
                    @Override public void close() { secondClosed.set(true); }
                };
            }
        };
        GgufInferenceEngine engine = new GgufInferenceEngine(List.of(provider));
        try {
            engine.loadModel("first", firstModel, Map.of("gguf.backend", "test"));
            engine.loadModel("second", secondModel, Map.of("gguf.backend", "test"));

            assertThrows(IllegalArgumentException.class, engine::close);
            assertTrue(secondClosed.get());
            assertDoesNotThrow(engine::close);
        } finally {
            Files.deleteIfExists(firstModel);
            Files.deleteIfExists(secondModel);
        }
    }

    private static GgufBackendProvider provider(BackendFactory factory) {
        return new GgufBackendProvider() {
            @Override public String id() { return "test"; }
            @Override public boolean isAvailable() { return true; }
            @Override public GgufBackendCapability probe(Path path) {
                return GgufBackendCapability.ready("test", Set.of(), Map.of());
            }
            @Override public GgufBackend create(Path path, Map<String, Object> config) throws Exception {
                return factory.create(path, config);
            }
        };
    }

    @FunctionalInterface
    private interface BackendFactory {
        GgufBackend create(Path path, Map<String, Object> config) throws Exception;
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
        // Model setup is outside the operation under test. Keep its load lifecycle
        // from polluting assertions for inference, streaming, embedding, and unload.
        ExecutionObserver operationObserver = event -> {
            if (!"gguf.model.load".equals(event.context().operation())) {
                observer.onEvent(event);
            }
        };
        GgufInferenceEngine engine = new GgufInferenceEngine(
                List.of(provider), tech.kayys.aljabr.diagnostics.ProviderDecisionObserver.NOOP, operationObserver);
        engine.loadModel("model", model, Map.of("gguf.backend", "test"));
        return engine;
    }

    private static EmbeddingRequest embeddingRequest() {
        return EmbeddingRequest.builder().requestId("request").model("model").input("private input").build();
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
