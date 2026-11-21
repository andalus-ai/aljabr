package tech.kayys.aljabr.models.embeddinggemma2;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import tech.kayys.aljabr.models.embeddinggemma2.input.EmbeddingGemma2InputRole;
import tech.kayys.aljabr.spi.embedding.EmbeddingModelRunner;
import tech.kayys.aljabr.spi.embedding.EmbeddingRequest;
import tech.kayys.aljabr.spi.embedding.EmbeddingResponse;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;

/**
 * Adapts an already-loaded EmbeddingGemma 2 text model to Aljabr's embedding
 * request contract. Model loading and lifecycle remain the responsibility of
 * the bundle loader and inference host.
 */
public final class EmbeddingGemma2TextModelRunner implements EmbeddingModelRunner {
    public static final String MODEL_FAMILY = "embedding_gemma2";

    private final EmbeddingGemma2TextModel model;

    public EmbeddingGemma2TextModelRunner(EmbeddingGemma2TextModel model) {
        this.model = Objects.requireNonNull(model, "model must not be null");
    }

    @Override
    public String modelFamily() {
        return MODEL_FAMILY;
    }

    @Override
    public CompletionStage<EmbeddingResponse> embed(EmbeddingRequest request) {
        Objects.requireNonNull(request, "request must not be null");
        try {
            List<String> inputs = request.inputs();
            List<String> titles = titles(request.parameters(), inputs.size());
            List<float[]> vectors = new ArrayList<>(inputs.size());
            int dimension = request.outputDimension() == null
                    ? model.nativeDimension() : request.outputDimension();
            EmbeddingGemma2InputRole role = roleFor(request.task());

            for (int index = 0; index < inputs.size(); index++) {
                vectors.add(model.embed(inputs.get(index), titles.get(index),
                        request.task(), role, request.outputDimension()));
            }

            return CompletableFuture.completedFuture(new EmbeddingResponse(
                    request.requestId(), request.model(), List.copyOf(vectors), dimension,
                    Map.of("model_family", MODEL_FAMILY, "task", request.task().name())));
        } catch (RuntimeException failure) {
            return CompletableFuture.failedFuture(failure);
        }
    }

    private static List<String> titles(Map<String, Object> parameters, int inputCount) {
        Object supplied = parameters.get("titles");
        if (supplied == null) {
            Object title = parameters.get("title");
            if (title != null && !(title instanceof String)) {
                throw new IllegalArgumentException("embedding parameter 'title' must be a string");
            }
            return java.util.Collections.nCopies(inputCount, (String) title);
        }
        if (!(supplied instanceof List<?> values) || values.size() != inputCount) {
            throw new IllegalArgumentException(
                    "embedding parameter 'titles' must be a list matching the number of inputs");
        }
        List<String> result = new ArrayList<>(inputCount);
        for (Object value : values) {
            if (value != null && !(value instanceof String)) {
                throw new IllegalArgumentException("every entry in embedding parameter 'titles' must be a string or null");
            }
            result.add((String) value);
        }
        return result;
    }

    static EmbeddingGemma2InputRole roleFor(EmbeddingTask task) {
        return switch (Objects.requireNonNull(task, "task must not be null")) {
            case RETRIEVAL_QUERY, QUESTION_ANSWERING -> EmbeddingGemma2InputRole.QUERY;
            case RETRIEVAL_DOCUMENT, CODE_RETRIEVAL, FACT_CHECKING -> EmbeddingGemma2InputRole.DOCUMENT;
            case UNSPECIFIED, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING ->
                    EmbeddingGemma2InputRole.SINGLE;
        };
    }
}
