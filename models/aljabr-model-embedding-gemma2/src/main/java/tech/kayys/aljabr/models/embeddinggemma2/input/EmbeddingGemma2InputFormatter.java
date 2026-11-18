package tech.kayys.aljabr.models.embeddinggemma2.input;

import java.util.Objects;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;

/**
 * Applies the task templates used by the EmbeddingGemma 2 text encoder.
 *
 * <p>This is model input preparation, not a serving prompt API. It deliberately
 * keeps query and document roles explicit because asymmetric retrieval tasks
 * use different formatting on each side.</p>
 */
public final class EmbeddingGemma2InputFormatter {
    private EmbeddingGemma2InputFormatter() {
    }

    /**
     * Formats one text input according to its task and role.
     *
     * @param input raw text or code
     * @param title optional document title; ignored for query and symmetric inputs
     * @param task semantic task
     * @param role query, document, or symmetric single-input role
     * @return model-ready text
     */
    public static String format(
            String input,
            String title,
            EmbeddingTask task,
            EmbeddingGemma2InputRole role) {
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(task, "task must not be null");
        Objects.requireNonNull(role, "role must not be null");

        return switch (task) {
            case UNSPECIFIED -> requireSingle(role, input);
            case RETRIEVAL_QUERY -> {
                requireRole(role, EmbeddingGemma2InputRole.QUERY, task);
                yield "task: search result | query: " + input;
            }
            case RETRIEVAL_DOCUMENT -> {
                requireRole(role, EmbeddingGemma2InputRole.DOCUMENT, task);
                yield document(title, input);
            }
            case CODE_RETRIEVAL -> switch (role) {
                case QUERY -> "task: code retrieval | query: " + input;
                case DOCUMENT -> document(title, input);
                case SINGLE -> throw incompatible(task, role);
            };
            case FACT_CHECKING -> switch (role) {
                case QUERY -> "task: fact checking | query: " + input;
                case DOCUMENT -> document(title, input);
                case SINGLE -> throw incompatible(task, role);
            };
            case SEMANTIC_SIMILARITY -> symmetric(
                    role, "task: sentence similarity | query: " + input);
            case CLASSIFICATION -> symmetric(
                    role, "task: classification | query: " + input);
            case CLUSTERING -> symmetric(
                    role, "task: clustering | query: " + input);
        };
    }

    private static String document(String title, String input) {
        String normalizedTitle = title == null || title.isBlank() ? "none" : title;
        return "title: " + normalizedTitle + " | text: " + input;
    }

    private static String symmetric(
            EmbeddingGemma2InputRole role, String formatted) {
        requireRole(role, EmbeddingGemma2InputRole.SINGLE, null);
        return formatted;
    }

    private static String requireSingle(EmbeddingGemma2InputRole role, String input) {
        requireRole(role, EmbeddingGemma2InputRole.SINGLE, EmbeddingTask.UNSPECIFIED);
        return input;
    }

    private static void requireRole(
            EmbeddingGemma2InputRole actual,
            EmbeddingGemma2InputRole expected,
            EmbeddingTask task) {
        if (actual != expected) {
            throw incompatible(task, actual);
        }
    }

    private static IllegalArgumentException incompatible(
            EmbeddingTask task, EmbeddingGemma2InputRole role) {
        return new IllegalArgumentException(
                "Input role " + role + " is not valid for task " + task);
    }
}
