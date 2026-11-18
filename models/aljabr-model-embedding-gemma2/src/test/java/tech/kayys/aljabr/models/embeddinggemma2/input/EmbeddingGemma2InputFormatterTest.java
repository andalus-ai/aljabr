package tech.kayys.aljabr.models.embeddinggemma2.input;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.embedding.EmbeddingTask;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class EmbeddingGemma2InputFormatterTest {
    @Test
    void formatsRetrievalQueryAndDocumentAsymmetrically() {
        assertEquals(
                "task: search result | query: how does GQA work?",
                format("how does GQA work?", null,
                        EmbeddingTask.RETRIEVAL_QUERY, EmbeddingGemma2InputRole.QUERY));
        assertEquals(
                "title: Attention | text: Grouped-query attention shares key/value heads.",
                format("Grouped-query attention shares key/value heads.", "Attention",
                        EmbeddingTask.RETRIEVAL_DOCUMENT, EmbeddingGemma2InputRole.DOCUMENT));
    }

    @Test
    void formatsCodeRetrievalAndFactChecking() {
        assertEquals("task: code retrieval | query: Java async stream",
                format("Java async stream", null,
                        EmbeddingTask.CODE_RETRIEVAL, EmbeddingGemma2InputRole.QUERY));
        assertEquals("title: none | text: public class Main {}",
                format("public class Main {}", null,
                        EmbeddingTask.CODE_RETRIEVAL, EmbeddingGemma2InputRole.DOCUMENT));
        assertEquals("task: fact checking | query: The API is thread safe.",
                format("The API is thread safe.", null,
                        EmbeddingTask.FACT_CHECKING, EmbeddingGemma2InputRole.QUERY));
    }

    @Test
    void formatsSymmetricTasksWithSameInstructionTemplate() {
        assertEquals("task: sentence similarity | query: hello",
                format("hello", null,
                        EmbeddingTask.SEMANTIC_SIMILARITY, EmbeddingGemma2InputRole.SINGLE));
        assertEquals("task: classification | query: hello",
                format("hello", null,
                        EmbeddingTask.CLASSIFICATION, EmbeddingGemma2InputRole.SINGLE));
        assertEquals("task: clustering | query: hello",
                format("hello", null,
                        EmbeddingTask.CLUSTERING, EmbeddingGemma2InputRole.SINGLE));
    }

    @Test
    void leavesUnspecifiedInputUntouched() {
        assertEquals("  raw input  ", format("  raw input  ", null,
                EmbeddingTask.UNSPECIFIED, EmbeddingGemma2InputRole.SINGLE));
    }

    @Test
    void rejectsRoleMismatchesInsteadOfSilentlyChangingTaskSemantics() {
        assertThrows(IllegalArgumentException.class, () -> format("x", null,
                EmbeddingTask.RETRIEVAL_QUERY, EmbeddingGemma2InputRole.DOCUMENT));
        assertThrows(IllegalArgumentException.class, () -> format("x", null,
                EmbeddingTask.CLASSIFICATION, EmbeddingGemma2InputRole.QUERY));
    }

    private static String format(
            String input, String title, EmbeddingTask task, EmbeddingGemma2InputRole role) {
        return EmbeddingGemma2InputFormatter.format(input, title, task, role);
    }
}
