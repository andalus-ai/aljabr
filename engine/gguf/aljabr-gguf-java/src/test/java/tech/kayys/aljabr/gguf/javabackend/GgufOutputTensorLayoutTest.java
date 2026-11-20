package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufOutputTensorLayoutTest {
    private static final int EMBEDDING = 128;
    private static final int VOCABULARY = 512;

    @Test
    void acceptsSeparateTokenAndOutputMatrices() {
        assertDoesNotThrow(() -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING),
                matrix("output.weight", EMBEDDING, VOCABULARY)));
    }

    @Test
    void acceptsTiedOutputEmbeddingWhenOutputWeightIsAbsent() {
        assertDoesNotThrow(() -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING), null));
    }

    @Test
    void rejectsTokenEmbeddingWithWrongInputWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING - 1, VOCABULARY),
                vector("output_norm.weight", EMBEDDING), null));
    }

    @Test
    void rejectsTokenEmbeddingWithWrongVocabularyRows() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY - 1),
                vector("output_norm.weight", EMBEDDING), null));
    }

    @Test
    void rejectsNonMatrixTokenEmbedding() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                new GGUFTensorInfo("token_embd.weight", new long[] {EMBEDDING}, 0, 0L, 0L),
                vector("output_norm.weight", EMBEDDING), null));
    }

    @Test
    void rejectsOutputNormWithWrongWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING - 1), null));
    }

    @Test
    void rejectsOutputNormWithNonVectorShape() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                new GGUFTensorInfo("output_norm.weight", new long[] {EMBEDDING, 1}, 0, 0L, 0L), null));
    }

    @Test
    void rejectsUntiedOutputProjectionWithWrongVocabularyRows() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING),
                matrix("output.weight", EMBEDDING, VOCABULARY - 1)));
    }

    @Test
    void rejectsUntiedOutputProjectionWithWrongEmbeddingWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING),
                matrix("output.weight", EMBEDDING - 1, VOCABULARY)));
    }

    @Test
    void rejectsNonPositiveVocabularyOrEmbeddingDimensions() {
        assertThrows(IllegalArgumentException.class, () -> GgufOutputTensorLayout.validate(
                matrix("token_embd.weight", EMBEDDING, VOCABULARY),
                vector("output_norm.weight", EMBEDDING), null, EMBEDDING, 0));
    }

    private static void validate(GGUFTensorInfo token, GGUFTensorInfo norm, GGUFTensorInfo output) {
        GgufOutputTensorLayout.validate(token, norm, output, EMBEDDING, VOCABULARY);
    }

    private static GGUFTensorInfo matrix(String name, int columns, int rows) {
        return new GGUFTensorInfo(name, new long[] {columns, rows}, 0, 0L,
                (long) columns * rows * Float.BYTES);
    }

    private static GGUFTensorInfo vector(String name, int dimension) {
        return new GGUFTensorInfo(name, new long[] {dimension}, 0, 0L,
                (long) dimension * Float.BYTES);
    }
}
