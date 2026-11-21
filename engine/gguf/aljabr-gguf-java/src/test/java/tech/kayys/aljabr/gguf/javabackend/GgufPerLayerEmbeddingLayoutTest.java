package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufPerLayerEmbeddingLayoutTest {
    private static final int EMBEDDING = 128;
    private static final int VOCABULARY = 512;
    private static final int LAYERS = 4;
    private static final int PER_LAYER = 16;
    private static final int PER_TOKEN = LAYERS * PER_LAYER;

    @Test
    void acceptsCompletePerLayerEmbeddingTensorGroup() {
        assertDoesNotThrow(() -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN),
                vector("per_layer_proj_norm.weight", PER_LAYER)));
    }

    @Test
    void permitsOptionalProjectionNormToBeAbsent() {
        assertDoesNotThrow(() -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN), null));
    }

    @Test
    void permitsAllPerLayerTensorsToBeAbsent() {
        assertDoesNotThrow(() -> validate(null, null, null));
    }

    @Test
    void rejectsTokenEmbeddingWithoutModelProjection() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY), null, null));
    }

    @Test
    void rejectsModelProjectionWithoutTokenEmbedding() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null, matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN), null));
    }

    @Test
    void rejectsProjectionNormWithoutPerLayerEmbeddingGroup() {
        assertThrows(IllegalArgumentException.class,
                () -> validate(null, null, vector("per_layer_proj_norm.weight", PER_LAYER)));
    }

    @Test
    void rejectsTokenEmbeddingWithWrongPerTokenWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN - 1, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN), null));
    }

    @Test
    void rejectsTokenEmbeddingWithWrongVocabularyRows() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY - 1),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN), null));
    }

    @Test
    void rejectsModelProjectionWithWrongEmbeddingInputWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING - 1, PER_TOKEN), null));
    }

    @Test
    void rejectsModelProjectionWithWrongPerTokenOutputWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN - 1), null));
    }

    @Test
    void rejectsProjectionNormWithWrongPerLayerWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN),
                vector("per_layer_proj_norm.weight", PER_LAYER - 1)));
    }

    @Test
    void rejectsNonVectorProjectionNorm() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN),
                new GGUFTensorInfo("per_layer_proj_norm.weight",
                        new long[] {PER_LAYER, 1}, 0, 0L, 0L)));
    }

    @Test
    void rejectsNonPositiveGeometryWhenTensorGroupIsPresent() {
        assertThrows(IllegalArgumentException.class, () -> GgufPerLayerEmbeddingLayout.validate(
                matrix("per_layer_token_embd.weight", PER_TOKEN, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, PER_TOKEN),
                null, EMBEDDING, VOCABULARY, LAYERS, 0));
    }

    @Test
    void rejectsPerTokenWidthOverflow() {
        assertThrows(IllegalArgumentException.class, () -> GgufPerLayerEmbeddingLayout.validate(
                matrix("per_layer_token_embd.weight", 1, VOCABULARY),
                matrix("per_layer_model_proj.weight", EMBEDDING, 1),
                null, EMBEDDING, VOCABULARY, Integer.MAX_VALUE, 2));
    }

    private static void validate(GGUFTensorInfo token, GGUFTensorInfo projection, GGUFTensorInfo norm) {
        GgufPerLayerEmbeddingLayout.validate(
                token, projection, norm, EMBEDDING, VOCABULARY, LAYERS, PER_LAYER);
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
