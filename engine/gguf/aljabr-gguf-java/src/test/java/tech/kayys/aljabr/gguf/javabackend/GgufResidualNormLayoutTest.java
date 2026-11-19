package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufResidualNormLayoutTest {
    @Test
    void acceptsRequiredAttentionNormAtEmbeddingWidth() {
        assertDoesNotThrow(() -> GgufResidualNormLayout.validateRequiredTensor(
                tensor(768), "blk.0.attn_norm.weight", 0, 768));
    }

    @Test
    void rejectsMissingRequiredAttentionNorm() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufResidualNormLayout.validateRequiredTensor(
                        null, "blk.0.attn_norm.weight", 0, 768));
    }

    @Test
    void rejectsAttentionNormWithWrongWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufResidualNormLayout.validateRequiredTensor(
                        tensor(512), "blk.0.attn_norm.weight", 0, 768));
    }

    @Test
    void rejectsAttentionNormWithNonVectorShape() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufResidualNormLayout.validateRequiredTensor(
                        new GGUFTensorInfo("blk.0.attn_norm.weight", new long[] {768, 1},
                                0, 0L, 768L * Float.BYTES),
                        "blk.0.attn_norm.weight", 0, 768));
    }

    @Test
    void permitsOmittedPostAttentionNorm() {
        assertDoesNotThrow(() -> GgufResidualNormLayout.validateOptionalTensor(
                null, "blk.0.post_attention_norm.weight", 0, 768));
    }

    @Test
    void validatesPresentPostAttentionNorm() {
        assertDoesNotThrow(() -> GgufResidualNormLayout.validateOptionalTensor(
                tensor(768), "blk.0.post_attention_norm.weight", 0, 768));
    }

    @Test
    void rejectsMalformedPostAttentionNorm() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufResidualNormLayout.validateOptionalTensor(
                        tensor(512), "blk.0.post_attention_norm.weight", 0, 768));
    }

    @Test
    void permitsOmittedPerLayerPostNorm() {
        assertDoesNotThrow(() -> GgufResidualNormLayout.validateOptionalTensor(
                null, "blk.0.post_norm.weight", 0, 768));
    }

    @Test
    void validatesPresentPerLayerPostNorm() {
        assertDoesNotThrow(() -> GgufResidualNormLayout.validateOptionalTensor(
                tensor(768), "blk.0.post_norm.weight", 0, 768));
    }

    @Test
    void rejectsNonPositiveEmbeddingDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufResidualNormLayout.validateRequiredTensor(
                        tensor(1), "blk.0.attn_norm.weight", 0, 0));
    }

    private static GGUFTensorInfo tensor(int dimension) {
        return new GGUFTensorInfo("blk.0.norm.weight",
                new long[] {dimension}, 0, 0L, (long) dimension * Float.BYTES);
    }
}
