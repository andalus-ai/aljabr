package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufFfnNormLayoutTest {
    @Test
    void acceptsRequiredFfnNormWithEmbeddingWidth() {
        assertDoesNotThrow(() -> GgufFfnNormLayout.validateRequiredTensor(
                tensor(768), "blk.0.ffn_norm.weight", 0, 768));
    }

    @Test
    void rejectsMissingRequiredFfnNorm() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateRequiredTensor(
                        null, "blk.0.ffn_norm.weight", 0, 768));
    }

    @Test
    void rejectsRequiredFfnNormWithWrongWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateRequiredTensor(
                        tensor(512), "blk.0.ffn_norm.weight", 0, 768));
    }

    @Test
    void rejectsRequiredFfnNormThatIsNotRankOne() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateRequiredTensor(
                        new GGUFTensorInfo("blk.0.ffn_norm.weight", new long[] {768, 1},
                                0, 0L, 768L * Float.BYTES),
                        "blk.0.ffn_norm.weight", 0, 768));
    }

    @Test
    void permitsOmittedPostFfnNorm() {
        assertDoesNotThrow(() -> GgufFfnNormLayout.validateOptionalTensor(
                null, "blk.0.post_ffw_norm.weight", 0, 768));
    }

    @Test
    void acceptsPresentPostFfnNormWithEmbeddingWidth() {
        assertDoesNotThrow(() -> GgufFfnNormLayout.validateOptionalTensor(
                tensor(768), "blk.0.post_ffw_norm.weight", 0, 768));
    }

    @Test
    void rejectsPresentPostFfnNormWithWrongWidth() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateOptionalTensor(
                        tensor(512), "blk.0.post_ffw_norm.weight", 0, 768));
    }

    @Test
    void rejectsPresentPostFfnNormThatIsNotRankOne() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateOptionalTensor(
                        new GGUFTensorInfo("blk.0.post_ffw_norm.weight", new long[] {768, 1},
                                0, 0L, 768L * Float.BYTES),
                        "blk.0.post_ffw_norm.weight", 0, 768));
    }

    @Test
    void rejectsNonPositiveEmbeddingDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufFfnNormLayout.validateRequiredTensor(
                        tensor(1), "blk.0.ffn_norm.weight", 0, 0));
    }

    private static GGUFTensorInfo tensor(int dimension) {
        return new GGUFTensorInfo("blk.0.ffn_norm.weight",
                new long[] {dimension}, 0, 0L, (long) dimension * Float.BYTES);
    }
}
