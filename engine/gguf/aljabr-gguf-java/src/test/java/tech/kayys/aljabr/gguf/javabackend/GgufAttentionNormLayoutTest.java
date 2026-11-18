package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufAttentionNormLayoutTest {
    @Test
    void acceptsMatchingQkvNormVectors() {
        assertDoesNotThrow(() -> {
            GgufAttentionNormLayout.validateTensor(tensor("attn_q_norm", 64), "blk.0.attn_q_norm.weight", 0, 64);
            GgufAttentionNormLayout.validateTensor(tensor("attn_k_norm", 64), "blk.0.attn_k_norm.weight", 0, 64);
            GgufAttentionNormLayout.validateTensor(tensor("attn_v_norm", 64), "blk.0.attn_v_norm.weight", 0, 64);
        });
    }

    @Test
    void permitsArchitectureSpecificOmissionOfEveryNormVector() {
        assertDoesNotThrow(() -> {
            GgufAttentionNormLayout.validateTensor(null, "blk.0.attn_q_norm.weight", 0, 64);
            GgufAttentionNormLayout.validateTensor(null, "blk.0.attn_k_norm.weight", 0, 64);
            GgufAttentionNormLayout.validateTensor(null, "blk.0.attn_v_norm.weight", 0, 64);
        });
    }

    @Test
    void rejectsQNormWithWrongDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionNormLayout.validateTensor(
                        tensor("attn_q_norm", 32), "blk.0.attn_q_norm.weight", 0, 64));
    }

    @Test
    void rejectsKNormWithWrongDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionNormLayout.validateTensor(
                        tensor("attn_k_norm", 128), "blk.1.attn_k_norm.weight", 1, 64));
    }

    @Test
    void rejectsVNormWithWrongDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionNormLayout.validateTensor(
                        tensor("attn_v_norm", 63), "blk.2.attn_v_norm.weight", 2, 64));
    }

    @Test
    void validatesHeterogeneousLayerDimensionsIndependently() {
        assertDoesNotThrow(() -> {
            GgufAttentionNormLayout.validateTensor(
                    tensor("attn_q_norm", 32), "blk.0.attn_q_norm.weight", 0, 32);
            GgufAttentionNormLayout.validateTensor(
                    tensor("attn_q_norm", 64), "blk.1.attn_q_norm.weight", 1, 64);
        });
    }

    @Test
    void rejectsNonVectorNormTensor() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionNormLayout.validateTensor(
                        new GGUFTensorInfo("blk.0.attn_q_norm.weight", new long[] {64, 1},
                                0, 0L, 64L * Float.BYTES),
                        "blk.0.attn_q_norm.weight", 0, 64));
    }

    @Test
    void rejectsNonPositiveExpectedHeadDimension() {
        assertThrows(IllegalArgumentException.class,
                () -> GgufAttentionNormLayout.validateTensor(
                        null, "blk.0.attn_q_norm.weight", 0, 0));
    }

    private static GGUFTensorInfo tensor(String name, int dimension) {
        return new GGUFTensorInfo("blk.0." + name + ".weight",
                new long[] {dimension}, 0, 0L, (long) dimension * Float.BYTES);
    }
}
