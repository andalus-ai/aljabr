package tech.kayys.aljabr.models.bert;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BertEncoderTest {
    private static BertConfig config() {
        return BertConfig.from(Map.of(
                "vocab_size", 32,
                "hidden_size", 8,
                "num_hidden_layers", 2,
                "num_attention_heads", 2,
                "intermediate_size", 16,
                "max_position_embeddings", 16,
                "type_vocab_size", 2,
                "pad_token_id", 0,
                "layer_norm_eps", 1.0e-12));
    }

    @Test
    void encodesBatchWithPaddingMask() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        Tensor ids = TensorFactory.of(new int[] {1, 2, 0, 3, 4, 5}, backend, 2, 3);
        Tensor mask = TensorFactory.of(new int[] {1, 1, 0, 1, 1, 1}, backend, 2, 3);
        Tensor output = null;
        try {
            output = encoder.forward(ids, mask, null);
            assertEquals(3, output.shape().rank());
            assertEquals(2, output.shape().dim(0));
            assertEquals(3, output.shape().dim(1));
            assertEquals(8, output.shape().dim(2));
            for (float value : output.toFloatArray()) assertTrue(Float.isFinite(value));
        } finally {
            if (output != null) output.release();
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }

    @Test
    void rejectsNonBinaryAttentionMask() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        Tensor ids = TensorFactory.of(new int[] {1, 2}, backend, 1, 2);
        Tensor mask = TensorFactory.of(new int[] {1, 2}, backend, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> encoder.forward(ids, mask, null));
        } finally {
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }

    @Test
    void rejectsMismatchedAttentionMaskShape() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        Tensor ids = TensorFactory.of(new int[] {1, 2, 3}, backend, 1, 3);
        Tensor mask = TensorFactory.of(new int[] {1, 1}, backend, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> encoder.forward(ids, mask, null));
        } finally {
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }
}
