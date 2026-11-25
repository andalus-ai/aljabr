package tech.kayys.aljabr.models.bert;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BertInputEmbeddingsTest {
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
    void combinesWordPositionAndDefaultSegmentEmbeddings() {
        CpuBackend backend = new CpuBackend();
        BertInputEmbeddings embeddings = new BertInputEmbeddings(backend, config());
        Tensor ids = TensorFactory.of(new int[] {1, 2, 3, 0, 4, 5}, backend, 2, 3);
        Tensor output = null;
        try {
            output = embeddings.forward(ids);
            assertEquals(3, output.shape().rank());
            assertEquals(2, output.shape().dim(0));
            assertEquals(3, output.shape().dim(1));
            assertEquals(8, output.shape().dim(2));
            assertEquals(backend, output.backend());
            for (float value : output.toFloatArray()) assertTrue(Float.isFinite(value));
        } finally {
            if (output != null) output.release();
            ids.release();
            embeddings.close();
            backend.close();
        }
    }

    @Test
    void acceptsExplicitTokenTypeIdsAndRejectsShapeMismatch() {
        CpuBackend backend = new CpuBackend();
        BertInputEmbeddings embeddings = new BertInputEmbeddings(backend, config());
        Tensor ids = TensorFactory.of(new int[] {1, 2, 3, 4}, backend, 1, 4);
        Tensor types = TensorFactory.of(new int[] {0, 0, 1, 1}, backend, 1, 4);
        Tensor wrong = TensorFactory.of(new int[] {0, 1}, backend, 1, 2);
        Tensor output = null;
        try {
            output = embeddings.forward(ids, types);
            assertEquals(8, output.shape().dim(2));
            assertThrows(IllegalArgumentException.class, () -> embeddings.forward(ids, wrong));
        } finally {
            if (output != null) output.release();
            wrong.release();
            types.release();
            ids.release();
            embeddings.close();
            backend.close();
        }
    }

    @Test
    void rejectsSequencesLongerThanConfiguredPositionTable() {
        CpuBackend backend = new CpuBackend();
        BertInputEmbeddings embeddings = new BertInputEmbeddings(backend, config());
        Tensor ids = TensorFactory.of(new int[17], backend, 1, 17);
        try {
            assertThrows(IllegalArgumentException.class, () -> embeddings.forward(ids));
        } finally {
            ids.release();
            embeddings.close();
            backend.close();
        }
    }
}
