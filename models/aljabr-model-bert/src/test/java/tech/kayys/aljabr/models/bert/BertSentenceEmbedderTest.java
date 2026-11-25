package tech.kayys.aljabr.models.bert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class BertSentenceEmbedderTest {
    private static BertConfig config() {
        return BertConfig.from(Map.of(
                "vocab_size", 32, "hidden_size", 8, "num_hidden_layers", 2,
                "num_attention_heads", 2, "intermediate_size", 16,
                "max_position_embeddings", 16, "type_vocab_size", 2,
                "pad_token_id", 0, "layer_norm_eps", 1.0e-12));
    }

    @Test
    void meanPoolsMaskedTokensAndReturnsUnitLengthVectors() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        BertSentenceEmbedder embedder = new BertSentenceEmbedder(encoder);
        Tensor ids = TensorFactory.of(new int[] {1, 2, 0, 3, 4, 5}, backend, 2, 3);
        Tensor mask = TensorFactory.of(new int[] {1, 1, 0, 1, 1, 1}, backend, 2, 3);
        try {
            float[][] vectors = embedder.embed(ids, mask);
            assertEquals(2, vectors.length);
            for (float[] vector : vectors) {
                assertEquals(8, vector.length);
                double norm = 0;
                for (float value : vector) {
                    assertTrue(Float.isFinite(value));
                    norm += (double) value * value;
                }
                assertEquals(1.0, Math.sqrt(norm), 1.0e-5);
            }
        } finally {
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }

    @Test
    void supportsTruncateThenNormalizeAndRejectsInvalidDimension() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        BertSentenceEmbedder embedder = new BertSentenceEmbedder(encoder);
        Tensor ids = TensorFactory.of(new int[] {1, 2}, backend, 1, 2);
        Tensor mask = TensorFactory.of(new int[] {1, 1}, backend, 1, 2);
        try {
            float[][] vectors = embedder.embed(ids, mask, null, 4);
            assertEquals(4, vectors[0].length);
            assertThrows(IllegalArgumentException.class, () -> embedder.embed(ids, mask, null, 9));
        } finally {
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }

    @Test
    void rejectsAnAllPaddingSequence() {
        CpuBackend backend = new CpuBackend();
        BertEncoder encoder = new BertEncoder(backend, config());
        BertSentenceEmbedder embedder = new BertSentenceEmbedder(encoder);
        Tensor ids = TensorFactory.of(new int[] {0, 0}, backend, 1, 2);
        Tensor mask = TensorFactory.of(new int[] {0, 0}, backend, 1, 2);
        try {
            assertThrows(IllegalArgumentException.class, () -> embedder.embed(ids, mask));
        } finally {
            mask.release();
            ids.release();
            encoder.close();
            backend.close();
        }
    }
}
