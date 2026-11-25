package tech.kayys.aljabr.models.bert;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.backend.cpu.CpuBackend;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.TensorFactory;

class MiniLmL6V2Test {
    @Test
    void exposesOfficialModelProfile() {
        BertConfig config = MiniLmL6V2.config();
        assertEquals(30_522, config.vocabularySize());
        assertEquals(384, config.hiddenSize());
        assertEquals(6, config.layerCount());
        assertEquals(12, config.attentionHeadCount());
        assertEquals(1_536, config.intermediateSize());
        assertEquals(512, config.maxPositionEmbeddings());
        assertEquals(256, MiniLmL6V2.MAX_SEQUENCE_LENGTH);
    }

    @Test
    void acceptsOnlyTheMatchingBertArchitecture() {
        MiniLmL6V2.validateCompatible(MiniLmL6V2.config());
        BertConfig incompatible = BertConfig.from(Map.of(
                "vocab_size", 30_522, "hidden_size", 768, "num_hidden_layers", 6,
                "num_attention_heads", 12, "intermediate_size", 3_072));
        assertThrows(IllegalArgumentException.class,
                () -> MiniLmL6V2.validateCompatible(incompatible));
    }

    @Test
    void embedsToTheExpectedDimensionAndEnforcesSentenceTransformerLimit() {
        CpuBackend backend = new CpuBackend();
        MiniLmL6V2 model = MiniLmL6V2.create(backend);
        Tensor ids = TensorFactory.of(new int[] {101, 2023, 102}, backend, 1, 3);
        Tensor mask = TensorFactory.of(new int[] {1, 1, 1}, backend, 1, 3);
        try {
            assertEquals(384, model.embed(ids, mask)[0].length);
            Tensor tooLong = TensorFactory.zeros(backend, 1, 257);
            Tensor tooLongMask = TensorFactory.zeros(backend, 1, 257);
            try {
                assertThrows(IllegalArgumentException.class, () -> model.embed(tooLong, tooLongMask));
            } finally {
                tooLongMask.release();
                tooLong.release();
            }
        } finally {
            mask.release();
            ids.release();
            model.encoder().close();
            backend.close();
        }
    }
}
