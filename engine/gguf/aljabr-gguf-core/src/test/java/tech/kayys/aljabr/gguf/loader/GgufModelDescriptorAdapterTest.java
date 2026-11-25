package tech.kayys.aljabr.gguf.loader;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.api.GgufModelDescriptor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class GgufModelDescriptorAdapterTest {

    @Test
    void adaptsParsedModelMetadataAndTensorInventory() {
        GGUFModel model = new GGUFModel(
                3,
                Map.of(
                        "general.architecture", "gemma4",
                        "tokenizer.ggml.model", "sentencepiece",
                        "tokenizer.chat_template", "{{ messages }}"),
                Map.of(),
                List.of(
                        new GGUFTensorInfo("token_embd.weight", new long[]{8, 8}, 0, 0, 256),
                        new GGUFTensorInfo("blk.0.attn_q.weight", new long[]{8, 8}, 0, 256, 256)),
                512,
                java.lang.foreign.MemorySegment.NULL,
                null);

        GgufModelDescriptor descriptor = GgufModelDescriptorAdapter.inspect(model);

        assertEquals("gemma4", descriptor.architecture());
        assertTrue(descriptor.tokenizerMetadataPresent());
        assertTrue(descriptor.chatTemplatePresent());
        assertTrue(descriptor.hasTensorPrefix("blk."));
        assertEquals(2, descriptor.tensorNames().size());
        assertEquals(List.of(8L, 8L),
                descriptor.tensorDescriptors().get("token_embd.weight").dimensions());
        assertEquals(0, descriptor.tensorDescriptors().get("token_embd.weight").ggmlTypeId());
    }

    @Test
    void rejectsNullModel() {
        assertThrows(NullPointerException.class,
                () -> GgufModelDescriptorAdapter.inspect(null));
    }
}
