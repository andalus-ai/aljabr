package tech.kayys.aljabr.gguf.api;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class GgufModelDescriptorTest {

    @Test
    void inspectCapturesArtifactFactsWithoutInferringCapabilities() {
        Map<String, Object> metadata = new HashMap<>();
        metadata.put("general.architecture", "  gemma4  ");
        metadata.put("general.name", "Saina test artifact");
        metadata.put("tokenizer.ggml.model", "sentencepiece");
        metadata.put("tokenizer.chat_template", "  {{ messages }}  ");
        metadata.put("general.file_type", 15);

        GgufModelDescriptor descriptor = GgufModelDescriptor.inspect(
                metadata, List.of("token_embd.weight", "blk.0.attn_q.weight"));

        assertEquals("gemma4", descriptor.architecture());
        assertEquals("Saina test artifact", descriptor.modelName());
        assertTrue(descriptor.tokenizerMetadataPresent());
        assertTrue(descriptor.chatTemplatePresent());
        assertTrue(descriptor.hasTensorPrefix("blk."));
        assertFalse(descriptor.hasTensorPrefix("vision."));
        assertEquals(Set.of(
                "general.architecture", "general.name", "general.file_type",
                "tokenizer.ggml.model", "tokenizer.chat_template"), descriptor.metadataKeys());
    }

    @Test
    void unknownOrNonStringArchitectureIsNotGuessed() {
        GgufModelDescriptor descriptor = GgufModelDescriptor.inspect(
                Map.of("general.architecture", 42), List.of());

        assertNull(descriptor.architecture());
        assertNull(descriptor.modelName());
        assertFalse(descriptor.tokenizerMetadataPresent());
        assertFalse(descriptor.chatTemplatePresent());
        assertTrue(descriptor.summary().contains("architecture=unknown"));
    }

    @Test
    void descriptorCollectionsAreImmutableAndDeterministic() {
        GgufModelDescriptor descriptor = GgufModelDescriptor.inspect(
                Map.of("general.architecture", "llama"),
                List.of("z.weight", "a.weight", "z.weight"));

        assertEquals(List.of("a.weight", "z.weight"), List.copyOf(descriptor.tensorNames()));
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.tensorNames().add("new.weight"));
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.metadataKeys().add("new.key"));
    }

    @Test
    void rejectsNullInputsAndNullEntries() {
        assertThrows(NullPointerException.class,
                () -> GgufModelDescriptor.inspect(null, List.of()));
        assertThrows(NullPointerException.class,
                () -> GgufModelDescriptor.inspect(Map.of(), null));
        assertThrows(NullPointerException.class,
                () -> GgufModelDescriptor.inspect(Map.of(), java.util.Arrays.asList("x", null)));
    }
}
