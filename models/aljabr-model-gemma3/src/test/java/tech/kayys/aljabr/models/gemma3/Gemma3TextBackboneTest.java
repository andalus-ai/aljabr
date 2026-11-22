package tech.kayys.aljabr.models.gemma3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.cache.GrowingHeapKVCacheStorage;
import tech.kayys.aljabr.cache.KVCacheLayout;
import tech.kayys.aljabr.spi.model.ModelConfig;

class Gemma3TextBackboneTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void executesCausalTextForwardAndReturnsOneLogitVectorPerToken() throws Exception {
        ModelConfig config = config();
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        putPatternMatrix(tensors, architecture.embedTokensWeight(), 5, 4);
        putVector(tensors, architecture.finalNormWeight(), 4, 0.0f);
        putMatrix(tensors, architecture.layerQueryWeight(0), 4, 4, 0.02f);
        putMatrix(tensors, architecture.layerKeyWeight(0), 2, 4, 0.015f);
        putMatrix(tensors, architecture.layerValueWeight(0), 2, 4, 0.03f);
        putMatrix(tensors, architecture.layerOutputWeight(0), 4, 4, 0.025f);
        putVector(tensors, architecture.layerAttentionNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostAttnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPreFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerQueryNormWeight(0), 2, 0.0f);
        putVector(tensors, architecture.layerKeyNormWeight(0), 2, 0.0f);
        putMatrix(tensors, architecture.layerFfnGateWeight(0), 6, 4, 0.01f);
        putMatrix(tensors, architecture.layerFfnUpWeight(0), 6, 4, 0.012f);
        putMatrix(tensors, architecture.layerFfnDownWeight(0), 4, 6, 0.008f);

        Gemma3TextCheckpointMapper.Weights mapped =
                Gemma3TextCheckpointMapper.map(config, architecture, tensors);
        Gemma3TextBackbone backbone = new Gemma3TextBackbone(config, mapped);

        float[][] logits = backbone.forward(new int[] { 0, 1, 2 });

        assertEquals(3, logits.length);
        assertEquals(config.vocabSize(), logits[0].length);
        assertEquals(1, backbone.layerCount());
        assertEquals(config.hiddenSize(), backbone.hiddenSize());
        for (float[] tokenLogits : logits) {
            for (float logit : tokenLogits) assertTrue(Float.isFinite(logit));
        }
        // Appending future tokens must not change logits at earlier causal positions.
        assertArrayEquals(backbone.forward(new int[] { 0 })[0], logits[0], 1.0e-6f);
    }


    @Test
    void incrementalDecodeMatchesFullCausalForwardAndKeepsSessionStateIsolated() throws Exception {
        ModelConfig config = config();
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        putPatternMatrix(tensors, architecture.embedTokensWeight(), 5, 4);
        putVector(tensors, architecture.finalNormWeight(), 4, 0.0f);
        putMatrix(tensors, architecture.layerQueryWeight(0), 4, 4, 0.02f);
        putMatrix(tensors, architecture.layerKeyWeight(0), 2, 4, 0.015f);
        putMatrix(tensors, architecture.layerValueWeight(0), 2, 4, 0.03f);
        putMatrix(tensors, architecture.layerOutputWeight(0), 4, 4, 0.025f);
        putVector(tensors, architecture.layerAttentionNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostAttnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPreFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerQueryNormWeight(0), 2, 0.0f);
        putVector(tensors, architecture.layerKeyNormWeight(0), 2, 0.0f);
        putMatrix(tensors, architecture.layerFfnGateWeight(0), 6, 4, 0.01f);
        putMatrix(tensors, architecture.layerFfnUpWeight(0), 6, 4, 0.012f);
        putMatrix(tensors, architecture.layerFfnDownWeight(0), 4, 6, 0.008f);
        AtomicReference<KVCacheLayout> requestedLayout = new AtomicReference<>();
        Gemma3TextBackbone backbone = new Gemma3TextBackbone(
                config,
                Gemma3TextCheckpointMapper.map(config, architecture, tensors),
                layout -> {
                    requestedLayout.set(layout);
                    return new GrowingHeapKVCacheStorage(layout);
                });

        int[] prompt = {0, 1, 2};
        float[][] expected = backbone.forward(prompt);
        try (Gemma3TextBackbone.DecodeSession session = backbone.newDecodeSession()) {
            assertNotNull(requestedLayout.get());
            assertEquals(config.maxPositionEmbeddings(), requestedLayout.get().maxSequenceLength());
            assertEquals(1, requestedLayout.get().layerCount());
            float[][] prefill = session.prefill(new int[] {0, 1});
            assertEquals(2, session.position());
            assertArrayEquals(expected[0], prefill[0], 1.0e-5f);
            assertArrayEquals(expected[1], prefill[1], 1.0e-5f);
            assertArrayEquals(expected[2], session.decode(2), 1.0e-5f);
            assertEquals(3, session.position());
        }

        int[] longerPrompt = new int[20];
        for (int index = 0; index < longerPrompt.length; index++) {
            longerPrompt[index] = index % config.vocabSize();
        }
        float[][] longExpected = backbone.forward(longerPrompt);
        try (Gemma3TextBackbone.DecodeSession growing = backbone.newDecodeSession()) {
            float[][] actual = growing.prefill(longerPrompt);
            assertEquals(longerPrompt.length, growing.position());
            for (int index = 0; index < longExpected.length; index++) {
                assertArrayEquals(longExpected[index], actual[index], 1.0e-5f);
            }
        }

        try (Gemma3TextBackbone.DecodeSession first = backbone.newDecodeSession();
                Gemma3TextBackbone.DecodeSession second = backbone.newDecodeSession()) {
            first.decode(0);
            assertEquals(1, first.position());
            assertEquals(0, second.position());
            assertArrayEquals(backbone.forward(new int[] {1})[0], second.decode(1), 1.0e-5f);
        }
    }

    @Test
    void prefillValidatesTheEntirePromptBeforeMutatingDecodeState() throws Exception {
        Gemma3TextBackbone backbone = new Gemma3TextBackbone(config(), emptyMappedWeights());
        try (Gemma3TextBackbone.DecodeSession session = backbone.newDecodeSession()) {
            int[] oversized = new int[33];
            java.util.Arrays.fill(oversized, 0);
            assertThrows(IllegalArgumentException.class, () -> session.prefill(oversized));
            assertEquals(0, session.position());

            assertThrows(IllegalArgumentException.class, () -> session.prefill(new int[] {0, 1, 5}));
            assertEquals(0, session.position());

            float[][] logits = session.prefill(new int[] {0, 1});
            assertEquals(2, logits.length);
            assertEquals(2, session.position());
        }
    }

    @Test
    void rejectsEmptyAndOutOfRangeTokenSequences() throws Exception {
        Gemma3TextBackbone backbone = new Gemma3TextBackbone(config(), emptyMappedWeights());
        assertThrows(IllegalArgumentException.class, () -> backbone.forward(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> backbone.forward(new int[] { 5 }));
    }

    private Gemma3TextCheckpointMapper.Weights emptyMappedWeights() throws Exception {
        ModelConfig config = config();
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        putMatrix(tensors, architecture.embedTokensWeight(), 5, 4, 0.0f);
        putVector(tensors, architecture.finalNormWeight(), 4, 0.0f);
        putMatrix(tensors, architecture.layerQueryWeight(0), 4, 4, 0.0f);
        putMatrix(tensors, architecture.layerKeyWeight(0), 2, 4, 0.0f);
        putMatrix(tensors, architecture.layerValueWeight(0), 2, 4, 0.0f);
        putMatrix(tensors, architecture.layerOutputWeight(0), 4, 4, 0.0f);
        putVector(tensors, architecture.layerAttentionNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostAttnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPreFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerPostFfnNormWeight(0), 4, 0.0f);
        putVector(tensors, architecture.layerQueryNormWeight(0), 2, 0.0f);
        putVector(tensors, architecture.layerKeyNormWeight(0), 2, 0.0f);
        putMatrix(tensors, architecture.layerFfnGateWeight(0), 6, 4, 0.0f);
        putMatrix(tensors, architecture.layerFfnUpWeight(0), 6, 4, 0.0f);
        putMatrix(tensors, architecture.layerFfnDownWeight(0), 4, 6, 0.0f);
        return Gemma3TextCheckpointMapper.map(config, architecture, tensors);
    }

    private ModelConfig config() throws Exception {
        Path path = temporaryDirectory.resolve("gemma3-config.json");
        Files.writeString(path, """
                {
                  "model_type": "gemma3_text",
                  "architectures": ["Gemma3ForCausalLM"],
                  "hidden_size": 4,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 2,
                  "intermediate_size": 6,
                  "vocab_size": 5,
                  "max_position_embeddings": 32,
                  "tie_word_embeddings": true,
                  "layer_types": ["full_attention"]
                }
                """);
        return ModelConfig.load(path, new ObjectMapper());
    }

    private static void putMatrix(
            Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors,
            String name, int rows, int columns, float value) {
        tensors.put(name, new Gemma3TextCheckpointMapper.CheckpointTensor(
                new int[] { rows, columns }, filled(rows * columns, value)));
    }

    private static void putPatternMatrix(
            Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors,
            String name, int rows, int columns) {
        float[] values = new float[rows * columns];
        for (int index = 0; index < values.length; index++) {
            values[index] = (index + 1) * 0.025f;
        }
        tensors.put(name, new Gemma3TextCheckpointMapper.CheckpointTensor(
                new int[] { rows, columns }, values));
    }

    private static void putVector(
            Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors,
            String name, int length, float value) {
        tensors.put(name, new Gemma3TextCheckpointMapper.CheckpointTensor(
                new int[] { length }, filled(length, value)));
    }

    private static float[] filled(int length, float value) {
        float[] values = new float[length];
        java.util.Arrays.fill(values, value);
        return values;
    }
}
