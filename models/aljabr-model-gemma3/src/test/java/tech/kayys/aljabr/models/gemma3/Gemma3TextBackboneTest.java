package tech.kayys.aljabr.models.gemma3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
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
                  "max_position_embeddings": 16,
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
