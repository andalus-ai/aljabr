package tech.kayys.aljabr.models.gemma3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tech.kayys.aljabr.spi.model.ModelConfig;

class Gemma3TextCheckpointMapperTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void mapsGemma3TextWeightsIntoCanonicalOutputMajorBundle() throws Exception {
        ModelConfig config = config();
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();

        putMatrix(tensors, architecture.embedTokensWeight(), 5, 4, 1.0f);
        putVector(tensors, architecture.finalNormWeight(), 4, 2.0f);
        putMatrix(tensors, architecture.layerQueryWeight(0), 4, 4, 3.0f);
        putMatrix(tensors, architecture.layerKeyWeight(0), 2, 4, 4.0f);
        putMatrix(tensors, architecture.layerValueWeight(0), 2, 4, 5.0f);
        putMatrix(tensors, architecture.layerOutputWeight(0), 4, 4, 6.0f);
        putVector(tensors, architecture.layerAttentionNormWeight(0), 4, 7.0f);
        putVector(tensors, architecture.layerPostAttnNormWeight(0), 4, 8.0f);
        putVector(tensors, architecture.layerPreFfnNormWeight(0), 4, 9.0f);
        putVector(tensors, architecture.layerPostFfnNormWeight(0), 4, 10.0f);
        putVector(tensors, architecture.layerQueryNormWeight(0), 2, 11.0f);
        putVector(tensors, architecture.layerKeyNormWeight(0), 2, 12.0f);
        putMatrix(tensors, architecture.layerFfnGateWeight(0), 6, 4, 13.0f);
        putMatrix(tensors, architecture.layerFfnUpWeight(0), 6, 4, 14.0f);
        putMatrix(tensors, architecture.layerFfnDownWeight(0), 4, 6, 15.0f);

        Gemma3TextCheckpointMapper.Weights weights =
                Gemma3TextCheckpointMapper.map(config, architecture, tensors);

        assertEquals(5, weights.tokenEmbeddings().length);
        assertEquals(4, weights.tokenEmbeddings()[0].length);
        assertEquals(1, weights.layers().size());
        assertEquals(4, weights.layers().getFirst().query().length);
        assertEquals(4, weights.layers().getFirst().query()[0].length);
        assertEquals(6, weights.layers().getFirst().feedForwardGate().length);
        assertEquals(6, weights.layers().getFirst().feedForwardDown()[0].length);
        assertEquals(1.0f, weights.tokenEmbeddings()[0][0]);
        assertEquals(13.0f, weights.layers().getFirst().feedForwardGate()[0][0]);
        assertEquals(weights.tokenEmbeddings(), weights.languageModelHead());
    }

    @Test
    void rejectsIncorrectTensorShape() throws Exception {
        ModelConfig config = config();
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors = new LinkedHashMap<>();
        putMatrix(tensors, architecture.embedTokensWeight(), 5, 4, 1.0f);
        putVector(tensors, architecture.finalNormWeight(), 4, 1.0f);
        putMatrix(tensors, architecture.layerQueryWeight(0), 3, 4, 1.0f);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> Gemma3TextCheckpointMapper.map(config, architecture, tensors));
        assertEquals(true, error.getMessage().contains(architecture.layerQueryWeight(0)));
    }

    private ModelConfig config() throws Exception {
        Path path = temporaryDirectory.resolve("config.json");
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
                  "tie_word_embeddings": true
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

    private static void putVector(
            Map<String, Gemma3TextCheckpointMapper.CheckpointTensor> tensors,
            String name, int length, float value) {
        tensors.put(name, new Gemma3TextCheckpointMapper.CheckpointTensor(
                new int[] { length }, filled(length, value)));
    }

    private static float[] filled(int length, float value) {
        float[] result = new float[length];
        java.util.Arrays.fill(result, value);
        return result;
    }
}
