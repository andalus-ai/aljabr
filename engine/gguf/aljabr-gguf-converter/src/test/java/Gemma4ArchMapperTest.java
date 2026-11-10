import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.converter.gguf.GemmaArchMapper;
import tech.kayys.aljabr.converter.gguf.HfConfigParser;
import tech.kayys.aljabr.gguf.core.GgufMetaValue;
import tech.kayys.aljabr.gguf.core.GgufModel;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class Gemma4ArchMapperTest {

    @Test
    void emitsGemma4MetadataForHeterogeneousAttention() {
        var raw = JsonParser.parseString("""
                {
                  "architectures": ["Gemma4ForConditionalGeneration"],
                  "model_type": "gemma4",
                  "text_config": {
                    "hidden_size": 1536,
                    "intermediate_size": 6144,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "num_kv_shared_layers": 2,
                    "max_position_embeddings": 131072,
                    "vocab_size": 262144,
                    "rms_norm_eps": 1e-6,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "hidden_size_per_layer_input": 256,
                    "use_double_wide_mlp": true,
                    "sliding_window": 512,
                    "final_logit_softcapping": 30.0,
                    "layer_types": [
                      "sliding_attention",
                      "sliding_attention",
                      "full_attention",
                      "sliding_attention",
                      "full_attention"
                    ],
                    "rope_parameters": {
                      "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0
                      },
                      "sliding_attention": {
                        "rope_theta": 10000.0
                      }
                    }
                  }
                }
                """).getAsJsonObject();
        var cfg = new HfConfigParser.ModelConfig(
                "gemma4",
                1536,
                6144,
                5,
                8,
                1,
                131072,
                262144,
                1e-6f,
                10000f,
                "gelu_pytorch_tanh",
                false,
                512,
                false,
                1.0f,
                256,
                raw);
        var tokenizer = new HfConfigParser.TokenizerData(
                List.of("<pad>", "<eos>", "where"),
                List.of(0f, 0f, 0f),
                List.of(1, 3, 1),
                "<bos>",
                "<eos>",
                2,
                1,
                List.of("w h"),
                "gpt2");

        GgufModel model = new GgufModel();
        GemmaArchMapper.applyConfig(model, cfg, tokenizer, "test");

        assertThat(model.architecture()).isEqualTo("gemma4");
        assertThat(model.getMeta("gemma4.attention.key_length").orElseThrow().asUInt32()).isEqualTo(512);
        assertThat(model.getMeta("gemma4.attention.key_length_swa").orElseThrow().asUInt32()).isEqualTo(256);
        assertThat(model.getMeta("gemma4.embedding_length_per_layer_input").orElseThrow().asUInt32()).isEqualTo(256);
        assertThat(model.getMeta("gemma4.rope.freq_base").orElseThrow().asFloat32()).isEqualTo(1000000.0f);
        assertThat(model.getMeta("gemma4.rope.freq_base_swa").orElseThrow().asFloat32()).isEqualTo(10000.0f);
        assertThat(uintArray(model, "gemma4.feed_forward_length")).containsExactly(6144L, 6144L, 6144L, 12288L, 12288L);
        assertThat(boolArray(model, "gemma4.attention.sliding_window_pattern")).containsExactly(true, true, false, true, false);
        assertThat(model.getMeta("tokenizer.ggml.model").orElseThrow().asString()).isEqualTo("gemma4");
        assertThat(model.getMeta("tokenizer.ggml.add_bos_token").orElseThrow().asBool()).isTrue();
    }

    @Test
    void mapsGemma4PerLayerAndNormTensors() {
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.embed_tokens_per_layer.weight", 35))
                .isEqualTo("per_layer_token_embd.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.per_layer_model_projection.weight", 35))
                .isEqualTo("per_layer_model_proj.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.layers.4.self_attn.q_proj.weight", 35))
                .isEqualTo("blk.4.attn_q.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.layers.4.post_attention_layernorm.weight", 35))
                .isEqualTo("blk.4.post_attention_norm.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.layers.4.pre_feedforward_layernorm.weight", 35))
                .isEqualTo("blk.4.ffn_norm.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.layers.4.layer_scalar", 35))
                .isEqualTo("blk.4.layer_output_scale.weight");
        assertThat(GemmaArchMapper.mapTensorName("model.language_model.layers.4.per_layer_input_gate.weight", 35))
                .isEqualTo("blk.4.inp_gate.weight");
    }

    private static List<Long> uintArray(GgufModel model, String key) {
        return model.getMeta(key).orElseThrow().asArray().stream()
                .map(GgufMetaValue::asUInt32)
                .toList();
    }

    private static List<Boolean> boolArray(GgufModel model, String key) {
        return model.getMeta(key).orElseThrow().asArray().stream()
                .map(GgufMetaValue::asBool)
                .toList();
    }
}
