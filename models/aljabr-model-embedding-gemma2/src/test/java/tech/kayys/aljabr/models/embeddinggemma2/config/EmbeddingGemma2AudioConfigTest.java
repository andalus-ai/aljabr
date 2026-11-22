package tech.kayys.aljabr.models.embeddinggemma2.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EmbeddingGemma2AudioConfigTest {
    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void parsesPublishedEmbeddingGemma2AudioConfiguration() throws Exception {
        EmbeddingGemma2AudioConfig c = EmbeddingGemma2AudioConfig.fromJson(model(), processor());
        assertEquals("gemma4_audio", c.modelType());
        assertEquals(1024, c.hiddenSize());
        assertEquals(12, c.numHiddenLayers());
        assertEquals(8, c.numAttentionHeads());
        assertEquals(1536, c.outputProjectionSize());
        assertEquals(5, c.convolutionKernelSize());
        assertEquals(java.util.List.of(128, 32), c.subsamplingConvolutionChannels());
        assertEquals(12, c.attentionChunkSize());
        assertEquals(13, c.attentionContextLeft());
        assertEquals(0, c.attentionContextRight());
        assertEquals(16000, c.samplingRate());
        assertEquals(128, c.featureSize());
        assertEquals(512, c.fftLength());
        assertEquals(320, c.frameLength());
        assertEquals(160, c.hopLength());
        assertEquals(8000.0, c.maxFrequency());
        assertEquals(280, c.audioSequenceLength());
        assertEquals(40, c.audioMillisecondsPerToken());
        assertEquals(258881, c.audioTokenId());
        assertEquals(256000, c.beginAudioTokenId());
        assertEquals(258883, c.endAudioTokenId());
        assertEquals(1.0e10, c.gradientClipping());
    }

    @Test
    void rejectsUnsupportedFeatureExtractor() throws Exception {
        JsonNode p=processor();
        ((com.fasterxml.jackson.databind.node.ObjectNode)p.path("feature_extractor"))
                .put("feature_extractor_type", "WhisperFeatureExtractor");
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioConfig.fromJson(model(), p));
    }

    @Test
    void rejectsSamplingRateMismatchWithNyquistRange() throws Exception {
        JsonNode p=processor();
        ((com.fasterxml.jackson.databind.node.ObjectNode)p.path("feature_extractor"))
                .put("sampling_rate", 8000);
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AudioConfig.fromJson(model(), p));
    }

    private JsonNode model() throws Exception {
        return mapper.readTree("""
          {
            "model_type":"embedding_gemma2","dtype":"bfloat16",
            "audio_token_id":258881,"boa_token_id":256000,"eoa_token_index":258883,
            "audio_config":{
              "model_type":"gemma4_audio","hidden_size":1024,"num_hidden_layers":12,
              "num_attention_heads":8,"output_proj_dims":1536,"conv_kernel_size":5,
              "subsampling_conv_channels":[128,32],"attention_chunk_size":12,
              "attention_context_left":13,"attention_context_right":0,
              "rms_norm_eps":0.000001,"residual_weight":0.5,"hidden_act":"silu",
              "dtype":"bfloat16","use_clipped_linears":true,"gradient_clipping":1.0e10
            }
          }
          """);
    }

    private JsonNode processor() throws Exception {
        return mapper.readTree("""
          {
            "audio_seq_length":280,"audio_ms_per_token":40,
            "feature_extractor":{
              "feature_extractor_type":"Gemma4AudioFeatureExtractor",
              "sampling_rate":16000,"feature_size":128,"fft_length":512,
              "frame_length":320,"hop_length":160,"min_frequency":0.0,
              "max_frequency":8000.0,"mel_floor":0.001,"padding_value":0.0,
              "input_scale_factor":1.0,"dither":0.0
            }
          }
          """);
    }
}
