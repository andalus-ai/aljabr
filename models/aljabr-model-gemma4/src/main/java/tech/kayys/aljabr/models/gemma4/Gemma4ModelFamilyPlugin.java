package tech.kayys.aljabr.models.gemma4;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.aljabr.spi.model.ModelArchitecture;
import tech.kayys.aljabr.spi.model.ModelFamilyCapability;
import tech.kayys.aljabr.spi.model.ModelFamilyDescriptor;
import tech.kayys.aljabr.spi.model.ModelFamilyPlugin;
import tech.kayys.aljabr.spi.model.ModelFamilyUnifiedRuntimeRequirement;
import tech.kayys.aljabr.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;
import static java.util.Map.entry;

/**
 * Optional Gemma 4 model-family descriptor for text, multimodal, QAT, and guarded direct runtime metadata.
 */
@ApplicationScoped
public class Gemma4ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gemma4",
                "Google Gemma 4",
                List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio",
                        "gemma4_unified", "gemma4_unified_text"),
                List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
                        "Gemma4Model", "Gemma4TextModel", "Gemma4VisionModel",
                        "Gemma4AudioModel", "Gemma4ForImageTextToText",
                        "Gemma4ForMultimodalLM", "Gemma4UnifiedForConditionalGeneration",
                        "Gemma4Processor", "Gemma4ImageProcessor",
                        "Gemma4ImageProcessorPil", "Gemma4VideoProcessor",
                        "Gemma4AudioFeatureExtractor"),
                List.of(ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION,
                        ModelFamilyCapability.AUDIO,
                        ModelFamilyCapability.MOE,
                        ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                Map.ofEntries(
                        entry("bundle_profile", "optional"),
                        entry("direct_safetensor", "ready_text_path_guarded_by_runtime"),
                        entry("direct_safetensor_scope", "text_only_gemma4_text_and_unified_text"),
                        entry("moe_direct_safetensor", "pending_packed_expert_router_runtime"),
                        entry("double_wide_mlp", "detected_from_text_config"),
                        entry("multimodal_direct_safetensor", "pending_audio_vision_video_embedder_runtime"),
                        entry("unified_model_type", "gemma4_unified"),
                        entry("unified_runtime", "metadata_processor_only_until_unified_embedder_runtime"),
                        entry("checkpoint_gemma_4_12b_it", "google/gemma-4-12B-it"),
                        entry("checkpoint_gemma_4_12b_it_model_type", "gemma4_unified"),
                        entry("checkpoint_gemma_4_12b_it_scope",
                                "tokenizer_processor_and_family_resolution_guarded_runtime"),
                        entry("checkpoint_gemma_4_12b_it_architecture",
                                "dense_unified_encoder_free_multimodal"),
                        entry("checkpoint_gemma_4_12b_it_local_target",
                                "dedicated_gpu_laptop_or_unified_memory_16gb"),
                        entry("checkpoint_gemma_4_12b_it_mtp",
                                "companion_multi_token_prediction_model_available"),
                        entry("checkpoint_gemma_4_12b_it_litert_lm",
                                "litert-community/gemma-4-12B-it-litert-lm"),
                        entry("checkpoint_gemma_4_12b_it_litert_lm_scope",
                                "metadata_only_pending_litert_lm_runner"),
                        entry("checkpoint_gemma_4_qat_q4_0_collection",
                                "google/gemma-4-qat-q4-0"),
                        entry("checkpoint_gemma_4_qat_q4_0_formats",
                                "unquantized,gguf,compressed_tensors_w4a16"),
                        entry("checkpoint_gemma_4_qat_q4_0_scope",
                                "metadata_only_pending_q4_0_weight_loader"),
                        entry("checkpoint_gemma_4_qat_mobile_collection",
                                "google/gemma-4-qat-mobile"),
                        entry("checkpoint_gemma_4_qat_mobile_formats",
                                "transformers,compressed_tensors_mobile"),
                        entry("checkpoint_gemma_4_qat_mobile_scope",
                                "metadata_only_pending_mobile_quant_loader"),
                        entry("modalities", "text,image,audio,video"),
                        entry("tokenizer", "gemma_sentencepiece_with_audio_vision_processor"),
                        entry("processor", "Gemma4Processor"),
                        entry("image_processor", "Gemma4ImageProcessor"),
                        entry("pil_image_processor", "Gemma4ImageProcessorPil"),
                        entry("video_processor", "Gemma4VideoProcessor"),
                        entry("feature_extractor", "Gemma4FeatureExtractor"),
                        entry("audio_feature_extractor", "Gemma4AudioFeatureExtractor"),
                        entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new Gemma4Family());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("gemma4-spm-bpe"));
    }

    @Override
    public List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
        return List.of(new ModelFamilyUnifiedRuntimeRequirement(
                "gemma4_unified",
                List.of("text", "image", "audio", "video"),
                true,
                "Gemma 4 unified execution requires the multimodal embedder runtime",
                Map.of(
                        "checkpoint", "google/gemma-4-12B-it",
                        "architecture", "dense_unified_encoder_free_multimodal",
                        "litert_lm_repo", "litert-community/gemma-4-12B-it-litert-lm",
                        "qat_q4_0_collection", "google/gemma-4-qat-q4-0",
                        "qat_mobile_collection", "google/gemma-4-qat-mobile",
                        "processor", "Gemma4Processor")));
    }
}
