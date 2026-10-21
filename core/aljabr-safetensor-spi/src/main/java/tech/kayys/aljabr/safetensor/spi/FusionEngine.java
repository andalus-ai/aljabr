package tech.kayys.aljabr.safetensor.spi;

import tech.kayys.aljabr.spi.inference.InferenceRequest;
import tech.kayys.aljabr.spi.model.ModelConfig;
import java.util.List;

/**
 * Service interface for multimodal fusion strategies.
 * 
 * <p>A FusionEngine takes pre-processed modality inputs (text embeddings, 
 * image patches, etc.) and merges them into a unified sequence that can 
 * be processed by the transformer forward pass.
 * @author bhangun
 */
public interface FusionEngine {

    /**
     * Unique ID of the fusion strategy (e.g., "early", "cross-attention").
     */
    String id();

    /**
     * Fuses multiple encoded inputs into a single sequence result.
     * 
     * @param inputs        list of modality-specific inputs
     * @param config        model configuration for dimension validation
     * @param request       original inference request for context
     * @return              fused sequence metadata
     */
    FusionResult fuse(List<EncodedInput> inputs, ModelConfig config, InferenceRequest request);

    /**
     * Result of a fusion operation.
     */
    record FusionResult(
        float[][] embeddings,
        int[] positionIds,
        int totalTokens,
        boolean requiresCustomAttentionMask
    ) {}
}
