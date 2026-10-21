package tech.kayys.aljabr.gguf.model.aljabr;

import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.nn.NNModule;
import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.core.tensor.WeightAdapter;
/**
 * 
 * Core class for aljabr module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class VisionProjector extends NNModule {
    private final Linear proj;

    public VisionProjector(WeightAdapter weights) {
        // PaliGemma uses a single Linear projection (often bias=true)
        Tensor wProj = weights.getWeight("mm_proj.weight");
        Tensor bProj = weights.getWeight("mm_proj.bias");

        proj = new Linear(wProj, bProj);
        registerModule("proj", proj);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: [batch, num_patches, embed_dim]
        // PaliGemma usually applies the projector directly to the normalized patch
        // embeddings
        return proj.forward(x);
    }
}
