package tech.kayys.aljabr.models.diffusion;

import java.util.Objects;

import tech.kayys.aljabr.core.tensor.Tensor;
import tech.kayys.aljabr.diffusion.api.PreparedDiffusionConditioningSet;
import tech.kayys.aljabr.diffusion.execution.api.SigmaStepContext;
import tech.kayys.aljabr.diffusion.execution.loop.DiffusionModelInvoker;
import tech.kayys.aljabr.models.UNet2DConditional;

/**
 * Adapts Aljabr's conditional 2D UNet to the generic diffusion invocation
 * contract.
 *
 * <p>Checkpoint-specific conditioning and sigma/timestep conversion are
 * delegated to {@link UNet2DConditioningMapper}. This adapter owns neither
 * the UNet nor request conditioning. It owns and closes the temporary model
 * input tensors returned by the mapper.</p>
 *
 * <p>The current UNet2DConditional implementation is a simplified scaffold,
 * not yet a checkpoint-compatible Stable Diffusion implementation. This
 * adapter provides the integration seam without claiming that limitation is
 * solved.</p>
 */
public final class UNet2DConditionalInvoker implements DiffusionModelInvoker {
    private final UNet2DConditional model;
    private final UNet2DConditioningMapper conditioningMapper;

    public UNet2DConditionalInvoker(
            UNet2DConditional model,
            UNet2DConditioningMapper conditioningMapper) {
        this.model = Objects.requireNonNull(model, "model");
        this.conditioningMapper = Objects.requireNonNull(
                conditioningMapper, "conditioningMapper");
    }

    @Override
    public Tensor predict(
            Tensor sample,
            PreparedDiffusionConditioningSet conditioning,
            SigmaStepContext context) {
        Objects.requireNonNull(sample, "sample");
        Objects.requireNonNull(conditioning, "conditioning");
        Objects.requireNonNull(context, "context");
        if (conditioning.isClosed()) {
            throw new IllegalStateException("conditioning was closed before model invocation");
        }

        UNet2DModelInputs inputs = Objects.requireNonNull(
                conditioningMapper.map(sample, conditioning, context),
                "conditioning mapper returned null inputs");
        if (inputs.timestepEmbedding() == sample
                || inputs.encoderHiddenStates() == sample) {
            inputs.close();
            throw new IllegalArgumentException(
                    "conditioning mapper must not return the borrowed latent sample");
        }

        try (inputs) {
            return Objects.requireNonNull(
                    model.forward(
                            sample,
                            inputs.timestepEmbedding(),
                            inputs.encoderHiddenStates()),
                    "UNet returned null prediction");
        }
    }
}
