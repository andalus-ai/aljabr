package tech.kayys.aljabr.gguf.loader;

import tech.kayys.aljabr.gguf.api.GgufModelDescriptor;

import java.util.Objects;

/**
 * Bridges the canonical GGUF parser model into the inference-facing
 * metadata-only descriptor contract.
 */
public final class GgufModelDescriptorAdapter {

    private GgufModelDescriptorAdapter() {
    }

    public static GgufModelDescriptor inspect(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        return GgufModelDescriptor.inspect(
                model.metadata(),
                model.tensors().stream().map(GGUFTensorInfo::name).toList());
    }
}
