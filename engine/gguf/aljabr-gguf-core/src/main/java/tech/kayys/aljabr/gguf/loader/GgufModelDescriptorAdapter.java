package tech.kayys.aljabr.gguf.loader;

import tech.kayys.aljabr.gguf.api.GgufModelDescriptor;
import tech.kayys.aljabr.gguf.api.GgufTensorDescriptor;

import java.util.Objects;

/** Bridges the canonical GGUF parser model into the inference-facing descriptor contract. */
public final class GgufModelDescriptorAdapter {

    private GgufModelDescriptorAdapter() {
    }

    public static GgufModelDescriptor inspect(GGUFModel model) {
        Objects.requireNonNull(model, "model");
        return GgufModelDescriptor.inspectWithTensors(
                model.metadata(),
                model.tensors().stream()
                        .map(tensor -> new GgufTensorDescriptor(
                                tensor.name(),
                                java.util.Arrays.stream(tensor.shape()).boxed().toList(),
                                tensor.typeId()))
                        .toList());
    }
}
