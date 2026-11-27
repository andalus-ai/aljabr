package tech.kayys.aljabr.gguf.loader;

import tech.kayys.aljabr.gguf.api.GgufModelDescriptor;
import tech.kayys.aljabr.gguf.api.GgufTensorDescriptor;

import java.util.Objects;

/**
 * Maps parser-owned GGUF metadata into the public structural descriptor API.
 *
 * <p>Physical payload location and size remain on the parser model and are
 * intentionally not copied into {@link GgufTensorDescriptor}.
 */
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
