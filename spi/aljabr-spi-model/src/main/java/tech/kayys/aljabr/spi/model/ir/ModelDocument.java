package tech.kayys.aljabr.spi.model.ir;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * @deprecated Use {@link ModelDescriptor}. This compatibility facade preserves
 * the original metadata-first constructor and accessors while AMIR adopts the
 * canonical model descriptor API.
 */
@Deprecated(forRemoval = false)
public final class ModelDocument {

    private final ModelDescriptor descriptor;

    public ModelDocument(
            String id,
            String architecture,
            String sourceFormat,
            Map<String, String> metadata,
            Map<String, String> extensions,
            List<TensorDescriptor> tensors) {
        this(new ModelDescriptor(
                ModelIdentity.of(id), architecture, sourceFormat, metadata, extensions, tensors));
    }

    public ModelDocument(ModelDescriptor descriptor) {
        this.descriptor = Objects.requireNonNull(descriptor, "descriptor must not be null");
    }

    public String id() {
        return descriptor.identity().id();
    }

    public String architecture() {
        return descriptor.architecture();
    }

    public String sourceFormat() {
        return descriptor.sourceFormat();
    }

    public Map<String, String> metadata() {
        return descriptor.metadata();
    }

    public Map<String, String> extensions() {
        return descriptor.extensions();
    }

    public List<TensorDescriptor> tensors() {
        return descriptor.tensors();
    }

    public ModelIdentity identity() {
        return descriptor.identity();
    }

    public ModelDescriptor toDescriptor() {
        return descriptor;
    }

    @Override
    public boolean equals(Object other) {
        return this == other || other instanceof ModelDocument that
                && descriptor.equals(that.descriptor);
    }

    @Override
    public int hashCode() {
        return descriptor.hashCode();
    }

    @Override
    public String toString() {
        return "ModelDocument[descriptor=" + descriptor + "]";
    }
}
