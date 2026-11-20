package tech.kayys.aljabr.spi.model.ir;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelDescriptorTest {

    @Test
    void createsStableIdentityWithoutInferringNamespaceOrRevision() {
        var identity = ModelIdentity.of("org.example/model");

        assertEquals("org.example/model", identity.id());
        assertNull(identity.namespace());
        assertNull(identity.revision());
    }

    @Test
    void normalizesOptionalIdentityQualifiers() {
        var identity = new ModelIdentity("model-1", "  ", null);

        assertNull(identity.namespace());
        assertNull(identity.revision());
    }

    @Test
    void descriptorIsImmutableAndPreservesOpaqueArchitectureAndFormatIds() {
        var metadata = new java.util.LinkedHashMap<>(Map.of("parameter_count", "1234"));
        var extensions = new java.util.LinkedHashMap<>(Map.of("vendor.custom", "{\"enabled\":true}"));
        var tensors = List.of(new TensorDescriptor(
                "weights", "F16", List.of(2L, 2L), "row-major", null, null));

        var descriptor = new ModelDescriptor(
                new ModelIdentity("model-1", "example.org", "revision-7"),
                "vendor.custom_architecture",
                "vendor.custom_format",
                metadata,
                extensions,
                tensors);

        metadata.put("mutated", "true");
        extensions.clear();

        assertEquals("vendor.custom_architecture", descriptor.architecture());
        assertEquals("vendor.custom_format", descriptor.sourceFormat());
        assertEquals(Map.of("parameter_count", "1234"), descriptor.metadata());
        assertEquals(Map.of("vendor.custom", "{\"enabled\":true}"), descriptor.extensions());
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.metadata().put("x", "y"));
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.extensions().put("x", "y"));
        assertThrows(UnsupportedOperationException.class,
                () -> descriptor.tensors().clear());
    }

    @Test
    void rejectsDuplicateTensorNames() {
        var first = new TensorDescriptor("weight", "F32", List.of(2L), "row-major", null, null);
        var second = new TensorDescriptor("weight", "F16", List.of(2L), "row-major", null, null);

        assertThrows(IllegalArgumentException.class, () -> new ModelDescriptor(
                ModelIdentity.of("model"), "unknown", "unknown",
                Map.of(), Map.of(), List.of(first, second)));
    }

    @Test
    void withIdentityReturnsNewDescriptorWithoutChangingModelContent() {
        var original = new ModelDescriptor(
                ModelIdentity.of("model-v1"), "decoder", "safetensors",
                Map.of("layers", "2"), Map.of(), List.of());

        var revised = original.withIdentity(new ModelIdentity("model-v2", "example.org", "v2"));

        assertNotSame(original, revised);
        assertEquals("model-v1", original.identity().id());
        assertEquals("model-v2", revised.identity().id());
        assertEquals(original.architecture(), revised.architecture());
        assertEquals(original.tensors(), revised.tensors());
    }

    @Test
    void legacyModelDocumentDelegatesToCanonicalDescriptor() {
        var document = new ModelDocument(
                "legacy-id", "decoder", "safetensors", Map.of(), Map.of(), List.of());

        assertEquals(ModelIdentity.of("legacy-id"), document.identity());
        assertEquals(document.toDescriptor(), new ModelDescriptor(
                ModelIdentity.of("legacy-id"), "decoder", "safetensors",
                Map.of(), Map.of(), List.of()));
    }

    @Test
    void rejectsInvalidIdentityAndNullMetadataValues() {
        assertThrows(IllegalArgumentException.class, () -> ModelIdentity.of(" "));
        assertThrows(IllegalArgumentException.class, () -> new ModelDescriptor(
                ModelIdentity.of("model"), "decoder", "safetensors",
                Map.of("invalid", null), Map.of(), List.of()));
    }
}
