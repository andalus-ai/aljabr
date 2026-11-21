package tech.kayys.aljabr.spi.model.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ModelIrSchemaVersionTest {

    private static ModelDescriptor model() {
        return new ModelDescriptor(
                ModelIdentity.of("sample"), "decoder", "native",
                Map.of("context_length", "4096"), Map.of(), java.util.List.of());
    }

    @Test
    void comparesVersionsSemantically() {
        ModelIrSchemaVersion v1 = new ModelIrSchemaVersion(1, 2, 3);
        assertTrue(v1.compareTo(new ModelIrSchemaVersion(1, 2, 4)) < 0);
        assertTrue(v1.compareTo(new ModelIrSchemaVersion(1, 3, 0)) < 0);
        assertTrue(v1.compareTo(new ModelIrSchemaVersion(2, 0, 0)) < 0);
        assertEquals("1.2.3", v1.toString());
        assertThrows(IllegalArgumentException.class, () -> new ModelIrSchemaVersion(1, -1, 0));
    }

    @Test
    void acceptsSameMajorAndRejectsIncompatibleMajor() {
        ModelIrSchemaVersion reader = new ModelIrSchemaVersion(1, 0, 0);
        assertTrue(reader.isCompatibleWith(new ModelIrSchemaVersion(1, 9, 7)));
        assertDoesNotThrow(() -> reader.requireCompatibleWith(new ModelIrSchemaVersion(1, 9, 7)));

        UnsupportedModelIrVersionException error = assertThrows(
                UnsupportedModelIrVersionException.class,
                () -> reader.requireCompatibleWith(new ModelIrSchemaVersion(2, 0, 0)));
        assertEquals(reader, error.readerVersion());
        assertEquals(new ModelIrSchemaVersion(2, 0, 0), error.documentVersion());
    }

    @Test
    void documentRetainsOpaqueExtensionsAcrossImmutableUpdates() {
        ModelIrDocument document = ModelIrDocument.current(model())
                .withExtension("vendor.example/custom", "{\"future\":[1,2,3]}")
                .withExtension("org.example/opaque", "YWJj");

        assertEquals(ModelIrSchemaVersion.CURRENT, document.schemaVersion());
        assertEquals("{\"future\":[1,2,3]}", document.extensions().get("vendor.example/custom"));
        assertThrows(UnsupportedOperationException.class,
                () -> document.extensions().put("new", "value"));

        ModelIrDocument changed = document.withModel(model().withIdentity(ModelIdentity.of("renamed")));
        assertEquals(document.extensions(), changed.extensions());
        assertEquals("renamed", changed.model().identity().id());
    }

    @Test
    void codecContractSupportsStreamingWithoutTakingStreamOwnership() throws IOException {
        final class TestCodec implements ModelIrCodec {
            @Override public String formatId() { return "test-text"; }
            @Override public Set<String> mediaTypes() { return Set.of("application/x-amir-test"); }

            @Override
            public ModelIrDocument read(java.io.InputStream input) throws IOException {
                String text = new String(input.readAllBytes(), StandardCharsets.UTF_8);
                String[] fields = text.split("\\n", 2);
                ModelIrSchemaVersion version = parse(fields[0]);
                version.requireCompatibleWith(ModelIrSchemaVersion.CURRENT);
                return new ModelIrDocument(version, model(),
                        fields.length == 1 ? Map.of() : Map.of("test/opaque", fields[1]));
            }

            @Override
            public void write(ModelIrDocument document, java.io.OutputStream output) throws IOException {
                ModelIrSchemaVersion.CURRENT.requireCompatibleWith(document.schemaVersion());
                output.write((document.schemaVersion() + "\\n"
                        + document.extensions().getOrDefault("test/opaque", ""))
                        .getBytes(StandardCharsets.UTF_8));
            }

            private ModelIrSchemaVersion parse(String value) {
                String[] parts = value.split("\\.");
                return new ModelIrSchemaVersion(
                        Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
            }
        }

        TestCodec codec = new TestCodec();
        byte[] bytes = "1.0.0\\nopaque-payload".getBytes(StandardCharsets.UTF_8);
        ByteArrayInputStream input = new ByteArrayInputStream(bytes);
        ModelIrDocument decoded = codec.read(input);
        assertEquals("test-text", codec.formatId());
        assertEquals("opaque-payload", decoded.extensions().get("test/opaque"));
        assertEquals(bytes.length, input.available());

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        codec.write(decoded, output);
        assertEquals(new String(bytes, StandardCharsets.UTF_8),
                output.toString(StandardCharsets.UTF_8));
    }
}
