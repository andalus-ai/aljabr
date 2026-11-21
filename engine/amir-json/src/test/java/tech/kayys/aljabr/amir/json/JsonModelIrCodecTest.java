package tech.kayys.aljabr.amir.json;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.spi.model.ir.ModelDescriptor;
import tech.kayys.aljabr.spi.model.ir.ModelIdentity;
import tech.kayys.aljabr.spi.model.ir.ModelIrDocument;
import tech.kayys.aljabr.spi.model.ir.ModelIrSchemaVersion;

class JsonModelIrCodecTest {

    private final JsonModelIrCodec codec = new JsonModelIrCodec();

    private static ModelDescriptor model() {
        return new ModelDescriptor(
                new ModelIdentity("sample", "org.example", "rev-1"),
                "decoder",
                "safetensors",
                Map.of("context_length", "4096", "hidden_size", "2048"),
                Map.of("vendor.example/source", "checkpoint-v2"),
                List.of());
    }

    @Test
    void roundTripsTypedDocumentAndOpaqueExtensions() throws IOException {
        ModelIrDocument original = ModelIrDocument.current(model())
                .withExtension("vendor.example/payload", "{\"future\":[1,2,3]}")
                .withExtension("org.example/opaque", "YWJj");

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        codec.write(original, output);
        ModelIrDocument decoded = codec.read(new ByteArrayInputStream(output.toByteArray()));

        assertEquals(original, decoded);
        assertEquals("amir-json", codec.formatId());
        assertTrue(codec.mediaTypes().contains("application/vnd.aljabr.amir+json"));
    }

    @Test
    void emitsDeterministicJsonForEquivalentDocuments() throws IOException {
        ModelIrDocument first = ModelIrDocument.current(model());
        ModelIrDocument second = ModelIrDocument.current(new ModelDescriptor(
                model().identity(), model().architecture(), model().sourceFormat(),
                Map.of("hidden_size", "2048", "context_length", "4096"),
                Map.of("vendor.example/source", "checkpoint-v2"), List.of()));

        ByteArrayOutputStream firstOutput = new ByteArrayOutputStream();
        ByteArrayOutputStream secondOutput = new ByteArrayOutputStream();
        codec.write(first, firstOutput);
        codec.write(second, secondOutput);

        assertEquals(firstOutput.toString(StandardCharsets.UTF_8),
                secondOutput.toString(StandardCharsets.UTF_8));
    }

    @Test
    void doesNotCloseCallerOwnedStreams() throws IOException {
        class TrackingInput extends ByteArrayInputStream {
            private boolean closed;

            TrackingInput(byte[] bytes) {
                super(bytes);
            }

            @Override
            public void close() {
                closed = true;
            }
        }
        class TrackingOutput extends ByteArrayOutputStream {
            private boolean closed;

            @Override
            public void close() throws IOException {
                closed = true;
                super.close();
            }
        }

        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
        codec.write(ModelIrDocument.current(model()), encoded);

        TrackingInput input = new TrackingInput(encoded.toByteArray());
        codec.read(input);
        assertFalse(input.closed);

        TrackingOutput output = new TrackingOutput();
        codec.write(ModelIrDocument.current(model()), output);
        assertFalse(output.closed);
    }

    @Test
    void rejectsIncompatibleSchemaMajorOnReadAndWrite() {
        String incompatible = """
                {
                  "schemaVersion": {"major": 2, "minor": 0, "patch": 0},
                  "model": {
                    "identity": {"id": "sample"},
                    "architecture": "decoder",
                    "sourceFormat": "safetensors",
                    "metadata": {},
                    "extensions": {},
                    "tensors": [],
                    "architectureDescriptor": null,
                    "graphCollection": null,
                    "assetCollection": null,
                    "componentCollection": null
                  },
                  "extensions": {}
                }
                """;

        assertThrows(IllegalArgumentException.class,
                () -> codec.read(new ByteArrayInputStream(incompatible.getBytes(StandardCharsets.UTF_8))));
        assertThrows(IllegalArgumentException.class,
                () -> codec.write(new ModelIrDocument(
                        new ModelIrSchemaVersion(2, 0, 0), model(), Map.of()),
                        new ByteArrayOutputStream()));
    }

    @Test
    void rejectsUnknownFieldsInsteadOfSilentlyDroppingThem() {
        String withUnknownField = """
                {
                  "schemaVersion": {"major": 1, "minor": 0, "patch": 0},
                  "model": {
                    "identity": {"id": "sample"},
                    "architecture": "decoder",
                    "sourceFormat": "safetensors",
                    "metadata": {},
                    "extensions": {},
                    "tensors": [],
                    "architectureDescriptor": null,
                    "graphCollection": null,
                    "assetCollection": null,
                    "componentCollection": null,
                    "futureNestedField": {"must": "not disappear"}
                  },
                  "extensions": {}
                }
                """;

        assertThrows(IOException.class,
                () -> codec.read(new ByteArrayInputStream(
                        withUnknownField.getBytes(StandardCharsets.UTF_8))));
    }
}
