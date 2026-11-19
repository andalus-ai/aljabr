package tech.kayys.aljabr.gguf.javabackend;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GgufFfnProjectionLayoutTest {
    private static final int EMBEDDING = 128;
    private static final int INTERMEDIATE = 256;

    @Test
    void acceptsGatedFfnWithExpectedProjectionGeometry() {
        assertDoesNotThrow(() -> validate(
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void acceptsUngatedFfnWhenGateProjectionIsAbsent() {
        assertDoesNotThrow(() -> validate(
                new GGUFTensorInfo[] {null},
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void acceptsAbsentGateArrayForUngatedArchitecture() {
        assertDoesNotThrow(() -> validate(
                null,
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void rejectsUpProjectionWithWrongInputWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null,
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING - 1)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void rejectsUpProjectionWithWrongIntermediateWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null,
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE - 1, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void rejectsGateProjectionWithWrongGeometry() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING - 1)},
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
    }

    @Test
    void rejectsDownProjectionWithWrongIntermediateWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null,
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE - 1)}));
    }

    @Test
    void rejectsDownProjectionWithWrongEmbeddingWidth() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null,
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING - 1, INTERMEDIATE)}));
    }

    @Test
    void rejectsMissingRequiredUpOrDownProjection() {
        assertThrows(IllegalArgumentException.class, () -> validate(
                null, new GGUFTensorInfo[] {null},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}));
        assertThrows(IllegalArgumentException.class, () -> validate(
                null, new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {null}));
    }

    @Test
    void validatesEachLayerIndependentlyAndRejectsArrayLengthMismatch() {
        assertDoesNotThrow(() -> GgufFfnProjectionLayout.validate(
                new GGUFTensorInfo[] {null, null},
                new GGUFTensorInfo[] {
                        tensor(0, INTERMEDIATE, EMBEDDING),
                        tensor(1, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {
                        tensor(0, EMBEDDING, INTERMEDIATE),
                        tensor(1, EMBEDDING, INTERMEDIATE)},
                EMBEDDING, INTERMEDIATE));
        assertThrows(IllegalArgumentException.class, () -> GgufFfnProjectionLayout.validate(
                new GGUFTensorInfo[] {null},
                new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING), tensor(1, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE), tensor(1, EMBEDDING, INTERMEDIATE)},
                EMBEDDING, INTERMEDIATE));
    }

    @Test
    void rejectsNonPositiveConfiguredDimensions() {
        assertThrows(IllegalArgumentException.class, () -> GgufFfnProjectionLayout.validate(
                null, new GGUFTensorInfo[] {tensor(0, INTERMEDIATE, EMBEDDING)},
                new GGUFTensorInfo[] {tensor(0, EMBEDDING, INTERMEDIATE)}, EMBEDDING, 0));
    }

    private static void validate(
            GGUFTensorInfo[] gate, GGUFTensorInfo[] up, GGUFTensorInfo[] down) {
        GgufFfnProjectionLayout.validate(gate, up, down, EMBEDDING, INTERMEDIATE);
    }

    private static GGUFTensorInfo tensor(int layer, int outputDimension, int inputDimension) {
        return new GGUFTensorInfo("blk." + layer + ".ffn.weight",
                new long[] {inputDimension, outputDimension}, 0, 0L,
                (long) inputDimension * outputDimension * Float.BYTES);
    }
}
