package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static tech.kayys.aljabr.gguf.runtime.GgufProfFx.assertSingleTypeReady;

class GgufProfCompactTest {
    @Test
    void treatsCompactQ8FamilyDecoderTensorsAsJavaRowDotReady() {
        assertSingleTypeReady(9, 36, 32, "Q8_1");
        assertSingleTypeReady(41, 18, 128, "Q1_0");
        assertSingleTypeReady(34, 54, 256, "TQ1_0");
        assertSingleTypeReady(35, 66, 256, "TQ2_0");
        assertSingleTypeReady(39, 17, 32, "MXFP4");
        assertSingleTypeReady(40, 36, 64, "NVFP4");
    }
}
