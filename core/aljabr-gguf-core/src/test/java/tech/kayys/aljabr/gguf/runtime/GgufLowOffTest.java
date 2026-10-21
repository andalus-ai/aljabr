package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static tech.kayys.aljabr.gguf.runtime.GgufOffFx.assertPreparedQ8Offset;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.*;

class GgufLowOffTest {
    @Test
    void buildsPreparedLowBitMatricesFromNonZeroTensorOffsets() {
        assertPreparedQ8Offset(
                41,
                128,
                18,
                (block, quant) -> writeQ1_0Block(block, (short) 0x3c00, quant),
                (byte) 0xFF,
                128.0f,
                132L);
        assertPreparedQ8Offset(
                34,
                256,
                54,
                (block, quant) -> writeTQ1_0Block(block, (short) 0x3c00, quant),
                (byte) 0xFF,
                256.0f,
                260L);
        assertPreparedQ8Offset(
                35,
                256,
                66,
                (block, quant) -> writeTQ2_0Block(block, (short) 0x3c00, quant),
                (byte) 0xAA,
                256.0f,
                260L);
    }
}
