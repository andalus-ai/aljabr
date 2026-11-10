package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import static tech.kayys.aljabr.gguf.runtime.GgufOffFx.assertPreparedQ8Offset;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.*;

class GgufQ8OffTest {
    @Test
    void buildsPreparedQ8MatricesFromNonZeroTensorOffsets() {
        assertPreparedQ8Offset(
                8,
                32,
                34,
                (block, quant) -> writeQ8Block(block, (short) 0x3c00, quant),
                (byte) 2,
                64.0f,
                36L);
        assertPreparedQ8Offset(
                9,
                32,
                36,
                (block, quant) -> writeQ8_1Block(block, (short) 0x3c00, quant),
                (byte) 2,
                64.0f,
                36L);
        assertPreparedQ8Offset(
                15,
                256,
                292,
                (block, quant) -> writeQ8KBlock(block, 1.0f, quant),
                (byte) 2,
                512.0f,
                260L);
    }
}
