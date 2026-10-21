package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.assertPreparedMatrixProbe;
import static tech.kayys.aljabr.gguf.runtime.GgufProbeFx.oneTensorModel;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeIQ4XSBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeMXFP4Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQFx.writeNVFP4Block;

class GgufProbeQ4Test {
    @Test
    void probesPreparedMXFP4MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(17);
            writeMXFP4Block(segment, (byte) 0xA5);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{32, 1}, 39, 17),
                    17,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "MXFP4", 1, 32, -2.0f);
        }
    }

    @Test
    void probesPreparedNVFP4MatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(36);
            writeNVFP4Block(segment, (byte) 0xA5);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{64, 1}, 40, 36),
                    36,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "NVFP4", 1, 64, -39.29412f);
        }
    }

    @Test
    void probesPreparedIQ4XSMatrixPath() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(136);
            writeIQ4XSBlock(segment, (byte) 0x88);

            GgufRuntimeProbe probe = GgufRuntimeProbe.fromModel(
                    oneTensorModel(segment, new long[]{256, 1}, 23, 136),
                    136,
                    1,
                    1);

            assertPreparedMatrixProbe(probe, "IQ4_XS", 1, 256, -0.47058824f);
        }
    }
}
