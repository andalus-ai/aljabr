package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_0_BLOCK_SIZE;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_1_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ4_1Block;
import static tech.kayys.aljabr.gguf.runtime.GgufQ32Fx.writeQ5_1Block;

class GgufQ32MetaTest {
    @Test
    void detectsQ4_1BlockBiasesOnlyInsideRequestedRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(Q4_1_BLOCK_BYTES * 2L);
            writeQ4_1Block(rows.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ4_1Block(
                    rows.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    (byte) 0x21);

            assertFalse(GgufQ32Meta.q4_1RowsHaveBlockBiases(rows, Q4_1_BLOCK_BYTES, Q4_0_BLOCK_SIZE, 1));
            assertTrue(GgufQ32Meta.q4_1RowsHaveBlockBiases(rows, Q4_1_BLOCK_BYTES, Q4_0_BLOCK_SIZE, 2));
            assertTrue(GgufQ32Meta.q4_1BlocksHaveBiases(rows, 2));
        }
    }

    @Test
    void detectsQ4_1BiasesInContiguousUnrolledBlockTail() {
        try (Arena arena = Arena.ofConfined()) {
            int blocks = 5;
            MemorySegment row = arena.allocate(Q4_1_BLOCK_BYTES * (long) blocks);
            for (int block = 0; block < blocks - 1; block++) {
                writeQ4_1Block(
                        row.asSlice(block * (long) Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                        (short) 0x3c00,
                        (short) 0,
                        (byte) 0x21);
            }
            writeQ4_1Block(
                    row.asSlice((blocks - 1L) * Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    (byte) 0x21);

            assertTrue(GgufQ32Meta.q4_1RowsHaveBlockBiases(
                    row, Q4_1_BLOCK_BYTES * (long) blocks, Q4_0_BLOCK_SIZE * blocks, 1));
            assertTrue(GgufQ32Meta.q4_1BlocksHaveBiases(row, blocks));
        }
    }

    @Test
    void paddedQ4_1RowsDoNotScanPaddingAsTensorBlocks() {
        try (Arena arena = Arena.ofConfined()) {
            long paddedRowBytes = Q4_1_BLOCK_BYTES * 2L;
            MemorySegment rows = arena.allocate(paddedRowBytes * 2L);
            writeQ4_1Block(rows.asSlice(0, Q4_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, (byte) 0x21);
            writeQ4_1Block(
                    rows.asSlice(Q4_1_BLOCK_BYTES, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    (byte) 0x21);
            writeQ4_1Block(
                    rows.asSlice(paddedRowBytes, Q4_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    (byte) 0x21);

            assertFalse(GgufQ32Meta.q4_1RowsHaveBlockBiases(rows, paddedRowBytes, Q4_0_BLOCK_SIZE, 1));
            assertTrue(GgufQ32Meta.q4_1RowsHaveBlockBiases(rows, paddedRowBytes, Q4_0_BLOCK_SIZE, 2));
        }
    }

    @Test
    void detectsQ5_1BlockBiasesOnlyInsideRequestedRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(Q5_1_BLOCK_BYTES * 2L);
            writeQ5_1Block(rows.asSlice(0, Q5_1_BLOCK_BYTES), (short) 0x3c00, (short) 0, -1, (byte) 0x21);
            writeQ5_1Block(
                    rows.asSlice(Q5_1_BLOCK_BYTES, Q5_1_BLOCK_BYTES),
                    (short) 0x3c00,
                    (short) 0x3800,
                    -1,
                    (byte) 0x21);

            assertFalse(GgufQ32Meta.q5_1RowsHaveBlockBiases(rows, Q5_1_BLOCK_BYTES, Q4_0_BLOCK_SIZE, 1));
            assertTrue(GgufQ32Meta.q5_1RowsHaveBlockBiases(rows, Q5_1_BLOCK_BYTES, Q4_0_BLOCK_SIZE, 2));
            assertTrue(GgufQ32Meta.q5_1BlocksHaveBiases(rows, 2));
        }
    }
}
