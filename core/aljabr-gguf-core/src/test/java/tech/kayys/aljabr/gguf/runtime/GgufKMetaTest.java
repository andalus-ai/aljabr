package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q2_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q4_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.Q5_K_BLOCK_BYTES;
import static tech.kayys.aljabr.gguf.runtime.GgufQuantFormats.QK_K;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ2KBlockWithMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlock;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ5KBlockWithMin;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeSimpleQ4KBlock;

class GgufKMetaTest {
    @Test
    void detectsQ2MinsOnlyInsideRequestedRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(Q2_K_BLOCK_BYTES * 2L);
            writeQ2KBlock(rows.asSlice(0, Q2_K_BLOCK_BYTES), (byte) 1, (byte) 0x55);
            writeQ2KBlockWithMin(rows.asSlice(Q2_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES), (byte) 0x12, (byte) 0x55);

            assertFalse(GgufKMeta.q2RowsHaveMins(rows, Q2_K_BLOCK_BYTES, QK_K, 1));
            assertTrue(GgufKMeta.q2RowsHaveMins(rows, Q2_K_BLOCK_BYTES, QK_K, 2));
            assertTrue(GgufKMeta.q2BlocksHaveMins(rows, 2));
        }
    }

    @Test
    void detectsQ4MinsOnlyInsideRequestedRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(Q4_K_BLOCK_BYTES * 2L);
            writeSimpleQ4KBlock(rows.asSlice(0, Q4_K_BLOCK_BYTES));
            writeQ4KBlockWithAllScalesAndMins(rows.asSlice(Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));

            assertFalse(GgufKMeta.q4RowsHaveMins(rows, Q4_K_BLOCK_BYTES, QK_K, 1));
            assertTrue(GgufKMeta.q4RowsHaveMins(rows, Q4_K_BLOCK_BYTES, QK_K, 2));
            assertTrue(GgufKMeta.q4BlocksHaveMins(rows, 2));
        }
    }

    @Test
    void detectsQ4MinsInContiguousUnrolledBlockTail() {
        try (Arena arena = Arena.ofConfined()) {
            int blocks = 5;
            MemorySegment row = arena.allocate(Q4_K_BLOCK_BYTES * (long) blocks);
            for (int block = 0; block < blocks - 1; block++) {
                writeSimpleQ4KBlock(row.asSlice(block * (long) Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
            }
            writeQ4KBlockWithAllScalesAndMins(
                    row.asSlice((blocks - 1L) * Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));

            assertTrue(GgufKMeta.q4RowsHaveMins(row, Q4_K_BLOCK_BYTES * (long) blocks, QK_K * blocks, 1));
            assertTrue(GgufKMeta.q4BlocksHaveMins(row, blocks));
        }
    }

    @Test
    void paddedQ4RowsDoNotScanPaddingAsTensorBlocks() {
        try (Arena arena = Arena.ofConfined()) {
            long paddedRowBytes = Q4_K_BLOCK_BYTES * 2L;
            MemorySegment rows = arena.allocate(paddedRowBytes * 2L);
            writeSimpleQ4KBlock(rows.asSlice(0, Q4_K_BLOCK_BYTES));
            writeQ4KBlockWithAllScalesAndMins(rows.asSlice(Q4_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES));
            writeQ4KBlockWithAllScalesAndMins(rows.asSlice(paddedRowBytes, Q4_K_BLOCK_BYTES));

            assertFalse(GgufKMeta.q4RowsHaveMins(rows, paddedRowBytes, QK_K, 1));
            assertTrue(GgufKMeta.q4RowsHaveMins(rows, paddedRowBytes, QK_K, 2));
        }
    }

    @Test
    void detectsQ4MinsFromLowPackedMinBytes() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment row = arena.allocate(Q4_K_BLOCK_BYTES);
            markK4DMin(row);
            row.set(ValueLayout.JAVA_BYTE, 8, (byte) 0x01);

            assertTrue(GgufKMeta.q4RowsHaveMins(row, Q4_K_BLOCK_BYTES, QK_K, 1));
        }
    }

    @Test
    void detectsQ4MinsFromHighPackedMinNibbles() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment row = arena.allocate(Q4_K_BLOCK_BYTES);
            markK4DMin(row);
            row.set(ValueLayout.JAVA_BYTE, 12, (byte) 0x10);

            assertTrue(GgufKMeta.q4RowsHaveMins(row, Q4_K_BLOCK_BYTES, QK_K, 1));
        }
    }

    @Test
    void detectsQ5MinsOnlyInsideRequestedRows() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment rows = arena.allocate(Q5_K_BLOCK_BYTES * 2L);
            writeQ5KBlock(rows.asSlice(0, Q5_K_BLOCK_BYTES), (byte) 0xFF, (byte) 0);
            writeQ5KBlockWithMin(
                    rows.asSlice(Q5_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES),
                    (byte) 2,
                    (byte) 1,
                    (byte) 0,
                    (byte) 0x11);

            assertFalse(GgufKMeta.q5RowsHaveMins(rows, Q5_K_BLOCK_BYTES, QK_K, 1));
            assertTrue(GgufKMeta.q5RowsHaveMins(rows, Q5_K_BLOCK_BYTES, QK_K, 2));
            assertTrue(GgufKMeta.q5BlocksHaveMins(rows, 2));
        }
    }

    @Test
    void detectsQ5MinsFromHighPackedMinNibbles() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment row = arena.allocate(Q5_K_BLOCK_BYTES);
            markK4DMin(row);
            row.set(ValueLayout.JAVA_BYTE, 12, (byte) 0x10);

            assertTrue(GgufKMeta.q5RowsHaveMins(row, Q5_K_BLOCK_BYTES, QK_K, 1));
        }
    }

    private static void markK4DMin(MemorySegment row) {
        row.set(GgufKFx.LE_SHORT, 2, (short) 0x3c00);
    }
}
