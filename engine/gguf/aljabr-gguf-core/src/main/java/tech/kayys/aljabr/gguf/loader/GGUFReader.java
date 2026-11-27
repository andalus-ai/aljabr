package tech.kayys.aljabr.gguf.loader;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Handles zero-copy memory mapping of GGUF model files using FFM.
 * @author bhangun
 */
public final class GGUFReader implements AutoCloseable {

    private final FileChannel channel;
    private final Arena arena;
    private final MemorySegment segment;
    private final boolean ownArena;

    public GGUFReader(Path path) throws IOException {
        this(path, Arena.ofShared(), true);
    }

    public GGUFReader(Path path, Arena arena) throws IOException {
        this(path, arena, false);
    }

    private GGUFReader(Path path, Arena arena, boolean ownArena) throws IOException {
        this.channel = FileChannel.open(path, StandardOpenOption.READ);
        this.arena = arena;
        this.ownArena = ownArena;
        long size = channel.size();
        this.segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
    }

    public MemorySegment segment() {
        return segment;
    }

    public Arena arena() {
        return arena;
    }

    @Override
    public void close() throws IOException {
        try {
            if (ownArena) {
                arena.close();
            }
        } finally {
            channel.close();
        }
    }

    /**
     * Binary cursor for reading through the memory segment.
     * GGUF numeric fields are little-endian regardless of host byte order.
     */
    public static final class Cursor {
        private final MemorySegment seg;
        private long pos;

        private static final ValueLayout.OfShort I16_UNALIGNED =
                ValueLayout.JAVA_SHORT.withByteAlignment(1).withOrder(ByteOrder.LITTLE_ENDIAN);
        private static final ValueLayout.OfInt I32_UNALIGNED =
                ValueLayout.JAVA_INT.withByteAlignment(1).withOrder(ByteOrder.LITTLE_ENDIAN);
        private static final ValueLayout.OfLong I64_UNALIGNED =
                ValueLayout.JAVA_LONG.withByteAlignment(1).withOrder(ByteOrder.LITTLE_ENDIAN);
        private static final ValueLayout.OfFloat F32_UNALIGNED =
                ValueLayout.JAVA_FLOAT.withByteAlignment(1).withOrder(ByteOrder.LITTLE_ENDIAN);
        private static final ValueLayout.OfDouble F64_UNALIGNED =
                ValueLayout.JAVA_DOUBLE.withByteAlignment(1).withOrder(ByteOrder.LITTLE_ENDIAN);

        public Cursor(MemorySegment seg) {
            this.seg = seg;
            this.pos = 0;
        }

        public byte i8() {
            byte value = seg.get(ValueLayout.JAVA_BYTE, pos);
            pos += 1;
            return value;
        }

        public short i16() {
            short value = seg.get(I16_UNALIGNED, pos);
            pos += 2;
            return value;
        }

        public int u16() {
            int value = seg.get(I16_UNALIGNED, pos) & 0xFFFF;
            pos += 2;
            return value;
        }

        public int i32() {
            int value = seg.get(I32_UNALIGNED, pos);
            pos += 4;
            return value;
        }

        public long u32() {
            long value = seg.get(I32_UNALIGNED, pos) & 0xFFFFFFFFL;
            pos += 4;
            return value;
        }

        public long i64() {
            long value = seg.get(I64_UNALIGNED, pos);
            pos += 8;
            return value;
        }

        public float f32() {
            float value = seg.get(F32_UNALIGNED, pos);
            pos += 4;
            return value;
        }

        public double f64() {
            double value = seg.get(F64_UNALIGNED, pos);
            pos += 8;
            return value;
        }

        public String str() {
            long length = i64();
            if (length < 0 || length > Integer.MAX_VALUE) {
                throw new IllegalStateException("Invalid string length: " + length);
            }
            byte[] bytes = new byte[(int) length];
            MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, pos, bytes, 0, (int) length);
            pos += length;
            return new String(bytes, StandardCharsets.UTF_8);
        }

        public void skip(long n) {
            pos += n;
        }

        public long position() {
            return pos;
        }

        public void seek(long p) {
            this.pos = p;
        }
    }
}
