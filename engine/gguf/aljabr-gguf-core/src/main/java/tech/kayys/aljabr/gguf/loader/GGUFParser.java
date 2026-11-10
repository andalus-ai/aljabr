package tech.kayys.aljabr.gguf.loader;

import tech.kayys.aljabr.gguf.core.GgmlType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure Java GGUF parser. Extracts model structure and metadata.
 * @author bhangun
 */
public final class GGUFParser {

    public GGUFModel parse(MemorySegment seg, Arena arena) {
        GGUFReader.Cursor c = new GGUFReader.Cursor(seg);

        // 1. Header
        byte[] magic = new byte[4];
        for (int i = 0; i < 4; i++) magic[i] = c.i8();
        if (!new String(magic).equals("GGUF")) {
            throw new IllegalArgumentException("Not a valid GGUF file");
        }

        int version = c.i32();
        long nTensors = c.i64();
        long nKv = c.i64();

        // 2. Metadata KV
        Map<String, Object> metadata = new HashMap<>();
        for (int i = 0; i < nKv; i++) {
            String key = c.str();
            int typeId = c.i32();
            metadata.put(key, readValue(c, typeId));
        }

        // 3. Tensor Info
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        for (int i = 0; i < nTensors; i++) {
            String name = c.str();
            int nDims = c.i32();
            long[] shape = new long[nDims];
            long numElements = 1;
            for (int d = 0; d < nDims; d++) {
                shape[d] = c.i64();
                numElements *= shape[d];
            }
            int typeId = c.i32();
            long offset = c.i64();
            long size = calculateTensorSize(numElements, typeId);
            GGUFTensorInfo info = new GGUFTensorInfo(name, shape, typeId, offset, size);
            tensors.add(info);
        }

        // Alignment - GGUF usually aligns data to 32 bytes by default
        int alignment = ((Number) metadata.getOrDefault("general.alignment", 32)).intValue();
        long pos = c.position();
        long padding = (alignment - (pos % alignment)) % alignment;
        long dataStart = pos + padding;

        return new GGUFModel(version, metadata, tensors, dataStart, seg, arena);
    }

    private Object readValue(GGUFReader.Cursor c, int typeId) {
        return switch (typeId) {
            case 0 -> (int) c.i8() & 0xFF; // UINT8
            case 1 -> (int) c.i8();        // INT8
            case 2 -> c.u16();             // UINT16
            case 3 -> (int) c.i16();       // INT16
            case 4 -> c.u32();             // UINT32
            case 5 -> c.i32();             // INT32
            case 6 -> c.f32();             // FLOAT32
            case 7 -> c.i8() != 0;         // BOOL
            case 8 -> c.str();             // STRING
            case 9 -> readArray(c);        // ARRAY
            case 10 -> c.i64();            // UINT64
            case 11 -> c.i64();            // INT64
            case 12 -> c.f64();            // FLOAT64
            default -> throw new UnsupportedOperationException("Unsupported GGUF KV type: " + typeId + " at pos " + c.position());
        };
    }

    private List<Object> readArray(GGUFReader.Cursor c) {
        int typeId = c.i32();
        long len = c.i64();
        List<Object> list = new ArrayList<>((int) len);
        for (int i = 0; i < len; i++) {
            list.add(readValue(c, typeId));
        }
        return list;
    }
    
    private long calculateTensorSize(long numElements, int typeId) {
        try {
            return GgmlType.fromId(typeId).bytesFor(numElements);
        } catch (IllegalArgumentException exception) {
            throw new UnsupportedOperationException(
                    "Unsupported or misaligned GGUF tensor type " + typeId
                            + " with " + numElements + " element(s)",
                    exception);
        }
    }
}
