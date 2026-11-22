package tech.kayys.aljabr.gguf.loader;

import tech.kayys.aljabr.gguf.core.GgmlType;
import tech.kayys.aljabr.gguf.core.GgufMetaType;
import tech.kayys.aljabr.gguf.core.GgufMetaValue;

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
        Map<String, GgufMetaValue> typedMetadata = new HashMap<>();
        for (int i = 0; i < nKv; i++) {
            String key = c.str();
            int typeId = c.i32();
            GgufMetaValue value = readTypedValue(c, typeId);
            typedMetadata.put(key, value);
            metadata.put(key, toLegacyValue(value));
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

        return new GGUFModel(version, metadata, typedMetadata, tensors, dataStart, seg, arena);
    }

    private GgufMetaValue readTypedValue(GGUFReader.Cursor c, int typeId) {
        return switch (GgufMetaType.fromId(typeId)) {
            case UINT8 -> new GgufMetaValue.UInt8Val((short) (c.i8() & 0xff));
            case INT8 -> new GgufMetaValue.Int8Val(c.i8());
            case UINT16 -> new GgufMetaValue.UInt16Val(c.u16());
            case INT16 -> new GgufMetaValue.Int16Val(c.i16());
            case UINT32 -> new GgufMetaValue.UInt32Val(c.u32());
            case INT32 -> new GgufMetaValue.Int32Val(c.i32());
            case FLOAT32 -> new GgufMetaValue.Float32Val(c.f32());
            case BOOL -> new GgufMetaValue.BoolVal(c.i8() != 0);
            case STRING -> new GgufMetaValue.StringVal(c.str());
            case ARRAY -> readArray(c);
            case UINT64 -> new GgufMetaValue.UInt64Val(c.i64());
            case INT64 -> new GgufMetaValue.Int64Val(c.i64());
            case FLOAT64 -> new GgufMetaValue.Float64Val(c.f64());
        };
    }

    private GgufMetaValue.ArrayVal readArray(GGUFReader.Cursor c) {
        GgufMetaType elementType = GgufMetaType.fromId(c.i32());
        if (elementType == GgufMetaType.ARRAY) {
            throw new UnsupportedOperationException("Nested GGUF metadata arrays are not supported");
        }
        long length = c.i64();
        if (length < 0 || length > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Invalid GGUF metadata array length: " + length);
        }
        List<GgufMetaValue> values = new ArrayList<>((int) length);
        for (int i = 0; i < length; i++) {
            values.add(readTypedValue(c, elementType.id));
        }
        return new GgufMetaValue.ArrayVal(elementType, values);
    }

    private Object toLegacyValue(GgufMetaValue value) {
        return switch (value) {
            case GgufMetaValue.UInt8Val v -> (int) v.value();
            case GgufMetaValue.Int8Val v -> (int) v.value();
            case GgufMetaValue.UInt16Val v -> v.value();
            case GgufMetaValue.Int16Val v -> (int) v.value();
            case GgufMetaValue.UInt32Val v -> v.value();
            case GgufMetaValue.Int32Val v -> v.value();
            case GgufMetaValue.Float32Val v -> v.value();
            case GgufMetaValue.BoolVal v -> v.value();
            case GgufMetaValue.StringVal v -> v.value();
            case GgufMetaValue.UInt64Val v -> v.value();
            case GgufMetaValue.Int64Val v -> v.value();
            case GgufMetaValue.Float64Val v -> v.value();
            case GgufMetaValue.ArrayVal v -> v.elements().stream()
                    .map(this::toLegacyValue).toList();
        };
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
