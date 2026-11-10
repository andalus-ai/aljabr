package tech.kayys.aljabr.gguf.loader.gguf;

import java.io.IOException;
import java.lang.foreign.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

import tech.kayys.aljabr.gguf.core.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.core.GgmlType;
/**
 * 
 * Core class for aljabr module.
 *
 * <p>Key functionality:
 * <ul>
 * <li>Provides core class operations
 * </ul>
 *
 * @author bhangun
 * @since 0.1.0
 */
public class GGUFLoader {
    
    public static GGUFFile load(Path path) throws IOException {
        // Use a persistent arena so tensor MemorySegment slices remain valid
        // after load() returns.  The caller (GGUFFile) is responsible for closing it.
        Arena arena = Arena.ofShared();
        try {
            FileChannel ch = FileChannel.open(path, StandardOpenOption.READ);
            long size = ch.size();
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
            ch.close();
            
            return parse(seg, arena);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }
    
    private static GGUFFile parse(MemorySegment seg, Arena arena) throws IOException {
        long pos = 0;
        
        // Magic
        int magic = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
        pos += 4;
        if (magic != 0x46554747) throw new IOException("Invalid GGUF magic");
        
        // Version
        int version = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
        pos += 4;
        
        // Counts
        long tensorCount = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
        pos += 8;
        long kvCount = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
        pos += 8;
        
        // Metadata
        Map<String, Object> metadata = new LinkedHashMap<>();
        for (long i = 0; i < kvCount; i++) {
            // Read key string
            long keyLen = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
            pos += 8;
            byte[] keyBytes = seg.asSlice(pos, keyLen).toArray(ValueLayout.JAVA_BYTE);
            pos += keyLen;
            String key = new String(keyBytes, StandardCharsets.UTF_8);
            
            // Read value type
            int valType = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
            pos += 4;
            
            Object value = readValue(seg, pos, valType, version);
            pos += getValueSize(seg, pos, valType, version);
            metadata.put(key, value);
        }
        
        // Tensor infos
        Map<String, GGUFTensorInfo> tensors = new LinkedHashMap<>();
        long dataStart = align(pos, 32);
        
        for (long i = 0; i < tensorCount; i++) {
            // Name
            long nameLen = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
            pos += 8;
            byte[] nameBytes = seg.asSlice(pos, nameLen).toArray(ValueLayout.JAVA_BYTE);
            pos += nameLen;
            String name = new String(nameBytes, StandardCharsets.UTF_8);
            
            // Dimensions
            int nDims = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
            pos += 4;
            long[] shape = new long[nDims];
            for (int d = 0; d < nDims; d++) {
                shape[d] = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
                pos += 8;
            }
            
            // Type
            int type = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
            pos += 4;
            
            // Offset
            long offset = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
            pos += 8;
            
            // Create tensor
            long nElements = 1;
            for (long d : shape) nElements *= d;
            long byteSize = tensorBytes(type, nElements);
            MemorySegment data = seg.asSlice(dataStart + offset, byteSize);
            tensors.put(name, new GGUFTensorInfo(name, shape, GgmlType.fromId(type), offset, data));
        }
        
        return new GGUFFile(version, metadata, tensors);
    }
    
    private static Object readValue(MemorySegment seg, long pos, int type, int version) {
        return switch (type) {
            case 0 -> seg.get(ValueLayout.JAVA_BYTE, pos) & 0xFF;           // UINT8
            case 1 -> seg.get(ValueLayout.JAVA_BYTE, pos);                  // INT8
            case 2 -> seg.get(ValueLayout.JAVA_SHORT_UNALIGNED, pos) & 0xFFFF; // UINT16
            case 3 -> seg.get(ValueLayout.JAVA_SHORT_UNALIGNED, pos);       // INT16
            case 4 -> seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos) & 0xFFFFFFFFL; // UINT32
            case 5 -> seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);         // INT32
            case 6 -> seg.get(ValueLayout.JAVA_FLOAT_UNALIGNED, pos);       // FLOAT32
            case 7 -> seg.get(ValueLayout.JAVA_BYTE, pos) != 0;              // BOOL
            case 8 -> readString(seg, pos);                                  // STRING
            case 9 -> readArray(seg, pos, version);                          // ARRAY
            case 10 -> seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);        // UINT64
            case 11 -> seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);        // INT64
            case 12 -> seg.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, pos);      // FLOAT64
            default -> throw new IllegalStateException("Unknown type: " + type);
        };
    }
    
    private static long getValueSize(MemorySegment seg, long pos, int type, int version) {
        return switch (type) {
            case 0,1,7 -> 1;
            case 2,3 -> 2;
            case 4,5,6 -> 4;
            case 8 -> 8 + getLen8(seg, pos);
            case 9 -> {
                int elemType = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
                long count = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos + 4);
                long headerSize = 12;
                long totalSize = headerSize;
                long cur = pos + headerSize;
                for (long i = 0; i < count; i++) {
                    long sz = getValueSize(seg, cur, elemType, version);
                    totalSize += sz;
                    cur += sz;
                }
                yield totalSize;
            }
            case 10,11 -> 8;
            case 12 -> 8;
            default -> 0;
        };
    }
    
    private static long getLen8(MemorySegment seg, long pos) {
        return seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
    }
    
    private static String readString(MemorySegment seg, long pos) {
        long len = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos);
        byte[] bytes = seg.asSlice(pos + 8, len).toArray(ValueLayout.JAVA_BYTE);
        return new String(bytes, StandardCharsets.UTF_8);
    }
    
    private static List<Object> readArray(MemorySegment seg, long pos, int version) {
        int elemType = seg.get(ValueLayout.JAVA_INT_UNALIGNED, pos);
        long count = seg.get(ValueLayout.JAVA_LONG_UNALIGNED, pos + 4);
        List<Object> list = new ArrayList<>();
        long cur = pos + 12;
        for (long i = 0; i < count; i++) {
            list.add(readValue(seg, cur, elemType, version));
            cur += getValueSize(seg, cur, elemType, version);
        }
        return list;
    }
    
    private static long align(long pos, int align) {
        long rem = pos % align;
        return rem == 0 ? pos : pos + (align - rem);
    }
    
    private static long tensorBytes(int type, long n) {
        return switch (type) {
            case 0 -> n * 4;           // F32
            case 1 -> n * 2;           // F16
            case 2 -> (n / 32) * 18;   // Q4_0
            case 3 -> (n / 32) * 20;   // Q4_1
            case 6 -> (n / 32) * 22;   // Q5_0
            case 7 -> (n / 32) * 24;   // Q5_1
            case 8 -> (n / 32) * 34;   // Q8_0
            case 10 -> (n / 256) * 84; // Q2_K
            case 11 -> (n / 256) * 110; // Q3_K
            case 12 -> (n / 256) * 144; // Q4_K
            case 13 -> (n / 256) * 176; // Q5_K
            case 14 -> (n / 256) * 210; // Q6_K
            default -> throw new IllegalArgumentException("Unknown type: " + type);
        };
    }
}
