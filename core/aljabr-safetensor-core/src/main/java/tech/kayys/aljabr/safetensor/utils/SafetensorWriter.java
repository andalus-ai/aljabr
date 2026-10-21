package tech.kayys.aljabr.safetensor.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;

/**
 * Utility for writing tensors in the Safetensor format.
 * <p>
 * Safetensors format:
 * <ul>
 * <li><b>Header Length:</b> 8-byte Little-Endian unsigned 64-bit integer N</li>
 * <li><b>Header:</b> N bytes of UTF-8 encoded JSON string</li>
 * <li><b>Data:</b> Raw binary data stored at offsets specified in the
 * header</li>
 * </ul>
 * @author bhangun
 */
public class SafetensorWriter {

    private static final Gson GSON = new GsonBuilder().create();

    /**
     * Save a map of tensors to a Safetensor file.
     *
     * @param path    the destination path
     * @param tensors named tensors to save
     * @throws IOException if writing fails
     */
    public static void save(Path path, Map<String, AccelTensor> tensors) throws IOException {
        JsonObject header = new JsonObject();
        long currentOffset = 0;

        // 1. Build JSON metadata for each tensor
        for (Map.Entry<String, AccelTensor> entry : tensors.entrySet()) {
            String name = entry.getKey();
            AccelTensor tensor = entry.getValue();

            JsonObject tensorMeta = new JsonObject();

            // Map DType to Safetensor string
            String dtypeStr = switch (tensor.quantType()) {
                case F32 -> "F32";
                case F16 -> "F16";
                case BF16 -> "BF16";
                case INT8 -> "I8";
                default -> "F32";
            };
            tensorMeta.addProperty("dtype", dtypeStr);

            com.google.gson.JsonArray shapeArr = new com.google.gson.JsonArray();
            for (long d : tensor.shape()) {
                shapeArr.add(d);
            }
            tensorMeta.add("shape", shapeArr);

            long byteSize = logicalByteSize(tensor);

            com.google.gson.JsonArray offsets = new com.google.gson.JsonArray();
            offsets.add(currentOffset);
            offsets.add(currentOffset + byteSize);
            tensorMeta.add("data_offsets", offsets);

            header.add(name, tensorMeta);
            currentOffset += byteSize;
        }

        // 2. Generate JSON string and handle alignment
        String jsonString = GSON.toJson(header);
        byte[] headerBytes = jsonString.getBytes(StandardCharsets.UTF_8);
        int headerLen = headerBytes.length;

        // Standard Safetensors recommendation: data segment should start at 8-byte
        // aligned offset.
        // data_start = 8 (for header link) + headerLen
        int alignmentPadding = (int) ((8 - (8 + headerLen) % 8) % 8);
        if (alignmentPadding > 0) {
            // Pad the JSON with space characters to reach alignment
            StringBuilder sb = new StringBuilder(jsonString);
            for (int i = 0; i < alignmentPadding; i++) {
                sb.append(" ");
            }
            headerBytes = sb.toString().getBytes(StandardCharsets.UTF_8);
            headerLen = headerBytes.length;
        }

        // 3. Write to file
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
                FileChannel channel = fos.getChannel()) {

            // Write 8-byte LE header length
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            lenBuf.putLong(headerLen);
            lenBuf.flip();
            while (lenBuf.hasRemaining()) {
                channel.write(lenBuf);
            }

            // Write JSON header
            channel.write(ByteBuffer.wrap(headerBytes));

            // Write tensor data segments directly from the backing MemorySegment.
            for (AccelTensor tensor : tensors.values()) {
                ByteBuffer dataBuf = tensor.dataSegment()
                        .asSlice(0L, logicalByteSize(tensor))
                        .asByteBuffer()
                        .duplicate()
                        .order(ByteOrder.LITTLE_ENDIAN);
                while (dataBuf.hasRemaining()) {
                    channel.write(dataBuf);
                }
            }
        }
    }

    private static long logicalByteSize(AccelTensor tensor) {
        long numel = tensor.numel();
        return switch (tensor.quantType()) {
            case F16, BF16 -> numel * 2L;
            case INT8, FP8 -> numel;
            case INT4, NF4 -> (numel + 1L) / 2L;
            case F32 -> numel * 4L;
        };
    }
}
