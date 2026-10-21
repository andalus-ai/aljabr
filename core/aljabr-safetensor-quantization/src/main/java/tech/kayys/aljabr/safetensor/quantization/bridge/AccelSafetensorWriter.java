package tech.kayys.aljabr.safetensor.quantization.bridge;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import tech.kayys.aljabr.safetensor.core.tensor.AccelTensor;
import org.jboss.logging.Logger;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Map;

/**
 * High-performance Safetensor writer optimized for AccelTensor (FFM MemorySegment).
 * @author bhangun
 */
public class AccelSafetensorWriter {

    private static final Logger log = Logger.getLogger(AccelSafetensorWriter.class);
    private static final ObjectMapper mapper = new ObjectMapper();

    /**
     * Save a map of AccelTensors to a Safetensor file.
     * 
     * @param path destination path
     * @param weights tensors to save
     * @throws IOException if writing fails
     */
    public void save(Path path, Map<String, AccelTensor> weights) throws IOException {
        ObjectNode header = mapper.createObjectNode();
        long currentOffset = 0;

        // 1. Build JSON metadata
        for (Map.Entry<String, AccelTensor> entry : weights.entrySet()) {
            String name = entry.getKey();
            AccelTensor tensor = entry.getValue();

            ObjectNode tensorMeta = header.putObject(name);
            tensorMeta.put("dtype", "F32"); // For now assuming F32, will expand for Quantized
            
            ArrayNode shapeArr = tensorMeta.putArray("shape");
            for (long d : tensor.shape()) {
                shapeArr.add(d);
            }

            long byteSize = tensor.numel() * 4;
            ArrayNode offsets = tensorMeta.putArray("data_offsets");
            offsets.add(currentOffset);
            offsets.add(currentOffset + byteSize);
            
            currentOffset += byteSize;
        }

        // 2. Prepare Header
        String jsonString = header.toString();
        byte[] headerBytes = jsonString.getBytes(StandardCharsets.UTF_8);
        int headerLen = headerBytes.length;
        
        // Alignment
        int alignmentPadding = (int) ((8 - (8 + headerLen) % 8) % 8);
        if (alignmentPadding > 0) {
            jsonString = jsonString + " ".repeat(alignmentPadding);
            headerBytes = jsonString.getBytes(StandardCharsets.UTF_8);
            headerLen = headerBytes.length;
        }

        // 3. Write binary data
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             FileChannel channel = fos.getChannel()) {
            
            // Header length
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            lenBuf.putLong(headerLen);
            lenBuf.flip();
            channel.write(lenBuf);

            // Header JSON
            channel.write(ByteBuffer.wrap(headerBytes));

            // Weights Data (Direct from MemorySegment)
            for (AccelTensor tensor : weights.values()) {
                // FFM to NIO channel write
                channel.write(tensor.dataSegment().asByteBuffer());
            }
        }
        
        log.infof("Safetensor saved: %s (%d tensors)", path, weights.size());
    }
}
