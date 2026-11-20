package tech.kayys.aljabr.gguf;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.inject.Inject;
import tech.kayys.aljabr.safetensor.loader.SafetensorFFMLoader;
import tech.kayys.aljabr.safetensor.loader.SafetensorHeader;
import tech.kayys.aljabr.safetensor.loader.SafetensorLoadResult;
import tech.kayys.aljabr.safetensor.loader.SafetensorShardIndex;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensor;
import tech.kayys.aljabr.safetensor.loader.SafetensorTensorInfo;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter;
import tech.kayys.aljabr.safetensor.utils.StreamingSafetensorWriter.TensorSpec;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Sequential, lossless Safetensors-to-Safetensors checkpoint copier.
 *
 * <p>It performs a metadata-only preflight, then transfers each tensor's raw
 * bytes in bounded chunks. At most one source shard is mapped at a time and
 * no tensor payload is accumulated in heap memory. This is the first reusable
 * source-to-target pipeline primitive for format conversion.
 */
public final class SafetensorCheckpointCopier {

    private static final long COPY_CHUNK_BYTES = 1024L * 1024L;

    private final SafetensorFFMLoader loader;
    private final ObjectMapper objectMapper;

    @Inject
    public SafetensorCheckpointCopier(SafetensorFFMLoader loader, ObjectMapper objectMapper) {
        this.loader = Objects.requireNonNull(loader, "loader must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
    }

    /**
     * Copy a single-file or indexed multi-shard checkpoint to one Safetensors file.
     *
     * @return summary of the completed copy
     */
    public CopyResult copy(Path input, Path output) throws IOException {
        Objects.requireNonNull(input, "input must not be null");
        Objects.requireNonNull(output, "output must not be null");
        Path source = input.toAbsolutePath().normalize();
        Path target = output.toAbsolutePath().normalize();
        if (source.equals(target)) {
            throw new IllegalArgumentException("Input and output paths must be different");
        }

        SourcePlan sourcePlan = inspect(source);
        List<TensorSpec> specs = new ArrayList<>(sourcePlan.tensors().size());
        long totalBytes = 0L;
        for (SourceTensor tensor : sourcePlan.tensors()) {
            totalBytes = Math.addExact(totalBytes, tensor.info().byteLength());
            long[] shape = tensor.info().shape();
            List<Long> dimensions = new ArrayList<>(shape.length);
            for (long dimension : shape) {
                dimensions.add(dimension);
            }
            specs.add(new TensorSpec(
                    tensor.info().name(),
                    tensor.info().dtype().jsonName(),
                    dimensions,
                    tensor.info().byteLength(),
                    channel -> sourcePlan.access().write(tensor, channel)));
        }

        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temp = Files.createTempFile(parent, "aljabr-" + target.getFileName(), ".partial");
        boolean completed = false;
        try (ShardAccess access = sourcePlan.access()) {
            StreamingSafetensorWriter.write(temp, specs, sourcePlan.fileMetadata());
            try {
                Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING);
            }
            completed = true;
        } finally {
            if (!completed) {
                Files.deleteIfExists(temp);
            }
        }
        return new CopyResult(specs.size(), totalBytes, target);
    }

    private SourcePlan inspect(Path source) throws IOException {
        if (Files.isRegularFile(source)) {
            SafetensorHeader header = loader.loadHeaderOnly(source);
            List<SourceTensor> tensors = header.tensors().values().stream()
                    .map(info -> new SourceTensor(source, info))
                    .toList();
            return new SourcePlan(tensors, header.fileMetadata(), new ShardAccess(loader));
        }
        if (!Files.isDirectory(source)) {
            throw new IOException("Input is not a Safetensors file or directory: " + source);
        }

        if (SafetensorShardIndex.isShardedModel(source)) {
            SafetensorShardIndex index = SafetensorShardIndex.load(
                    SafetensorShardIndex.resolveIndexPath(source), objectMapper);
            Map<String, SafetensorHeader> headers = new LinkedHashMap<>();
            for (String shardName : index.shardFileNames()) {
                headers.put(shardName, loader.loadHeaderOnly(index.shardPath(shardName)));
            }
            index.validateShardHeaders(headers);

            Map<String, String> metadata = mergeMetadata(headers);
            List<SourceTensor> tensors = new ArrayList<>(index.tensorCount());
            for (String shardName : index.shardFileNames()) {
                SafetensorHeader header = headers.get(shardName);
                for (SafetensorTensorInfo info : header.tensors().values()) {
                    tensors.add(new SourceTensor(index.shardPath(shardName), info));
                }
            }
            return new SourcePlan(List.copyOf(tensors), metadata, new ShardAccess(loader));
        }

        try (var files = Files.list(source)) {
            List<Path> candidates = files
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".safetensors"))
                    .sorted()
                    .toList();
            if (candidates.size() != 1) {
                throw new IOException("Expected exactly one .safetensors file in " + source
                        + " but found " + candidates.size());
            }
            return inspect(candidates.get(0));
        }
    }

    private static Map<String, String> mergeMetadata(Map<String, SafetensorHeader> headers) throws IOException {
        Map<String, String> merged = new LinkedHashMap<>();
        for (Map.Entry<String, SafetensorHeader> entry : headers.entrySet()) {
            for (Map.Entry<String, String> metadata : entry.getValue().fileMetadata().entrySet()) {
                String previous = merged.putIfAbsent(metadata.getKey(), metadata.getValue());
                if (previous != null && !previous.equals(metadata.getValue())) {
                    throw new IOException("Conflicting Safetensors file metadata key '"
                            + metadata.getKey() + "' across shards");
                }
            }
        }
        return Map.copyOf(merged);
    }

    public record CopyResult(int tensorCount, long payloadBytes, Path output) {
    }

    private record SourceTensor(Path shard, SafetensorTensorInfo info) {
    }

    private record SourcePlan(
            List<SourceTensor> tensors,
            Map<String, String> fileMetadata,
            ShardAccess access) {
    }

    private static final class ShardAccess implements AutoCloseable {
        private final SafetensorFFMLoader loader;
        private Path openPath;
        private SafetensorLoadResult openResult;

        private ShardAccess(SafetensorFFMLoader loader) {
            this.loader = loader;
        }

        private void write(SourceTensor source, WritableByteChannel output) throws IOException {
            if (!source.shard().equals(openPath)) {
                closeCurrent();
                openResult = loader.load(source.shard());
                openPath = source.shard();
            }
            SafetensorTensor tensor = openResult.tensor(source.info().name());
            MemorySegment segment = tensor.segment();
            long offset = 0L;
            while (offset < segment.byteSize()) {
                long length = Math.min(COPY_CHUNK_BYTES, segment.byteSize() - offset);
                ByteBuffer buffer = segment.asSlice(offset, length).asByteBuffer();
                while (buffer.hasRemaining()) {
                    output.write(buffer);
                }
                offset += length;
            }
        }

        private void closeCurrent() {
            if (openResult != null) {
                openResult.close();
                openResult = null;
                openPath = null;
            }
        }

        @Override
        public void close() {
            closeCurrent();
        }
    }
}
