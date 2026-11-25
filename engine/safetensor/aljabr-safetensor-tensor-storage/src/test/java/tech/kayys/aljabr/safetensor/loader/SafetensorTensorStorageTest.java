package tech.kayys.aljabr.safetensor.loader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.safetensor.tensor.SafetensorTensorStorage;

/** Verifies zero-copy binding and independent storage ownership. */
class SafetensorTensorStorageTest {

    @Test
    void storageKeepsMappedBytesAliveAfterSessionOwnerCloses() {
        Arena arena = Arena.ofShared();
        byte[] headerBytes = """
                {"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);
        MemorySegment file = arena.allocate(Long.BYTES + headerBytes.length + 2L * Float.BYTES);
        MemorySegment.copy(prefix.array(), 0, file, ValueLayout.JAVA_BYTE, 0, Long.BYTES);
        MemorySegment.copy(headerBytes, 0, file, ValueLayout.JAVA_BYTE, Long.BYTES, headerBytes.length);
        long payloadOffset = Long.BYTES + headerBytes.length;
        file.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), payloadOffset, 3.5f);
        file.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), payloadOffset + Float.BYTES, 7.25f);

        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper())
                .parse(file, Path.of("storage-test.safetensors"));
        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("storage-test.safetensors"), header, file, arena, SafetensorLoadResult.LoadMode.COPY);
        SafetensorShardLoader.SafetensorShardSession session =
                SafetensorShardLoader.SafetensorShardSession.singleFile(result);
        SafetensorTensor original = session.tensor("weight");
        SafetensorTensorStorage storage = SafetensorTensorStorage.open(session, "weight");

        assertThat(storage.segment()).isSameAs(original.segment());
        assertThat(storage.sizeBytes()).isEqualTo(8);
        assertThat(storage.shape().dims()).containsExactly(2);
        assertThat(storage.dtype()).isEqualTo(DType.F32);

        storage.retain();
        session.close();
        storage.release();

        assertThat(storage.segment().get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), 0))
                .isEqualTo(3.5f);

        storage.release();

        assertThatThrownBy(storage::segment).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(storage::retain).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> original.segment()).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void unsupportedDtypesAreRejectedWithoutReinterpretation() {
        Arena arena = Arena.ofShared();
        byte[] headerBytes = """
                {"weight":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);
        MemorySegment file = arena.allocate(Long.BYTES + headerBytes.length + Double.BYTES);
        MemorySegment.copy(prefix.array(), 0, file, ValueLayout.JAVA_BYTE, 0, Long.BYTES);
        MemorySegment.copy(headerBytes, 0, file, ValueLayout.JAVA_BYTE, Long.BYTES, headerBytes.length);
        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper())
                .parse(file, Path.of("unsupported-dtype.safetensors"));
        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("unsupported-dtype.safetensors"), header, file, arena, SafetensorLoadResult.LoadMode.COPY);
        var session = SafetensorShardLoader.SafetensorShardSession.singleFile(result);
        var storage = SafetensorTensorStorage.open(session, "weight");

        assertThatThrownBy(storage::dtype)
                .isInstanceOf(UnsupportedOperationException.class)
                .hasMessageContaining("F64");

        storage.release();
        session.close();
    }
}
