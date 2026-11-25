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

/** Verifies that retained consumers keep mapped tensor memory alive. */
class SafetensorShardSessionLeaseTest {

    @Test
    void retainedLeaseDefersResourceCloseUntilLastRelease() {
        Arena arena = Arena.ofShared();
        byte[] headerBytes = """
                {"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);
        MemorySegment file = arena.allocate(Long.BYTES + headerBytes.length + Float.BYTES);
        MemorySegment.copy(prefix.array(), 0, file, ValueLayout.JAVA_BYTE, 0, Long.BYTES);
        MemorySegment.copy(headerBytes, 0, file, ValueLayout.JAVA_BYTE, Long.BYTES, headerBytes.length);
        file.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
                Long.BYTES + headerBytes.length, 42.5f);

        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper())
                .parse(file, Path.of("lease-test.safetensors"));
        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("lease-test.safetensors"), header, file, arena, SafetensorLoadResult.LoadMode.COPY);
        SafetensorShardLoader.SafetensorShardSession session =
                SafetensorShardLoader.SafetensorShardSession.singleFile(result);

        SafetensorShardLoader.SafetensorShardSession.SessionLease lease = session.retain();
        SafetensorTensor tensor = session.tensor("weight");

        session.close();

        assertThat(session.resourcesClosed()).isFalse();
        assertThat(tensor.getFloat(0)).isEqualTo(42.5f);
        assertThatThrownBy(() -> session.tensor("weight"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");

        lease.close();

        assertThat(lease.isClosed()).isTrue();
        assertThat(session.resourcesClosed()).isTrue();
        assertThatThrownBy(() -> tensor.getFloat(0)).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void leaseCloseIsIdempotentAndSessionCloseIsIdempotent() {
        Arena arena = Arena.ofShared();
        byte[] headerBytes = """
                {"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);
        MemorySegment file = arena.allocate(Long.BYTES + headerBytes.length + Float.BYTES);
        MemorySegment.copy(prefix.array(), 0, file, ValueLayout.JAVA_BYTE, 0, Long.BYTES);
        MemorySegment.copy(headerBytes, 0, file, ValueLayout.JAVA_BYTE, Long.BYTES, headerBytes.length);
        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper())
                .parse(file, Path.of("lease-idempotent.safetensors"));
        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("lease-idempotent.safetensors"), header, file, arena, SafetensorLoadResult.LoadMode.COPY);
        SafetensorShardLoader.SafetensorShardSession session =
                SafetensorShardLoader.SafetensorShardSession.singleFile(result);

        var lease = session.retain();
        session.close();
        session.close();
        lease.close();
        lease.close();

        assertThat(session.resourcesClosed()).isTrue();
        assertThatThrownBy(session::retain)
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }
}
