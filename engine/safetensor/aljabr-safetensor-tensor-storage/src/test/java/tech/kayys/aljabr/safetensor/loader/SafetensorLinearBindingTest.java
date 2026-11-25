package tech.kayys.aljabr.safetensor.loader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.nn.Linear;
import tech.kayys.aljabr.core.memory.HostAccessibleStorage;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.safetensor.tensor.SafetensorLinearBinding;

class SafetensorLinearBindingTest {
    @Test
    void createsLinearModuleWithoutCopyAndRetainsWeightUntilModuleClose() {
        Arena arena = Arena.ofShared();
        byte[] headerBytes = """
                {"projection.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}
                """.strip().getBytes(StandardCharsets.UTF_8);
        ByteBuffer prefix = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        prefix.putLong(headerBytes.length);
        MemorySegment file = arena.allocate(Long.BYTES + headerBytes.length + 4L * Float.BYTES);
        MemorySegment.copy(prefix.array(), 0, file, ValueLayout.JAVA_BYTE, 0, Long.BYTES);
        MemorySegment.copy(headerBytes, 0, file, ValueLayout.JAVA_BYTE, Long.BYTES, headerBytes.length);
        long payloadOffset = Long.BYTES + headerBytes.length;
        for (int i = 0; i < 4; i++) {
            file.set(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN),
                    payloadOffset + (long) i * Float.BYTES, i + 1f);
        }

        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper())
                .parse(file, Path.of("linear-binding.safetensors"));
        SafetensorLoadResult result = new SafetensorLoadResult(
                Path.of("linear-binding.safetensors"), header, file, arena, SafetensorLoadResult.LoadMode.COPY);
        var session = SafetensorShardLoader.SafetensorShardSession.singleFile(result);
        Linear linear = SafetensorLinearBinding.openLinear(session, "projection.weight", cpuBackend());

        assertThat(linear.inFeatures()).isEqualTo(2);
        assertThat(linear.outFeatures()).isEqualTo(2);
        assertThat(linear.parameters()).containsKey("weight");
        HostAccessibleStorage storage = (HostAccessibleStorage) linear.parameters().get("weight").storage();
        assertThat(storage.segment().get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), 0)).isEqualTo(1f);

        session.close();
        assertThat(storage.segment().get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), 3L * Float.BYTES))
                .isEqualTo(4f);

        linear.close();
        assertThatThrownBy(storage::segment).isInstanceOf(IllegalStateException.class);
    }

    private static ComputeBackend cpuBackend() {
        return (ComputeBackend) Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] { ComputeBackend.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("deviceType") && method.getParameterCount() == 0) {
                        return DeviceType.CPU;
                    }
                    if (method.getDeclaringClass() == Object.class) {
                        return switch (method.getName()) {
                            case "toString" -> "TestCpuBackend";
                            case "hashCode" -> System.identityHashCode(proxy);
                            case "equals" -> proxy == args[0];
                            default -> null;
                        };
                    }
                    throw new UnsupportedOperationException("Unexpected backend call: " + method.getName());
                });
    }
}
