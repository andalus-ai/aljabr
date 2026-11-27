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
import java.lang.reflect.Proxy;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Verifies cache retirement does not invalidate memory held by caller leases. */
class SafetensorLoadCacheLeaseTest {

    @Test
    void evictionDefersMappedMemoryCloseUntilCallerLeaseCloses() {
        SafetensorLoadCache cache = cache(true, 1);
        SafetensorLoadResult first = result("cache-first.safetensors", 11.5f);
        SafetensorLoadResult second = result("cache-second.safetensors", 22.5f);
        SafetensorFFMLoader loader = loader(Map.of(
                Path.of("first.safetensors"), first,
                Path.of("second.safetensors"), second));

        SafetensorLoadCache.LoadLease firstLease =
                cache.acquire(Path.of("first.safetensors"), loader);
        SafetensorTensor tensor = firstLease.result().tensor("weight");

        SafetensorLoadCache.LoadLease secondLease =
                cache.acquire(Path.of("second.safetensors"), loader);

        assertThat(cache.size()).isEqualTo(1);
        assertThat(first.isClosed()).isFalse();
        assertThat(tensor.getFloat(0)).isEqualTo(11.5f);

        firstLease.close();

        assertThat(first.isClosed()).isTrue();
        assertThatThrownBy(() -> tensor.getFloat(0)).isInstanceOf(IllegalStateException.class);
        secondLease.close();
    }

    @Test
    void invalidationDefersCloseUntilCallerLeaseCloses() {
        SafetensorLoadCache cache = cache(true, 2);
        SafetensorLoadResult result = result("cache-invalidate.safetensors", 7.25f);
        SafetensorLoadCache.LoadLease lease = cache.acquire(
                Path.of("invalidate.safetensors"),
                loader(Map.of(Path.of("invalidate.safetensors"), result)));
        SafetensorTensor tensor = lease.result().tensor("weight");

        cache.invalidate(Path.of("invalidate.safetensors"));

        assertThat(cache.size()).isZero();
        assertThat(result.isClosed()).isFalse();
        assertThat(tensor.getFloat(0)).isEqualTo(7.25f);

        lease.close();

        assertThat(result.isClosed()).isTrue();
    }

    @Test
    void shutdownRetiresCacheOwnerButPreservesActiveLease() {
        SafetensorLoadCache cache = cache(true, 2);
        SafetensorLoadResult result = result("cache-shutdown.safetensors", 3.5f);
        SafetensorLoadCache.LoadLease lease = cache.acquire(
                Path.of("shutdown.safetensors"),
                loader(Map.of(Path.of("shutdown.safetensors"), result)));

        cache.invalidateAll();

        assertThat(result.isClosed()).isFalse();
        assertThat(lease.result().tensor("weight").getFloat(0)).isEqualTo(3.5f);

        lease.close();

        assertThat(result.isClosed()).isTrue();
    }

    @Test
    void disabledCacheReturnsLeaseThatOwnsItsResource() {
        SafetensorLoadCache cache = cache(false, 1);
        SafetensorLoadResult result = result("cache-disabled.safetensors", 5.0f);
        SafetensorLoadCache.LoadLease lease = cache.acquire(
                Path.of("disabled.safetensors"),
                loader(Map.of(Path.of("disabled.safetensors"), result)));

        assertThat(cache.size()).isZero();
        assertThat(result.isClosed()).isFalse();
        assertThat(lease.result().tensor("weight").getFloat(0)).isEqualTo(5.0f);

        lease.close();

        assertThat(result.isClosed()).isTrue();
    }

    private static SafetensorLoadCache cache(boolean enabled, int maxSize) {
        SafetensorLoaderConfig.Cache cacheConfig = (SafetensorLoaderConfig.Cache) Proxy.newProxyInstance(
                SafetensorLoaderConfig.Cache.class.getClassLoader(),
                new Class<?>[] { SafetensorLoaderConfig.Cache.class },
                (proxy, method, args) -> switch (method.getName()) {
                    case "enabled" -> enabled;
                    case "maxSize" -> maxSize;
                    case "ttlSeconds" -> 0L;
                    default -> throw new UnsupportedOperationException(method.getName());
                });
        SafetensorLoaderConfig config = (SafetensorLoaderConfig) Proxy.newProxyInstance(
                SafetensorLoaderConfig.class.getClassLoader(),
                new Class<?>[] { SafetensorLoaderConfig.class },
                (proxy, method, args) -> {
                    if (method.getName().equals("cache")) return cacheConfig;
                    throw new UnsupportedOperationException(method.getName());
                });
        SafetensorLoadCache cache = new SafetensorLoadCache();
        cache.config = config;
        cache.metrics = new SafetensorMetrics();
        return cache;
    }

    private static SafetensorFFMLoader loader(Map<Path, SafetensorLoadResult> results) {
        return new SafetensorFFMLoader() {
            @Override
            public SafetensorLoadResult load(Path path) {
                SafetensorLoadResult result = results.get(path);
                if (result == null) throw new IllegalArgumentException("Unexpected test path: " + path);
                return result;
            }
        };
    }

    private static SafetensorLoadResult result(String fileName, float value) {
        Path path = Path.of(fileName);
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
                Long.BYTES + headerBytes.length, value);
        SafetensorHeader header = SafetensorHeaderParser.create(new ObjectMapper()).parse(file, path);
        return new SafetensorLoadResult(path, header, file, arena, SafetensorLoadResult.LoadMode.COPY);
    }
}
