package tech.kayys.aljabr.integration.kv;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.Shape;
import tech.kayys.aljabr.core.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class KVCacheTensorIntegrationTest {

    @Test
    public void tensorBackedByCacheAndUnpinsOnRelease() throws Exception {
        // Instantiate PagedKVCache reflectively to avoid compile-time dependency
        Class<?> cacheCls = Class.forName("tech.kayys.aljabr.cache.PagedKVCache");
        Object cache = cacheCls.getConstructor(long.class).newInstance(1024L * 1024L);

        long key = 123L;
        int len = 256;
        float[] data = new float[len];
        for (int i = 0; i < len; i++)
            data[i] = i + 1;
        cacheCls.getMethod("put", long.class, float[].class).invoke(cache, key, data);

        // Use null backend — adapter only stores the backend reference and does not
        // call it in this test
        Shape shape = new Shape(new long[] { len });

        // create tensor from cache (adapter pins page and arranges unpin on
        // buffer.release())
        Tensor t = KVCacheTensorAdapter.tensorFromCache(cache, key, shape, DType.F32, null);
        Assertions.assertNotNull(t);
        Assertions.assertTrue(t instanceof DefaultTensor);

        DefaultTensor dt = (DefaultTensor) t;
        MemorySegment seg = dt.buffer().segment();
        // read few values to ensure zero-copy view is valid
        float v0 = seg.get(ValueLayout.JAVA_FLOAT, 0L);
        float vmid = seg.get(ValueLayout.JAVA_FLOAT, (len / 2) * 4L);
        Assertions.assertEquals(1.0f, v0);
        Assertions.assertEquals((float) (len / 2 + 1), vmid);

        // add eviction listener for this key; after release, eviction should be
        // possible
        CountDownLatch latch = new CountDownLatch(1);
        java.lang.reflect.Method addListener = cacheCls.getMethod("addEvictionListener",
                java.util.function.Consumer.class);
        addListener.invoke(cache, (Consumer<Long>) (k -> {
            if (k == key)
                latch.countDown();
        }));

        // release tensor which should trigger buffer.release -> cache.unpin via adapter
        // callback
        dt.release();

        // fill cache to force eviction
        java.lang.reflect.Method putM = cacheCls.getMethod("put", long.class, float[].class);
        for (long k = 0; k < 1000; k++) {
            float[] d = new float[1024];
            putM.invoke(cache, k + 1000, d);
        }

        boolean evicted = latch.await(3, TimeUnit.SECONDS);
        Assertions.assertTrue(evicted, "eviction should happen after tensor.release() unpins the page");
    }
}
