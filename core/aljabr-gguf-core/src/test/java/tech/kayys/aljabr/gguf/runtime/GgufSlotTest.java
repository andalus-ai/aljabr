package tech.kayys.aljabr.gguf.runtime;

import org.junit.jupiter.api.Test;
import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;
import tech.kayys.aljabr.gguf.runtime.GgufTensorOps.PreparedMatrix;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static tech.kayys.aljabr.gguf.runtime.GgufKFx.writeQ4KBlockWithAllScalesAndMins;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class GgufSlotTest {
    @Test
    void repeatedMatVecCacheHitUsesRecentMatrixWithoutCacheProbe() {
        GgufSlot.clearRecentSlotCaches();
        assertEquals(0, GgufSlot.recentMatrixFastCacheSize());
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> new Matrix(320L));

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KBlockWithAllScalesAndMins(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.one", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufKey key = GgufKey.from(tensor);

            Matrix matrix = slot.forMatVec(model, tensor, key, 320L);
            assertEquals(1, GgufSlot.recentMatrixFastCacheSize());
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            for (int i = 0; i < 4; i++) {
                assertSame(matrix, slot.forMatVec(model, tensor, key, 320L));
            }
            assertEquals(0, cache.gets);
            assertEquals(1, GgufSlot.recentMatrixFastCacheSize());
        } finally {
            GgufSlot.clearRecentSlotCaches();
        }
    }

    @Test
    void alternatingMatVecCacheHitsUseRecentMatricesWithoutCacheProbe() {
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> new Matrix(320L));

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);
            GgufKey firstKey = GgufKey.from(firstTensor);
            GgufKey secondKey = GgufKey.from(secondTensor);

            Matrix first = slot.forMatVec(model, firstTensor, firstKey, 640L);
            Matrix second = slot.forMatVec(model, secondTensor, secondKey, 640L);
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            for (int i = 0; i < 4; i++) {
                assertSame(first, slot.forMatVec(model, firstTensor, firstKey, 640L));
                assertSame(second, slot.forMatVec(model, secondTensor, secondKey, 640L));
            }
            assertEquals(0, cache.gets);
        }
    }

    @Test
    void rotatingMatVecCacheHitsUseRecentMatrixTableWithoutCacheProbe() {
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> new Matrix(320L));

        int tensorCount = 6;
        List<GGUFTensorInfo> tensors = rotatingQ4Tensors(tensorCount);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(tensorCount * 144L);
            for (int index = 0; index < tensorCount; index++) {
                writeQ4KBlockWithAllScalesAndMins(segment.asSlice(index * 144L, 144));
            }
            GGUFModel model = new GGUFModel(3, Map.of(), tensors, 0, segment, null);
            long maxBytes = tensorCount * 320L;
            Matrix[] matrices = new Matrix[tensorCount];
            for (int index = 0; index < tensorCount; index++) {
                GGUFTensorInfo tensor = tensors.get(index);
                matrices[index] = slot.forMatVec(model, tensor, GgufKey.from(tensor), maxBytes);
            }
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            for (int pass = 0; pass < 4; pass++) {
                for (int index = 0; index < tensorCount; index++) {
                    GGUFTensorInfo tensor = tensors.get(index);
                    assertSame(matrices[index], slot.forMatVec(model, tensor, GgufKey.from(tensor), maxBytes));
                }
            }
            assertEquals(0, cache.gets);
        }
    }

    @Test
    void matVecAdmissionMissAvoidsExtraCacheProbeBeforePreparing() {
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> new Matrix(320L));

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            slot.forMatVec(model, firstTensor, GgufKey.from(firstTensor), 640L);
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            slot.forMatVec(model, secondTensor, GgufKey.from(secondTensor), 640L);

            assertEquals(3, cache.gets);
        }
    }

    @Test
    void rejectedMatVecAdmissionDoesNotPrepareMatrix() {
        AtomicInteger prepares = new AtomicInteger();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                CountingCache::new,
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(320L);
                });

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);

            slot.forMatVec(model, firstTensor, GgufKey.from(firstTensor), 320L);
            assertNull(slot.forMatVec(model, secondTensor, GgufKey.from(secondTensor), 320L));

            assertEquals(1, prepares.get());
        }
    }

    @Test
    void oversizedPreparedMatrixIsReturnedOnceThenRememberedAsRejected() {
        AtomicInteger prepares = new AtomicInteger();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                CountingCache::new,
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(640L);
                });

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KBlockWithAllScalesAndMins(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.oversized", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);
            GgufKey key = GgufKey.from(tensor);

            assertEquals(new Matrix(640L), slot.forMatVec(model, tensor, key, 320L));
            assertNull(slot.forMatVec(model, tensor, key, 320L));

            assertEquals(1, prepares.get());
            assertEquals(0, slot.size(model));
        }
    }

    @Test
    void repeatedRejectedMatVecAdmissionUsesRecentRejectionWithoutCacheProbe() {
        GgufSlot.clearRecentSlotCaches();
        assertEquals(0, GgufSlot.recentRejectFastCacheSize());
        AtomicInteger prepares = new AtomicInteger();
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(320L);
                });

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);
            GgufKey secondKey = GgufKey.from(secondTensor);

            slot.forMatVec(model, firstTensor, GgufKey.from(firstTensor), 320L);
            assertNull(slot.forMatVec(model, secondTensor, secondKey, 320L));
            assertEquals(1, GgufSlot.recentRejectFastCacheSize());
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            for (int i = 0; i < 4; i++) {
                assertNull(slot.forMatVec(model, secondTensor, secondKey, 320L));
            }
            assertEquals(0, cache.gets);
            assertEquals(1, prepares.get());
            assertEquals(1, GgufSlot.recentRejectFastCacheSize());
        } finally {
            GgufSlot.clearRecentSlotCaches();
        }
    }

    @Test
    void rotatingRejectedMatVecAdmissionsUseRecentRejectTableWithoutCacheProbe() {
        AtomicInteger prepares = new AtomicInteger();
        AtomicReference<CountingCache> cacheRef = new AtomicReference<>();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                () -> {
                    CountingCache cache = new CountingCache();
                    cacheRef.set(cache);
                    return cache;
                },
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(320L);
                });

        int tensorCount = 7;
        List<GGUFTensorInfo> tensors = rotatingQ4Tensors(tensorCount);
        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(tensorCount * 144L);
            for (int index = 0; index < tensorCount; index++) {
                writeQ4KBlockWithAllScalesAndMins(segment.asSlice(index * 144L, 144));
            }
            GGUFModel model = new GGUFModel(3, Map.of(), tensors, 0, segment, null);
            long maxBytes = 320L;

            GGUFTensorInfo admitted = tensors.get(0);
            slot.forMatVec(model, admitted, GgufKey.from(admitted), maxBytes);
            for (int index = 1; index < tensorCount; index++) {
                GGUFTensorInfo tensor = tensors.get(index);
                assertNull(slot.forMatVec(model, tensor, GgufKey.from(tensor), maxBytes));
            }
            CountingCache cache = cacheRef.get();
            cache.gets = 0;

            for (int pass = 0; pass < 4; pass++) {
                for (int index = 1; index < tensorCount; index++) {
                    GGUFTensorInfo tensor = tensors.get(index);
                    assertNull(slot.forMatVec(model, tensor, GgufKey.from(tensor), maxBytes));
                }
            }
            assertEquals(0, cache.gets);
            assertEquals(1, prepares.get());
        }
    }

    @Test
    void clearInvalidatesRecentRejectedMatVecAdmission() {
        AtomicInteger prepares = new AtomicInteger();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                CountingCache::new,
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(320L);
                });

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(2L * 144);
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(0, 144));
            writeQ4KBlockWithAllScalesAndMins(segment.asSlice(144, 144));
            GGUFTensorInfo firstTensor = new GGUFTensorInfo("q4.first", new long[]{256, 1}, 12, 0, 144);
            GGUFTensorInfo secondTensor = new GGUFTensorInfo("q4.second", new long[]{256, 1}, 12, 144, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(firstTensor, secondTensor), 0, segment, null);
            GgufKey secondKey = GgufKey.from(secondTensor);

            slot.forMatVec(model, firstTensor, GgufKey.from(firstTensor), 320L);
            assertNull(slot.forMatVec(model, secondTensor, secondKey, 320L));
            assertEquals(1, prepares.get());

            assertEquals(1, slot.clear(model));
            assertEquals(new Matrix(320L), slot.forMatVec(model, secondTensor, secondKey, 320L));
            assertEquals(2, prepares.get());
        }
    }

    @Test
    void clearInvalidatesLastModelFastPath() {
        AtomicInteger prepares = new AtomicInteger();
        GgufSlot<Matrix, CountingCache> slot = new GgufSlot<>(
                GgufPreparedCachePolicy.Family.Q4K,
                CountingCache::new,
                (model, tensor) -> {
                    prepares.incrementAndGet();
                    return new Matrix(320L);
                });

        try (Arena arena = Arena.ofShared()) {
            MemorySegment segment = arena.allocate(144);
            writeQ4KBlockWithAllScalesAndMins(segment);
            GGUFTensorInfo tensor = new GGUFTensorInfo("q4.one", new long[]{256, 1}, 12, 0, 144);
            GGUFModel model = new GGUFModel(3, Map.of(), List.of(tensor), 0, segment, null);

            slot.forMatVec(model, tensor, GgufKey.from(tensor), 320L);
            assertEquals(1, slot.size(model));
            assertEquals(320L, slot.bytes(model));

            assertEquals(1, slot.clear(model));
            assertEquals(0, slot.size(model));
            assertEquals(0L, slot.bytes(model));

            slot.forMatVec(model, tensor, GgufKey.from(tensor), 320L);
            assertEquals(2, prepares.get());
        }
    }

    private static final class CountingCache extends GgufPreparedMatrixCache<GgufKey, Matrix> {
        private int gets;

        @Override
        Matrix get(GgufKey key) {
            gets++;
            return super.get(key);
        }
    }

    private record Matrix(long estimatedBytes) implements PreparedMatrix {
    }

    private static List<GGUFTensorInfo> rotatingQ4Tensors(int count) {
        List<GGUFTensorInfo> tensors = new ArrayList<>();
        boolean[] slots = new boolean[256];
        int candidate = 0;
        while (tensors.size() < count) {
            int index = tensors.size();
            GGUFTensorInfo tensor = new GGUFTensorInfo(
                    "q4.rotate." + candidate,
                    new long[]{256, 1},
                    12,
                    index * 144L,
                    144);
            int slot = GgufKey.from(tensor).hashCode() & 255;
            if (!slots[slot]) {
                slots[slot] = true;
                tensors.add(tensor);
            }
            candidate++;
        }
        return tensors;
    }
}
