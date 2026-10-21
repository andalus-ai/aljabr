package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFModel;
import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.foreign.MemorySegment;
import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Model/tensor lookup helpers for GGUF runtime kernels.
 *
 * <p>This class keeps bounds-checked tensor slicing close to the runtime so
 * callers do not repeat offset arithmetic against the shared model segment.</p>
 * @author bhangun
 */
final class GgufTensorData {
    private static final int RECENT_SLICE_SLOTS = 256;
    private static final int RECENT_SLICE_MASK = RECENT_SLICE_SLOTS - 1;
    private static final ThreadLocal<RecentSlices> RECENT_SLICES = ThreadLocal.withInitial(RecentSlices::new);
    private static final Map<GGUFModel, ModelDataCache> MODEL_DATA = new WeakHashMap<>();
    private static volatile LastModelDataCache lastModelData = LastModelDataCache.empty();

    private GgufTensorData() {
    }

    static GGUFTensorInfo findTensor(GGUFModel model, String name) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(name, "name");
        GGUFTensorInfo tensor = cacheFor(model).tensor(model, name);
        if (tensor == null) {
            throw new IllegalArgumentException("Tensor not found: " + name);
        }
        return tensor;
    }

    static MemorySegment tensorData(GGUFModel model, GGUFTensorInfo tensor) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(tensor, "tensor");
        RecentSlices recent = RECENT_SLICES.get();
        MemorySegment slice = recent.get(model, tensor);
        if (slice != null) {
            return slice;
        }
        ModelDataCache cache = cachedModelCache(model);
        if (cache == null) {
            cache = cacheFor(model);
        }
        slice = cache.slice(model, tensor);
        recent.put(model, tensor, slice);
        return slice;
    }

    static int recentSliceCacheSize() {
        return RECENT_SLICES.get().size();
    }

    static int recentSliceFastCacheSize() {
        return RECENT_SLICES.get().fastSize();
    }

    static void clearRecentSliceCache() {
        RECENT_SLICES.get().clear();
    }

    private static ModelDataCache cachedModelCache(GGUFModel model) {
        LastModelDataCache last = lastModelData;
        GGUFModel cachedModel = last.model().get();
        if (cachedModel == model) {
            return last.cache();
        }
        if (cachedModel == null && last.cache() != null) {
            lastModelData = LastModelDataCache.empty();
        }
        return null;
    }

    private static ModelDataCache cacheFor(GGUFModel model) {
        ModelDataCache cache = cachedModelCache(model);
        if (cache != null) {
            return cache;
        }
        synchronized (MODEL_DATA) {
            cache = MODEL_DATA.computeIfAbsent(model, ignored -> new ModelDataCache());
            remember(model, cache);
            return cache;
        }
    }

    private static void remember(GGUFModel model, ModelDataCache cache) {
        lastModelData = new LastModelDataCache(new WeakReference<>(model), cache);
    }

    private static MemorySegment checkedSlice(GGUFModel model, GGUFTensorInfo tensor) {
        long start = model.dataStart() + tensor.offset();
        long end = start + tensor.sizeInBytes();
        if (start < 0 || end < start || end > model.segment().byteSize()) {
            throw new IllegalArgumentException("Tensor data range is outside model segment: " + tensor.name());
        }
        return model.segment().asSlice(start, tensor.sizeInBytes());
    }

    private static final class ModelDataCache {
        private final IdentityHashMap<GGUFTensorInfo, MemorySegment> slices = new IdentityHashMap<>();
        private Map<String, GGUFTensorInfo> tensorsByName;
        private volatile LastNamedTensor lastTensor;
        private volatile LastNamedTensor previousTensor;
        private volatile LastTensorSlice lastSlice;
        private volatile LastTensorSlice previousSlice;

        GGUFTensorInfo tensor(GGUFModel model, String name) {
            LastNamedTensor last = lastTensor;
            if (last != null && last.name().equals(name)) {
                return last.tensor();
            }
            LastNamedTensor previous = previousTensor;
            if (previous != null && previous.name().equals(name)) {
                return previous.tensor();
            }
            return cachedTensor(model, name);
        }

        private synchronized GGUFTensorInfo cachedTensor(GGUFModel model, String name) {
            Map<String, GGUFTensorInfo> index = tensorsByName;
            if (index == null) {
                index = tensorIndex(model);
                tensorsByName = index;
            }
            GGUFTensorInfo tensor = index.get(name);
            if (tensor != null) {
                rememberTensor(name, tensor);
            }
            return tensor;
        }

        MemorySegment slice(GGUFModel model, GGUFTensorInfo tensor) {
            LastTensorSlice last = lastSlice;
            if (last != null && last.tensor() == tensor) {
                return last.slice();
            }
            LastTensorSlice previous = previousSlice;
            if (previous != null && previous.tensor() == tensor) {
                return previous.slice();
            }
            return cachedSlice(model, tensor);
        }

        private void rememberTensor(String name, GGUFTensorInfo tensor) {
            LastNamedTensor last = lastTensor;
            if (last != null && last.name().equals(name)) {
                return;
            }
            if (last != null) {
                previousTensor = last;
            }
            lastTensor = new LastNamedTensor(name, tensor);
        }

        private synchronized MemorySegment cachedSlice(GGUFModel model, GGUFTensorInfo tensor) {
            MemorySegment slice = slices.get(tensor);
            if (slice == null) {
                slice = checkedSlice(model, tensor);
                slices.put(tensor, slice);
            }
            rememberSlice(tensor, slice);
            return slice;
        }

        private void rememberSlice(GGUFTensorInfo tensor, MemorySegment slice) {
            LastTensorSlice last = lastSlice;
            if (last != null && last.tensor() == tensor) {
                return;
            }
            if (last != null) {
                previousSlice = last;
            }
            lastSlice = new LastTensorSlice(tensor, slice);
        }

        private static Map<String, GGUFTensorInfo> tensorIndex(GGUFModel model) {
            Map<String, GGUFTensorInfo> index = new HashMap<>(Math.max(16, model.tensors().size()));
            for (GGUFTensorInfo tensor : model.tensors()) {
                index.putIfAbsent(tensor.name(), tensor);
            }
            return index;
        }
    }

    private record LastNamedTensor(
            String name,
            GGUFTensorInfo tensor) {
    }

    private record LastTensorSlice(
            GGUFTensorInfo tensor,
            MemorySegment slice) {
    }

    private static final class RecentSlices {
        private final RecentSlice[] slices = new RecentSlice[RECENT_SLICE_SLOTS];
        private RecentSlice last;

        private MemorySegment get(GGUFModel model, GGUFTensorInfo tensor) {
            MemorySegment slice = sliceIfMatches(last, model, tensor);
            if (slice != null) {
                return slice;
            }
            int slot = slot(model, tensor);
            RecentSlice recent = slices[slot];
            slice = sliceIfMatches(recent, model, tensor);
            if (slice != null) {
                last = recent;
                return slice;
            }
            if (recent != null && recentExpired(recent)) {
                slices[slot] = null;
            }
            return null;
        }

        private void put(GGUFModel model, GGUFTensorInfo tensor, MemorySegment slice) {
            RecentSlice recent = new RecentSlice(
                    new WeakReference<>(model),
                    tensor,
                    new WeakReference<>(slice));
            last = recent;
            slices[slot(model, tensor)] = recent;
        }

        private int size() {
            int size = 0;
            for (int i = 0; i < slices.length; i++) {
                RecentSlice recent = slices[i];
                if (recent == null) {
                    continue;
                }
                if (recent.model().get() == null || recent.slice().get() == null) {
                    slices[i] = null;
                } else {
                    size++;
                }
            }
            return size;
        }

        private int fastSize() {
            if (last == null) {
                return 0;
            }
            if (recentExpired(last)) {
                last = null;
                return 0;
            }
            return 1;
        }

        private void clear() {
            last = null;
            for (int i = 0; i < slices.length; i++) {
                slices[i] = null;
            }
        }

        private static MemorySegment sliceIfMatches(RecentSlice recent, GGUFModel model, GGUFTensorInfo tensor) {
            if (recent == null || recent.tensor() != tensor) {
                return null;
            }
            GGUFModel cachedModel = recent.model().get();
            MemorySegment slice = recent.slice().get();
            if (cachedModel == model && slice != null) {
                return slice;
            }
            return null;
        }

        private static boolean recentExpired(RecentSlice recent) {
            return recent.model().get() == null || recent.slice().get() == null;
        }

        private static int slot(GGUFModel model, GGUFTensorInfo tensor) {
            int modelHash = System.identityHashCode(model);
            int tensorHash = System.identityHashCode(tensor);
            return (31 * modelHash + tensorHash) & RECENT_SLICE_MASK;
        }
    }

    private record RecentSlice(
            WeakReference<GGUFModel> model,
            GGUFTensorInfo tensor,
            WeakReference<MemorySegment> slice) {
    }

    private record LastModelDataCache(
            WeakReference<GGUFModel> model,
            ModelDataCache cache) {
        static LastModelDataCache empty() {
            return new LastModelDataCache(new WeakReference<>(null), null);
        }
    }
}
