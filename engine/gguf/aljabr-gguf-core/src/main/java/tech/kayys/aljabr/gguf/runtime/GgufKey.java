package tech.kayys.aljabr.gguf.runtime;

import tech.kayys.aljabr.gguf.loader.GGUFTensorInfo;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;

/**
 * Stable tensor identity shared by GGUF prepared-matrix caches and estimate caches.
 * @author bhangun
 */
final class GgufKey {
    private static final int RECENT_KEY_SLOTS = 256;
    private static final int RECENT_KEY_MASK = RECENT_KEY_SLOTS - 1;
    private static final ThreadLocal<RecentKeys> RECENT_KEYS = ThreadLocal.withInitial(RecentKeys::new);
    private static final Map<GGUFTensorInfo, TensorKeys> KEYS = new WeakHashMap<>();
    private static volatile LastKey lastKey = LastKey.empty();

    private final String name;
    private final int typeId;
    private final long offset;
    private final long sizeInBytes;
    private final long columns;
    private final long rows;
    private final int hash;

    private GgufKey(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
        this.name = name;
        this.typeId = typeId;
        this.offset = offset;
        this.sizeInBytes = sizeInBytes;
        this.columns = columns;
        this.rows = rows;
        this.hash = computeHash(name, typeId, offset, sizeInBytes, columns, rows);
    }

    static GgufKey from(GGUFTensorInfo tensor) {
        return from(tensor, GgufTensorShape.matrixColumns(tensor), GgufTensorShape.matrixRows(tensor));
    }

    static GgufKey from(GGUFTensorInfo tensor, long columns, long rows) {
        RecentKeys recent = RECENT_KEYS.get();
        GgufKey key = recent.get(tensor, columns, rows);
        if (key != null) {
            return key;
        }
        LastKey last = lastKey;
        GGUFTensorInfo cachedTensor = last.tensor().get();
        key = last.key();
        if (cachedTensor == tensor && key != null && key.matches(columns, rows)) {
            recent.put(tensor, key);
            return key;
        }
        if (cachedTensor == null && key != null) {
            lastKey = LastKey.empty();
        }
        synchronized (KEYS) {
            TensorKeys keys = KEYS.computeIfAbsent(tensor, ignored -> new TensorKeys());
            key = keys.getOrCreate(tensor, columns, rows);
            lastKey = new LastKey(new WeakReference<>(tensor), key);
            recent.put(tensor, key);
            return key;
        }
    }

    static int recentKeyCacheSize() {
        return RECENT_KEYS.get().size();
    }

    static int recentKeyFastCacheSize() {
        return RECENT_KEYS.get().fastSize();
    }

    static void clearCaches() {
        RECENT_KEYS.get().clear();
        synchronized (KEYS) {
            KEYS.clear();
        }
        lastKey = LastKey.empty();
    }

    private static GgufKey create(GGUFTensorInfo tensor, long columns, long rows) {
        return new GgufKey(
                tensor.name(),
                tensor.typeId(),
                tensor.offset(),
                tensor.sizeInBytes(),
                columns,
                rows);
    }

    private boolean matches(long columns, long rows) {
        return this.columns == columns && this.rows == rows;
    }

    @Override
    public boolean equals(Object value) {
        if (this == value) {
            return true;
        }
        if (!(value instanceof GgufKey other)) {
            return false;
        }
        return typeId == other.typeId
                && offset == other.offset
                && sizeInBytes == other.sizeInBytes
                && columns == other.columns
                && rows == other.rows
                && Objects.equals(name, other.name);
    }

    @Override
    public int hashCode() {
        return hash;
    }

    private static int computeHash(
            String name,
            int typeId,
            long offset,
            long sizeInBytes,
            long columns,
            long rows) {
        int result = name == null ? 0 : name.hashCode();
        result = 31 * result + typeId;
        result = 31 * result + Long.hashCode(offset);
        result = 31 * result + Long.hashCode(sizeInBytes);
        result = 31 * result + Long.hashCode(columns);
        result = 31 * result + Long.hashCode(rows);
        return result;
    }

    private static final class RecentKeys {
        private final GGUFTensorInfo[] tensors = new GGUFTensorInfo[RECENT_KEY_SLOTS];
        private final TensorKeys[] keys = new TensorKeys[RECENT_KEY_SLOTS];
        private GGUFTensorInfo lastTensor;
        private TensorKeys lastKeys;

        private GgufKey get(GGUFTensorInfo tensor, long columns, long rows) {
            if (lastTensor == tensor && lastKeys != null) {
                return lastKeys.find(columns, rows);
            }
            int slot = slot(tensor);
            TensorKeys tensorKeys = tensors[slot] == tensor ? keys[slot] : null;
            if (tensorKeys != null) {
                lastTensor = tensor;
                lastKeys = tensorKeys;
            }
            return tensorKeys == null ? null : tensorKeys.find(columns, rows);
        }

        private void put(GGUFTensorInfo tensor, GgufKey key) {
            int slot = slot(tensor);
            TensorKeys tensorKeys = tensors[slot] == tensor ? keys[slot] : null;
            if (tensorKeys == null) {
                tensorKeys = new TensorKeys();
                tensors[slot] = tensor;
                keys[slot] = tensorKeys;
            }
            lastTensor = tensor;
            lastKeys = tensorKeys;
            tensorKeys.remember(key);
        }

        private int size() {
            int size = 0;
            for (GGUFTensorInfo tensor : tensors) {
                if (tensor != null) {
                    size++;
                }
            }
            return size;
        }

        private int fastSize() {
            return lastTensor == null ? 0 : 1;
        }

        private void clear() {
            lastTensor = null;
            lastKeys = null;
            for (int index = 0; index < tensors.length; index++) {
                tensors[index] = null;
                keys[index] = null;
            }
        }

        private static int slot(GGUFTensorInfo tensor) {
            return System.identityHashCode(tensor) & RECENT_KEY_MASK;
        }
    }

    private static final class TensorKeys {
        private GgufKey primary;
        private GgufKey secondary;

        private GgufKey getOrCreate(GGUFTensorInfo tensor, long columns, long rows) {
            GgufKey key = find(columns, rows);
            if (key != null) {
                return key;
            }
            key = create(tensor, columns, rows);
            remember(key);
            return key;
        }

        private GgufKey find(long columns, long rows) {
            if (primary != null && primary.matches(columns, rows)) {
                return primary;
            }
            if (secondary != null && secondary.matches(columns, rows)) {
                return secondary;
            }
            return null;
        }

        private void remember(GgufKey key) {
            if (primary != null && primary.matches(key.columns, key.rows)) {
                return;
            }
            if (secondary != null && secondary.matches(key.columns, key.rows)) {
                return;
            }
            if (primary == null) {
                primary = key;
            } else {
                secondary = key;
            }
        }
    }

    private record LastKey(WeakReference<GGUFTensorInfo> tensor, GgufKey key) {
        static LastKey empty() {
            return new LastKey(new WeakReference<>(null), null);
        }
    }
}
