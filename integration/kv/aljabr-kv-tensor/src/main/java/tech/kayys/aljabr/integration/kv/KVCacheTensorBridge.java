package tech.kayys.aljabr.integration.kv;

import java.util.Objects;

import tech.kayys.aljabr.cache.KVCacheSession;
import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.DType;
import tech.kayys.aljabr.core.tensor.DefaultTensor;
import tech.kayys.aljabr.core.tensor.DeviceType;
import tech.kayys.aljabr.core.tensor.Shape;

/**
 * Materializes committed KV-cache ranges as ordinary host-backed Aljabr tensors.
 *
 * <p>This is an explicit copy boundary, not a zero-copy adapter. The cache
 * contract currently exposes range reads into Java arrays; this bridge keeps
 * that contract honest while allowing tensor consumers to use a stable
 * snapshot independent of subsequent cache mutation or release.</p>
 */
public final class KVCacheTensorBridge {
    private KVCacheTensorBridge() {
    }

    /**
     * Copies one layer's committed K/V range into independently owned F32 CPU tensors.
     *
     * @param session sequence-scoped cache session
     * @param layer transformer layer index
     * @param firstPosition first committed sequence position to copy
     * @param tokenCount number of positions to copy; must be positive
     * @param backend backend metadata used by the resulting tensors
     */
    public static KVCacheTensorSnapshot snapshot(
            KVCacheSession session,
            int layer,
            int firstPosition,
            int tokenCount,
            ComputeBackend backend) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(backend, "backend");
        if (firstPosition < 0 || tokenCount <= 0
                || firstPosition > session.committedLength()
                || tokenCount > session.committedLength() - firstPosition) {
            throw new IllegalArgumentException(
                    "Snapshot range must be a non-empty subset of the committed cache prefix");
        }

        int heads = session.layout().kvHeads(layer);
        int headDimension = session.layout().headDimension(layer);
        int elements = Math.multiplyExact(tokenCount, session.layout().layerStride(layer));
        float[] keyValues = new float[elements];
        float[] valueValues = new float[elements];
        session.readKeyRange(layer, firstPosition, tokenCount, keyValues, 0);
        session.readValueRange(layer, firstPosition, tokenCount, valueValues, 0);

        Shape shape = new Shape(tokenCount, heads, headDimension);
        DefaultTensor keys = createTensor(keyValues, shape, backend);
        try {
            DefaultTensor values = createTensor(valueValues, shape, backend);
            return new KVCacheTensorSnapshot(keys, values, layer, firstPosition, tokenCount);
        } catch (RuntimeException | Error failure) {
            keys.release();
            throw failure;
        }
    }

    private static DefaultTensor createTensor(float[] values, Shape shape, ComputeBackend backend) {
        OwnedHostStorage storage = new OwnedHostStorage(values);
        try {
            return new DefaultTensor(shape, DType.F32, DeviceType.CPU, storage, backend);
        } catch (RuntimeException | Error failure) {
            storage.release();
            throw failure;
        }
    }
}
