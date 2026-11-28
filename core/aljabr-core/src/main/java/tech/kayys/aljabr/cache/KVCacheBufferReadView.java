package tech.kayys.aljabr.cache;

import java.util.concurrent.CompletionStage;

/**
 * Borrowed backend-aware view of target-owned KV storage.
 *
 * <p>The returned regions are valid only until this view is closed. Closing
 * the view releases the borrow/lease, not the target-owned allocation.
 * Providers must keep the allocation alive and prevent unsafe mutation or
 * relocation for the duration of the lease.</p>
 *
 * <p>This contract does not promise that a consumer can access a region
 * directly. A compatible backend must recognize the opaque handle and memory
 * domain. Cross-backend access requires an explicit transfer and must not be
 * represented as zero-copy.</p>
 */
public interface KVCacheBufferReadView extends AutoCloseable {

    /** Logical cache geometry shared with the ordinary read-view contract. */
    KVCacheLayout layout();

    /** Committed prefix length visible through this lease. */
    int visibleSequenceLength();

    /** Borrowed K region for one layer. */
    KVCacheBufferRegion keyRegion(int layer);

    /** Borrowed V region for one layer. */
    KVCacheBufferRegion valueRegion(int layer);

    /**
     * Logical layout of the borrowed K region. The default describes the
     * sequence-major layout used by the built-in CPU and Metal providers.
     */
    default KVCacheBufferLayout keyLayout(int layer) {
        return KVCacheBufferLayout.sequenceMajor(
                visibleSequenceLength(), layout().kvHeads(layer), layout().headDimension(layer));
    }

    /** Logical layout of the borrowed V region. */
    default KVCacheBufferLayout valueLayout(int layer) {
        return KVCacheBufferLayout.sequenceMajor(
                visibleSequenceLength(), layout().kvHeads(layer), layout().headDimension(layer));
    }

    /**
     * Completes when all target writes represented by this lease are visible
     * to a compatible consumer. Consumers must establish this dependency
     * before submitting work that reads the borrowed regions.
     */
    CompletionStage<Void> readiness();

    /** Backend that owns/exports the borrowed regions. */
    String backendId();

    /** Releases the lease without releasing target-owned storage. */
    @Override
    void close();
}
