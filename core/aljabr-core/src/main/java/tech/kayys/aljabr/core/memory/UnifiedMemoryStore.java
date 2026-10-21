package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Optional;

/**
 * Unified abstract interface for zero-copy off-heap memory storage across the
 * ecosystem.
 * Backed by native storage engines like RocksDB or HelixDB.
 * @author bhangun
 */
public interface UnifiedMemoryStore extends AutoCloseable {

    /**
     * Stores data directly from a MemorySegment into the off-heap DB.
     */
    void put(byte[] key, MemorySegment valueSegment);

    /**
     * Retrieves the data as a zero-copy MemorySegment pinned to a specific Arena
     * lifecycle.
     * This memory segment can be passed directly to Aljabr's FFM native bindings or
     * Tafkir.
     */
    Optional<MemorySegment> getZeroCopy(byte[] key, long expectedSize, Arena arena);

    /**
     * Deletes the data associated with the key.
     */
    void delete(byte[] key);

    /**
     * Flushes the store.
     */
    void flush();
}
