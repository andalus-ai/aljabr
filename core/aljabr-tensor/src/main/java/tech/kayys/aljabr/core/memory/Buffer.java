package tech.kayys.aljabr.core.memory;

import java.lang.foreign.Arena;

/**
 * Compatibility contract for FFM-backed, host-accessible tensor buffers.
 *
 * <p>New backend-neutral code should depend on {@link TensorStorage}; code
 * that needs host access should depend on {@link HostAccessibleStorage}.
 * This interface remains for callers that require the legacy {@link Arena}
 * accessor.</p>
 */
public interface Buffer extends HostAccessibleStorage {

    /** Returns the arena associated with this legacy FFM buffer. */
    Arena arena();
}
