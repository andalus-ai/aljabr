package tech.kayys.aljabr.diffusion.api;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Caller-owned collection of prepared conditioning data.
 *
 * <p>Results preserve input order and are closed in reverse preparation
 * order. Closing is idempotent. If multiple resources fail to close, the
 * first failure is thrown and later failures are attached as suppressed
 * exceptions.</p>
 */
public final class PreparedDiffusionConditioningSet implements AutoCloseable {

    private final List<DiffusionConditioningData> data;
    private boolean closed;

    PreparedDiffusionConditioningSet(List<? extends DiffusionConditioningData> data) {
        Objects.requireNonNull(data, "data");
        var copy = new ArrayList<DiffusionConditioningData>(data.size());
        for (var item : data) {
            copy.add(Objects.requireNonNull(item, "prepared data"));
        }
        this.data = List.copyOf(copy);
    }

    public List<DiffusionConditioningData> data() {
        return data;
    }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public synchronized void close() throws Exception {
        if (closed) {
            return;
        }
        closed = true;
        Exception failure = null;
        for (int i = data.size() - 1; i >= 0; i--) {
            try {
                data.get(i).close();
            } catch (Exception closeFailure) {
                if (failure == null) {
                    failure = closeFailure;
                } else {
                    failure.addSuppressed(closeFailure);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
