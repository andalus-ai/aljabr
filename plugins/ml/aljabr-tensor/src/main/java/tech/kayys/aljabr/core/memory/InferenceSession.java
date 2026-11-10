package tech.kayys.aljabr.core.memory;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.Tensor;

import java.lang.foreign.Arena;
import java.util.function.Function;
import java.util.logging.Logger;

/**
 * A scoped container for a single inference forward-pass.
 *
 * <p>
 * {@code InferenceSession} bundles a {@link ComputeBackend} with a shared
 * {@link OffHeapBufferPool} so that all intermediate tensors (activations,
 * attention scores, norms, etc.) produced during the session share one Arena.
 * When the session is closed, <strong>all</strong> off-heap memory is freed in
 * a single syscall — eliminating per-tensor GC overhead entirely.
 *
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * // 1. Build the backend once (heavy; reuse across sessions).
 * ComputeBackend backend = new CpuBackend();
 *
 * // 2. Create a session for each forward pass.
 * try (InferenceSession session = InferenceSession.of(backend)) {
 *     Tensor output = session.run(model::forward, inputTokens);
 *     float[] logits = output.toFloatArray(); // data must be read inside the try block
 * }
 * // ← every intermediate tensor freed here in O(1)
 * }</pre>
 *
 * <h2>Thread safety</h2>
 * <p>
 * Each session must be used from a single thread. For parallel token
 * generation, create one {@code InferenceSession} per request/thread.
 * @author bhangun
 */
public final class InferenceSession implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(InferenceSession.class.getName());

    private final ComputeBackend backend;
    private final OffHeapBufferPool pool;
    private volatile boolean closed = false;

    private InferenceSession(ComputeBackend sessionBackend, OffHeapBufferPool pool) {
        this.backend = sessionBackend;
        this.pool = pool;
    }

    // ── Factory methods ────────────────────────────────────────────────────────

    /**
     * Creates a session using the given backend.
     *
     * <p>
     * If the backend supports session-scoped pools (i.e. it accepts an
     * {@link OffHeapBufferPool} in its constructor), callers should prefer
     * {@link #of(java.util.function.Function)} so the pool is wired in.
     *
     * <p>
     * This overload wraps {@code backend} directly and creates an internal
     * pool for tracking stats.
     *
     * @param backend the compute backend to use
     * @return a new {@code InferenceSession}
     */
    public static InferenceSession of(ComputeBackend backend) {
        OffHeapBufferPool pool = new OffHeapBufferPool(Arena.ofShared());
        return new InferenceSession(backend, pool);
    }

    /**
     * Creates a session by constructing a pool-aware backend via the supplied
     * factory.
     *
     * <pre>{@code
     * try (InferenceSession session = InferenceSession.of(CpuBackend::new)) {
     *     Tensor output = session.run(model::forward, input);
     * }
     * }</pre>
     *
     * @param backendFactory a function that produces a backend from a pool
     * @return a new {@code InferenceSession}
     */
    public static InferenceSession of(Function<OffHeapBufferPool, ComputeBackend> backendFactory) {
        OffHeapBufferPool pool = new OffHeapBufferPool(Arena.ofShared());
        ComputeBackend backend = backendFactory.apply(pool);
        return new InferenceSession(backend, pool);
    }

    // ── Core API ───────────────────────────────────────────────────────────────

    /**
     * Returns the session-scoped {@link ComputeBackend}.
     *
     * <p>
     * Use this to run individual tensor operations. Results are valid only
     * within the {@code try-with-resources} block.
     *
     * @return the compute backend for this session
     */
    public ComputeBackend backend() {
        ensureOpen();
        return backend;
    }

    /**
     * Runs a model forward pass inside this session and returns the output.
     *
     * <p>
     * The output tensor's memory is still backed by this session's arena.
     * If you need the data to outlive the session, copy it to a heap array
     * before closing:
     * 
     * <pre>{@code
     * float[] logits;
     * try (InferenceSession s = InferenceSession.of(CpuBackend::new)) {
     *     logits = s.run(model::forward, input).toFloatArray();
     * }
     * // logits[] is safe here — it lives on the Java heap.
     * }</pre>
     *
     * @param <I>   input type
     * @param model the model forward function
     * @param input the input tensor
     * @return the model output
     */
    public <I extends Tensor> Tensor run(Function<I, Tensor> model, I input) {
        ensureOpen();
        return model.apply(input);
    }

    /**
     * Returns the underlying memory pool (for passing to sub-operations that
     * accept a pool directly).
     *
     * @return the {@link OffHeapBufferPool} used by this session
     */
    public OffHeapBufferPool pool() {
        ensureOpen();
        return pool;
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    /**
     * Closes the session, freeing all off-heap memory allocated during inference.
     *
     * <p>
     * After calling {@code close()}, any tensor produced within this session
     * must not be accessed.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            LOG.fine(() -> "InferenceSession closing — " + pool.stats());
            pool.close();
        }
    }

    // ── Diagnostics ────────────────────────────────────────────────────────────

    /**
     * Returns a human-readable pool statistics string for this session.
     *
     * @return pool hit/miss/pooled stats
     */
    public String stats() {
        return pool.stats();
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private void ensureOpen() {
        if (closed)
            throw new IllegalStateException("InferenceSession is already closed");
    }
}
