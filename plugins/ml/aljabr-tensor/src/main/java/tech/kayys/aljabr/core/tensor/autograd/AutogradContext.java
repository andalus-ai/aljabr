package tech.kayys.aljabr.core.tensor.autograd;

/**
 * Thread-local control for recording mathematical operations.
 *
 * <p>Reverse-mode VJP implementations use ordinary Tensor operations to build
 * gradient values. Those operations must not recursively enter the forward
 * graph while a backward pass is running.</p>
 */
public final class AutogradContext {

    private static final ThreadLocal<Integer> RECORDING_PAUSE_DEPTH =
            ThreadLocal.withInitial(() -> 0);

    private AutogradContext() {
    }

    public static boolean isRecordingEnabled() {
        return RECORDING_PAUSE_DEPTH.get() == 0;
    }

    public static Scope pauseRecording() {
        RECORDING_PAUSE_DEPTH.set(RECORDING_PAUSE_DEPTH.get() + 1);
        return new Scope();
    }

    public static final class Scope implements AutoCloseable {
        private boolean closed;

        private Scope() {
        }

        @Override
        public void close() {
            if (!closed) {
                int depth = RECORDING_PAUSE_DEPTH.get() - 1;
                if (depth <= 0) {
                    RECORDING_PAUSE_DEPTH.remove();
                } else {
                    RECORDING_PAUSE_DEPTH.set(depth);
                }
                closed = true;
            }
        }
    }
}
